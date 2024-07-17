# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import build_geom_dataset
from configs.datasets_config import geom_with_h, get_dataset_info
import copy
import utils
import yaml
import argparse
import wandb
from os.path import join
from qm9.models import get_optim, get_model, get_autoencoder, get_latent_diffusion, get_controlled_latent_diffusion
from equivariant_diffusion import en_diffusion, control_en_diffusion

from equivariant_diffusion import utils as diffusion_utils
import torch
import time
import pickle
from copy import deepcopy

from qm9.utils import prepare_context, compute_mean_mad
import train_test
from global_registry import PARAM_REGISTRY, Config


CONTROLNET = 'ControlNet'
LDM = "LDM"
VAE = "VAE"


def main():
    parser = argparse.ArgumentParser(description='e3_diffusion')
    parser.add_argument('--config_file', type=str, default='custom_config/base_geom_config.yaml')
    opt = parser.parse_args()

    with open(opt.config_file, 'r') as file:
        args_dict = yaml.safe_load(file)
    args = Config(**args_dict)


    # # priority check goes here
    # if args.remove_h:
    #     raise NotImplementedError()
    # else:
    #     dataset_info = geom_with_h
    dataset_info = get_dataset_info(dataset_name=args.dataset, remove_h=args.remove_h)


    # device settings
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    args.device_ = "cuda" if args.cuda else "cpu"


    # dtype settings
    module_name, dtype_name = args.dtype.split('.')
    dtype = getattr(torch, dtype_name)
    args.dtype = dtype
    torch.set_default_dtype(dtype)


    # params global registry for easy access
    PARAM_REGISTRY.update_from_config(args)


    # pre-computed data file
    data_file = args.data_file
    print(">> Loading data from:", data_file)
    split_data = build_geom_dataset.load_split_data(data_file, 
                                                    val_proportion=0.1, 
                                                    test_proportion=0.1, 
                                                    filter_size=args.filter_molecule_size, 
                                                    permutation_file_path=args.permutation_file_path, 
                                                    dataset_name=args.dataset,
                                                    training_mode=args.training_mode,
                                                    filter_pocket_size=args.filter_pocket_size)
    # ~!to ~!mp
    # ['positions'], ['one_hot'], ['charges'], ['atom_mask'] are added here
    transform = build_geom_dataset.GeomDrugsTransform(dataset_info, args.include_charges, args.device, args.sequential)

    dataloaders = {}
    for key, data_list in zip(['train', 'val', 'test'], split_data):
        dataset = build_geom_dataset.GeomDrugsDataset(data_list, transform=transform, training_mode=args.training_mode)
        # shuffle = (key == 'train') and not args.sequential
        shuffle = (key == 'train')

        # Sequential dataloading disabled for now.
        dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
            sequential=args.sequential, dataset=dataset, batch_size=args.batch_size,
            shuffle=shuffle, training_mode=args.training_mode, drop_last=True)
    del split_data

    # additional data extracted for stability and quickvina tests on controlnet
    if args.training_mode == CONTROLNET:
        split = args.n_stability_eval_split
        # Ligands' & Pockets' ['positions'], ['one_hot'], ['charges'], ['atom_mask'] are already available
        controlnet_eval_datalist = [deepcopy(dataloaders[split].dataset[idx]) for idx in range(args.n_stability_samples)]


    atom_encoder = dataset_info['atom_encoder']
    atom_decoder = dataset_info['atom_decoder']

    # args, unparsed_args = parser.parse_known_args()
    args.wandb_usr = utils.get_wandb_username(args.wandb_usr)


    # resume
    if args.resume is not None:
        exp_name = args.exp_name + '_resume'
        start_epoch = args.start_epoch
        resume = args.resume
        wandb_usr = args.wandb_usr

        with open(join(args.resume, 'args.pickle'), 'rb') as f:
            args = pickle.load(f)
        args.resume = resume
        args.break_train_epoch = False
        args.exp_name = exp_name
        args.start_epoch = start_epoch
        args.wandb_usr = wandb_usr

    utils.create_folders(args)


    # Wandb config
    if args.no_wandb:
        mode = 'disabled'
    else:
        mode = 'online' if args.online else 'offline'
    proj_name = args.proj_name if hasattr(args, 'proj_name') else 'e3_diffusion_geom'
    kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': proj_name, 'config': args,
            'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
    wandb.init(**kwargs)
    wandb.save('*.txt')


    if len(args.conditioning) > 0:
        raise NotImplementedError()
        # print(f'Conditioning on {args.conditioning}')
        # data_dummy = next(iter(dataloaders['train']))
        # property_norms = compute_mean_mad(dataloaders, args.conditioning)
        # context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
        # context_node_nf = context_dummy.size(2)
    else:
        context_node_nf = 0
        property_norms = None

    args.context_node_nf = context_node_nf


    # Create Control-LDM, Latent Diffusion Model or Autoencoder
    if args.training_mode == CONTROLNET:
        model, nodes_dist, prop_dist = get_controlled_latent_diffusion(args, args.device, dataset_info, dataloaders['train'])
    elif args.training_mode == LDM:
        model, nodes_dist, prop_dist = get_latent_diffusion(args, args.device, dataset_info, dataloaders['train'])
    elif args.training_mode == VAE:
        model, nodes_dist, prop_dist = get_autoencoder(args, args.device, dataset_info, dataloaders['train'])
    else:
        raise NotImplementedError()

    model = model.to(args.device)
    optim = get_optim(args, model)


    gradnorm_queue = utils.Queue()
    gradnorm_queue.add(3000)  # Add large value that will be flushed.
    
    if args.resume is not None:
        flow_state_dict = torch.load(join(args.resume, 'flow.npy'))
        dequantizer_state_dict = torch.load(join(args.resume, 'dequantizer.npy'))
        optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)

    # Initialize dataparallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1 and args.cuda:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = diffusion_utils.EMA(args.ema_decay)

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp
    
    
    # model details logging
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes
    mem_mb, mem_gb = mem/(1024**2), mem/(1024**3)
    print(f"Model running on device  : {args.device}")
    # print(f"Mixed precision training : {args.mixed_precision_training}")
    print(f"Model running on dtype   : {args.dtype}")
    print(f"Model Size               : {mem_gb} GB  /  {mem_mb} MB  /  {mem} Bytes")
    print(f"Training Dataset Name    : {args.dataset}")
    print(f"Model Training Mode      : {args.training_mode}")
    print(f"Fusion Blocks Zero Weight: {args.zero_fusion_block_weights}") if args.training_mode == CONTROLNET else None
    print(f"================================")
    print(model)
    
    
    
    best_nll_val = 1e8
    best_nll_test = 1e8
    nth_iter = 0
    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()
        if args.training_mode == CONTROLNET:
            n_iters = train_test.train_epoch_controlnet(args, dataloaders['train'], epoch, model, model_dp, model_ema, ema, device, dtype,
                                                        property_norms, optim, nodes_dist, gradnorm_queue, dataset_info,
                                                        prop_dist)
        else:
            n_iters = train_test.train_epoch(args, dataloaders['train'], epoch, model, model_dp, model_ema, ema, device, dtype,
                                             property_norms, optim, nodes_dist, gradnorm_queue, dataset_info,
                                             prop_dist)
        print(f">>> Epoch took {time.time() - start_epoch:.1f} seconds.")
        nth_iter += n_iters

        if epoch % args.test_epochs == 0:
            if isinstance(model, en_diffusion.EnVariationalDiffusion) or isinstance(model, control_en_diffusion.ControlEnLatentDiffusion):
                wandb.log(model.log_info(), commit=True)

            if not args.break_train_epoch and args.training_mode in [LDM, CONTROLNET]:
                start  = time.time()
                print(">>> Entering analyze_and_save")
                if args.training_mode == CONTROLNET:
                    train_test.analyze_and_save_controlnet(epoch, model_ema, nodes_dist, args, device, dataset_info, prop_dist,
                                                           n_samples=args.n_stability_samples, batch_size=args.n_stability_samples_batch_size, 
                                                           pair_dict_list=controlnet_eval_datalist)
                else:
                    train_test.analyze_and_save(epoch, model_ema, nodes_dist, args, device,
                                                dataset_info, prop_dist, n_samples=args.n_stability_samples)
                print(f">>> analyze_and_save took {time.time() - start:.1f} seconds.")

            start  = time.time()
            if args.training_mode == CONTROLNET:
                nll_val = train_test.test_controlnet(args, dataloaders['val'], epoch, model_ema_dp, device, dtype,
                                                     property_norms, nodes_dist, partition='Val')
            else:
                nll_val = train_test.test(args, dataloaders['val'], epoch, model_ema_dp, device, dtype,
                                          property_norms, nodes_dist, partition='Val')
            print(f">>> validation set test took {time.time() - start:.1f} seconds.")

            start  = time.time()
            if args.training_mode == CONTROLNET:
                nll_test = train_test.test_controlnet(args, dataloaders['test'], epoch, model_ema_dp, device, dtype,
                                                      property_norms, nodes_dist, partition='Test')
            else:
                nll_test = train_test.test(args, dataloaders['test'], epoch, model_ema_dp, device, dtype,
                                           property_norms, nodes_dist, partition='Test')
            print(f">>> testing set test took {time.time() - start:.1f} seconds.")


            if nll_val < best_nll_val:
                best_nll_val = nll_val
                best_nll_test = nll_test
                if args.save_model:
                    args.current_epoch = epoch + 1
                    utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                    utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                    with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)

            if args.save_model:
                utils.save_model(optim, 'outputs/%s/optim_%d_iter_%d.npy' % (args.exp_name, epoch, nth_iter))
                utils.save_model(model, 'outputs/%s/generative_model_%d_iter_%d.npy' % (args.exp_name, epoch, nth_iter))
                if args.ema_decay > 0:
                    utils.save_model(model_ema, 'outputs/%s/generative_model_ema_%d_iter_%d.npy' % (args.exp_name, epoch, nth_iter))
                with open('outputs/%s/args_%d_iter_%d.pickle' % (args.exp_name, epoch, nth_iter), 'wb') as f:
                    pickle.dump(args, f)
            print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
            print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))
            wandb.log({"Val loss ": nll_val}, commit=True)
            wandb.log({"Test loss ": nll_test}, commit=True)
            wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)


if __name__ == "__main__":
    main()
