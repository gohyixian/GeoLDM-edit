# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import build_geom_dataset
from configs.datasets_config import geom_with_h
import copy
import utils
import yaml
import argparse
import wandb
from os.path import join
from qm9.models import get_optim, get_model, get_autoencoder, get_latent_diffusion
from equivariant_diffusion import en_diffusion

from equivariant_diffusion import utils as diffusion_utils
import torch
import time
import pickle

from qm9.utils import prepare_context, compute_mean_mad
import train_test

from global_registry import PARAM_REGISTRY, Config




def main():
    parser = argparse.ArgumentParser(description='e3_diffusion')
    parser.add_argument('--config_file', type=str, default='custom_config/base_geom_config.yaml')
    opt = parser.parse_args()

    with open(opt.config_file, 'r') as file:
        args_dict = yaml.safe_load(file)
    args = Config(**args_dict)
    

    # priority check goes here
    if args.remove_h:
        raise NotImplementedError()
    else:
        dataset_info = geom_with_h
    

    # device settings
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    args.device_ = "cuda" if args.cuda else "cpu"
    print(f">> Model running on device {args.device}")
    
    
    # mixed precision training
    # ~!mp
    if args.mixed_precision_training:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        print(f">> Mixed precision training enabled")
    else:
        scaler = None
        print(f">> Mixed precision training disabled")
    

    # dtype settings
    module_name, dtype_name = args.dtype.split('.')
    dtype = getattr(torch, dtype_name)
    args.dtype = dtype
    torch.set_default_dtype(dtype)
    print(f">> Model running on dtype {args.dtype}")
    
    
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
                                                    dataset_name=args.dataset)
    # ~!to ~!mp
    transform = build_geom_dataset.GeomDrugsTransform(dataset_info, args.include_charges, args.device, args.sequential)
    # transform = build_geom_dataset.GeomDrugsTransform(dataset_info, args.include_charges, torch.device("cpu"), args.sequential)

    dataloaders = {}
    for key, data_list in zip(['train', 'val', 'test'], split_data):
        dataset = build_geom_dataset.GeomDrugsDataset(data_list, transform=transform)
        shuffle = (key == 'train') and not args.sequential

        # Sequential dataloading disabled for now.
        dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
            sequential=args.sequential, dataset=dataset, batch_size=args.batch_size,
            shuffle=shuffle)
    del split_data


    # # not used
    # atom_encoder = dataset_info['atom_encoder']
    # atom_decoder = dataset_info['atom_decoder']


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
    kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'e3_diffusion_geom', 'config': args,
            'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
    wandb.init(**kwargs)
    wandb.save('*.txt')


    data_dummy = next(iter(dataloaders['train']))
    if len(args.conditioning) > 0:
        print(f'Conditioning on {args.conditioning}')
        property_norms = compute_mean_mad(dataloaders, args.conditioning)
        context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
        context_node_nf = context_dummy.size(2)
    else:
        context_node_nf = 0
        property_norms = None

    args.context_node_nf = context_node_nf


    # Create Latent Diffusion Model or Audoencoder
    if args.train_diffusion:
        model, nodes_dist, prop_dist = get_latent_diffusion(args, args.device, dataset_info, dataloaders['train'])
    else:
        model, nodes_dist, prop_dist = get_autoencoder(args, args.device, dataset_info, dataloaders['train'])

    model = model.to(args.device)
    optim = get_optim(args, model)
    # print(model)


    gradnorm_queue = utils.Queue(dtype=args.dtype)
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
    
    
    best_nll_val = 1e8
    best_nll_test = 1e8
    nth_iter = 0
    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()
        n_iters = train_test.train_epoch(args, dataloaders['train'], epoch, model, model_dp, model_ema, ema, device, dtype,
                               property_norms, optim, nodes_dist, gradnorm_queue, dataset_info,
                               prop_dist, scaler=scaler)
        print(f">>> Epoch took {time.time() - start_epoch:.1f} seconds.")
        nth_iter += n_iters

        if epoch % args.test_epochs == 0:
            if isinstance(model, en_diffusion.EnVariationalDiffusion):
                wandb.log(model.log_info(), commit=True)
            
            if not args.break_train_epoch and args.train_diffusion:
                start  = time.time()
                train_test.analyze_and_save(epoch, model_ema, nodes_dist, args, device,
                                            dataset_info, prop_dist, n_samples=args.n_stability_samples)
                print(f">>> analyze_and_save took {time.time() - start:.1f} seconds.")
                
            start  = time.time()
            nll_val = train_test.test(args, dataloaders['val'], epoch, model_ema_dp, device, dtype,
                                      property_norms, nodes_dist, partition='Val')
            print(f">>> validation set test took {time.time() - start:.1f} seconds.")
            
            start  = time.time()
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
