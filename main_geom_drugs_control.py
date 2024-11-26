# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass

import time
import math
import copy
import yaml
import wandb
import torch
import pickle
import random
import argparse
import numpy as np
from os.path import join

import utils
import train_test
import build_geom_dataset
from configs.datasets_config import get_dataset_info
from equivariant_diffusion import utils as diffusion_utils
from equivariant_diffusion import en_diffusion, control_en_diffusion
from qm9.models import get_optim, get_controlled_latent_diffusion
from global_registry import PARAM_REGISTRY, Config



MMSEQ2_SPLIT = "MMseq2_split"

def main():
    parser = argparse.ArgumentParser(description='e3_diffusion')
    parser.add_argument('--config_file', type=str, default='custom_config/base_geom_config.yaml')
    opt = parser.parse_args()

    with open(opt.config_file, 'r') as file:
        args_dict = yaml.safe_load(file)
    
    # args = Config(**args_dict)
    args = Config(
        **{k: 
            Config(**v) if isinstance(v, dict) \
            else v \
            for k, v in args_dict.items()
        }
    )

    # set random seed
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # load pre-computed dataset configs
    ligand_dataset_info = get_dataset_info(dataset_name=args.dataset, remove_h=args.remove_h)
    pocket_dataset_info = get_dataset_info(dataset_name=args.pocket_vae.dataset, remove_h=args.pocket_vae.remove_h)

    # device settings
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    args.device_ = "cuda" if args.cuda else "cpu"

    # dtype settings
    _, dtype_name = args.dtype.split('.')
    dtype = getattr(torch, dtype_name)
    args.dtype = dtype
    torch.set_default_dtype(dtype)

    # add missing configs with default values
    args = utils.add_missing_configs_controlnet(args, dtype, ligand_dataset_info, pocket_dataset_info)

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
                                                    filter_pocket_size=args.filter_pocket_size,
                                                    data_splitted=args.data_splitted,
                                                    return_ids=args.match_raw_file_by_id)
    # ~!to ~!mp
    # ['positions'], ['one_hot'], ['charges'], ['atom_mask'] are added here
    ligand_transform = build_geom_dataset.GeomDrugsTransform(ligand_dataset_info, args.include_charges, args.device, args.sequential)
    pocket_transform = build_geom_dataset.GeomDrugsTransform(pocket_dataset_info, args.pocket_vae.include_charges, args.device, args.sequential)

    ids = {}
    dataloaders = {}
    for key, data_list in zip(['train', 'val', 'test'], split_data):
        dataset = build_geom_dataset.GeomDrugsDataset(
            data_list, 
            transform=ligand_transform, 
            pocket_transform=pocket_transform, 
            training_mode=args.training_mode
        )
        if args.match_raw_file_by_id:
            ids[key] = data_list['ids']
        # shuffle = (key == 'train') and not args.sequential
        shuffle = (key == 'train')

        # Sequential dataloading disabled for now.
        dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
            sequential=args.sequential, dataset=dataset, batch_size=args.batch_size,
            shuffle=shuffle, training_mode=args.training_mode, drop_last=True)
        
        if args.vis_activations and key == 'val':
            dataloaders['vis_activations'] = build_geom_dataset.GeomDrugsDataLoader(
            sequential=args.sequential, dataset=dataset, batch_size=args.vis_activations_batch_size,
            shuffle=False, training_mode=args.training_mode, drop_last=True)

    del split_data

    # additional data extracted for stability and quickvina tests on controlnet
    split = args.n_stability_eval_split   # train, test, val
    # Ligands' & Pockets' ['positions'], ['one_hot'], ['charges'], ['atom_mask'] are already available
    controlnet_eval_datalist = [copy.deepcopy(dataloaders[split].dataset[idx]) for idx in range(args.n_stability_samples)]
    if args.match_raw_file_by_id:
        controlnet_eval_datalist_ids = [ids[split][idx] for idx in range(args.n_stability_samples)]
    else:
        controlnet_eval_datalist_ids = []

    # args, unparsed_args = parser.parse_known_args()

    # resume
    if args.resume is not None:
        exp_name = args.exp_name + '_resume'
        args.exp_name = exp_name
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


    # Create Control-LDM
    model, nodes_dist, prop_dist = get_controlled_latent_diffusion(args, args.device, ligand_dataset_info, pocket_dataset_info, dataloaders['train'])

    # move model to gpu
    model = model.to(args.device)

    # set optimiser
    optim = get_optim(args, model)

    # load model weights if resume training
    if args.resume is not None:
        if args.resume_model_ckpt is not None:
            model_state_dict = join(args.resume, args.resume_model_ckpt)
        else:
            if args.ema_decay > 0:
                model_state_dict = join(args.resume, 'generative_model_ema.npy')
            else:
                model_state_dict = join(args.resume, 'generative_model.npy')

        if args.resume_optim_ckpt is not None:
            optim_state_dict = join(args.resume, args.resume_optim_ckpt)
        else:
            optim_state_dict = join(args.resume, 'optim.npy')

        print(f">> Loading {args.training_mode} weights from {model_state_dict}")
        print(f">> Loading Optimizer State Dict from {optim_state_dict}")
        model.load_state_dict(torch.load(model_state_dict))
        optim.load_state_dict(torch.load(optim_state_dict))
        # dequantizer_state_dict = torch.load(join(args.resume, 'dequantizer.npy'))

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

    # gradient norm tracking for clipping
    gradnorm_queue = utils.Queue()
    gradnorm_queue.add(3000)  # Add large value that will be flushed.

    # model details logging
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes
    mem_mb, mem_gb = mem/(1024**2), mem/(1024**3)
    print(f"Model running on device  : {args.device}")
    print(f"Mixed precision training : {args.mixed_precision_training}")
    print(f"Mixed precision autocast dtype : {args.mixed_precision_autocast_dtype}") if args.mixed_precision_training else None
    print(f"Model running on dtype   : {args.dtype}")
    print(f"Model Size               : {mem_gb} GB  /  {mem_mb} MB  /  {mem} Bytes")
    print(f"Training Dataset Name    : {args.dataset}")
    print(f"Model Training Mode      : {args.training_mode}")
    print(f"Fusion Blocks Zero Weight: {args.zero_fusion_block_weights}")
    print(f"================================")
    print(model)
    
    
    
    best_nll_val = math.inf
    best_nll_test = math.inf
    nth_iter = 0
    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()
        n_iters = train_test.train_epoch_controlnet(args, dataloaders['train'], None, epoch, model, model_dp, model_ema, ema, device, dtype,
                                                    args.property_norms, optim, nodes_dist, gradnorm_queue, ligand_dataset_info,
                                                    prop_dist)

        print(f">>> Epoch took {time.time() - start_epoch:.1f} seconds.")
        nth_iter += n_iters

        if epoch % args.test_epochs == 0:
            if isinstance(model, en_diffusion.EnVariationalDiffusion) or isinstance(model, control_en_diffusion.ControlEnLatentDiffusion):
                wandb.log(model.log_info(), commit=True)

            if not args.break_train_epoch:
                start  = time.time()
                print(">>> Entering analyze_and_save")
                train_test.analyze_and_save_controlnet(epoch, model_ema, nodes_dist, args, device, ligand_dataset_info, prop_dist,
                                                        n_samples=args.n_stability_samples, batch_size=args.n_stability_samples_batch_size, 
                                                        pair_dict_list=controlnet_eval_datalist, pair_dict_list_ids=controlnet_eval_datalist_ids,
                                                        output_dir=f"outputs/{args.exp_name}/analysis/epoch_{epoch}_iter_{nth_iter}")
                print(f">>> analyze_and_save took {time.time() - start:.1f} seconds.")

            start  = time.time()
            nll_val = train_test.test_controlnet(args, dataloaders['val'], epoch, model_ema_dp, device, dtype,
                                                    args.property_norms, nodes_dist, partition='Val')
            print(f">>> validation set test took {time.time() - start:.1f} seconds.")

            # MMseq2 only has train:val set (test & val are the same, hence skip)
            if MMSEQ2_SPLIT not in str(args.data_file):
                start  = time.time()
                nll_test = train_test.test_controlnet(args, dataloaders['test'], epoch, model_ema_dp, device, dtype,
                                                    args.property_norms, nodes_dist, partition='Test')
                print(f">>> testing set test took {time.time() - start:.1f} seconds.")
            else:
                nll_test = None


            if nll_val < best_nll_val:
                best_nll_val = nll_val
                if MMSEQ2_SPLIT not in str(args.data_file):
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
            if MMSEQ2_SPLIT not in str(args.data_file):
                print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
                print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))
            else:
                print('Val loss: %.4f' % (nll_val))
                print('Best val loss: %.4f' % (best_nll_val))
            wandb.log({"Val loss ": nll_val}, commit=True)
            wandb.log({"Test loss ": nll_test}, commit=True) if MMSEQ2_SPLIT not in str(args.data_file) else None
            wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True) if MMSEQ2_SPLIT not in str(args.data_file) else None


if __name__ == "__main__":
    main()
