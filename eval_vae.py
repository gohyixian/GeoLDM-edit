# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import build_geom_dataset
from configs.datasets_config import geom_with_h, get_dataset_info
import utils
import yaml
import os
import re
import argparse
from os.path import join
from qm9.models import get_optim, get_model, get_autoencoder, get_latent_diffusion
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask

import gc
import math
import torch
from torch import nn
import time
import shutil
from tqdm import tqdm
from datetime import datetime
import qm9.utils as qm9utils
from qm9.sampling import sample
from train_test import check_mask_correct
from global_registry import PARAM_REGISTRY, Config



def save_xyz_file(path, one_hot, charges, positions, dataset_info, id_from=0, name='molecule', node_mask=None):
    try:
        os.makedirs(path)
    except OSError:
        pass

    if node_mask is not None:
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [one_hot.size(1)] * one_hot.size(0)

    for batch_i in range(one_hot.size(0)):
        f = open(os.path.join(path, name + '_' + "%03d.xyz" % (batch_i + id_from)), "w")
        f.write("%d\n\n" % atomsxmol[batch_i])
        atoms = torch.argmax(one_hot[batch_i], dim=1)
        # print(one_hot[batch_i], atoms, dataset_info['atom_decoder']) if name=='REC' else None
        n_atoms = int(atomsxmol[batch_i])
        for atom_i in range(n_atoms):
            atom = atoms[atom_i]
            atom = dataset_info['atom_decoder'][atom]
            # print(atom) if name=='REC' else None
            f.write("%s %.9f %.9f %.9f\n" % (atom, positions[batch_i, atom_i, 0], positions[batch_i, atom_i, 1], positions[batch_i, atom_i, 2]))
        f.close()



def main():
    parser = argparse.ArgumentParser(description='e3_diffusion')
    parser.add_argument('--config_file', type=str, default='custom_config/base_geom_config.yaml')
    parser.add_argument('--load_last', action='store_true', help='load weights of model of last epoch')
    parser.add_argument('--n_samples', type=int, default=500, help='number of samples to generate')
    parser.add_argument('--store_samples', action='store_true', help='keeps samples from the same model in previous runs, else deleted')
    opt = parser.parse_args()

    with open(opt.config_file, 'r') as file:
        args_dict = yaml.safe_load(file)
    args = Config(**args_dict)

    dataset_info = get_dataset_info(dataset_name=args.dataset, remove_h=args.remove_h)

    # vae encoder n layers
    if not hasattr(args, 'encoder_n_layers'):
        args.encoder_n_layers = 1


    # additional & override settings for sparsity plots
    args.batch_size = 1  # must be 1 for this script
    args.save_samples_dir = f'eval_vae/{args.exp_name}/{args.training_mode}/'

    # remove activations from previous runs
    if not opt.store_samples:
        if os.path.exists(args.save_samples_dir):
            print(f">>> Removing activations saved from previous runs as {args.save_samples_dir}")
            shutil.rmtree(args.save_samples_dir)

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


    # mp autocast dtype
    if args.mixed_precision_training == True:
        _, mp_dtype_name = args.mixed_precision_autocast_dtype.split('.')
        mp_dtype = getattr(torch, mp_dtype_name)
        args.mixed_precision_autocast_dtype = mp_dtype
    else:
        args.mixed_precision_autocast_dtype = dtype


    # gradient accumulation
    if not hasattr(args, 'grad_accumulation_steps'):
        args.grad_accumulation_steps = 1  # call optim every step


    # vae data mode
    if not hasattr(args, 'vae_data_mode'):
        args.vae_data_mode = 'all'


    # loss analysis
    if not hasattr(args, 'loss_analysis'):
        args.loss_analysis = False
    args.loss_analysis_modes = ['VAE']


    # loss analysis usage
    atom_encoder = dataset_info['atom_encoder']
    atom_decoder = dataset_info['atom_decoder']
    args.atom_encoder = atom_encoder
    args.atom_decoder = atom_decoder


    # intermediate activations analysis usage
    args.vis_activations_instances = (nn.Linear)
    args.save_activations_path = 'vis_activations'
    args.vis_activations_bins = 200
    if not hasattr(args, 'vis_activations_specific_ylim'):
        args.vis_activations_specific_ylim = [0, 40]
    if not hasattr(args, 'vis_activations'):
        args.vis_activations = False
    if not hasattr(args, 'vis_activations_batch_samples'):
        args.vis_activations_batch_samples = 0
    if not hasattr(args, 'vis_activations_batch_size'):
        args.vis_activations_batch_size = 1


    # class-imbalance loss reweighting
    if not hasattr(args, 'reweight_class_loss'):  # supported: "inv_class_freq"
        args.reweight_class_loss = None
    if args.reweight_class_loss == "inv_class_freq":
        class_freq_dict = dataset_info['atom_types']
        sorted_keys = sorted(class_freq_dict.keys())
        frequencies = torch.tensor([class_freq_dict[key] for key in sorted_keys], dtype=args.dtype)
        inverse_frequencies = 1.0 / frequencies
        class_weights = inverse_frequencies / inverse_frequencies.sum()  # normalize
        args.class_weights = class_weights
        [print(f"{atom_decoder[sorted_keys[i]]} freq={class_freq_dict[sorted_keys[i]]} \
            inv_freq={inverse_frequencies[i]} \weight={class_weights[i]}") for i in sorted_keys]

    # scaling of coordinates/x
    if not hasattr(args, 'vae_normalize_x'):
        args.vae_normalize_x = False
    if not hasattr(args, 'vae_normalize_method'):  # supported: "scale" | "linear"
        args.vae_normalize_method = None
    if not hasattr(args, 'vae_normalize_fn_points'):  # [x_min, y_min, x_max, y_max]
        args.vae_normalize_fn_points = None

    # data splits
    if not hasattr(args, 'data_splitted'):
        args.data_splitted = False


    # params global registry for easy access
    PARAM_REGISTRY.update_from_config(args)

    atom_encoder = dataset_info['atom_encoder']
    atom_decoder = dataset_info['atom_decoder']

    utils.create_folders(args)

    context_node_nf = 0
    property_norms = None
    args.context_node_nf = context_node_nf


    if args.train_diffusion:
        raise NotImplementedError()
    else:
        assert len(args.conditioning) == 0, "Conditioning not supported"
        model, nodes_dist, prop_dist = get_autoencoder(args, args.device, dataset_info, None)   # prop_dist = None

    model = model.to(args.device)
    
    if args.ae_path is None:
        args.ae_path = join(os.getcwd(), 'outputs', args.exp_name)

    if opt.load_last:
        if args.ema_decay > 0:
            pattern_str = r'generative_model_ema_(\d+)_iter_(\d+)\.npy'
        else:
            pattern_str = r'generative_model_(\d+)_iter_(\d+)\.npy'
            
        filtered_files = [f for f in os.listdir(args.ae_path) if re.compile(pattern_str).match(f)]
        filtered_files.sort(key=lambda x: (
            int(re.search(pattern_str, x).group(1)),
            int(re.search(pattern_str, x).group(2))
        ))
        fn = filtered_files[-1]
    else:
        fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    
    print(f">> Loading VAE weights from {join(args.ae_path, fn)}")
    flow_state_dict = torch.load(join(args.ae_path, fn), map_location=device)
    model.load_state_dict(flow_state_dict)

    
    
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
    print(f"================================")
    print(model)


    model.eval()

    # ~!mp
    with torch.no_grad():
        assert opt.n_samples % args.batch_size == 1
        start  = time.time()
        for i in tqdm(range(int(opt.n_samples/args.batch_size))):
            
            # sample number of nodes
            nodesxsample = nodes_dist.sample(args.batch_size)
            
            # sample from model
            one_hot, charges, x, node_mask = sample(
                args, device, model, dataset_info, prop_dist=prop_dist, nodesxsample=nodesxsample)

            save_mol_path = os.path.join(args.save_samples_dir, datetime.now().strftime('%Y%m%d%H%M%S%f') + '_' + str(i).zfill(6))
            
            if not os.path.exists(save_mol_path):
                os.makedirs(save_mol_path)
            
            # save molecules
            save_xyz_file(path=save_mol_path,
                          one_hot=one_hot.detach().cpu(),
                          positions=x.detach().cpu(),
                          charges=None,
                          dataset_info=dataset_info,
                          id_from=0, 
                          name='GEN', 
                          node_mask=node_mask)


            # cleanup
            del x, h, node_mask, edge_mask, one_hot, charges, nll
            torch.cuda.empty_cache()
            gc.collect()

    print(f">>> Generation {time.time() - start:.1f} seconds.")
    print("DONE.")



if __name__ == "__main__":
    main()
