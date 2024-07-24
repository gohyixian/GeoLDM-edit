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
import argparse
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from qm9.models import get_optim, get_model, get_autoencoder, get_latent_diffusion

import torch
import time
import train_test

from global_registry import PARAM_REGISTRY, Config

def plot_activation_distribution(tensor: np.ndarray, title: str, save_path: str, filename: str, save_tensor=False, bins=200):
    tensor_flat = tensor.flatten()
    plt.hist(tensor_flat, bins=bins)
    plt.title(title)
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, filename+".png"))
    plt.close()

    if save_tensor:
        with open(os.path.join(save_path, filename+".txt"), 'w') as f:
            original_options = np.get_printoptions()
            np.set_printoptions(threshold=np.inf, linewidth=np.inf)
            print(f"\n\n{filename}\n=============================")
            print(tensor)
            print(tensor, file=f)
            np.set_printoptions(**original_options)



def main():
    parser = argparse.ArgumentParser(description='e3_diffusion')
    parser.add_argument('--config_file', type=str, default='custom_config/base_geom_config.yaml')
    opt = parser.parse_args()

    with open(opt.config_file, 'r') as file:
        args_dict = yaml.safe_load(file)
    args = Config(**args_dict)


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

    # additional & override settings for sparsity plots
    args.plot_sparsity = True
    args.save_tensor = True
    args.batch_size = 1
    args.plot_counter = 0
    args.plot_func_bins = 200
    args.plot_func = plot_activation_distribution
    args.save_plot_dir = f'sparsity_check/plots/{args.exp_name}/{args.training_mode}/'


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
    # ['positions'], ['one_hot'], ['charges'], ['atonm_mask'], ['edge_mask'] are added here
    dataset_info = get_dataset_info(dataset_name=args.dataset, remove_h=args.remove_h)
    transform = build_geom_dataset.GeomDrugsTransform(dataset_info, args.include_charges, args.device, args.sequential)

    dataloaders = {}
    for key, data_list in zip(['train', 'val', 'test'], split_data):
        dataset = build_geom_dataset.GeomDrugsDataset(data_list, transform=transform, training_mode=args.training_mode)
        # shuffle = (key == 'train') and not args.sequential
        shuffle = (key == 'train')

        # Sequential dataloading disabled for now.
        dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
            sequential=args.sequential, dataset=dataset, batch_size=args.batch_size,
            shuffle=shuffle, training_mode=args.training_mode)
    del split_data


    atom_encoder = dataset_info['atom_encoder']
    atom_decoder = dataset_info['atom_decoder']

    utils.create_folders(args)

    context_node_nf = 0
    property_norms = None
    args.context_node_nf = context_node_nf


    if args.train_diffusion:
        raise NotImplementedError()
    else:
        model, nodes_dist, prop_dist = get_autoencoder(args, args.device, dataset_info, dataloaders['train'])

    model = model.to(args.device)
    
    print(f">> Loading VAE weights from {args.ae_path}")
    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
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


    epoch = 0
    start  = time.time()
    nll_val = train_test.test(args, dataloaders['val'], epoch, model, device, dtype,
                                property_norms, nodes_dist, partition='Val')
    print(f">>> validation set test took {time.time() - start:.1f} seconds.")



if __name__ == "__main__":
    main()
