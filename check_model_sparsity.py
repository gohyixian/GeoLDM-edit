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
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from qm9.models import get_optim, get_model, get_autoencoder, get_latent_diffusion

import torch
import time
import train_test
from tqdm import tqdm
from datetime import datetime
from global_registry import PARAM_REGISTRY, Config


def save_activations(activations: np.ndarray, save_path, filename):
    original_options = np.get_printoptions()
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    activations_str = np.array2string(activations, separator=', ')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    unique_id = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    file = os.path.join(save_path, f"{filename}___t{unique_id}.txt")
    with open(file, 'w') as f:
        f.write(activations_str)
    np.set_printoptions(**original_options)


def plot_activation_distribution(tensor: np.ndarray, title: str, save_path: str, filename: str, save_tensor=False, bins=200):
    tensor_flat = tensor.flatten()
    print(tensor_flat.shape)
    plt.hist(tensor_flat, bins=bins)
    plt.title(title, fontsize=7)
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
    parser.add_argument('--load_last', action='store_true', help='load weights of model of last epoch')
    parser.add_argument('--data_size', type=float, default=0.001, help='portion of val data split to use for test')
    opt = parser.parse_args()

    with open(opt.config_file, 'r') as file:
        args_dict = yaml.safe_load(file)
    args = Config(**args_dict)


    # additional & override settings for sparsity plots
    args.batch_size = 1
    args.layer_id_counter = 0
    args.plot_func_bins = 200
    args.save_plot_dir = f'sparsity_check/plots/{args.exp_name}/{args.training_mode}/'
    # plot
    args.plot_sparsity = False
    args.save_tensor = False
    args.plot_func = plot_activation_distribution
    # save activations for combined plot
    args.save_activations = True
    args.save_act_func = save_activations
    args.save_act_dir = f'sparsity_check/saved_activations/{args.exp_name}/{args.training_mode}/'


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


    # params global registry for easy access
    PARAM_REGISTRY.update_from_config(args)


    # pre-computed data file
    data_file = args.data_file
    print(">> Loading data from:", data_file)
    split_data = build_geom_dataset.load_split_data(data_file, 
                                                    val_proportion=opt.data_size,   # will only be using this
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


    epoch = 0
    start  = time.time()
    print(f">>> validation dataset number of samples: {len(dataloaders['val'].dataset)}")
    nll_val = train_test.test(args, dataloaders['val'], epoch, model, device, dtype,
                                property_norms, nodes_dist, partition='Val')
    print(f">>> validation set test took {time.time() - start:.1f} seconds.")

    # plot combined_sparsity_plots:
    if args.save_activations:
        # get unique layers' names
        to_omit = ['.DS_Store']
        unique_layers = sorted(list(set([f.split("___t")[0] for f in os.listdir(args.save_act_dir) if f not in to_omit])))
        
        # loop through every unique layer's saved activations
        for layer in unique_layers:
            layer_activations = sorted([f for f in os.listdir(args.save_act_dir) if f not in to_omit and f.startswith(layer)])
            print(f"Working with {layer}'s {len(layer_activations)} activations ...")
            activation_tensors = []
            for file in tqdm(layer_activations):
                with open(os.path.join(args.save_act_dir, file), 'r') as f:
                    activation_str = f.read()
                recreated_activation = eval(f"np.array({activation_str})")
                activation_tensors.append(recreated_activation.flatten())
            activation_tensors = np.concatenate(activation_tensors)
            plot_activation_distribution(tensor=activation_tensors, 
                                         title=f"Combined Distribution of Activations for {layer} over {len(layer_activations)} samples  (bins={args.plot_func_bins})", 
                                         save_path=args.save_plot_dir, 
                                         filename=f"combined_ActDist_{layer}__bins{args.plot_func_bins}", 
                                         save_tensor=False, 
                                         bins=args.plot_func_bins)
    print("DONE.")


if __name__ == "__main__":
    main()
