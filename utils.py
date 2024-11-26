import numpy as np
import getpass
import os
import re
import torch
from torch import nn
from global_registry import PARAM_REGISTRY


def add_missing_configs_controlnet(args, dtype, ligand_dataset_info, pocket_dataset_info):

    # mp autocast dtype
    if args.mixed_precision_training == True:
        _, mp_dtype_name = args.mixed_precision_autocast_dtype.split('.')
        mp_dtype = getattr(torch, mp_dtype_name)
        args.mixed_precision_autocast_dtype = mp_dtype
    else:
        args.mixed_precision_autocast_dtype = dtype

    if len(args.conditioning) > 0:
        raise NotImplementedError()
        # print(f'Conditioning on {args.conditioning}')
        # data_dummy = next(iter(dataloaders['train']))
        # property_norms = compute_mean_mad(dataloaders, args.conditioning)
        # context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
        # context_node_nf = context_dummy.size(2)
    else:
        args.context_node_nf = 0
        args.property_norms = None
    
    # [Pocket VAE]
    if len(args.pocket_vae.conditioning) > 0:
        raise NotImplementedError()
    else:
        args.pocket_vae.context_node_nf = 0
        args.pocket_vae.property_norms = 0

    # gradient accumulation
    if not hasattr(args, 'grad_accumulation_steps'):
        args.grad_accumulation_steps = 1  # call optim every step

    # vae data mode
    if not hasattr(args, 'vae_data_mode'):
        args.vae_data_mode = 'all'
    if not hasattr(args.pocket_vae, 'vae_data_mode'):
        args.pocket_vae.vae_data_mode = 'all'

    # vae encoder n layers
    if not hasattr(args, 'encoder_n_layers'):
        args.encoder_n_layers = 1
    if not hasattr(args.pocket_vae, 'encoder_n_layers'):
        args.pocket_vae.encoder_n_layers = 1

    # grad prenalty
    if not hasattr(args, 'grad_penalty'):
        args.grad_penalty = False

    # loss analysis
    if not hasattr(args, 'loss_analysis'):
        args.loss_analysis = False
    args.loss_analysis_modes = ['VAE', 'LDM']

    # loss analysis usage
    args.atom_encoder = ligand_dataset_info['atom_encoder']
    args.atom_decoder = ligand_dataset_info['atom_decoder']
    # [Pocket VAE]
    args.pocket_vae.atom_encoder = pocket_dataset_info['atom_encoder']
    args.pocket_vae.atom_decoder = pocket_dataset_info['atom_decoder']

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

    # data splits
    if not hasattr(args, 'data_splitted'):
        args.data_splitted = False

    # visualise sample chain
    if not hasattr(args, 'visualize_sample_chain'):
        args.visualize_sample_chain = False
    if not hasattr(args, 'visualize_sample_chain_epochs'):
        args.visualize_sample_chain_epochs = 1

    # [Ligand VAE] scaling of coordinates/x
    if not hasattr(args, 'vae_normalize_x'):
        args.vae_normalize_x = False
    if not hasattr(args, 'vae_normalize_method'):  # supported: "scale" | "linear"
        args.vae_normalize_method = None
    if not hasattr(args, 'vae_normalize_factors'):
        args.vae_normalize_factors = [1, 1, 1]

    # [Ligand VAE] class-imbalance loss reweighting
    if not hasattr(args, 'reweight_class_loss'):  # supported: "inv_class_freq"
        args.reweight_class_loss = None
    if not hasattr(args, 'reweight_coords_loss'):  # supported: "inv_class_freq"
        args.reweight_coords_loss = None
    if not hasattr(args, 'smoothing_factor'):  # smoothing: (0. - 1.]
        args.smoothing_factor = None
    if args.reweight_class_loss == "inv_class_freq":
        class_freq_dict = ligand_dataset_info['atom_types']
        sorted_keys = sorted(class_freq_dict.keys())
        frequencies = torch.tensor([class_freq_dict[key] for key in sorted_keys], dtype=args.dtype)
        inverse_frequencies = 1.0 / frequencies

        if args.smoothing_factor is not None:
            smoothing_factor = float(args.smoothing_factor)
            inverse_frequencies = torch.pow(inverse_frequencies, smoothing_factor)

        class_weights = inverse_frequencies / inverse_frequencies.sum()  # normalize
        args.class_weights = class_weights
        [print(f"{args.atom_decoder[sorted_keys[i]]} freq={class_freq_dict[sorted_keys[i]]} \
            inv_freq={inverse_frequencies[i]} \weight={class_weights[i]}") for i in sorted_keys]
    else:
        args.class_weights = None

    # [Ligand VAE] coordinates loss weighting
    if not hasattr(args, 'error_x_weight'):
        args.error_x_weight = None
    # [Ligand VAE] atom types loss weighting
    if not hasattr(args, 'error_h_weight'):
        args.error_h_weight = None


    # [Pocket VAE] scaling of coordinates/x
    if not hasattr(args.pocket_vae, 'vae_normalize_x'):
        args.pocket_vae.vae_normalize_x = False
    if not hasattr(args.pocket_vae, 'vae_normalize_method'):  # supported: "scale" | "linear"
        args.pocket_vae.vae_normalize_method = None
    if not hasattr(args.pocket_vae, 'vae_normalize_factors'):
        args.pocket_vae.vae_normalize_factors = [1, 1, 1]
    
    # [Pocket VAE] class-imbalance loss reweighting
    if not hasattr(args.pocket_vae, 'reweight_class_loss'):  # supported: "inv_class_freq"
        args.pocket_vae.reweight_class_loss = None
    if not hasattr(args.pocket_vae, 'reweight_coords_loss'):  # supported: "inv_class_freq"
        args.pocket_vae.reweight_coords_loss = None
    if not hasattr(args.pocket_vae, 'smoothing_factor'):  # smoothing: (0. - 1.]
        args.pocket_vae.smoothing_factor = None
    if args.pocket_vae.reweight_class_loss == "inv_class_freq":
        class_freq_dict = pocket_dataset_info['atom_types']
        sorted_keys = sorted(class_freq_dict.keys())
        frequencies = torch.tensor([class_freq_dict[key] for key in sorted_keys], dtype=args.dtype)
        inverse_frequencies = 1.0 / frequencies

        if args.pocket_vae.smoothing_factor is not None:
            smoothing_factor = float(args.pocket_vae.smoothing_factor)
            inverse_frequencies = torch.pow(inverse_frequencies, smoothing_factor)

        class_weights = inverse_frequencies / inverse_frequencies.sum()  # normalize
        args.pocket_vae.class_weights = class_weights
        [print(f"{args.pocket_vae.atom_decoder[sorted_keys[i]]} freq={class_freq_dict[sorted_keys[i]]} \
            inv_freq={inverse_frequencies[i]} \weight={class_weights[i]}") for i in sorted_keys]

    # [Pocket VAE] coordinates loss weighting
    if not hasattr(args.pocket_vae, 'error_x_weight'):
        args.pocket_vae.error_x_weight = None
    # [Pocket VAE] atom types loss weighting
    if not hasattr(args.pocket_vae, 'error_h_weight'):
        args.pocket_vae.error_h_weight = None


    # [ControlNet] time noisy: t/2 for main network, adopted from [https://arxiv.org/abs/2405.06659]
    if not hasattr(args, 'time_noisy'):
        args.time_noisy = False


    # [ControlNet] match ligand & pocket raw files by ids
    if not hasattr(args, 'match_raw_file_by_id'):
        args.match_raw_file_by_id = False
    if not hasattr(args, 'compute_qvina'):
        args.compute_qvina = False
    if not hasattr(args, 'qvina_search_size'):
        args.qvina_search_size = 20
    if not hasattr(args, 'qvina_exhaustiveness'):
        args.qvina_exhaustiveness = 16
    if not hasattr(args, 'qvina_cleanup_files'):
        args.qvina_cleanup_files = True
    if not hasattr(args, 'qvina_save_csv'):
        args.qvina_save_csv = True
    if not hasattr(args, 'pocket_pdb_dir'):
        args.pocket_pdb_dir = ""
    if not hasattr(args, 'match_raw_file_by_id'):
        args.match_raw_file_by_id = True
    if not hasattr(args, 'mgltools_env_name'):
        args.mgltools_env_name = 'mgltools-python2'
    if not hasattr(args, 'ligand_add_H'):
        args.ligand_add_H = False
    if not hasattr(args, 'pocket_add_H'):
        args.pocket_add_H = False
    if not hasattr(args, 'pocket_remove_nonstd_resi'):
        args.pocket_remove_nonstd_resi = False

    return args




import periodictable
# https://periodictable.readthedocs.io/en/latest/guide/using.html
# https://www.geeksforgeeks.org/get-the-details-of-an-element-by-atomic-number-using-python/

def get_periodictable_list():
    atom_num = []
    symbol = []
    
    for element in periodictable.elements:
        atom_num.append(int(element.number))
        symbol.append(str(element.symbol))
    
    an2s = dict(zip(atom_num, symbol))
    s2an = dict(zip(symbol, atom_num))
    
    return an2s, s2an


def get_nvidia_smi_usage(smi_txt: str):
    gpu_id_pattern = r'^\|\s*(\d+)\s+'
    memory_pattern = r'(\d+)MiB\s*/\s*(\d+)MiB'

    gpu_memory_usage = {}
    gpu_ids = re.findall(gpu_id_pattern, smi_txt, re.MULTILINE)
    memories = re.findall(memory_pattern, smi_txt)

    # assuming GPU IDs and memories are in the same order
    for gpu_id, (used_memory, total_memory) in zip(gpu_ids, memories):
        gpu_memory_usage[int(gpu_id)] = {
            'used_mem': int(used_memory),
            'total_mem': int(total_memory)
        }
    return gpu_memory_usage


# Folders
def create_folders(args):
    try:
        os.makedirs('outputs')
    except OSError:
        pass

    try:
        os.makedirs('outputs/' + args.exp_name)
    except OSError:
        pass


# Model checkpoints
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


#Gradient clipping
class Queue():
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def gradient_clipping(flow, gradnorm_queue):
    # Allow gradient norm to be 150% + 2 * stdev of the recent history.
    max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()

    # Clips gradient and returns the norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        flow.parameters(), max_norm=max_grad_norm, norm_type=2.0)

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    if float(grad_norm) > max_grad_norm:
        print(f'Clipped gradient with value {grad_norm:.1f} '
              f'while allowed {max_grad_norm:.1f}')
    return grad_norm


# Rotation data augmntation
def random_rotation(x):
    bs, n_nodes, n_dims = x.size()
    device = x.device
    angle_range = np.pi * 2
    if n_dims == 2:
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        R_row0 = torch.cat([cos_theta, -sin_theta], dim=2)
        R_row1 = torch.cat([sin_theta, cos_theta], dim=2)
        R = torch.cat([R_row0, R_row1], dim=1)

        x = x.transpose(1, 2)
        x = torch.matmul(R, x)
        x = x.transpose(1, 2)

    elif n_dims == 3:

        # Build Rx
        Rx = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rx[:, 1:2, 1:2] = cos
        Rx[:, 1:2, 2:3] = sin
        Rx[:, 2:3, 1:2] = - sin
        Rx[:, 2:3, 2:3] = cos

        # Build Ry
        Ry = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Ry[:, 0:1, 0:1] = cos
        Ry[:, 0:1, 2:3] = -sin
        Ry[:, 2:3, 0:1] = sin
        Ry[:, 2:3, 2:3] = cos

        # Build Rz
        Rz = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rz[:, 0:1, 0:1] = cos
        Rz[:, 0:1, 1:2] = sin
        Rz[:, 1:2, 0:1] = -sin
        Rz[:, 1:2, 1:2] = cos

        x = x.transpose(1, 2)
        x = torch.matmul(Rx, x)
        #x = torch.matmul(Rx.transpose(1, 2), x)
        x = torch.matmul(Ry, x)
        #x = torch.matmul(Ry.transpose(1, 2), x)
        x = torch.matmul(Rz, x)
        #x = torch.matmul(Rz.transpose(1, 2), x)
        x = x.transpose(1, 2)
    else:
        raise Exception("Not implemented Error")

    return x.contiguous()



if __name__ == "__main__":


    ## Test random_rotation
    bs = 2
    n_nodes = 16
    n_dims = 3
    x = torch.randn(bs, n_nodes, n_dims)
    print(x)
    x = random_rotation(x)
    #print(x)
