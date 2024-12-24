# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import utils
import argparse
from qm9 import dataset
from qm9.models import get_model, get_autoencoder, get_latent_diffusion
import os
import re
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
import torch
import time
import pickle
from configs.dataset_configs.datasets_config import get_dataset_info
from os.path import join
from qm9.sampling import sample
from qm9.analyze import compute_molecule_metrics, analyze_node_distribution
from qm9.utils import prepare_context, compute_mean_mad
from qm9 import visualizer as qm9_visualizer
import qm9.losses as losses
from global_registry import PARAM_REGISTRY, Config

try:
    from qm9 import rdkit_functions
except ModuleNotFoundError:
    print('Not importing rdkit functions.')


def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)


def analyze_and_save(args, eval_args, device, generative_model,
                     nodes_dist, prop_dist, dataset_info, n_samples=10,
                     batch_size=10, save_to_xyz=False):
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    start_time = time.time()
    for i in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(
            args, device, generative_model, dataset_info, prop_dist=prop_dist, nodesxsample=nodesxsample)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

        current_num_samples = (i+1) * batch_size
        secs_per_sample = (time.time() - start_time) / current_num_samples
        print('\t %d/%d Molecules generated at %.2f secs/sample' % (
            current_num_samples, n_samples, secs_per_sample))

        if save_to_xyz:
            id_from = i * batch_size
            qm9_visualizer.save_xyz_file(
                join(eval_args.save_path, os.path.basename(eval_args.model_path), 'analyzed_molecules/'),
                one_hot, charges, x, dataset_info, id_from, name='molecule',
                node_mask=node_mask)

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    print(f"one_hot: {molecules['one_hot'].shape}")
    print(f"x: {molecules['x'].shape}")
    print(f"node_mask: {molecules['node_mask'].shape}")
    # one_hot: torch.Size([10, 106, 8])
    # x: torch.Size([10, 106, 3])
    # node_mask: torch.Size([10, 106, 1])
    
    metrics_dict = compute_molecule_metrics(molecules, dataset_info)

    return metrics_dict




# python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path /mnt/c/Users/PC/Desktop/yixian/geoldm-edit/outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent2_nf256_epoch200_bs36_lr1e-4_EMA-0.99__VAE_DecOnly_KL-0__20240623__10A_7x_resume
# python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path /mnt/c/Users/PC/Desktop/yixian/geoldm-edit/outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent2_nf256_epoch200_bs36_lr1e-4_EMA-0.99__VAE_DecOnly_KL-0__20240623__10A_8x_resume

# python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path /mnt/c/Users/PC/Desktop/yixian/geoldm-edit/outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent2_nf256_epoch200_bs36_lr1e-4_NoEMA__VAE_DecOnly_KL-0__20240623__10A_7x_resume
# python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path /mnt/c/Users/PC/Desktop/yixian/geoldm-edit/outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent2_nf256_epoch200_bs36_lr1e-4_NoEMA__VAE_DecOnly_KL-0__20240623__10A_8x_resume
# python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path /mnt/c/Users/PC/Desktop/yixian/geoldm-edit/outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent2_nf256_epoch200_bs36_lr1e-4_NoEMA__VAE_DecOnly_KL-0__20240623__10A_9x_resume

# python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path /mnt/c/Users/PC/Desktop/yixian/geoldm-edit/outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent8_nf128_epoch200_bs64_lr1e-4_EMA-0.99__20240623__10A_3x_resume
# python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path /mnt/c/Users/PC/Desktop/yixian/geoldm-edit/outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent8_nf128_epoch200_bs64_lr1e-4_EMA-0.99__20240623__10A_4x_resume
# python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path /mnt/c/Users/PC/Desktop/yixian/geoldm-edit/outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent8_nf128_epoch200_bs64_lr1e-4_EMA-0.99__20240623__10A_5x_resume
# python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path /mnt/c/Users/PC/Desktop/yixian/geoldm-edit/outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent8_nf128_epoch200_bs64_lr1e-4_EMA-0.99__20240623__10A_6x_resume
# python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path /mnt/c/Users/PC/Desktop/yixian/geoldm-edit/outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent8_nf128_epoch200_bs64_lr1e-4_EMA-0.99__20240623__10A_7x_resume

# python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path /mnt/c/Users/PC/Desktop/yixian/geoldm-edit/outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent8_nf128_epoch200_bs64_lr1e-4_NoEMA__20240623__10A_3x_resume
# python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path /mnt/c/Users/PC/Desktop/yixian/geoldm-edit/outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent8_nf128_epoch200_bs64_lr1e-4_NoEMA__20240623__10A_4x_resume
# python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path /mnt/c/Users/PC/Desktop/yixian/geoldm-edit/outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent8_nf128_epoch200_bs64_lr1e-4_NoEMA__20240623__10A_5x_resume
# python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path /mnt/c/Users/PC/Desktop/yixian/geoldm-edit/outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent8_nf128_epoch200_bs64_lr1e-4_NoEMA__20240623__10A_6x_resume




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="outputs/edm_1",
                        help='Specify model path')
    parser.add_argument('--Load_last', action='store_true', 
                        help='load weights of model of last epoch')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of samples to generate & evaluate')
    parser.add_argument('--batch_size_gen', type=int, default=100,
                        help='Sampling batch size')
    parser.add_argument('--save_path', type=str, default='eval_ldm',
                        help='Path to save xyz files.')
    eval_args, unparsed_args = parser.parse_known_args()
    eval_args.save_to_xyz = True
    
    print(f"Eval Args (model_path)    : {eval_args.model_path}")
    print(f"Eval Args (n_samples)     : {eval_args.n_samples}")
    print(f"Eval Args (batch_size_gen): {eval_args.batch_size_gen}")
    print(f"Eval Args (save_to_xyz)   : {eval_args.save_to_xyz}")
    

    assert eval_args.model_path is not None

    with open(join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)
    
    dataset_info = get_dataset_info(dataset_name=args.dataset, remove_h=args.remove_h)

    # CAREFUL with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    # ================================================================================ #
    # For code compatibility                                                           #
    # ================================================================================ #

    # device settings
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    args.device_ = "cuda" if args.cuda else "cpu"


    # dtype settings
    dtype = args.dtype
    torch.set_default_dtype(dtype)


    # mp autocast dtype
    if args.mixed_precision_training == True:
        pass
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
    args.loss_analysis_modes = ['VAE', 'LDM']


    # loss analysis usage
    atom_encoder = dataset_info['atom_encoder']
    atom_decoder = dataset_info['atom_decoder']
    args.atom_encoder = atom_encoder
    args.atom_decoder = atom_decoder


    # intermediate activations analysis usage
    args.vis_activations_instances = (torch.nn.Linear)
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
    if not hasattr(args, 'reweight_coords_loss'):  # supported: "inv_class_freq"
        args.reweight_coords_loss = None
    if not hasattr(args, 'smoothing_factor'):  # smoothing: (0. - 1.]
        args.smoothing_factor = None
    if args.reweight_class_loss == "inv_class_freq":
        class_freq_dict = dataset_info['atom_types']
        sorted_keys = sorted(class_freq_dict.keys())
        frequencies = torch.tensor([class_freq_dict[key] for key in sorted_keys], dtype=args.dtype)
        inverse_frequencies = 1.0 / frequencies

        if args.smoothing_factor is not None:
            smoothing_factor = float(args.smoothing_factor)
            inverse_frequencies = torch.pow(inverse_frequencies, smoothing_factor)

        class_weights = inverse_frequencies / inverse_frequencies.sum()  # normalize
        args.class_weights = class_weights
        [print(f"{atom_decoder[sorted_keys[i]]} freq={class_freq_dict[sorted_keys[i]]} \
            inv_freq={inverse_frequencies[i]} \weight={class_weights[i]}") for i in sorted_keys]
    else:
        args.class_weights = None

    # coordinates loss weighting
    if not hasattr(args, 'error_x_weight'):
        args.error_x_weight = None
    # atom types loss weighting
    if not hasattr(args, 'error_h_weight'):
        args.error_h_weight = None

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

    # ================================================================================ #
    # ================================================================================ #

    utils.create_folders(args)
    print(args)

    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    # Load model
    if args.training_mode == "LDM":
        assert len(args.conditioning) == 0, "Conditioning not supported"
        generative_model, nodes_dist, prop_dist = get_latent_diffusion(args, args.device, dataset_info)   # prop_dist = None
    else:
        raise NotImplementedError()

    generative_model.to(device)

    if eval_args.load_last:
        if args.ema_decay > 0:
            pattern_str = r'generative_model_ema_(\d+)_iter_(\d+)\.npy'
        else:
            pattern_str = r'generative_model_(\d+)_iter_(\d+)\.npy'
            
        filtered_files = [f for f in os.listdir(eval_args.model_path) if re.compile(pattern_str).match(f)]
        filtered_files.sort(key=lambda x: (
            int(re.search(pattern_str, x).group(1)),
            int(re.search(pattern_str, x).group(2))
        ))
        fn = filtered_files[-1]
    else:
        fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    
    print(f">> Loading model weights from {join(eval_args.model_path, fn)}")
    flow_state_dict = torch.load(join(eval_args.model_path, fn), map_location=device)
    generative_model.load_state_dict(flow_state_dict)

    # Analyze stability, validity, uniqueness and novelty
    metrics_dict = analyze_and_save(
        args, eval_args, device, generative_model, nodes_dist,
        prop_dist, dataset_info, n_samples=eval_args.n_samples,
        batch_size=eval_args.batch_size_gen, save_to_xyz=eval_args.save_to_xyz)
    print(metrics_dict)

    with open(join(eval_args.save_path, os.path.basename(eval_args.model_path), 'eval_log.txt'), 'w') as f:
        text = \
            f"Molecule Stability : {metrics_dict['mol_stable']}\n" + \
            f"Atom Stability     : {metrics_dict['atm_stable']}\n\n" + \
            f"Validity           : {metrics_dict['validity']}\n" + \
            f"Uniqueness         : {metrics_dict['uniqueness']}\n" + \
            f"Novelty            : {metrics_dict['novelty']}\n" + \
            f"Diversity          : {metrics_dict['diversity']}\n\n" + \
            f"Connectivity       : {metrics_dict['connectivity']}\n" + \
            f"QED                : {metrics_dict['QED']}\n" + \
            f"SA                 : {metrics_dict['SA']}\n" + \
            f"LogP               : {metrics_dict['logP']}\n" + \
            f"Lipinski           : {metrics_dict['lipinski']}\n"
        print(text, file=f)

if __name__ == "__main__":
    main()
