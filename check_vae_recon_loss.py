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
    parser.add_argument('--data_size', type=float, default=0.1, help='portion of val data split to use for test')
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
    args.batch_size = 1 # must be 1 for this script
    args.save_samples_dir = f'recon_loss_analysis/{args.exp_name}/{args.training_mode}/'
    args.save_samples_dir_wrong = f'recon_loss_analysis/{args.exp_name}/{args.training_mode}/classwise_accuracy_mistake/'

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
                                                    filter_pocket_size=args.filter_pocket_size,
                                                    data_splitted=args.data_splitted)
    # ~!to ~!mp
    # ['positions'], ['one_hot'], ['charges'], ['atonm_mask'], ['edge_mask'] are added here
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


    start  = time.time()
    print(f">>> validation dataset number of samples: {len(dataloaders['val'].dataset)}")

    loader = dataloaders['val']
    model.eval()

    # ~!mp
    with torch.no_grad():
        NLL = []
        error = []
        error_x = []
        error_h = []
        error_charges = []
        overall_accuracy = []
        overall_recall = []
        overall_f1 = []
        classwise_accuracy = {}

        for i, data in tqdm(enumerate(loader)):
            x = data['positions'].to(device, dtype)
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

            if args.augment_noise > 0:
                raise NotImplementedError()
                # # Add noise eps ~ N(0, augment_noise) around points.
                # eps = sample_center_gravity_zero_gaussian_with_mask(x.size(),
                #                                                     x.device,
                #                                                     node_mask)
                # x = x + eps * args.augment_noise

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {'categorical': one_hot, 'integer': charges}

            if len(args.conditioning) > 0:
                context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
                assert_correctly_masked(context, node_mask)
            else:
                context = None

            bs, n_nodes, n_dims = x.size()
            edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
            assert_correctly_masked(x, node_mask)

            # Here x is a position tensor, and h is a dictionary with keys
            # 'categorical' and 'integer'.
            nll, nll_dict = model.forward(x, h, node_mask, edge_mask, context, loss_analysis=True)
            # loss_dict: {
            #     'loss_t': loss.squeeze(), 
            #     'rec_error': loss_recon.squeeze(), 
            #     'recon_loss_dict': recon_loss_dict,  # additional for rec loss analysis and xh saving
            #     'xh_gt': xh_unpack_dict: {'x': x, 'h_cat': h_cat, 'h_int': h_int}
            #     'xh_rec': xh_rec_unpack_dict: {'x': x, 'h_cat': h_cat, 'h_int': h_int}
            # }
            # recon_loss_dict: {
            #     'error': error,
            #     'error_x': error_x, 
            #     'error_h_cat': error_h_cat, 
            #     'error_h_int': error_h_int,
            #     'denom': denom,
            #     'n_dims': self.n_dims,
            #     'in_node_nf': self.in_node_nf,
            #     'num_atoms': xh.shape[1],
            #     'overall_accuracy': overall_accuracy,
            #     'overall_recall': overall_recall,
            #     'overall_f1': overall_f1,
            #     'classwise_accuracy': classwise_accuracy
            # }

            N = node_mask.squeeze(2).sum(1).long()

            log_pN = nodes_dist.log_prob(N)

            assert nll.size() == log_pN.size()
            nll = nll - log_pN

            # Average over batch.
            nll = nll.mean(0)
            
            perfect_classwise_accuracy = True
            for cls, acc in nll_dict['recon_loss_dict']['classwise_accuracy'].items():
                if not math.isnan(acc):
                    if acc < 1.0:
                        perfect_classwise_accuracy = False
            
            if perfect_classwise_accuracy:
                save_mol_path = os.path.join(args.save_samples_dir, datetime.now().strftime('%Y%m%d%H%M%S%f') + '_' + str(i).zfill(6))
            else:
                save_mol_path = os.path.join(args.save_samples_dir_wrong, datetime.now().strftime('%Y%m%d%H%M%S%f') + '_' + str(i).zfill(6))
            
            if not os.path.exists(save_mol_path):
                os.makedirs(save_mol_path)
            
            # save molecules (GT & REC)
            save_xyz_file(path=save_mol_path, 
                          one_hot=nll_dict['xh_gt']['h_cat'],
                          positions=nll_dict['xh_gt']['x'],
                          charges=None,
                          dataset_info=dataset_info,
                          id_from=0, 
                          name='GT', 
                          node_mask=node_mask)
            save_xyz_file(path=save_mol_path, 
                          one_hot=nll_dict['xh_rec']['h_cat'],
                          positions=nll_dict['xh_rec']['x'],
                          charges=None,
                          dataset_info=dataset_info,
                          id_from=0, 
                          name='REC', 
                          node_mask=node_mask)
            # save losses
            with open(os.path.join(save_mol_path, "loss.txt"), "w") as f:
                print(f"NLL             : {nll.item()}", file=f)
                print(f"Error           : {nll_dict['recon_loss_dict']['error'].item()}", file=f)
                print(f"Error X         : {nll_dict['recon_loss_dict']['error_x'].item()}", file=f)
                print(f"Error H         : {nll_dict['recon_loss_dict']['error_h_cat'].item()}", file=f)
                print(f"Error H Charges : {nll_dict['recon_loss_dict']['error_h_int'].item()}", file=f) if args.include_charges else None
                print(f"Overall Accuracy: {nll_dict['recon_loss_dict']['overall_accuracy']}", file=f)
                print(f"Overall Recall  : {nll_dict['recon_loss_dict']['overall_recall']}", file=f)
                print(f"Overall F1      : {nll_dict['recon_loss_dict']['overall_f1']}", file=f)
                print(f"Classwise Accuracy", file=f)
                for cls, acc in nll_dict['recon_loss_dict']['classwise_accuracy'].items():
                    print(f"    {cls} : {acc}", file=f)
                print(f"\n=================", file=f)
                print(f"denom      : {nll_dict['recon_loss_dict']['denom']}", file=f)
                print(f"n_dims     : {nll_dict['recon_loss_dict']['n_dims']}", file=f)
                print(f"in_node_nf : {nll_dict['recon_loss_dict']['in_node_nf']}", file=f)
                print(f"num_atoms  : {nll_dict['recon_loss_dict']['num_atoms']}", file=f)
            
            NLL.append(nll.item())
            error.append(nll_dict['recon_loss_dict']['error'].item())
            error_x.append(nll_dict['recon_loss_dict']['error_x'].item())
            error_h.append(nll_dict['recon_loss_dict']['error_h_cat'].item())
            if args.include_charges:
                error_charges.append(nll_dict['recon_loss_dict']['error_h_int'].item())
            overall_accuracy.append(nll_dict['recon_loss_dict']['overall_accuracy'])
            overall_recall.append(nll_dict['recon_loss_dict']['overall_recall'])
            overall_f1.append(nll_dict['recon_loss_dict']['overall_f1'])
            for cls, acc in nll_dict['recon_loss_dict']['classwise_accuracy'].items():
                if not math.isnan(acc):
                    classwise_accuracy[str(cls)] = classwise_accuracy.get(str(cls), []) + [acc]

            # cleanup
            del x, h, node_mask, edge_mask, one_hot, charges, nll
            torch.cuda.empty_cache()
            gc.collect()

        mean_NLL = sum(NLL) / len(NLL)
        mean_error = sum(error) / len(error)
        mean_error_x = sum(error_x) / len(error_x)
        mean_error_h = sum(error_h) / len(error_h)
        if args.include_charges:
            mean_error_charges = sum(error_charges) / len(error_charges)
        mean_overall_accuracy = sum(overall_accuracy) / len(overall_accuracy)
        mean_overall_recall = sum(overall_recall) / len(overall_recall)
        mean_overall_f1 = sum(overall_f1) / len(overall_f1)
        mean_classwise_accuracy = {}
        for cls, acc_list in classwise_accuracy.items():
            if len(acc_list) > 0:
                mean_classwise_accuracy[cls] = sum(acc_list) / len(acc_list)
            else:
                mean_classwise_accuracy[cls] = float('nan')
        
        # save overall average loss
        with open(os.path.join(args.save_samples_dir, "loss.txt"), "w") as f:
            print(f"num samples          : {len(error)}", file=f)
            print(f"mean NLL             : {mean_NLL}", file=f)
            print(f"mean Error           : {mean_error}", file=f)
            print(f"mean Error X         : {mean_error_x}", file=f)
            print(f"mean Error H         : {mean_error_h}", file=f)
            print(f"mean Error H Charges : {mean_error_charges}", file=f) if args.include_charges else None
            print(f"mean Overall Accuracy: {mean_overall_accuracy}", file=f)
            print(f"mean Overall Recall  : {mean_overall_recall}", file=f)
            print(f"mean Overall F1      : {mean_overall_f1}", file=f)
            print(f"mean Classwise Accuracy", file=f)
            for cls, acc in mean_classwise_accuracy.items():
                print(f"    {cls} : {acc}", file=f)
            
            

    print(f">>> validation set test took {time.time() - start:.1f} seconds.")
    print("DONE.")



if __name__ == "__main__":
    main()
