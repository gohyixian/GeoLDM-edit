import wandb
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import numpy as np
import qm9.visualizer as vis
from qm9.analyze import analyze_stability_for_molecules
from qm9.sampling import sample_chain, sample, sample_sweep_conditional, sample_controlnet
import utils
import qm9.utils as qm9utils
from qm9 import losses
import time
import math
import torch
from global_registry import PARAM_REGISTRY
import subprocess
import gc
import os
import matplotlib.pyplot as plt
from tqdm import tqdm



def train_epoch_controlnet(args, loader, epoch, model, model_dp, model_ema, ema, device, dtype, property_norms, optim,
                           nodes_dist, gradnorm_queue, dataset_info, prop_dist):
    model_dp.train()
    model.train()
    nll_epoch = []
    n_iterations = len(loader)
    optim.zero_grad()

    for i, data in enumerate(loader):
        lg_x = data['ligand']['positions'].to(device, dtype)
        lg_node_mask = data['ligand']['atom_mask'].to(device, dtype).unsqueeze(2)
        lg_edge_mask = data['ligand']['edge_mask'].to(device, dtype)
        lg_one_hot = data['ligand']['one_hot'].to(device, dtype)
        lg_charges = (data['ligand']['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

        pkt_x = data['pocket']['positions'].to(device, dtype)
        pkt_node_mask = data['pocket']['atom_mask'].to(device, dtype).unsqueeze(2)
        pkt_edge_mask = data['pocket']['edge_mask'].to(device, dtype)
        pkt_one_hot = data['pocket']['one_hot'].to(device, dtype)
        pkt_charges = (data['pocket']['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

        joint_edge_mask = data['joint_edge_mask'].to(device, dtype)
        
        lg_x = remove_mean_with_mask(lg_x, lg_node_mask)
        pkt_x = remove_mean_with_mask(pkt_x, pkt_node_mask)
        

        if args.augment_noise > 0:
            raise NotImplementedError()
            # # Add noise eps ~ N(0, augment_noise) around points.
            # eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            # x = x + eps * args.augment_noise

        lg_x = remove_mean_with_mask(lg_x, lg_node_mask)
        pkt_x = remove_mean_with_mask(pkt_x, pkt_node_mask)
        
        if args.data_augmentation:
            lg_x = utils.random_rotation(lg_x).detach()
            pkt_x = utils.random_rotation(pkt_x).detach()

        check_mask_correct([lg_x, lg_one_hot, lg_charges], lg_node_mask)
        check_mask_correct([pkt_x, pkt_one_hot, pkt_charges], pkt_node_mask)
        assert_mean_zero_with_mask(lg_x, lg_node_mask)
        assert_mean_zero_with_mask(pkt_x, pkt_node_mask)

        lg_h = {'categorical': lg_one_hot, 'integer': lg_charges}
        pkt_h = {'categorical': pkt_one_hot, 'integer': pkt_charges}

        if len(args.conditioning) > 0:
            raise NotImplementedError()
            # context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            # assert_correctly_masked(context, node_mask)
        else:
            context = None

        # optim.zero_grad()

        # ~!mp
        # transform batch through flow
        nll, reg_term, mean_abs_z = losses.compute_loss_and_nll_controlnet(
            args=args,
            generative_model=model_dp,
            nodes_dist=nodes_dist,
            lg_x=lg_x, lg_h=lg_h,
            pkt_x=pkt_x, pkt_h=pkt_h,
            lg_node_mask=lg_node_mask,
            pkt_node_mask=pkt_node_mask,
            lg_edge_mask=lg_edge_mask,
            pkt_edge_mask=pkt_edge_mask,
            joint_edge_mask=joint_edge_mask,
            context=context
        )

        # standard nll from forward KL
        loss = nll + args.ode_regularization * reg_term

        # gradient penalty
        if args.grad_penalty:
            # Creates gradients
            grad_params = torch.autograd.grad(outputs=loss,
                                            inputs=model_dp.parameters(),
                                            create_graph=True)

            # Computes the penalty term and adds it to the loss
            with torch.autocast(device_type=PARAM_REGISTRY.get('device_'), dtype=PARAM_REGISTRY.get('mixed_precision_autocast_dtype', alt=torch.float16), enabled=PARAM_REGISTRY.get('mixed_precision_training')):
                grad_norm_gp = 0
                for grad in grad_params:
                    grad_norm_gp += grad.pow(2).sum()
                grad_norm_gp = grad_norm_gp.sqrt()
                loss = loss + grad_norm_gp

        # gradient accumulation loss scaling
        if args.grad_accumulation_steps > 0:
            loss = loss / int(args.grad_accumulation_steps)

        # ~!mp
        loss.backward()
        
        print(f"1. NaN in .grad  = {any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)}")
        print(f"2. NaN in params = {any(torch.isnan(p).any() for p in model.parameters())}")
        
        # gpu usage monitoring
        smi_txt = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8')
        
        # if args.clip_grad:
        #     grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        # else:
        grad_norm = 0.

        # ~!mp
        if ((i+1) % args.grad_accumulation_steps == 0) or ((i+1) == len(loader)) or args.break_train_epoch:
            if args.clip_grad:
                grad_norm = utils.gradient_clipping(model, gradnorm_queue)
            
            optim.step()
            optim.zero_grad()
            print(f">> Optimizer Step taken")
            
            # Update EMA if enabled.
            if args.ema_decay > 0:
                ema.update_model_average(model_ema, model)


        print(f"3. NaN in params = {any(torch.isnan(p).any() for p in model.parameters())}")


        if i % args.n_report_steps == 0:
            print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                  f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                  f"RegTerm: {reg_term.item():.1f}, "
                  f"GradNorm: {grad_norm:.1f}")

            # nvidia-smi 
            print(f">> MEM Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB    Reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
            print(smi_txt)

        nll_epoch.append(nll.item())
        nll_item = nll.item()

        # cleanup first
        del lg_x, lg_h, lg_node_mask, lg_edge_mask, lg_one_hot, lg_charges
        del pkt_x, pkt_h, pkt_node_mask, pkt_edge_mask, pkt_one_hot, pkt_charges
        del joint_edge_mask, loss, nll, reg_term, mean_abs_z, grad_norm
        torch.cuda.empty_cache()
        gc.collect()

        smi_dict = utils.get_nvidia_smi_usage(smi_txt)
        wandb_dict = {}
        for k,v in smi_dict.items():
            wandb_dict[f"gpu/{k}-{v.get('total_mem')}MiB"] = v.get('used_mem')
        wandb_dict["Batch NLL"] = nll_item
        wandb.log(wandb_dict, commit=True)

        if args.break_train_epoch:
            break

    optim.zero_grad()
    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)

    # cleanup
    del nll_epoch
    torch.cuda.empty_cache()
    gc.collect()

    return n_iterations



def train_epoch(args, loader, loader_vis_activations, epoch, model, model_dp, model_ema, ema, device, dtype, property_norms, optim,
                nodes_dist, gradnorm_queue, dataset_info, prop_dist):
    
    model_dp.train()
    model.train()
    nll_epoch = []
    n_iterations = len(loader)
    optim.zero_grad()
    training_mode = PARAM_REGISTRY.get('training_mode')
    loss_analysis = PARAM_REGISTRY.get('loss_analysis')
    loss_analysis_modes = PARAM_REGISTRY.get('loss_analysis_modes')
    
    for i, data in enumerate(loader):
        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
        
        x = remove_mean_with_mask(x, node_mask)

        if args.augment_noise > 0:
            raise NotImplementedError()
            # # Add noise eps ~ N(0, augment_noise) around points.
            # eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            # x = x + eps * args.augment_noise

        x = remove_mean_with_mask(x, node_mask)
        if args.data_augmentation:
            x = utils.random_rotation(x).detach()

        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'integer': charges}

        if len(args.conditioning) > 0:
            context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None

        # optim.zero_grad()

        # ~!mp
        # transform batch through flow
        if (training_mode in loss_analysis_modes) and loss_analysis:
            nll, reg_term, mean_abs_z, loss_dict = losses.compute_loss_and_nll(args, model_dp, nodes_dist,
                                                                    x, h, node_mask, edge_mask, context, loss_analysis)
        else:
            nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(args, model_dp, nodes_dist,
                                                                    x, h, node_mask, edge_mask, context)
        # standard nll from forward KL
        loss = nll + args.ode_regularization * reg_term

        # gradient penalty
        if args.grad_penalty:
            # Creates gradients
            grad_params = torch.autograd.grad(outputs=loss,
                                            inputs=model_dp.parameters(),
                                            create_graph=True)

            # Computes the penalty term and adds it to the loss
            with torch.autocast(device_type=PARAM_REGISTRY.get('device_'), dtype=PARAM_REGISTRY.get('mixed_precision_autocast_dtype', alt=torch.float16), enabled=PARAM_REGISTRY.get('mixed_precision_training')):
                grad_norm_gp = 0
                for grad in grad_params:
                    grad_norm_gp += grad.pow(2).sum()
                grad_norm_gp = grad_norm_gp.sqrt()
                loss = loss + grad_norm_gp

        # gradient accumulation loss scaling
        if args.grad_accumulation_steps > 0:
            loss = loss / int(args.grad_accumulation_steps)

        # ~!mp
        loss.backward()
        
        print(f"1. NaN in .grad  = {any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)}")
        print(f"2. NaN in params = {any(torch.isnan(p).any() for p in model.parameters())}")
        
        # gpu usage monitoring
        smi_txt = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8')
        
        
        # if args.clip_grad:
        #     grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        # else:
        grad_norm = 0.

        # ~!mp
        if ((i+1) % args.grad_accumulation_steps == 0) or ((i+1) == len(loader)) or args.break_train_epoch:
            if args.clip_grad:
                grad_norm = utils.gradient_clipping(model, gradnorm_queue)
            
            optim.step()
            optim.zero_grad()
            print(f">> Optimizer Step taken")
            
            # Update EMA if enabled.
            if args.ema_decay > 0:
                ema.update_model_average(model_ema, model)


        print(f"3. NaN in params = {any(torch.isnan(p).any() for p in model.parameters())}")

        if i % args.n_report_steps == 0:
            print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                  f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                  f"RegTerm: {reg_term.item():.1f}, "
                  f"GradNorm: {grad_norm:.1f}")
            
            # nvidia-smi 
            print(f">> MEM Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB    Reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
            print(smi_txt)
            

        nll_epoch.append(nll.item())
        nll_item = nll.item()
        
        # cleanup first
        del x, h, node_mask, edge_mask, one_hot, charges, loss, nll, reg_term, mean_abs_z, grad_norm
        torch.cuda.empty_cache()
        gc.collect()
        
        if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and not (epoch == 0 and i == 0) and args.train_diffusion:
            start = time.time()
            if len(args.conditioning) > 0:
                save_and_sample_conditional(args, device, model_ema, prop_dist, dataset_info, epoch=epoch)
            save_and_sample_chain(model_ema, args, device, dataset_info, prop_dist, epoch=epoch,
                                  batch_id=str(i))
            sample_different_sizes_and_save(model_ema, nodes_dist, args, device, dataset_info,
                                            prop_dist, epoch=epoch)
            print(f'Sampling took {time.time() - start:.2f} seconds')

            # vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_{i}", dataset_info=dataset_info, wandb=wandb)
            # vis.visualize_chain(f"outputs/{args.exp_name}/epoch_{epoch}_{i}/chain/", dataset_info, wandb=wandb)
            # if len(args.conditioning) > 0:
            #     vis.visualize_chain("outputs/%s/epoch_%d/conditional/" % (args.exp_name, epoch), dataset_info,
            #                         wandb=wandb, mode='conditional')

        # VAE visualise activations
        if args.vis_activations and (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and not (i == 0):

            # handle for saving intermediary activations
            handles = model_ema._register_hooks()
            save_and_vis_activations(args, loader_vis_activations, epoch, i, model_ema,\
                                     device, dtype, property_norms, nodes_dist)
            for handle in handles:
                handle.remove()


        smi_dict = utils.get_nvidia_smi_usage(smi_txt)
        wandb_dict = {}
        for k,v in smi_dict.items():
            wandb_dict[f"gpu/{k}-{v.get('total_mem')}MiB"] = v.get('used_mem')
        wandb_dict["Batch NLL"] = nll_item
        # loss_analysis
        if (training_mode in loss_analysis_modes) and loss_analysis:
            recon_loss_dict = loss_dict['recon_loss_dict']
            # print(f"training_test.py {recon_loss_dict['error_x'].shape}")
            wandb_dict['Train/error_x'] = recon_loss_dict['error_x'].mean().item()
            wandb_dict['Train/error_h_cat'] = recon_loss_dict['error_h_cat'].mean().item()
            if args.include_charges:
                wandb_dict['Train/error_h_int'] = recon_loss_dict['error_h_int'].mean().item()
            wandb_dict['Train/overall_accuracy'] = recon_loss_dict['overall_accuracy']
            wandb_dict['Train/overall_recall'] = recon_loss_dict['overall_recall']
            wandb_dict['Train/overall_f1'] = recon_loss_dict['overall_f1']
            for cls, metric in recon_loss_dict['classwise_accuracy'].items():
                wandb_dict[f'Train_classwise_accuracy/ {cls}'] = metric

        wandb.log(wandb_dict, commit=True)
        
        # cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        if args.break_train_epoch:
            break
    
    optim.zero_grad()
    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)
    
    # cleanup
    del nll_epoch
    torch.cuda.empty_cache()
    gc.collect()
    
    return n_iterations


def save_and_vis_activations(args, loader, epoch, iter, model_ema, device, dtype, property_norms, nodes_dist):

    for i, data in tqdm(enumerate(loader)):
        if i >= args.vis_activations_batch_samples:
            break

        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
        
        x = remove_mean_with_mask(x, node_mask)

        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'integer': charges}

        if len(args.conditioning) > 0:
            context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None

        # ~!mp
        # transform batch through flow
        nll, _, _ = losses.compute_loss_and_nll(args, model_ema, nodes_dist,
                                                x, h, node_mask, edge_mask, context)

        base_path_activations = os.path.join(args.save_activations_path, args.exp_name, "saved_activations", \
            f"epoch_{str(epoch).zfill(3)}_iter_{str(iter).zfill(10)}", f"sample_{str(i).zfill(3)}")
        base_path_plots = os.path.join(args.save_activations_path, args.exp_name, "plots", \
            f"epoch_{str(epoch).zfill(3)}_iter_{str(iter).zfill(10)}", f"sample_{str(i).zfill(3)}")
        if not os.path.exists(base_path_activations):
            os.makedirs(base_path_activations)
        if not os.path.exists(base_path_plots):
            os.makedirs(base_path_plots)

        # input activation
        for name, activations in model_ema.input_activations.items():
            original_options = np.get_printoptions()
            np.set_printoptions(threshold=np.inf, linewidth=np.inf)
            activations_str = np.array2string(activations, separator=', ')
            with open(os.path.join(base_path_activations, f"{name}__input.txt"), 'w') as f:
                f.write(activations_str)
            np.set_printoptions(**original_options)
            
            tensor_flat = activations.ravel()
            fig, ax = plt.subplots()
            ax.hist(tensor_flat, bins=args.vis_activations_bins)
            ax.set_title(f"{name} (Input)", fontsize=7)
            ax.set_xlabel('Activation Value')
            ax.set_ylabel('Frequency')
            plt.savefig(os.path.join(base_path_plots, f"{name}__input.png"))

            if args.vis_activations_specific_ylim:
                ymin = args.vis_activations_specific_ylim[0]
                ymax = args.vis_activations_specific_ylim[1]
                ax.set_ylim(ymin, ymax)
                plt.savefig(os.path.join(base_path_plots, f"{name}__input__y{ymin}_{ymax}.png"))
                plt.close()
            else:
                plt.close()
        # clear
        model_ema.input_activations = {}

        # output activation
        for name, activations in model_ema.output_activations.items():
            original_options = np.get_printoptions()
            np.set_printoptions(threshold=np.inf, linewidth=np.inf)
            activations_str = np.array2string(activations, separator=', ')
            with open(os.path.join(base_path_activations, f"{name}__output.txt"), 'w') as f:
                f.write(activations_str)
            np.set_printoptions(**original_options)
            
            tensor_flat = activations.ravel()
            fig, ax = plt.subplots()
            ax.hist(tensor_flat, bins=args.vis_activations_bins)
            ax.set_title(f"{name} (Output)", fontsize=7)
            ax.set_xlabel('Activation Value')
            ax.set_ylabel('Frequency')
            plt.savefig(os.path.join(base_path_plots, f"{name}__output.png"))

            if args.vis_activations_specific_ylim:
                ymin = args.vis_activations_specific_ylim[0]
                ymax = args.vis_activations_specific_ylim[1]
                ax.set_ylim(ymin, ymax)
                plt.savefig(os.path.join(base_path_plots, f"{name}__output__y{ymin}_{ymax}.png"))
                plt.close()
            else:
                plt.close()
        # clear
        model_ema.output_activations = {}

        # cleanup
        torch.cuda.empty_cache()
        gc.collect()



def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def test(args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist, partition='Test'):
    eval_model.eval()
    training_mode = PARAM_REGISTRY.get('training_mode')
    loss_analysis = PARAM_REGISTRY.get('loss_analysis')
    loss_analysis_modes = PARAM_REGISTRY.get('loss_analysis_modes')

    if (training_mode in loss_analysis_modes) and loss_analysis:
        error_x = []
        error_h_cat = []
        error_h_int = [] if args.include_charges else None
        overall_accuracy, overall_recall, overall_f1 = [], [], []
        classwise_accuracy = {}

    # ~!mp
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
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

            # transform batch through flow
            if (training_mode in loss_analysis_modes) and loss_analysis:
                nll, _, _, loss_dict = losses.compute_loss_and_nll(args, eval_model, nodes_dist, x, h,
                                                                   node_mask, edge_mask, context, loss_analysis)
            else:
                nll, _, _ = losses.compute_loss_and_nll(args, eval_model, nodes_dist, x, h,
                                                        node_mask, edge_mask, context)
            # standard nll from forward KL

            nll_epoch += (nll.item() * batch_size)
            n_samples += batch_size
            if i % args.n_report_steps == 0:
                print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"NLL: {nll_epoch/n_samples:.2f}")
            
            # cleanup
            del x, h, node_mask, edge_mask, one_hot, charges, nll
            torch.cuda.empty_cache()
            gc.collect()

            if (training_mode in loss_analysis_modes) and loss_analysis:
                recon_loss_dict = loss_dict['recon_loss_dict']
                error_x.append(recon_loss_dict['error_x'].mean().item())
                error_h_cat.append(recon_loss_dict['error_h_cat'].mean().item())
                error_h_int.append(recon_loss_dict['error_h_int'].mean().item()) if args.include_charges else None
                overall_accuracy.append(recon_loss_dict['overall_accuracy'])
                overall_recall.append(recon_loss_dict['overall_recall'])
                overall_f1.append(recon_loss_dict['overall_f1'])
                for cls, metric in recon_loss_dict['classwise_accuracy'].items():
                    if not math.isnan(metric):
                        classwise_accuracy[str(cls)] = classwise_accuracy.get(str(cls), []) + [metric]
                    else:
                        classwise_accuracy[str(cls)] = classwise_accuracy.get(str(cls), [])

    if (training_mode in loss_analysis_modes) and loss_analysis:
        wandb_dict = {}
        # loss_analysis
        if (training_mode in loss_analysis_modes) and loss_analysis:
            wandb_dict[f'{partition}/error_x'] = (sum(error_x) / len(error_x)) if len(error_x) > 0 else float('nan')
            wandb_dict[f'{partition}/error_h_cat'] = (sum(error_h_cat) / len(error_h_cat)) if len(error_h_cat) > 0 else float('nan')
            if args.include_charges:
                wandb_dict[f'{partition}/error_h_int'] = (sum(error_h_int) / len(error_h_int)) if len(error_h_int) > 0 else float('nan')
            wandb_dict[f'{partition}/overall_accuracy'] = (sum(overall_accuracy) / len(overall_accuracy)) if len(overall_accuracy) > 0 else float('nan')
            wandb_dict[f'{partition}/overall_recall'] = (sum(overall_recall) / len(overall_recall)) if len(overall_recall) > 0 else float('nan')
            wandb_dict[f'{partition}/overall_f1'] = (sum(overall_f1) / len(overall_f1)) if len(overall_f1) > 0 else float('nan')
            for cls, metric in classwise_accuracy.items():
                wandb_dict[f'{partition}_classwise_accuracy/ {cls}'] = (sum(metric) / len(metric)) if len(metric) > 0 else float('nan')

        return nll_epoch/n_samples, wandb_dict

    return nll_epoch/n_samples, None


def test_controlnet(args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist, partition='Test'):
    eval_model.eval()
    
    # ~!mp
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            lg_x = data['ligand']['positions'].to(device, dtype)
            lg_batch_size = lg_x.size(0)
            lg_node_mask = data['ligand']['atom_mask'].to(device, dtype).unsqueeze(2)
            lg_edge_mask = data['ligand']['edge_mask'].to(device, dtype)
            lg_one_hot = data['ligand']['one_hot'].to(device, dtype)
            lg_charges = (data['ligand']['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

            pkt_x = data['pocket']['positions'].to(device, dtype)
            pkt_batch_size = pkt_x.size(0)
            pkt_node_mask = data['pocket']['atom_mask'].to(device, dtype).unsqueeze(2)
            pkt_edge_mask = data['pocket']['edge_mask'].to(device, dtype)
            pkt_one_hot = data['pocket']['one_hot'].to(device, dtype)
            pkt_charges = (data['pocket']['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

            joint_edge_mask = data['joint_edge_mask'].to(device, dtype)

            assert lg_batch_size == pkt_batch_size, f"Different batch_size encountered! lg_batch_size={lg_batch_size} pkt_batch_size={pkt_batch_size}"

            if args.augment_noise > 0:
                raise NotImplementedError()
                # # Add noise eps ~ N(0, augment_noise) around points.
                # eps = sample_center_gravity_zero_gaussian_with_mask(x.size(),
                #                                                     x.device,
                #                                                     node_mask)
                # x = x + eps * args.augment_noise

            lg_x = remove_mean_with_mask(lg_x, lg_node_mask)
            pkt_x = remove_mean_with_mask(pkt_x, pkt_node_mask)
            check_mask_correct([lg_x, lg_one_hot, lg_charges], lg_node_mask)
            check_mask_correct([pkt_x, pkt_one_hot, pkt_charges], pkt_node_mask)
            assert_mean_zero_with_mask(lg_x, lg_node_mask)
            assert_mean_zero_with_mask(pkt_x, pkt_node_mask)

            lg_h = {'categorical': lg_one_hot, 'integer': lg_charges}
            pkt_h = {'categorical': pkt_one_hot, 'integer': pkt_charges}

            if len(args.conditioning) > 0:
                raise NotImplementedError()
                # context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
                # assert_correctly_masked(context, node_mask)
            else:
                context = None

            # transform batch through flow
            nll, _, _ = losses.compute_loss_and_nll_controlnet(
                args=args,
                generative_model=eval_model,
                nodes_dist=nodes_dist,
                lg_x=lg_x, lg_h=lg_h,
                pkt_x=pkt_x, pkt_h=pkt_h,
                lg_node_mask=lg_node_mask,
                pkt_node_mask=pkt_node_mask,
                lg_edge_mask=lg_edge_mask,
                pkt_edge_mask=pkt_edge_mask,
                joint_edge_mask=joint_edge_mask,
                context=context
            )

            # standard nll from forward KL
            nll_epoch += (nll.item() * lg_batch_size)
            n_samples += lg_batch_size
            if i % args.n_report_steps == 0:
                print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"NLL: {nll_epoch/n_samples:.2f}")

            # cleanup
            del lg_x, lg_h, lg_node_mask, lg_edge_mask, lg_one_hot, lg_charges
            del pkt_x, pkt_h, pkt_node_mask, pkt_edge_mask, pkt_one_hot, pkt_charges
            del joint_edge_mask, nll
            torch.cuda.empty_cache()
            gc.collect()

    return nll_epoch/n_samples



def save_and_sample_chain(model, args, device, dataset_info, prop_dist,
                          epoch=0, id_from=0, batch_id=''):
    # ~!mp
    one_hot, charges, x = sample_chain(args=args, device=device, flow=model,
                                    n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist)

    vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/',
                      one_hot, charges, x, dataset_info, id_from, name='chain')

    return one_hot, charges, x


def sample_different_sizes_and_save(model, nodes_dist, args, device, dataset_info, prop_dist,
                                    n_samples=5, epoch=0, batch_size=100, batch_id=''):
    batch_size = min(batch_size, n_samples)
    for counter in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)

        # ~!mp
        one_hot, charges, x, node_mask = sample(args, device, model, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample,
                                                dataset_info=dataset_info)
        print(f"Generated molecule: Positions {x[:-1, :, :]}")
        vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/', one_hot, charges, x, dataset_info,
                          batch_size * counter, name='molecule')
        
        del one_hot, charges, x, node_mask  # cleanup
        torch.cuda.empty_cache()
        gc.collect()


def analyze_and_save(epoch, model_sample, nodes_dist, args, device, dataset_info, prop_dist,
                     n_samples=1000, batch_size=100):
    print(f'Analyzing molecule stability at epoch {epoch}...')
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    for i in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        
        # ~!mp
        one_hot, charges, x, node_mask = sample(args, device, model_sample, dataset_info, prop_dist,
                                                nodesxsample=nodesxsample)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info)

    wandb.log(validity_dict)
    if rdkit_tuple is not None:
        wandb.log({'Validity': rdkit_tuple[0][0], 'Uniqueness': rdkit_tuple[0][1], 'Novelty': rdkit_tuple[0][2]})
    return validity_dict



def analyze_and_save_controlnet(epoch, model_sample, nodes_dist, args, device, dataset_info, prop_dist,
                                n_samples=1000, batch_size=100, pair_dict_list=[]):
    print(f'Analyzing molecule stability at epoch {epoch}...')
    batch_size = min(batch_size, n_samples)
    assert len(pair_dict_list) == n_samples
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    batch_id = 0
    for i in range(int(n_samples/batch_size)):
        # this returns the number of nodes. i.e. n_samples=3, return=tensor([16, 17, 15]) / tensor([14, 15, 19]) / tensor([17, 27, 18])
        nodesxsample = nodes_dist.sample(batch_size)

        pocket_dict_list = []
        for j in range(batch_size):
            pocket_dict_list.append(pair_dict_list[j + batch_id]['pocket'])

        # ~!mp
        one_hot, charges, x, node_mask = sample_controlnet(args, device, model_sample, dataset_info, prop_dist,
                                                nodesxsample=nodesxsample, context=None, fix_noise=False, pocket_dict_list=pocket_dict_list)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())
        batch_id += batch_size

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info)

    wandb.log(validity_dict)
    if rdkit_tuple is not None:
        wandb.log({'Validity': rdkit_tuple[0][0], 'Uniqueness': rdkit_tuple[0][1], 'Novelty': rdkit_tuple[0][2]})
    return validity_dict



def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, epoch=0, id_from=0):
    # ~!mp
    one_hot, charges, x, node_mask = sample_sweep_conditional(args, device, model, dataset_info, prop_dist)

    vis.save_xyz_file(
        'outputs/%s/epoch_%d/conditional/' % (args.exp_name, epoch), one_hot, charges, x, dataset_info,
        id_from, name='conditional', node_mask=node_mask)

    return one_hot, charges, x
