import wandb
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import numpy as np
import qm9.visualizer as vis
from qm9.analyze import analyze_stability_for_molecules
from qm9.sampling import sample_chain, sample, sample_sweep_conditional
import utils
import qm9.utils as qm9utils
from qm9 import losses
import time
import torch
from global_registry import PARAM_REGISTRY
import subprocess
import gc


def train_epoch(args, loader, epoch, model, model_dp, model_ema, ema, device, dtype, property_norms, optim,
                nodes_dist, gradnorm_queue, dataset_info, prop_dist, scaler):
    
    model_dp.train()
    model.train()
    nll_epoch = []
    n_iterations = len(loader)
    
    for i, data in enumerate(loader):
        # tmp
        if i > 500:
            break
        # ~!to ~!mp
        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
        
        x = remove_mean_with_mask(x, node_mask)

        # not used
        if args.augment_noise > 0:  # 0
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * args.augment_noise

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

        optim.zero_grad()

        # ~!mp
        print("01/5 - compute_loss_and_nll") if args.verbose else None
        if args.mixed_precision_training:
            with torch.autocast(device_type='cuda', dtype=torch.get_default_dtype()):
                # transform batch through flow
                nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(args, model_dp, nodes_dist,
                                                                        x, h, node_mask, edge_mask, context)
                # standard nll from forward KL
                loss = nll + args.ode_regularization * reg_term
        else:
            nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(args, model_dp, nodes_dist,
                                                                    x, h, node_mask, edge_mask, context)
            loss = nll + args.ode_regularization * reg_term
        print(f"%%%%% MASTER LOSS {torch.isnan(torch.Tensor(loss)).any()}") if args.verbose else None
        
        
        # ~!mp
        print("02/5 - loss.backward") if args.verbose else None
        if args.mixed_precision_training:
            # first scale loss by scale_factor, then compute backward pass for gradients
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        
        if args.clip_grad:
            print("03/5 - utils.gradient_clipping") if args.verbose else None
            # manually unscaling gradients for correct gradient clipping
            # https://pytorch.org/docs/2.2/notes/amp_examples.html#gradient-clipping
            if args.mixed_precision_training:
                scaler.unscale_(optim)
            grad_norm = utils.gradient_clipping(model, gradnorm_queue)
            
            if args.verbose:
                def check_grad(w):
                    if w.requires_grad == True:
                        if w.grad is not None:
                            return 1
                        else:
                            return 0
                    else:
                        return 1
                grad_check_clipped = [check_grad(w) for name,w in model.named_parameters()]
                print(f"GRADIENTS is not None:  Clipped={sum(grad_check_clipped)}/{len(grad_check_clipped)}")
                [print("    ", name, w.grad is not None) for name,w in model.named_parameters()]
        else:
            grad_norm = 0.

        # ~!mp
        print("04/5 - optim.step()") if args.verbose else None
        print(f"%%%%% MASTER WEIGHTS (B4) {int(bool(sum([1 if torch.isnan(w).any() else 0 for w in model.parameters()])))}") if args.verbose else None
        if args.mixed_precision_training:
            scaler.step(optim)
            scaler.update()
        else:
            optim.step()

        print(f"%%%%% MASTER WEIGHTS (A3) {int(bool(sum([1 if torch.isnan(w).any() else 0 for w in model.parameters()])))}") if args.verbose else None

        # Update EMA if enabled.
        if args.ema_decay > 0:
            print("05/5 - ema.update_model_average") if args.verbose else None
            ema.update_model_average(model_ema, model)

        if i % args.n_report_steps == 0:
            print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                  f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                  f"RegTerm: {reg_term.item():.1f}, "
                  f"GradNorm: {grad_norm:.1f}")
            # nvidia-smi 
            print(f">> MEM Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB    Reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
            print(subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8'))

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

            vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_{i}", dataset_info=dataset_info, wandb=wandb)
            vis.visualize_chain(f"outputs/{args.exp_name}/epoch_{epoch}_{i}/chain/", dataset_info, wandb=wandb)
            if len(args.conditioning) > 0:
                vis.visualize_chain("outputs/%s/epoch_%d/conditional/" % (args.exp_name, epoch), dataset_info,
                                    wandb=wandb, mode='conditional')
        # if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and not (epoch == 0 and i == 0) and args.train_diffusion:
        #     TEST_PATH = '/home/user/yixian.goh/geoldm-edit/outputs/20240602_EPOCH_WITH_TEST_cleanup_bf_VisChain_VisChain_loadMoleculeXYZ_with_adjusted_test/epoch_0_200'
        #     vis.visualize(TEST_PATH, dataset_info=dataset_info, wandb=wandb)
        #     vis.visualize_chain(f"{TEST_PATH}/chain/", dataset_info, wandb=wandb)
        #     if len(args.conditioning) > 0:
        #         vis.visualize_chain("outputs/%s/epoch_%d/conditional/" % (args.exp_name, epoch), dataset_info,
        #                             wandb=wandb, mode='conditional')
        
        # wandb.log({"Batch NLL": nll.item()}, commit=True)
        wandb.log({"Batch NLL": nll_item}, commit=True)
        
        
        # cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        if args.break_train_epoch:
            break
        
    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)
    
    # cleanup
    del nll_epoch
    torch.cuda.empty_cache()
    gc.collect()
    
    return n_iterations


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def test(args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist, partition='Test'):
    eval_model.eval()
    
    # ~!mp
    # with torch.no_grad(), torch.cuda.amp.autocast():
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            # tmp
            if i > 500:
                break
            x = data['positions'].to(device, dtype)
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

            if args.augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(x.size(),
                                                                    x.device,
                                                                    node_mask)
                x = x + eps * args.augment_noise

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
            nll, _, _ = losses.compute_loss_and_nll(args, eval_model, nodes_dist, x, h,
                                                    node_mask, edge_mask, context)
            # standard nll from forward KL

            nll_epoch += nll.item() * batch_size
            n_samples += batch_size
            if i % args.n_report_steps == 0:
                print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"NLL: {nll_epoch/n_samples:.2f}")
            
            # cleanup
            del x, h, node_mask, edge_mask, one_hot, charges, nll
            torch.cuda.empty_cache()
            gc.collect()

    return nll_epoch/n_samples


def save_and_sample_chain(model, args, device, dataset_info, prop_dist,
                          epoch=0, id_from=0, batch_id=''):
    # ~!mp
    # with torch.cuda.amp.autocast():
    one_hot, charges, x = sample_chain(args=args, device=device, flow=model,
                                    n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist)

    vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/',
                      one_hot, charges, x, dataset_info, id_from, name='chain')

    # return one_hot, charges, x
    del one_hot, charges, x   # cleanup
    torch.cuda.empty_cache()
    gc.collect()


def sample_different_sizes_and_save(model, nodes_dist, args, device, dataset_info, prop_dist,
                                    n_samples=5, epoch=0, batch_size=100, batch_id=''):
    batch_size = min(batch_size, n_samples)
    for counter in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        
        # ~!mp
        # with torch.cuda.amp.autocast():
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
        # with torch.cuda.amp.autocast():
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


def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, epoch=0, id_from=0):
    # ~!mp
    # with torch.cuda.amp.autocast():
    one_hot, charges, x, node_mask = sample_sweep_conditional(args, device, model, dataset_info, prop_dist)

    vis.save_xyz_file(
        'outputs/%s/epoch_%d/conditional/' % (args.exp_name, epoch), one_hot, charges, x, dataset_info,
        id_from, name='conditional', node_mask=node_mask)

    # return one_hot, charges, x
    del one_hot, charges, x, node_mask # cleanup
    torch.cuda.empty_cache()
    gc.collect()
