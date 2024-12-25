import numpy as np
import torch
import torch.nn.functional as F
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
from qm9.analyze import check_stability
from qm9.data import collate as qm9_collate


def rotate_chain(z):
    assert z.size(0) == 1

    z_h = z[:, :, 3:]

    n_steps = 30
    theta = 0.6 * np.pi / n_steps
    Qz = torch.tensor(
        [[np.cos(theta), -np.sin(theta), 0.],
         [np.sin(theta), np.cos(theta), 0.],
         [0., 0., 1.]]
    ).float()
    Qx = torch.tensor(
        [[1., 0., 0.],
         [0., np.cos(theta), -np.sin(theta)],
         [0., np.sin(theta), np.cos(theta)]]
    ).float()
    Qy = torch.tensor(
        [[np.cos(theta), 0., np.sin(theta)],
         [0., 1., 0.],
         [-np.sin(theta), 0., np.cos(theta)]]
    ).float()

    Q = torch.mm(torch.mm(Qz, Qx), Qy)

    Q = Q.to(z.device)

    results = []
    results.append(z)
    for i in range(n_steps):
        z_x = results[-1][:, :, :3]
        # print(z_x.size(), Q.size())
        new_x = torch.matmul(z_x.view(-1, 3), Q.T).view(1, -1, 3)
        # print(new_x.size())
        new_z = torch.cat([new_x, z_h], dim=2)
        results.append(new_z)

    results = torch.cat(results, dim=0)
    return results


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


def sample_chain(args, device, flow, n_tries, dataset_info, prop_dist=None):
    n_samples = 1
    if args.dataset == 'qm9' or args.dataset == 'qm9_second_half' or args.dataset == 'qm9_first_half':
        n_nodes = 19
    elif args.dataset == 'geom' or 'ligand' in args.dataset.lower():
        n_nodes = 44  # applies to both Crossdocked & BindingMOAD, refer EDA
    else:
        raise NotImplementedError()

    # TODO FIX: This conditioning just zeros.
    if args.context_node_nf > 0:
        context = prop_dist.sample(n_nodes).unsqueeze(1).unsqueeze(0)
        context = context.repeat(1, n_nodes, 1).to(device)
        #context = torch.zeros(n_samples, n_nodes, args.context_node_nf).to(device)
    else:
        context = None

    node_mask = torch.ones(n_samples, n_nodes, 1).to(device)

    edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
    edge_mask = edge_mask.repeat(n_samples, 1, 1).view(-1, 1).to(device)

    if args.probabilistic_model == 'diffusion':
        one_hot, charges, x = None, None, None
        for i in range(n_tries):
            chain = flow.sample_chain(n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=100)
            chain = reverse_tensor(chain)

            # Repeat last frame to see final sample better.
            chain = torch.cat([chain, chain[-1:].repeat(10, 1, 1)], dim=0)
            x = chain[-1:, :, 0:3]
            one_hot = chain[-1:, :, 3:-1]
            one_hot = torch.argmax(one_hot, dim=2)

            atom_type = one_hot.squeeze(0).cpu().detach().numpy()
            x_squeeze = x.squeeze(0).cpu().detach().numpy()
            mol_stable = check_stability(x_squeeze, atom_type, dataset_info)[0]

            # Prepare entire chain.
            x = chain[:, :, 0:3]
            one_hot = chain[:, :, 3:-1]
            one_hot = F.one_hot(torch.argmax(one_hot, dim=2), num_classes=len(dataset_info['atom_decoder']))
            charges = torch.round(chain[:, :, -1:]).long()

            if mol_stable:
                print('Found stable molecule to visualize :)')
                break
            elif i == n_tries - 1:
                print('Did not find stable molecule, showing last sample.')

    else:
        raise ValueError

    return one_hot, charges, x


def sample(args, device, generative_model, dataset_info,
           prop_dist=None, nodesxsample=torch.tensor([10]), context=None,
           fix_noise=False):
    max_n_nodes = dataset_info['max_n_nodes']  # this is the maximum node_size in QM9

    assert int(torch.max(nodesxsample)) <= max_n_nodes
    batch_size = len(nodesxsample)

    node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        node_mask[i, 0:nodesxsample[i]] = 1

    # Compute edge_mask

    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)

    # TODO FIX: This conditioning just zeros.
    if args.context_node_nf > 0:
        if context is None:
            context = prop_dist.sample_batch(nodesxsample)
        context = context.unsqueeze(1).repeat(1, max_n_nodes, 1).to(device) * node_mask
    else:
        context = None

    generative_model.eval()
    with torch.no_grad():
        if args.probabilistic_model == 'diffusion':
            x, h = generative_model.sample(batch_size, max_n_nodes, node_mask, edge_mask, context, fix_noise=fix_noise)

            assert_correctly_masked(x, node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            one_hot = h['categorical']
            charges = h['integer']

            assert_correctly_masked(one_hot.float(), node_mask)
            if args.include_charges:
                assert_correctly_masked(charges.float(), node_mask)

        else:
            raise ValueError(args.probabilistic_model)

    return one_hot, charges, x, node_mask



def sample_controlnet(args, device, generative_model, dataset_info,
                      nodesxsample=torch.tensor([10]), context=None,
                      fix_noise=False, pocket_dict_list=[]):

    max_n_nodes = dataset_info['max_n_nodes']  # this is the maximum node_size

    # Pockets' ['positions'], ['one_hot'], ['charges'], ['atom_mask'] are already available
    assert int(torch.max(nodesxsample)) <= max_n_nodes
    batch_size = len(nodesxsample)

    assert batch_size == len(pocket_dict_list), f"Different batch_size encountered! batch_size={batch_size}, len(pocket_dict_list)={len(pocket_dict_list)}"

    # Ligand node_mask
    lg_node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        lg_node_mask[i, 0:nodesxsample[i]] = 1

    # Ligand edge_mask
    lg_edge_mask = lg_node_mask.unsqueeze(1) * lg_node_mask.unsqueeze(2)
    lg_diag_mask = ~torch.eye(lg_edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    lg_edge_mask *= lg_diag_mask
    lg_edge_mask = lg_edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
    lg_node_mask = lg_node_mask.unsqueeze(2).to(device)

    # Pocket: zero padding done here
    pocket_batch = {prop: qm9_collate.batch_stack([mol[prop] for mol in pocket_dict_list])
                    for prop in pocket_dict_list[0].keys()}
    pkt_x = pocket_batch['positions'].to(device, dtype=args.dtype)
    pkt_h_one_hot = pocket_batch['one_hot'].to(device, dtype=args.dtype)
    pkt_h_charges = (pocket_batch['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype=args.dtype)
    pkt_h = {'categorical': pkt_h_one_hot, 'integer': pkt_h_charges}

    pkt_node_mask = pocket_batch['atom_mask'].to(device)
    bs, pkt_n_nodes = pkt_node_mask.size()
    assert batch_size == bs, f"Different batch_size encountered! ligand={batch_size}, pocket={bs}"
    pkt_edge_mask = pkt_node_mask.unsqueeze(1) * pkt_node_mask.unsqueeze(2)
    pkt_diag_mask = ~torch.eye(pkt_edge_mask.size(1), dtype=torch.bool).unsqueeze(0).to(device)
    pkt_edge_mask *= pkt_diag_mask
    pkt_edge_mask = pkt_edge_mask.view(batch_size * pkt_n_nodes * pkt_n_nodes, 1).to(device)
    pkt_node_mask = pkt_node_mask.unsqueeze(2).to(device)

    joint_edge_mask = pkt_node_mask.unsqueeze(1) * lg_node_mask.unsqueeze(2)
    joint_edge_mask = joint_edge_mask.view(batch_size * max_n_nodes * pkt_n_nodes, 1).to(device)
    # ~!joint_edge_mask tested, same as:
    # edge_index = get_adj_matrix(n_nodes_1=3, n_nodes_2=2, batch_size=2)
    # n1, n2 = edge_index
    # joint_edge_mask_3 = ligand_atom_mask_batched[n1] * pocket_atom_mask_batched[n2]

    # center pocket coordinates
    pkt_x = remove_mean_with_mask(pkt_x, pkt_node_mask)
    assert_mean_zero_with_mask(pkt_x, pkt_node_mask)

    if args.context_node_nf > 0:
        raise NotImplementedError()
    else:
        context = None

    # set model to eval mode
    generative_model.eval()
    
    with torch.no_grad():
        if args.probabilistic_model == 'diffusion':
            x, h = generative_model.sample(n_samples=batch_size, 
                                        n_nodes=max_n_nodes, 
                                        x2=pkt_x, 
                                        h2=pkt_h, 
                                        node_mask_1=lg_node_mask, 
                                        node_mask_2=pkt_node_mask, 
                                        edge_mask_1=lg_edge_mask, 
                                        edge_mask_2=pkt_edge_mask, 
                                        joint_edge_mask=joint_edge_mask, 
                                        context=context, 
                                        fix_noise=fix_noise)

            assert_correctly_masked(x, lg_node_mask)
            assert_mean_zero_with_mask(x, lg_node_mask)

            one_hot = h['categorical']
            charges = h['integer']

            assert_correctly_masked(one_hot.float(), lg_node_mask)
            if args.include_charges:
                assert_correctly_masked(charges.float(), lg_node_mask)

        else:
            raise ValueError(args.probabilistic_model)

    return one_hot, charges, x, lg_node_mask



def sample_sweep_conditional(args, device, generative_model, dataset_info, prop_dist, n_nodes=19, n_frames=100):
    nodesxsample = torch.tensor([n_nodes] * n_frames)

    context = []
    for key in prop_dist.distributions:
        min_val, max_val = prop_dist.distributions[key][n_nodes]['params']
        mean, mad = prop_dist.normalizer[key]['mean'], prop_dist.normalizer[key]['mad']
        min_val = (min_val - mean) / (mad)
        max_val = (max_val - mean) / (mad)
        context_row = torch.tensor(np.linspace(min_val, max_val, n_frames)).unsqueeze(1)
        context.append(context_row)
    context = torch.cat(context, dim=1).float().to(device)

    one_hot, charges, x, node_mask = sample(args, device, generative_model, dataset_info, prop_dist, nodesxsample=nodesxsample, context=context, fix_noise=True)
    return one_hot, charges, x, node_mask