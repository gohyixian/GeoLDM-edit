import torch


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    # ~!fp16
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8
    


def compute_loss_and_nll(args, generative_model, nodes_dist, x, h, node_mask, edge_mask, context):
    bs, n_nodes, n_dims = x.size()


    if args.probabilistic_model == 'diffusion':
        edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

        assert_correctly_masked(x, node_mask)

        # Here x is a position tensor, and h is a dictionary with keys
        # 'categorical' and 'integer'.
        # returns neg_log_pxh / negatve log likelihood
        nll = generative_model(x, h, node_mask, edge_mask, context)

        N = node_mask.squeeze(2).sum(1).long()

        log_pN = nodes_dist.log_prob(N)

        assert nll.size() == log_pN.size()
        nll = nll - log_pN

        # Average over batch.
        nll = nll.mean(0)

        reg_term = torch.tensor([0.]).to(nll.device)
        mean_abs_z = 0.
    else:
        raise ValueError(args.probabilistic_model)

    return nll, reg_term, mean_abs_z



def compute_loss_and_nll_controlnet(args, generative_model, nodes_dist, lg_x, lg_h, pkt_x, pkt_h, lg_node_mask, pkt_node_mask, lg_edge_mask, pkt_edge_mask, joint_edge_mask, context):
    lg_bs, lg_n_nodes, lg_n_dims = lg_x.size()
    pkt_bs, pkt_n_nodes, pkt_n_dims = pkt_x.size()

    assert lg_bs == pkt_bs, f"Different batch_size encountered! lg_bs={lg_bs} pkt_bs={pkt_bs}"
    assert lg_n_dims == pkt_n_dims, f"Different num embeddings encountered! lg_n_dims={lg_n_dims} pkt_n_dims={pkt_n_dims}"

    if args.probabilistic_model == 'diffusion':
        lg_edge_mask = lg_edge_mask.view(lg_bs, lg_n_nodes * lg_n_nodes)
        pkt_edge_mask = pkt_edge_mask.view(pkt_bs, pkt_n_nodes * pkt_n_nodes)
        joint_edge_mask = joint_edge_mask.view(lg_bs, lg_n_nodes * pkt_n_nodes)

        assert_correctly_masked(lg_x, lg_node_mask)
        assert_correctly_masked(pkt_x, pkt_node_mask)

        # Here x is a position tensor, and h is a dictionary with keys
        # 'categorical' and 'integer'.
        # returns neg_log_pxh / negatve log likelihood
        nll = generative_model(
            x1=lg_x, h1=lg_h, 
            x2=pkt_x, h2=pkt_h, 
            node_mask_1=lg_node_mask, 
            node_mask_2=pkt_node_mask, 
            edge_mask_1=lg_edge_mask, 
            edge_mask_2=pkt_edge_mask, 
            joint_edge_mask=joint_edge_mask, 
            context=context
            )

        N = lg_node_mask.squeeze(2).sum(1).long()

        log_pN = nodes_dist.log_prob(N)

        assert nll.size() == log_pN.size()
        nll = nll - log_pN

        # Average over batch.
        nll = nll.mean(0)

        reg_term = torch.tensor([0.]).to(nll.device)
        mean_abs_z = 0.
    else:
        raise ValueError(args.probabilistic_model)

    return nll, reg_term, mean_abs_z