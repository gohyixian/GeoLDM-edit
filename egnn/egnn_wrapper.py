from torch import nn
import torch
import math
from torch.utils.checkpoint import checkpoint
from global_registry import PARAM_REGISTRY
from egnn.egnn_new import EGNN, GNN, SinusoidsEmbeddingNew, low_vram_forward, \
                          checkpoint_equiv_block, coord2diff, unsorted_segment_sum
from egnn.egnn_fusion import EGNN_Fusion, coord2diff_fusion, checkpoint_fusion_block


class ControlNet_Arch_Wrapper(nn.Module):
    """A wrapper class that manages the forward pass for LDM, Controlnet and Fusion Blocks"""
    def __init__(self, diffusion_network, control_network, fusion_network, fusion_weights=[], fusion_mode='scaled_sum', noise_injection_weights=[0.5, 0.5], noise_injection_aggregation_method='mean', noise_injection_normalization_factor=1.):
        super(ControlNet_Arch_Wrapper, self).__init__()
        
        self.allowed_fusion_modes = ['scaled_sum',      # (h1_i,x1_i) = (h1_i,x1_i) + w_i * (f_h1_i,f_x1_i)
                                     'balanced_sum',    # (h1_i,x1_i) = [(1 - w_i) * (h1_i,x1_i)] + [w_i * (f_h1_i,f_x1_i)]
                                     'replace'          # (h1_i,x1_i) = (f_h1_i,f_x1_i)
                                     ]
        
        if not isinstance(diffusion_network, EGNN):
            raise NotImplementedError()
        if not isinstance(control_network, EGNN):
            raise NotImplementedError()
        if not isinstance(fusion_network, EGNN_Fusion):
            raise NotImplementedError()
        if fusion_mode not in self.allowed_fusion_modes:
            raise NotImplementedError()
        
        assert (diffusion_network.n_layers == control_network.n_layers == fusion_network.n_layers), \
            f"Different Number of Blocks encountered: diff={diffusion_network.n_layers} control={control_network.n_layers} fusion={fusion_network.n_layers}"
        assert len(fusion_weights) == fusion_network.n_layers, \
            f"Different Number of weights for Fusion encountered: len(fusion_weights)={len(fusion_weights)} fusion={fusion_network.n_layers}"
        
        self.diffusion_net = diffusion_network
        self.control_net = control_network
        self.fusion_net = fusion_network
        self.fusion_mode = fusion_mode
        if self.fusion_mode in ['scaled_sum', 'balanced_sum']:
            self.fusion_weights = [float(w) for w in fusion_weights]
        else:
            self.fusion_weights = None
        self.n_layers = diffusion_network.n_layers
        self.noise_injection_weights = noise_injection_weights
        self.noise_injection_aggregation_method = noise_injection_aggregation_method
        self.noise_injection_normalization_factor = noise_injection_normalization_factor


    def forward(self, h1, x1, h2, x2, node_mask_1=None, node_mask_2=None, edge_mask_1=None, edge_mask_2=None, 
                edge_index_1=None, edge_index_2=None, joint_edge_index=None, joint_edge_mask=None):

        # === Embeddings & Edge Attrs ===
        # Ligands (h1,x1)
        distances_1, _ = coord2diff(x1, edge_index_1)
        if self.diffusion_net.sin_embedding is not None:
            distances_1 = self.diffusion_net.sin_embedding(distances_1)
        h1 = self.diffusion_net.embedding(h1)

        # Pockets (h2,x2)
        distances_2, _ = coord2diff(x2, edge_index_2)
        if self.control_net.sin_embedding is not None:
            distances_2 = self.control_net.sin_embedding(distances_2)
        h2 = self.control_net.embedding(h2)

        # Fusion
        distances_joint, _ = coord2diff_fusion(x1, x2, joint_edge_index)
        if self.fusion_net.sin_embedding is not None:
            distances_joint = self.fusion_net.sin_embedding(distances_joint)


        # === Initial Noise Injection into Clean Condition ===
        # initial injection of LDM noise into Condition / essential feedback for ControlNet
        # TODO: improve 
        n1, n2 = joint_edge_index  # n1:ligands n2:pockets
        h2_shape, x2_shape = h2.shape, x2.shape
        assert h1[n1].shape == h2[n2].shape, f"Different sizes! h1[n1]={h1[n1].shape} h2[n2]={h2[n2].shape}"
        assert x1[n1].shape == x2[n2].shape, f"Different sizes! x1[n1]={x1[n1].shape} x2[n2]={x2[n2].shape}"

        noise_injected_h2 = (self.noise_injection_weights[0] * h1[n1]) + (self.noise_injection_weights[1] * h2[n2])
        noise_injected_x2 = (self.noise_injection_weights[0] * x1[n1]) + (self.noise_injection_weights[1] * x2[n2])  # embedded x, use 50/50 weights to maintain value range

        agg_h = unsorted_segment_sum(noise_injected_h2, n2, num_segments=h2_shape[0],
                                     normalization_factor=self.noise_injection_normalization_factor,  # 1. (unused)
                                     aggregation_method=self.noise_injection_aggregation_method)      # mean
        agg_x = unsorted_segment_sum(noise_injected_x2, n2, num_segments=x2_shape[0],
                                     normalization_factor=self.noise_injection_normalization_factor,  # 1. (unused)
                                     aggregation_method=self.noise_injection_aggregation_method)      # mean

        assert h2.shape == agg_h.shape, f"Different sizes! h2={h2.shape} agg_h={agg_h.shape}"
        assert x2.shape == agg_x.shape, f"Different sizes! x2={x2.shape} agg_x={agg_x.shape}"

        h2 = (0. * h2) + (1. * agg_h)
        x2 = (0. * x2) + (1. * agg_x)


        # === Layers Foward Pass ===
        use_ckpt = PARAM_REGISTRY.get('use_checkpointing')
        ckpt_mode = PARAM_REGISTRY.get('checkpointing_mode')

        for i in range(0, self.n_layers):
            # Pocket (h2,x2) ControlNet
            if use_ckpt and (ckpt_mode == 'sqrt') and ((i+1) % int(math.sqrt(self.n_layers)) == 0) and self.n_layers > 1:
                print(f"            >>> EGNN [ControlNet] e_block_{i} ... h2:{h2.shape}   x2:{x2.shape} ... CHECKPOINTING") if PARAM_REGISTRY.get('verbose')==True else None
                h2, x2 = checkpoint(checkpoint_equiv_block, 
                                  (self.control_net._modules["e_block_%d" % i], h2, x2, edge_index_2, node_mask_2, edge_mask_2, distances_2), 
                                  use_reentrant=False)
            elif use_ckpt and (ckpt_mode == 'all'):
                print(f"            >>> EGNN [ControlNet] e_block_{i} ... h2:{h2.shape}   x2:{x2.shape} ... CHECKPOINTING") if PARAM_REGISTRY.get('verbose')==True else None
                h2, x2 = checkpoint(checkpoint_equiv_block, 
                                  (self.control_net._modules["e_block_%d" % i], h2, x2, edge_index_2, node_mask_2, edge_mask_2, distances_2), 
                                  use_reentrant=False)
            else:
                print(f"            >>> EGNN [ControlNet] e_block_{i} ... h2:{h2.shape}   x2:{x2.shape}") if PARAM_REGISTRY.get('verbose')==True else None
                h2, x2 = self.control_net._modules["e_block_%d" % i](h2, x2, edge_index_2, node_mask=node_mask_2, edge_mask=edge_mask_2, edge_attr=distances_2)



            # Fusion (fh,fx) Fusion Blocks
            if use_ckpt and (ckpt_mode == 'sqrt') and ((i+1) % int(math.sqrt(self.n_layers)) == 0) and self.n_layers > 1:
                print(f"            >>> EGNN [FusionBlock] fusion_e_block_{i} ... h2:{h2.shape}   x2:{x2.shape}   h1:{h1.shape}   x1:{x1.shape} ... CHECKPOINTING") if PARAM_REGISTRY.get('verbose')==True else None
                fh, fx = checkpoint(checkpoint_fusion_block, 
                                  (self.fusion_net._modules["fusion_e_block_%d" % i], h1, h2, x1, x2, joint_edge_index, node_mask_1, joint_edge_mask, distances_joint), 
                                  use_reentrant=False)
            elif use_ckpt and (ckpt_mode == 'all'):
                print(f"            >>> EGNN [FusionBlock] fusion_e_block_{i} ... h2:{h2.shape}   x2:{x2.shape}   h1:{h1.shape}   x1:{x1.shape} ... CHECKPOINTING") if PARAM_REGISTRY.get('verbose')==True else None
                fh, fx = checkpoint(checkpoint_fusion_block, 
                                  (self.fusion_net._modules["fusion_e_block_%d" % i], h1, h2, x1, x2, joint_edge_index, node_mask_1, joint_edge_mask, distances_joint), 
                                  use_reentrant=False)
            else:
                print(f"            >>> EGNN [FusionBlock] fusion_e_block_{i} ... h2:{h2.shape}   x2:{x2.shape}   h1:{h1.shape}   x1:{x1.shape}") if PARAM_REGISTRY.get('verbose')==True else None
                fh, fx = self.fusion_net._modules["fusion_e_block_%d" % i](h1, h2, x1, x2, joint_edge_index, node_mask_1, joint_edge_mask, distances_joint)


            assert fh.shape == h1.shape, f"Different sizes! fh={fh.shape} h1={h1.shape}"
            assert fx.shape == x1.shape, f"Different sizes! fx={fx.shape} x1={x1.shape}"


            # fusion
            if self.fusion_mode == 'scaled_sum':
                h1 = h1 + (self.fusion_weights[i] * fh)
                x1 = x1 + (self.fusion_weights[i] * fx)
            elif self.fusion_mode == 'balanced_sum':
                h1 = ((1. - self.fusion_weights[i]) * h1) + (self.fusion_weights[i] * fh)
                x1 = ((1. - self.fusion_weights[i]) * x1) + (self.fusion_weights[i] * fx)
            elif self.fusion_mode == 'replace':
                h1 = (0. * h1) + (1. * fh)
                x1 = (0. * x1) + (1. * fx)
            else:
                raise NotImplementedError()


            # Ligand (h1,x1) Diffusion LDM
            if use_ckpt and (ckpt_mode == 'sqrt') and ((i+1) % int(math.sqrt(self.n_layers)) == 0) and self.n_layers > 1:
                print(f"            >>> EGNN [Diffusion] e_block_{i} ... h1:{h1.shape}   x1:{x1.shape} ... CHECKPOINTING") if PARAM_REGISTRY.get('verbose')==True else None
                h1, x1 = checkpoint(checkpoint_equiv_block, 
                                  (self.diffusion_net._modules["e_block_%d" % i], h1, x1, edge_index_1, node_mask_1, edge_mask_1, distances_1), 
                                  use_reentrant=False)
            elif use_ckpt and (ckpt_mode == 'all'):
                print(f"            >>> EGNN [Diffusion] e_block_{i} ... h1:{h1.shape}   x1:{x1.shape} ... CHECKPOINTING") if PARAM_REGISTRY.get('verbose')==True else None
                h1, x1 = checkpoint(checkpoint_equiv_block, 
                                  (self.diffusion_net._modules["e_block_%d" % i], h1, x1, edge_index_1, node_mask_1, edge_mask_1, distances_1), 
                                  use_reentrant=False)
            else:
                print(f"            >>> EGNN [Diffusion] e_block_{i} ... h1:{h1.shape}   x1:{x1.shape}") if PARAM_REGISTRY.get('verbose')==True else None
                h1, x1 = self.diffusion_net._modules["e_block_%d" % i](h1, x1, edge_index_1, node_mask=node_mask_1, edge_mask=edge_mask_1, edge_attr=distances_1)

        # Important, the bias of the last linear might be non-zero
        h1 = self.diffusion_net.embedding_out(h1)
        # h = low_vram_forward(self.embedding_out, h)
        
        if node_mask_1 is not None:
            h1 = h1 * node_mask_1
        return h1, x1