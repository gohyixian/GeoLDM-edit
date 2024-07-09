from torch import nn
import torch
import math
from torch.utils.checkpoint import checkpoint
from global_registry import PARAM_REGISTRY


def low_vram_forward(layer, tensor):
    """Chunks tensor into smaller sizes along dim=0, and performs forward 
       propagation before combining them back together.

    Args:
        layer (nn.module): Layer for forward propagation.
        tensor (nn.tensor): Input to layer.
        max_tensor_size (int, optional): Maximum chunk size, defaults to 50000.

    Returns:
        _type_: _description_
    """

    max_tensor_size = int(PARAM_REGISTRY.get('forward_tensor_chunk_size'))
    splits = list(torch.split(tensor, max_tensor_size, dim=0))

    for i, split in enumerate(splits):
        # ~!to
        # splits[i] = layer(split.to(layer_device)).to(tensor_device)
        splits[i] = layer(split)

    tensor = torch.cat(splits, dim=0)
    return tensor



def checkpoint_fusion_block(inputs):
    """Wrapper function for Fusion block checkpointing, used
       in EGNN.forward().

    Args:
        inputs (_type_): block & inputs wrapped in a single tuple

    Returns:
        _type_: Fusion_block(...inputs)
    """
    block, h1, h2, x1, x2, edge_index, node_mask_1, joint_edge_mask, distances = inputs
    return block(h1=h1, h2=h2, x1=x1, x2=x2, edge_index=edge_index, node_mask_1=node_mask_1, 
                 joint_edge_mask=joint_edge_mask, edge_attr=distances)



class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False):
        super(GCL, self).__init__()
        input_edge = input_nf * 2     # 256*2 = 512
        self.normalization_factor = normalization_factor    # 1
        self.aggregation_method = aggregation_method   # sum
        self.attention = attention   # true

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),   # 256*2+0=512, 256
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),    # 256, 256
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),    # 256+256+0=512, 256
            act_fn,
            nn.Linear(hidden_nf, output_nf))    # 256, 256

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),   # 256, 1
                nn.Sigmoid())
            # TODO: implement softmax here

    def edge_model(self, source, target, edge_attr, joint_edge_mask):
        # h1[n1]=source, h2[n2]=target
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)  # torch.Size([bs*27*27, 256+256])
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)
        # mij = low_vram_forward(self.edge_mlp, out)

        if self.attention:
            att_val = self.att_mlp(mij)
            # att_val = low_vram_forward(self.att_mlp, mij)
            
            out = mij * att_val
        else:
            out = mij

        if joint_edge_mask is not None:
            out = out * joint_edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        n1, n2 = edge_index
        # edge_attr: [bs*27*27, 256]
        # aggregate: sum / normalization_factor=1
        agg = unsorted_segment_sum(edge_attr, n1, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,  # 1
                                   aggregation_method=self.aggregation_method)      # sum
        if node_attr is not None: # None
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        # out = x + low_vram_forward(self.node_mlp, agg)
        
        return out, agg

    def forward(self, h1, h2, edge_index, edge_attr=None, node_attr=None, node_mask_1=None, joint_edge_mask=None):
        n1, n2 = edge_index
        # node_attr = None
        # bs=64, n_nodes=27
        # 256 because there is an embedding layer in EGNN called self.embedding
        # print(">>", h.shape,       row.shape,          col.shape,          h[row].shape,            h[col].shape,            edge_attr.shape,       edge_mask.shape)
        # >> torch.Size([1728, 256]) torch.Size([46656]) torch.Size([46656]) torch.Size([46656, 256]) torch.Size([46656, 256]) torch.Size([46656, 2]) torch.Size([46656, 1])
        #                64x27                   64x27x27                                64x27x27
        edge_feat, mij = self.edge_model(h1[n1], h2[n2], edge_attr, joint_edge_mask)
        h1, agg = self.node_model(h1, edge_index, edge_feat, node_attr)
        if node_mask_1 is not None:
            h1 = h1 * node_mask_1
        return h1, mij


class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range     # 15
        input_edge = hidden_nf * 2 + edges_in_d    # 256*2 + 2 = 514
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h1, h2, coord1, coord2, edge_index, coord_diff, edge_attr, joint_edge_mask):
        n1, n2 = edge_index
        input_tensor = torch.cat([h1[n1], h2[n2], edge_attr], dim=1)
        if self.tanh:  # true
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
            # trans = coord_diff * torch.tanh(low_vram_forward(self.coord_mlp, input_tensor)) * self.coords_range
            
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
            # trans = coord_diff * low_vram_forward(self.coord_mlp, input_tensor)
        if joint_edge_mask is not None:
            trans = trans * joint_edge_mask
        agg = unsorted_segment_sum(trans, n1, num_segments=coord1.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        coord1 = coord1 + agg
        return coord1

    def forward(self, h1, h2, coord1, coord2, edge_index, coord_diff, edge_attr=None, node_mask_1=None, joint_edge_mask=None):
        coord1 = self.coord_model(h1, h2, coord1, coord2, edge_index, coord_diff, edge_attr, joint_edge_mask)
        if node_mask_1 is not None:
            coord1 = coord1 * node_mask_1
        return coord1


class FusionBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum'):
        super(FusionBlock, self).__init__()
        self.hidden_nf = hidden_nf  # 256
        self.device = device
        self.n_layers = n_layers    # 1
        self.coords_range_layer = float(coords_range)   # 15
        self.norm_diff = norm_diff   # true
        self.norm_constant = norm_constant  # 1
        self.sin_embedding = sin_embedding  # false
        self.normalization_factor = normalization_factor  # 1
        self.aggregation_method = aggregation_method  # sum

        for i in range(0, n_layers):  # 1
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,  # 2
                                              act_fn=act_fn, attention=attention,  #
                                              normalization_factor=self.normalization_factor,  # 1
                                              aggregation_method=self.aggregation_method))     # sum
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, act_fn=nn.SiLU(), tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method))
        self.to(self.device)

    def forward(self, h1=None, h2=None, x1=None, x2=None, edge_index=None, node_mask_1=None, 
                joint_edge_mask=None, edge_attr=None):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff_fusion(x1, x2, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(0, self.n_layers):
            h1, _ = self._modules["gcl_%d" % i](h1, h2, edge_index, edge_attr, node_attr=None, node_mask_1=node_mask_1, joint_edge_mask=joint_edge_mask)
        x1 = self._modules["gcl_equiv"](h1, h2, x1, x2, edge_index, coord_diff, edge_attr, node_mask_1, joint_edge_mask)

        # Important, the bias of the last linear might be non-zero
        if node_mask_1 is not None:
            h1 = h1 * node_mask_1
        return h1, x1


class EGNN_Fusion(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super(EGNN_Fusion, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf  # 2+(nf+?) for ldm
        self.hidden_nf = hidden_nf  # 256
        self.device = device
        self.n_layers = n_layers    # 1 for enc, 9 for dec & ldm
        self.coords_range_layer = float(coords_range/n_layers) if n_layers > 0 else float(coords_range)  # 15
        self.norm_diff = norm_diff  # True
        self.normalization_factor = normalization_factor  # 1
        self.aggregation_method = aggregation_method      # sum

        if sin_embedding:   # false
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)        # [6+(nf+?), 256] for enc, [1+(nf+?), 256] for dec, # [2+(nf+?), 256] for ldm
        # self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)   # [256,256] for enc [256,6] for dec, [256, 2+(nf+?)] for ldm
        for i in range(0, n_layers):   # 1 for enc, 9 for dec & ldm
            self.add_module("fusion_e_block_%d" % i, FusionBlock(hidden_nf,     # 256
                                                                 edge_feat_nf=edge_feat_nf,    # 2
                                                                 device=device,
                                                                 act_fn=act_fn,     # torch.nn.SiLU()
                                                                 n_layers=inv_sublayers,  # 1
                                                                 attention=attention,  # true
                                                                 norm_diff=norm_diff,  # true
                                                                 tanh=tanh,    # true
                                                                 coords_range=coords_range,   # 15
                                                                 norm_constant=norm_constant,  # 1
                                                                 sin_embedding=self.sin_embedding,  # false
                                                                 normalization_factor=self.normalization_factor, # 1
                                                                 aggregation_method=self.aggregation_method))  # sum
        self.to(self.device)

    def forward(self, h1, x1, h2, x2, edge_index, node_mask_1=None, joint_edge_mask=None):
        # Edit Emiel: Remove velocity as input
        distances, _ = coord2diff_fusion(x1, x2, edge_index)
        if self.sin_embedding is not None:      # none
            distances = self.sin_embedding(distances)
        h1 = self.embedding(h1)
        h2 = self.embedding(h2)
        # h = low_vram_forward(self.embedding, h)
        
        intermediate_conditions = []
        
        for i in range(0, self.n_layers):
            # checkpointing at multiples of sqrt(n_layers) provides best perf (~30% wall time inc, ~60% vram decrease)
            if PARAM_REGISTRY.get('use_checkpointing') and \
                (PARAM_REGISTRY.get('checkpointing_mode') == 'sqrt') and \
                ((i+1) % int(math.sqrt(self.n_layers)) == 0) and \
                self.n_layers > 1:
                    
                print(f"            >>> EGNN fusion_e_block_{i} ... h1:{h1.shape} x1:{x1.shape}  h2:{h2.shape} x2:{x2.shape} ... CHECKPOINTING") if PARAM_REGISTRY.get('verbose')==True else None
                h1, x1 = checkpoint(checkpoint_fusion_block, 
                                  (self._modules["fusion_e_block_%d" % i], h1, h2, x1, x2, edge_index, node_mask_1, joint_edge_mask, distances), 
                                  use_reentrant=False)
                
            # checkpointing all blocks (not so optimal but helps if input size is too large)
            elif PARAM_REGISTRY.get('use_checkpointing') and \
                (PARAM_REGISTRY.get('checkpointing_mode') == 'all'):
                    
                print(f"            >>> EGNN fusion_e_block_{i} ... h1:{h1.shape} x1:{x1.shape}  h2:{h2.shape} x2:{x2.shape} ... CHECKPOINTING") if PARAM_REGISTRY.get('verbose')==True else None
                h1, x1 = checkpoint(checkpoint_fusion_block, 
                                  (self._modules["fusion_e_block_%d" % i], h1, h2, x1, x2, edge_index, node_mask_1, joint_edge_mask, distances), 
                                  use_reentrant=False)
                
            # no checkpointing done
            else:
                print(f"            >>> EGNN fusion_e_block_{i} ... h1:{h1.shape} x1:{x1.shape}  h2:{h2.shape} x2:{x2.shape}") if PARAM_REGISTRY.get('verbose')==True else None
                h1, x1 = self._modules["fusion_e_block_%d" % i](h1, h2, x1, x2, edge_index, node_mask_1, joint_edge_mask, distances)

            if node_mask_1 is not None:
                h1 = h1 * node_mask_1

            intermediate_conditions.append((h1, x1))
        return intermediate_conditions



class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        # ~!fp16
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


def coord2diff_fusion(x1, x2, edge_index, norm_constant=1):
    n1, n2 = edge_index
    coord_diff = x1[n1] - x2[n2]     # feeding this to model, relative difference
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    # ~!fp16
    norm = torch.sqrt(radial + 1e-8)
    
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff



    
def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor of shape result_shape, all with value 0, and same data type as data
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))   # [bs*n_nodes*n_nodes, 256]
    segment_ids = segment_ids.to(data.device)

    result.scatter_add_(0, segment_ids, data)      # runs on dim=0, collapes dim 0 from bs*n_nodes*n_nodes to bs*n_nodes
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))   # N, sum up number of elems, i.e.  (..sum..) / N  <--
        norm[norm == 0] = 1  # 0 div error
        result = result / norm
    return result
