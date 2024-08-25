import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch import nn
import torch
import math
from torch.utils.checkpoint import checkpoint
# from global_registry import PARAM_REGISTRY
PARAM_REGISTRY = {}




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

    def edge_model(self, source, target, edge_attr, edge_mask):
        # h[row]=source, h[col]=target
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

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        # edge_attr: [bs*27*27, 256]
        # aggregate: sum / normalization_factor=1
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,  # 1
                                   aggregation_method=self.aggregation_method)      # sum
        if node_attr is not None: # None
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        # out = x + low_vram_forward(self.node_mlp, agg)
        
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        # node_attr = None
        # bs=64, n_nodes=27
        # 256 because there is an embedding layer in EGNN called self.embedding
        # print(">>", h.shape,       row.shape,          col.shape,          h[row].shape,            h[col].shape,            edge_attr.shape,       edge_mask.shape)
        # >> torch.Size([1728, 256]) torch.Size([46656]) torch.Size([46656]) torch.Size([46656, 256]) torch.Size([46656, 256]) torch.Size([46656, 2]) torch.Size([46656, 1])
        #                64x27                   64x27x27                                64x27x27
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


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

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:  # true
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
            # trans = coord_diff * torch.tanh(low_vram_forward(self.coord_mlp, input_tensor)) * self.coords_range
            
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
            # trans = coord_diff * low_vram_forward(self.coord_mlp, input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        coord = coord + agg
        return coord

    def forward(self, h, coord, edge_index, coord_diff, edge_attr=None, node_mask=None, edge_mask=None):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum'):
        super(EquivariantBlock, self).__init__()
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

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask)
        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super(EGNN, self).__init__()
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
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)   # [256,256] for enc [256,6] for dec, [256, 2+(nf+?)] for ldm
        for i in range(0, n_layers):   # 1 for enc, 9 for dec & ldm
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf,     # 256
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
        
        # sparsity plot check
        # if PARAM_REGISTRY.get('plot_sparsity') == True or PARAM_REGISTRY.get('save_activations') == True:
        #     self.layer_id = PARAM_REGISTRY._registry.get('layer_id_counter')
        #     PARAM_REGISTRY._registry['layer_id_counter'] = PARAM_REGISTRY._registry.get('layer_id_counter') + 1  # increase for next EGNN block


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


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]     # feeding this to model, relative difference
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    # ~!fp16
    norm = torch.sqrt(radial + 1e-8)
    
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff


    
def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    # unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),..)
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    # num_segments: bs * num_nodes
    # data.size(1): 256
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor of shape result_shape, all with value 0, and same data type as data
    #         say (100)       (100, 1)      (100, 256)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))   # [bs*n_nodes*n_nodes, 256]
    segment_ids = segment_ids.to(data.device)
    # since here we have n_nodes=5, meaning each molecule has 5 atoms / 5 nodes
    #
    #               Molecule 1: [0,1,2,3,4]                                                        Molecule 2: [5,6,7,8,9]
    #
    #  row: tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,     5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9]), 
    #               <----------->  <----------->  <----------->  <----------->  <----------->      <----------->  <----------->  <----------->  <----------->  <----------->  
    #               sum-to-idx-0   sum-to-idx-1   sum-to-idx-2   sum-to-idx-3   sum-to-idx-4       sum-to-idx-5   sum-to-idx-6   sum-to-idx-7   sum-to-idx-8   sum-to-idx-9   
    #               <----------------------------------------------------------------------->      <----------------------------------------------------------------------->
    #                           all possible node combinations in molecule 1                                   all possible node combinations in molecule 2
    #               <------------------------------------------------------------------------------------------------------------------------------------------------------>
    #                                                                        batch size = 2   (2 molecules per batch)
    result.scatter_add_(0, segment_ids, data)      # runs on dim=0, collapes dim 0 from bs*n_nodes*n_nodes to bs*n_nodes
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))   # N, sum up number of elems, i.e.  (..sum..) / N  <--
        norm[norm == 0] = 1  # 0 div error
        result = result / norm
    return result



class EGNN_encoder_QM9(nn.Module):
    def __init__(self, in_node_nf, context_node_nf, out_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 tanh=False, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum',
                 include_charges=True):
        '''
        :param in_node_nf: Number of invariant features for input nodes.'''
        super().__init__()

        include_charges = int(include_charges)      # 1
        num_classes = in_node_nf - include_charges      # 6-1 = 5 [HCNOF]

        self.mode = mode
        if mode == 'egnn_dynamics':   # this
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf,    # 6+(nf+?)
                out_node_nf=hidden_nf,      # 256
                in_edge_nf=1, 
                hidden_nf=hidden_nf,     # 256
                device=device, 
                act_fn=act_fn,    # torch.nn.SiLU()
                n_layers=n_layers,   # 1
                attention=attention,   # true
                tanh=tanh,     # true
                norm_constant=norm_constant,  # 1
                inv_sublayers=inv_sublayers,  # 1
                sin_embedding=sin_embedding,  # false
                normalization_factor=normalization_factor,  # 1
                aggregation_method=aggregation_method)      # sum
            self.in_node_nf = in_node_nf
        elif mode == 'gnn_dynamics':
            raise NotImplementedError()
            # self.gnn = GNN(
            #     in_node_nf=in_node_nf + context_node_nf + 3, 
            #     out_node_nf=hidden_nf + 3, 
            #     in_edge_nf=0, 
            #     hidden_nf=hidden_nf, 
            #     device=device,
            #     act_fn=act_fn, 
            #     n_layers=n_layers, 
            #     attention=attention,
            #     normalization_factor=normalization_factor, 
            #     aggregation_method=aggregation_method)
        
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),    # 256, 256
            act_fn,
            nn.Linear(hidden_nf, out_node_nf * 2 + 1))   # 256, 1*2+1=3

        self.num_classes = num_classes           # 6-1 = 5 [HCNOF]
        self.include_charges = include_charges   # 1
        self.context_node_nf = context_node_nf   # nf+?
        self.device = device
        self.n_dims = n_dims                     # 3
        self._edges_dict = {}
        self.out_node_nf = out_node_nf           # 1
        # self.condition_time = condition_time
        
        # Dictionary to store activations
        self.input_activations = {}
        self.output_activations = {}

        # Register hooks to save activations
        # self._register_hooks()

    def _register_hooks(self):
        self.hook_handles = []
        def hook_fn(module, input, output, name):
            self.input_activations[name] = input[0].clone().to(torch.float32).detach().cpu().numpy()
            self.output_activations[name] = output.clone().to(torch.float32).detach().cpu().numpy()

        # register hooks on all layers to track, i.e. nn.Linear 
        for name, layer in self.named_modules():
            if isinstance(layer, PARAM_REGISTRY.get('vis_activations_instances')):
                handle = layer.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))
                print(name, handle)
                self.hook_handles.append(handle)
        return self.hook_handles







# Example input (adjust according to your actual input size)
x = torch.randn(32, 512)  # batch_size, feature_size

# Instantiate your GCL module
# gcl = GCL(input_nf=256, output_nf=256, hidden_nf=256, normalization_factor=1, aggregation_method='sum', attention=True)

# egnn = EGNN(in_node_nf=128, in_edge_nf=128, hidden_nf=128, device='cpu', act_fn=nn.SiLU(), n_layers=4, attention=True,
#                  norm_diff=True, out_node_nf=None, tanh=True, coords_range=15, norm_constant=1, inv_sublayers=1,
#                  sin_embedding=False, normalization_factor=1, aggregation_method='sum')

encoder = EGNN_encoder_QM9(in_node_nf=128, context_node_nf=128, out_node_nf=128,
                 n_dims=3, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=True,
                 tanh=True, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=1, sin_embedding=False, normalization_factor=100, aggregation_method='sum',
                 include_charges=False)
# # Run the forward pass (replace with actual forward pass)
# gcl(x)

# # Plot histograms of the activations
# for name, activations in gcl.activations.items():
#     plt.figure()
#     plt.hist(activations.flatten(), bins=50)
#     plt.title(f'Activation distribution: {name}')
#     plt.show()
