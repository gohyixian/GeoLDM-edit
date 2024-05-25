import torch
import torch.nn as nn
from egnn.egnn_new import EGNN, GNN, low_vram_forward
from equivariant_diffusion.utils import remove_mean, remove_mean_with_mask
import numpy as np
from global_registry import PARAM_REGISTRY


class EGNN_dynamics_QM9(nn.Module):
    def __init__(self, in_node_nf, context_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super().__init__()
        self.mode = mode
        if mode == 'egnn_dynamics':
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf,   # (args.latent_nf+time) + (nf+?) = 2+0 = 2
                in_edge_nf=1,  # unused
                hidden_nf=hidden_nf,   # 256
                device=device, 
                act_fn=act_fn,  # torch.nn.SiLU()
                n_layers=n_layers,    # 9
                attention=attention,  # true
                tanh=tanh,            # true
                norm_constant=norm_constant,  # 1
                inv_sublayers=inv_sublayers,  # 1
                sin_embedding=sin_embedding,  # false
                normalization_factor=normalization_factor, # 1
                aggregation_method=aggregation_method # sum
                )
            self.in_node_nf = in_node_nf
        elif mode == 'gnn_dynamics':
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3, in_edge_nf=0,
                hidden_nf=hidden_nf, out_node_nf=3 + in_node_nf, device=device,
                act_fn=act_fn, n_layers=n_layers, attention=attention,
                normalization_factor=normalization_factor, aggregation_method=aggregation_method)

        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, xh, node_mask, edge_mask, context):
        print(f"        >>> DYNAMICS t:{torch.isnan(t).any()}  xh:{torch.isnan(xh).any()}  node_mask:{torch.isnan(node_mask).any()}  edge_mask:{torch.isnan(edge_mask).any()}") if PARAM_REGISTRY.get('verbose')==True else None

        bs, n_nodes, dims = xh.shape
        # 64, 25, 4
        h_dims = dims - self.n_dims  # 4-3 = 1
        # 1
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        # ~!to ~!mp
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)    # [1600, 1]
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)  # [40000, 1]
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        # [1600, 3]
        if h_dims == 0:
            # ~!to ~!mp
            h = torch.ones(bs*n_nodes, 1).to(self.device)
            # h = torch.ones(bs*n_nodes, 1)
        else:
            h = xh[:, self.n_dims:].clone()
            # [1600, 1]
        
        # t.size()
        # random time samples of shape [64, 1] : bs, values ranging from 0. - 1.

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t value is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:  # this
                # t value is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)   # [64, 25]
                h_time = h_time.view(bs * n_nodes, 1) # [1600, 1]
            # h_time.shape
            # [1600, 1]
            h = torch.cat([h, h_time], dim=1)
            # [1600, 2]

        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)
        # h.shape
        # [1600, 2+nf+?]

        if self.mode == 'egnn_dynamics':
            print(f"        >>> DYNAMICS (B4) h:{torch.isnan(h).any()} x:{torch.isnan(x).any()}  node_mask:{torch.isnan(node_mask).any()}  edge_mask:{torch.isnan(edge_mask).any()}") if PARAM_REGISTRY.get('verbose')==True else None
            h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
            print(f"        >>> DYNAMICS (A3) h_:{torch.isnan(h_final).any()} x_:{torch.isnan(x_final).any()}  node_mask:{torch.isnan(node_mask).any()}  edge_mask:{torch.isnan(edge_mask).any()}") if PARAM_REGISTRY.get('verbose')==True else None
            # [1600, 2]  [1600, 3]
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        if context is not None:
            # Slice off context size:
            h_final = h_final[:, :-self.context_node_nf]
            
        # h_final.shape
        # [1600, 2]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]
        # h_final.shape
        # [1600, 1]

        vel = vel.view(bs, n_nodes, -1)
        # [64, 25, 3]

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero. (EGNN_dynamics_QM9)')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if h_dims == 0:
            return vel
        else: # this
            h_final = h_final.view(bs, n_nodes, -1)
            return torch.cat([vel, h_final], dim=2)

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                # ~!to ~!mp
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                # edges = [torch.LongTensor(rows),
                        #  torch.LongTensor(cols)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)


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
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3, 
                out_node_nf=hidden_nf + 3, 
                in_edge_nf=0, 
                hidden_nf=hidden_nf, 
                device=device,
                act_fn=act_fn, 
                n_layers=n_layers, 
                attention=attention,
                normalization_factor=normalization_factor, 
                aggregation_method=aggregation_method)
        
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

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, xh, node_mask, edge_mask, context):  
        print(f"        >>> ENCODER xh:{torch.isnan(xh).any()}  node_mask:{torch.isnan(node_mask).any()}  edge_mask:{torch.isnan(edge_mask).any()}") if PARAM_REGISTRY.get('verbose')==True else None
        bs, n_nodes, dims = xh.shape     # 64, 29, ?
        h_dims = dims - self.n_dims      # ? - 3
        edges = self.get_adj_matrix(n_nodes, bs, self.device)   # [row[bs*n_nodes*n_nodes], col[bs*n_nodes*n_nodes]]
        
        # everything passed into the model is flattened, for example to shape [bs*29, 1], [bs*29, ?]
        # ~!to ~!mp
        edges = [x.to(self.device) for x in edges]   # row, col
        node_mask = node_mask.view(bs*n_nodes, 1)    # flatten
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)    # flatten
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone() # crop out
        if h_dims == 0:
            # ~!to ~!mp
            h = torch.ones(bs*n_nodes, 1).to(self.device)      # pass ones if no node embeddings h
            # h = torch.ones(bs*n_nodes, 1)      # pass ones if no node embeddings h
        else:
            h = xh[:, self.n_dims:].clone() # crop out

        if context is not None:   # condition
            # We're conditioning, awesome!
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)         # conditioning based on concatenation with node features

        # EGNN model
        # print(">>", h.shape,     x.shape,              node_mask.shape,      edge_mask.shape)
        # >> torch.Size([1600, 6]) torch.Size([1600, 3]) torch.Size([1600, 1]) torch.Size([40000, 1])
        #                64x25                                                             64x25x25
        if self.mode == 'egnn_dynamics':
            h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)   # feed to model
            vel = x_final * node_mask  # This masking operation is redundant but just in case
        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]
        else:
            raise Exception("Wrong mode %s" % self.mode)

        # print("shape", bs, n_nodes, dims, h_dims, h_final.shape, x_final.shape)
        #                64  1-29     9     6       (1600, 256)    (1600, 3)         for n_nodes=25
 
        vel = vel.view(bs, n_nodes, -1)     # reshape back

        # check for nan
        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero. (EGNN_encoder_QM9)')
            vel = torch.zeros_like(vel)

        # normalise mean from velocity (remove unwanted axis translations during generation - velocity)
        if node_mask is None:
            vel = remove_mean(vel)    # return x - torch.mean(x, dim=1, keepdim=True)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))    # same remove mean, but mean of masked item only

        # final out mlp
        h_final = low_vram_forward(self.final_mlp, h_final)      # h_final still flat
        h_final = h_final * node_mask if node_mask is not None else h_final
        h_final = h_final.view(bs, n_nodes, -1)    # reshape back

        vel_mean = vel
        
        # arbitrarily set the first element of last dim to vel_std (make model learn)
        vel_std = h_final[:, :, :1].sum(dim=1, keepdim=True).expand(-1, n_nodes, -1)    # [64, 29, 1]
        vel_std = torch.exp(0.5 * vel_std)

        # arbitrary choice too
        h_mean = h_final[:, :, 1:1 + self.out_node_nf]    # [:, :, 1:2]
        # arbitrary choice too
        h_std = torch.exp(0.5 * h_final[:, :, 1 + self.out_node_nf:])

        if torch.any(torch.isnan(vel_std)):
            print('Warning: detected nan in vel_std, resetting to one. (EQNN_encoder_QM9)')
            vel_std = torch.ones_like(vel_std)
        
        if torch.any(torch.isnan(h_std)):
            print('Warning: detected nan in h_std, resetting to one. (EGNN_encoder_QM9)')
            h_std = torch.ones_like(h_std)
        
        # Note: only vel_mean and h_mean are correctly masked
        # vel_std and h_std are not masked, but that's fine:

        # For calculating KL: vel_std will be squeezed to 1D
        # h_std will be masked

        # For sampling: both stds will be masked in reparameterization

        # print("!!!", vel_mean.shape, vel_std.shape, h_mean.shape, h_std.shape)
        #             [64, 25, 3]      [64, 25, 1]    [64, 25, 1]   [64, 25, 1]
        return vel_mean, vel_std, h_mean, h_std
    
    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]     # {}
            # cache
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                # for this batch of molecules
                for batch_idx in range(batch_size):    # [0-64]
                    # for each molecule
                    for i in range(n_nodes):           # 29
                        for j in range(n_nodes):       # 29
                            rows.append(i + batch_idx * n_nodes)    # [0-29) + (batch_idx * 29)
                            cols.append(j + batch_idx * n_nodes)
                            # originally:
                            #    rows.apend(i)
                            #    cols.append(j)
                            # For row: however, since now we need a row  vector to store all [0-29] valued rows, 
                            #          we add the term (batch_idx * n_nodes) to get the below. Why? because
                            #          the molecules' nodes are concatenated together into a long list in the
                            #          dataloader, hence need for the (batch_idx * n_nodes) to differentiate
                            #          between which molecule is which.
                            #
                            #    [1,2,3,4...29, 30,31,32...39, 40,41,42...49]
                            #     ^^^^^^^^^^^^  ^^^^^^^^^^^^^  ^^^^^^^^^^^^^
                            #     molecule 1    molecule 2     molecule 3
                            # For col: same as row
                            
                            
                            # example: n_nodes=5, batch_size=2
                            # since here we have n_nodes=5, meaning each molecule has 5 atoms / 5 nodes
                            #
                            #               Molecule 1: [0,1,2,3,4]                                                        Molecule 2: [5,6,7,8,9]
                            #
                            #  row: tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,     5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9]), 
                            #  col: tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4,     5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9])]
                            #               <----------->  <----------->  <----------->  <----------->  <----------->      <----------->  <----------->  <----------->  <----------->  <----------->  
                            #               node-0-vs-all  node-1-vs-all  node-2-vs-all  node-3-vs-all  node-4-vs-all      node-0-vs-all  node-1-vs-all  node-2-vs-all  node-3-vs-all  node-4-vs-all
                            #               <----------------------------------------------------------------------->      <----------------------------------------------------------------------->
                            #                           all possible node combinations in molecule 1                                   all possible node combinations in molecule 2
                            #               <------------------------------------------------------------------------------------------------------------------------------------------------------>
                            #                                                                        batch size = 2   (2 molecules per batch)
                            
                            
                            
                            # later in EGNN model (..etc), used to index out h into h[row], h[col]:
                            # self.edge_model(h[row], h[col], edge_attr, edge_mask)
                            #
                            # then concatenated together:
                            # if edge_attr is None:  # Unused.
                            #     out = torch.cat([source, target], dim=1)        # torch.Size([46656, 256+256])
                            # else:
                            #     out = torch.cat([source, target, edge_attr], dim=1)
                                                
                            # For each sample, it iterates over all possible pairs of nodes and generates 
                            # edges between them. The edge lists are then stored as tensors representing 
                            # the rows and columns of the adjacency matrix.
                # ~!to ~!mp
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                # edges = [torch.LongTensor(rows),
                        #  torch.LongTensor(cols)]     # [row[bs*n_nodes*n_nodes], col[bs*n_nodes*n_nodes]]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}    # precomputed adj matrix for diff batch sizes
            return self.get_adj_matrix(n_nodes, batch_size, device)


class EGNN_decoder_QM9(nn.Module):
    def __init__(self, in_node_nf, context_node_nf, out_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 tanh=False, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum',
                 include_charges=True):
        super().__init__()

        include_charges = int(include_charges)        # 1
        num_classes = out_node_nf - include_charges   # 6-1 = 5 [HCNOF]

        self.mode = mode
        if mode == 'egnn_dynamics':  # this
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf,    # 1+nf+?
                out_node_nf=out_node_nf,                    # 6
                in_edge_nf=1, 
                hidden_nf=hidden_nf,     # 256
                device=device, 
                act_fn=act_fn,           # torch.nn.SiLU()
                n_layers=n_layers,       # 9
                attention=attention,     # true
                tanh=tanh, 
                norm_constant=norm_constant,   # 1
                inv_sublayers=inv_sublayers,   # 1
                sin_embedding=sin_embedding,   # false
                normalization_factor=normalization_factor,   # 1
                aggregation_method=aggregation_method        # sum
                )
            self.in_node_nf = in_node_nf
        elif mode == 'gnn_dynamics':
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3, out_node_nf=out_node_nf + 3, 
                in_edge_nf=0, hidden_nf=hidden_nf, device=device,
                act_fn=act_fn, n_layers=n_layers, attention=attention,
                normalization_factor=normalization_factor, aggregation_method=aggregation_method)

        self.num_classes = num_classes   # 5
        self.include_charges = include_charges # 1
        self.context_node_nf = context_node_nf # nf+?
        self.device = device
        self.n_dims = n_dims  # 3
        self._edges_dict = {}
        # self.condition_time = condition_time

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, xh, node_mask, edge_mask, context):
        print(f"        >>> DECODER xh:{torch.isnan(xh).any()}  node_mask:{torch.isnan(node_mask).any()}  edge_mask:{torch.isnan(edge_mask).any()}") if PARAM_REGISTRY.get('verbose')==True else None

        bs, n_nodes, dims = xh.shape
        # 64, 27, 4
        
        h_dims = dims - self.n_dims
        # 4-3 = 1
        
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        # ~!to ~!mp
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            # ~!to ~!mp
            h = torch.ones(bs*n_nodes, 1).to(self.device)
            # h = torch.ones(bs*n_nodes, 1)
        else:
            h = xh[:, self.n_dims:].clone()
            # [1728, 1]  64*27
        

        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)
            # [1728, 1]
        

        if self.mode == 'egnn_dynamics':
            print(f"        >>> DECODER (B4) h:{torch.isnan(h).any()} x:{torch.isnan(x).any()}  node_mask:{torch.isnan(node_mask).any()}  edge_mask:{torch.isnan(edge_mask).any()}") if PARAM_REGISTRY.get('verbose')==True else None
            h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
            print(f"        >>> DECODER (A3) h_:{torch.isnan(h_final).any()} x_:{torch.isnan(x_final).any()}  node_mask:{torch.isnan(node_mask).any()}  edge_mask:{torch.isnan(edge_mask).any()}") if PARAM_REGISTRY.get('verbose')==True else None

            # [1728, 6]    [1728, 3]
            vel = x_final * node_mask  # This masking operation is redundant but just in case
        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        vel = vel.view(bs, n_nodes, -1)
        # [64, 27, 3]

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero. (EGNN_decoder_QM9)')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else: # this
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if node_mask is not None:
            h_final = h_final * node_mask
        h_final = h_final.view(bs, n_nodes, -1)

        #      x,   h
        #      [64, 27, 3], [64, 27, 6]
        return vel, h_final
    
    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                # ~!to ~!mp
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                # edges = [torch.LongTensor(rows),
                        #  torch.LongTensor(cols)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)
