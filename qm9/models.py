import torch
from torch.distributions.categorical import Categorical

import numpy as np
from egnn.models import EGNN_dynamics_QM9, EGNN_encoder_QM9, EGNN_decoder_QM9

from equivariant_diffusion.en_diffusion import EnVariationalDiffusion, EnHierarchicalVAE, EnLatentDiffusion

import pickle
from os.path import join

def get_model(args, device, dataset_info, dataloader_train):
    histogram = dataset_info['n_nodes']
    in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
    nodes_dist = DistributionNodes(histogram)

    prop_dist = None
    if len(args.conditioning) > 0:
        prop_dist = DistributionProperty(dataloader_train, args.conditioning)

    if args.condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf

    net_dynamics = EGNN_dynamics_QM9(
        in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
        attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method)

    if args.probabilistic_model == 'diffusion':
        vdm = EnVariationalDiffusion(
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=args.diffusion_steps,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            include_charges=args.include_charges
            )

        return vdm, nodes_dist, prop_dist

    else:
        raise ValueError(args.probabilistic_model)


def get_autoencoder(args, device, dataset_info, dataloader_train):
    histogram = dataset_info['n_nodes']
    in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)     # len[H,C,N,O,F] + int(True) = 6
    nodes_dist = DistributionNodes(histogram)

    prop_dist = None
    if len(args.conditioning) > 0:
        prop_dist = DistributionProperty(dataloader_train, args.conditioning)

    # if args.condition_time:
    #     dynamics_in_node_nf = in_node_nf + 1
    # else:
    print('Autoencoder models are _not_ conditioned on time.')
        # dynamics_in_node_nf = in_node_nf
    
    encoder = EGNN_encoder_QM9(
        in_node_nf=in_node_nf,     # 6
        context_node_nf=args.context_node_nf,   # nf: len(args.conditioning) + ?
        out_node_nf=args.latent_nf,     # 1
        n_dims=3, 
        device=device, 
        hidden_nf=args.nf,   # 256
        act_fn=torch.nn.SiLU(), 
        n_layers=1,
        attention=args.attention,   # true
        tanh=args.tanh, 
        mode=args.model,     # egnn_dynamics
        norm_constant=args.norm_constant,   # 1
        inv_sublayers=args.inv_sublayers,   # 1
        sin_embedding=args.sin_embedding,   # false
        normalization_factor=args.normalization_factor,   # 1
        aggregation_method=args.aggregation_method,   # sum
        include_charges=args.include_charges     # true
        )
    
    decoder = EGNN_decoder_QM9(
        in_node_nf=args.latent_nf,      # 1
        context_node_nf=args.context_node_nf,     # nf: len(args.conditioning) + ?
        out_node_nf=in_node_nf,      # 6
        n_dims=3, 
        device=device, 
        hidden_nf=args.nf,     # 256
        act_fn=torch.nn.SiLU(), 
        n_layers=args.n_layers,   # 9
        attention=args.attention,    # true
        tanh=args.tanh, 
        mode=args.model,    # egnn_dynamics
        norm_constant=args.norm_constant,  # 1
        inv_sublayers=args.inv_sublayers,  # 1
        sin_embedding=args.sin_embedding,  # false
        normalization_factor=args.normalization_factor,   # 1
        aggregation_method=args.aggregation_method,   # sum
        include_charges=args.include_charges    # true
        )

    vae = EnHierarchicalVAE(
        encoder=encoder,
        decoder=decoder,
        in_node_nf=in_node_nf,  # 6
        n_dims=3,
        latent_node_nf=args.latent_nf,  # 1
        kl_weight=args.kl_weight,   # 0.01
        norm_values=args.normalize_factors,   # 1,4,10
        include_charges=args.include_charges  # true
        )

    return vae, nodes_dist, prop_dist


def get_latent_diffusion(args, device, dataset_info, dataloader_train):

    # Create (and load) the first stage model (Autoencoder).
    if args.ae_path is not None:
        with open(join(args.ae_path, 'args.pickle'), 'rb') as f:
            first_stage_args = pickle.load(f)
    else:
        first_stage_args = args
    
    # CAREFUL with this -->
    if not hasattr(first_stage_args, 'normalization_factor'):
        first_stage_args.normalization_factor = 1
    if not hasattr(first_stage_args, 'aggregation_method'):
        first_stage_args.aggregation_method = 'sum'

    device = torch.device("cuda" if first_stage_args.cuda else "cpu")

    first_stage_model, nodes_dist, prop_dist = get_autoencoder(
        first_stage_args, device, dataset_info, dataloader_train)
    first_stage_model.to(device)

    if args.ae_path is not None:   # null
        fn = 'generative_model_ema.npy' if first_stage_args.ema_decay > 0 else 'generative_model.npy'
        flow_state_dict = torch.load(join(args.ae_path, fn),
                                        map_location=device)
        first_stage_model.load_state_dict(flow_state_dict)

    # Create the second stage model (Latent Diffusions).
    args.latent_nf = first_stage_args.latent_nf
    in_node_nf = args.latent_nf  # 1

    if args.condition_time:   # true
        dynamics_in_node_nf = in_node_nf + 1  # 2
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf
    
    net_dynamics = EGNN_dynamics_QM9(
        in_node_nf=dynamics_in_node_nf,  # args.latent_nf + time = 2
        context_node_nf=args.context_node_nf,  # nf+? = 0
        n_dims=3, 
        device=device, # cuda
        hidden_nf=args.nf,  # 256
        act_fn=torch.nn.SiLU(), 
        n_layers=args.n_layers,  # 9
        attention=args.attention,  # true
        tanh=args.tanh,    # true
        mode=args.model,   # egnn_dynamics
        norm_constant=args.norm_constant,  # 1
        inv_sublayers=args.inv_sublayers,  # 1
        sin_embedding=args.sin_embedding,  # false
        normalization_factor=args.normalization_factor, # 1
        aggregation_method=args.aggregation_method  # sum
        )

    if args.probabilistic_model == 'diffusion':
        vdm = EnLatentDiffusion(
            vae=first_stage_model,    # VAE model
            trainable_ae=args.trainable_ae,    # true
            dynamics=net_dynamics,    # LDM model
            in_node_nf=in_node_nf,    # 1
            n_dims=3,
            timesteps=args.diffusion_steps,    # 1000
            noise_schedule=args.diffusion_noise_schedule,  # polynomial_2
            noise_precision=args.diffusion_noise_precision, # 1.0e-05
            loss_type=args.diffusion_loss_type,  # L2
            norm_values=args.normalize_factors,  # [1,4,10]
            include_charges=args.include_charges # true
            )

        return vdm, nodes_dist, prop_dist

    else:
        raise ValueError(args.probabilistic_model)


def get_optim(args, generative_model):
    # ~!fp16
    optim = torch.optim.AdamW(
        generative_model.parameters(),
        lr=args.lr, amsgrad=True,
        weight_decay=1e-12)
    # optim = torch.optim.AdamW(
    #     generative_model.parameters(),
    #     lr=args.lr, amsgrad=True,
    #     weight_decay=1e-4)

    return optim


# Probability Distribution for graph nodes
class DistributionNodes:
    def __init__(self, histogram):

        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            # i: index 0,1,2,..
            # nodes: keys of n_nodes dict 22,17,23,21,..
            # where n_nodes dict = {node_num: frequency, ...},  ref: configs/datasets_config.py
            self.n_nodes.append(nodes)    # [key, ..]
            self.keys[nodes] = i          # {key: enum index}
            prob.append(histogram[nodes]) # [value, ..]
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob/np.sum(prob)    # sum to 1.

        self.prob = torch.from_numpy(prob).float()

        # ~!fp16
        entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))   # entropy calculation, unused
        # entropy = torch.sum(self.prob * torch.log(self.prob + 1e-4))   # entropy calculation, unused
        
        print("Entropy of n_nodes: H[N]", entropy.item())   # i.e. -2.475700616836548

        self.m = Categorical(torch.tensor(prob))    # output distribution for classification models

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))    # this would sample n_samples of random indexes based on the probability.
        return self.n_nodes[idx]             # this returns the number of nodes. i.e. n_samples=3, return=tensor([16, 17, 15]) / tensor([14, 15, 19]) / tensor([17, 27, 18])

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = [self.keys[i.item()] for i in batch_n_nodes]     # [enum idx, ..]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        # ~!fp16
        log_p = torch.log(self.prob + 1e-30)    # computes log probability. 1e-30 epsilon is to prevent log(0)
        # log_p = torch.log(self.prob + 1e-4)    # computes log probability. 1e-30 epsilon is to prevent log(0)

        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]    # get required log probs only

        return log_probs


# Probability Distribution for conditioning properties
class DistributionProperty:
    #                  dataloader['train'], args.conditioning
    def __init__(self, dataloader,          properties,       num_bins=1000, normalizer=None):
        self.num_bins = num_bins   # default 1000
        self.distributions = {}
        self.properties = properties
        for prop in properties:
            self.distributions[prop] = {}
            self._create_prob_dist(dataloader.dataset.data['num_atoms'],
                                   dataloader.dataset.data[prop],
                                   self.distributions[prop])        # for each property in args.conditioning, # for each num_of_nodes, {categorical_prob_dist_of_property, [property_min_val, max_val]}

        self.normalizer = normalizer  # None

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    # helper method
    def _create_prob_dist(self, nodes_arr, values, distribution):
        min_nodes, max_nodes = torch.min(nodes_arr), torch.max(nodes_arr)     # dataloader.dataset.data['num_atoms']
        for n_nodes in range(int(min_nodes), int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes        # get index according to current num_of_nodes
            values_filtered = values[idxs]     # to filter out required property values only
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)
                distribution[n_nodes] = {'probs': probs, 'params': params}       # for each num_of_nodes, {'probs':categorical_prob_dist_of_property, 'params':[property_min_val, max_val]}

    # helper method: create categorical probability distribution of conditioning properties
    def _create_prob_given_nodes(self, values):
        n_bins = self.num_bins #min(self.num_bins, len(values))     1000
        prop_min, prop_max = torch.min(values), torch.max(values)   # filtered property min max
        
        # ~!fp16
        prop_range = prop_max - prop_min + 1e-12      # epsilon to prevent 0
        # prop_range = prop_max - prop_min + 1e-4      # epsilon to prevent 0
        
        histogram = torch.zeros(n_bins)     # shape [1000]
        for val in values:
            i = int((val - prop_min)/prop_range * n_bins)     # norm to [0-1], then scale to [0-n_bins]
            # Because of numerical precision, one sample can fall in bin int(n_bins) instead of int(n_bins-1)
            # We move it to bin int(n_bind-1 if that happens)
            if i == n_bins:
                i = n_bins - 1
            histogram[i] += 1    # increase count/frequency of property 
        probs = histogram / torch.sum(histogram)  # sum to 1.
        probs = Categorical(torch.tensor(probs))
        params = [prop_min, prop_max]
        return probs, params

    # say n_nodes=19, then sample from the property distribution with num_of_nodes=19
    def sample(self, n_nodes=19):
        vals = []
        for prop in self.properties:     # args.conditioning
            dist = self.distributions[prop][n_nodes]  # for each property in args.conditioning, # for each num_of_nodes, {'probs':categorical_prob_dist_of_property, 'params':[property_min_val, max_val]}
            idx = dist['probs'].sample((1,))
            val = self._idx2value(idx, params=dist['params'], n_bins=len(dist['probs'].probs))    # dist['probs'].probs --> Categorial.probs
            val = self.normalize_tensor(val, prop)
            vals.append(val)
        vals = torch.cat(vals)
        return vals       # shape: [num_args_conditioning]

    def sample_batch(self, nodesxsample):
        vals = []
        for n_nodes in nodesxsample:
            vals.append(self.sample(int(n_nodes)).unsqueeze(0))
        vals = torch.cat(vals, dim=0)
        return vals       # shape: [batch_size, num_args_conditioning]

    def _idx2value(self, idx, params, n_bins):
        prop_range = params[1] - params[0]     # max - min
        left = float(idx) / n_bins * prop_range + params[0]
        right = float(idx + 1) / n_bins * prop_range + params[0]
        val = torch.rand(1) * (right - left) + left       # random [0-1) float
        return val
    
    def normalize_tensor(self, tensor, prop):
        assert self.normalizer is not None
        mean = self.normalizer[prop]['mean']      # precomputed mean & mad in main_qm9.py, line 196, property_norms = compute_mean_mad(..)
        mad = self.normalizer[prop]['mad']
        return (tensor - mean) / mad



if __name__ == '__main__':
    dist_nodes = DistributionNodes()
    print(dist_nodes.n_nodes)
    print(dist_nodes.prob)
    for i in range(10):
        print(dist_nodes.sample())
