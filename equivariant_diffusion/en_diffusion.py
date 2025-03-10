from equivariant_diffusion import utils
import numpy as np
import math
import torch
from egnn import models
from torch.nn import functional as F
from equivariant_diffusion import utils as diffusion_utils
from global_registry import PARAM_REGISTRY
from sklearn.metrics import accuracy_score, recall_score, f1_score


# Defining some useful util functions.

# exp(x) - 1
def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)

# log(1 + exp(x)), smooth approximation of ReLU
def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    # make sure first timestep is 1.
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    # computes (at / at-1) for each timestep
    alphas_step = (alphas2[1:] / alphas2[:-1])

    # clip in range
    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    
    # cumulative product: [1,2,3,4] -> [1,2,6,24]
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    # +1 to account for init condition
    steps = timesteps + 1
    
    # start=0, end=step, num_of_samples=step  (end inc)
    # [0,1,2,3,4...step]
    x = np.linspace(0, steps, steps)
    
    # main poly eqn
    alphas2 = (1 - np.power(x / steps, power))**2

    # computes (at / at-1) for each timestep
    # clip in range 0.001-1.
    # cumulative product: [1,2,3,4] -> [1,2,6,24]
    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    # s = scaling factor
    # Computes a precision value used for scaling the noise schedule. 
    # 1 - 2(1e-4) = 0.9998
    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def gaussian_entropy(mu, sigma):
    # In case sigma needed to be broadcast (which is very likely in this code).
    zeros = torch.zeros_like(mu)
    return sum_except_batch(
        zeros + 0.5 * torch.log(2 * np.pi * sigma**2) + 0.5
    )


def gaussian_KL(q_mu, q_sigma, p_mu, p_sigma, node_mask):
    """Computes the KL distance between two normal distributions, taking into account all dimensions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    # ~!fp16
    return sum_except_batch(
            (
                torch.log(p_sigma / (q_sigma + 1e-8) + 1e-8)
                + 0.5 * (q_sigma**2 + (q_mu - p_mu)**2) / (p_sigma**2)
                - 0.5
            ) * node_mask
        )



def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
    """Computes the KL distance between two normal distributions, taking into account only a single dimension.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    mu_norm2 = sum_except_batch((q_mu - p_mu)**2)
    assert len(q_sigma.size()) == 1, print(q_sigma.size())
    assert len(p_sigma.size()) == 1, print(p_sigma.size())
    # ~!fp16
    return (d * torch.log(p_sigma / (q_sigma + 1e-8) + 1e-8) 
            + 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2) 
            - 0.5 * d
            )


class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: int = -2):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze() * 1000
        assert len(x.shape) == 1
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """
    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:  # polynomial_2
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            
            # precomputed values for alphas (cumulative product)
            # [0,1,2,3,4...step]
            # alphas2 = (1 - np.power(x / steps, power))**2
            # computes (at / at-1) for each timestep
            # clip in range 0.001-1.
            # cumulative product: [1,2,3,4] -> [1,2,6,24]
            
            # basically timestep:[1,2,3,4] -> cum_prod(polynomial(timestep):[0.9, 0.5, 0.2, 0.1])
            # note that timesteps are discrete integers, not floats, hence we can precompute the
            # required finite amount (1,2,3...T) of values beforehand
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        print('alphas2:\n', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)
        
        # gamma = -log(alphas2 / sigmas2)
        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        print('gamma:\n', -log_alphas2_to_sigmas2)

        # ~!fp16
        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        # t in range [0-1], scale to actual discrete timesteps for lookup of poly
        # values in precomputed gamma list
        # .long() is 64bit int, .int() is 32bit int, .long() to ensure that the data 
        # type can accommodate a wide range of values without overflowing
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


class GammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper."""
    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma


def cdf_standard_gaussian(x):
    """
    Calculates the cumulative distribution function (CDF) of the standard Gaussian (normal) distribution.
    
    Args:
        x: value at which to evaluate the CDF of the standard Gaussian distribution.
    
    x / math.sqrt(2): scales input by 1/sqrt(2), necessary to conform to the 
                      standard deviation of the standard Gaussian distribution.
    torch.erf(...): Computes the error function of the scaled input - a mathematical function 
                    that describes the probability of a random variable falling within a particular 
                    range. In this case, it represents the integral of the Gaussian probability 
                    density function from negative infinity to x.
    0.5 * (.. + 1.) : scale the result between 0 and 1, representing the cumulative probability.
    """
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))


class EnVariationalDiffusion(torch.nn.Module):
    """
    The E(n) Diffusion Module.
    """
    def __init__(
            self,
            dynamics, in_node_nf: int, n_dims: int,
            timesteps: int = 1000, parametrization='eps', noise_schedule='learned',
            noise_precision=1e-4, loss_type='vlb', norm_values=(1., 1., 1.),
            norm_biases=(None, 0., 0.), include_charges=True):
        super().__init__()

        assert isinstance(dynamics, models.EGNN_dynamics_QM9) or isinstance(dynamics, models.ControlNet_Module_Wrapper)
        assert loss_type in {'vlb', 'l2'}

        self.loss_type = loss_type   # L2
        self.include_charges = include_charges  # true
        if noise_schedule == 'learned':
            assert loss_type == 'vlb', 'A noise schedule can only be learned' \
                                       ' with a vlb objective.'

        # Only supported parametrization.
        assert parametrization == 'eps'

        if noise_schedule == 'learned':
            self.gamma = GammaNetwork()
        else: # polynomial_2
            # Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
            self.gamma = PredefinedNoiseSchedule(noise_schedule, # polynomial_2
                                                 timesteps=timesteps, # 1000
                                                 precision=noise_precision # 1.0e-05
                                                 )

        # The network that will predict the denoising.
        self.dynamics = dynamics    # LDM model

        self.in_node_nf = in_node_nf # 2
        self.n_dims = n_dims         # 3
        self.num_classes = self.in_node_nf - self.include_charges   # 2-0 = 2

        self.T = timesteps    # 1000
        self.parametrization = parametrization   # eps

        self.norm_values = norm_values   # [1,4,10]
        self.norm_biases = norm_biases   # (None, 0., 0.)
        self.register_buffer('buffer', torch.zeros(1))

        if noise_schedule != 'learned':
            self.check_issues_norm_values()

        # dictionary to store activations
        self.input_activations = {}
        self.output_activations = {}

    def _register_hooks(self):
        self.hook_handles = []
        def hook_fn(module, input, output, name):
            self.input_activations[name] = input[0].clone().to(torch.float32).detach().cpu().numpy()
            self.output_activations[name] = output.clone().to(torch.float32).detach().cpu().numpy()

        # register hooks on all layers to track, i.e. nn.Linear 
        for name, layer in self.named_modules():
            if isinstance(layer, PARAM_REGISTRY.get('vis_activations_instances')):
                handle = layer.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))
                print(name)
                self.hook_handles.append(handle)
        return self.hook_handles

    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        max_norm_value = max(self.norm_values[1], self.norm_values[2])

        if sigma_0 * num_stdevs > 1. / max_norm_value:
            raise ValueError(
                f'Value for normalization value {max_norm_value} probably too '
                f'large with sigma_0 {sigma_0:.5f} and '
                f'1 / norm_value = {1. / max_norm_value}')

    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims
        # (29 - 1) * 3

    def normalize(self, x, h, node_mask):
        x = x / self.norm_values[0]
        #              -[(29 - 1) * 3] * log(norm_x_value)
        delta_log_px = -self.subspace_dimensionality(node_mask) * np.log(self.norm_values[0])

        # Casting to float in case h still has long or int type.
        h_cat = (h['categorical'].float() - self.norm_biases[1]) / self.norm_values[1] * node_mask
        h_int = (h['integer'].float() - self.norm_biases[2]) / self.norm_values[2]

        if self.include_charges:
            h_int = h_int * node_mask

        # Create new h dictionary.
        h = {'categorical': h_cat, 'integer': h_int}

        return x, h, delta_log_px

    def unnormalize(self, x, h_cat, h_int, node_mask):
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
        h_cat = h_cat * node_mask
        h_int = h_int * self.norm_values[2] + self.norm_biases[2]

        if self.include_charges:
            h_int = h_int * node_mask

        return x, h_cat, h_int

    def unnormalize_z(self, z, node_mask):
        # Parse from z
        x     = z[:, :, 0:self.n_dims]
        h_cat = z[:, :, self.n_dims:self.n_dims+self.num_classes]
        h_int = z[:, :, self.n_dims+self.num_classes:self.n_dims+self.num_classes+1]
        assert h_int.size(2) == self.include_charges

        # Unnormalize
        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)
        output = torch.cat([x, h_cat, h_int], dim=2)
        return output

    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -expm1(softplus(gamma_s) - softplus(gamma_t)), target_tensor
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(
            alpha_t_given_s, target_tensor)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def kl_prior(self, xh, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((xh.size(0), 1), device=xh.device)

        # ~!fp16
        # self.gamma = self.gamma.to(xh.device)

        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh)

        # Compute means.
        mu_T = alpha_T * xh
        mu_T_x, mu_T_h = mu_T[:, :, :self.n_dims], mu_T[:, :, self.n_dims:]

        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_x = self.sigma(gamma_T, mu_T_x).squeeze()  # Remove inflate, only keep batch dimension for x-part.
        sigma_T_h = self.sigma(gamma_T, mu_T_h)

        # Compute KL for h-part.
        zeros, ones = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
        kl_distance_h = gaussian_KL(mu_T_h, sigma_T_h, zeros, ones, node_mask)

        # Compute KL for x-part.
        zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
        subspace_d = self.subspace_dimensionality(node_mask)
        kl_distance_x = gaussian_KL_for_dimension(mu_T_x, sigma_T_x, zeros, ones, d=subspace_d)

        return kl_distance_x + kl_distance_h

    def log_constants_p_x_given_z0(self, x, node_mask):
        """Computes p(x|z0)."""
        batch_size = x.size(0)

        n_nodes = node_mask.squeeze(2).sum(1)  # N has shape [B]
        assert n_nodes.size() == (batch_size,)
        degrees_of_freedom_x = (n_nodes - 1) * self.n_dims

        zeros = torch.zeros((x.size(0), 1), device=x.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))

    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        # z_x = utils.sample_center_gravity_zero_gaussian_with_mask(..)
        # z_h = utils.sample_gaussian_with_mask(..)
        # eps = torch.cat([z_x, z_h], dim=2)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps

    def log_pxh_given_z0_without_constants(self, x, h, z_t, gamma_0, eps, net_out, node_mask, epsilon=1e-10):
    # ~!fp16
    
        """Calculates loss_term_0 at timestep t=0."""
        
        # Discrete properties are predicted directly from z_t.
        z_h_cat = z_t[:, :, self.n_dims:-1] if self.include_charges else z_t[:, :, self.n_dims:]
        z_h_int = z_t[:, :, -1:] if self.include_charges else torch.zeros(0).to(z_t.device)

        # Take only part over x.
        eps_x = eps[:, :, :self.n_dims]
        net_x = net_out[:, :, :self.n_dims]

        # Compute sigma_0 and rescale to the integer scale of the data.
        sigma_0 = self.sigma(gamma_0, target_tensor=z_t)
        sigma_0_cat = sigma_0 * self.norm_values[1]
        sigma_0_int = sigma_0 * self.norm_values[2]

        # Computes the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.
        log_p_x_given_z_without_constants = -0.5 * self.compute_error(net_x, gamma_0, eps_x)

        # Compute delta indicator masks.
        h_integer = torch.round(h['integer'] * self.norm_values[2] + self.norm_biases[2]).long()
        onehot = h['categorical'] * self.norm_values[1] + self.norm_biases[1]

        estimated_h_integer = z_h_int * self.norm_values[2] + self.norm_biases[2]
        estimated_h_cat = z_h_cat * self.norm_values[1] + self.norm_biases[1]
        assert h_integer.size() == estimated_h_integer.size()

        h_integer_centered = h_integer - estimated_h_integer

        # Compute integral from -0.5 to 0.5 of the normal distribution
        # N(mean=h_integer_centered, stdev=sigma_0_int)
        log_ph_integer = torch.log(
            cdf_standard_gaussian((h_integer_centered + 0.5) / sigma_0_int)
            - cdf_standard_gaussian((h_integer_centered - 0.5) / sigma_0_int)
            + epsilon)
        log_ph_integer = sum_except_batch(log_ph_integer * node_mask)

        # Centered h_cat around 1, since onehot encoded.
        centered_h_cat = estimated_h_cat - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=z_h_cat, stdev=sigma_0_cat)
        log_ph_cat_proportional = torch.log(
            cdf_standard_gaussian((centered_h_cat + 0.5) / sigma_0_cat)
            - cdf_standard_gaussian((centered_h_cat - 0.5) / sigma_0_cat)
            + epsilon)

        # Normalize the distribution over the categories.
        log_Z = torch.logsumexp(log_ph_cat_proportional, dim=2, keepdim=True)
        log_probabilities = log_ph_cat_proportional - log_Z

        # Select the log_prob of the current category usign the onehot
        # representation.
        log_ph_cat = sum_except_batch(log_probabilities * onehot * node_mask)

        # Combine categorical and integer log-probabilities.
        log_p_h_given_z = log_ph_integer + log_ph_cat

        # Combine log probabilities for x and h.
        log_p_xh_given_z = log_p_x_given_z_without_constants + log_p_h_given_z

        return log_p_xh_given_z
    
    def phi(self, x, t, node_mask, edge_mask, context):
        net_out = self.dynamics._forward(t, x, node_mask, edge_mask, context)

        return net_out
    
    def compute_x_pred(self, net_out, zt, gamma_t):
        """Commputes x_pred, i.e. the most likely prediction of x."""
        if self.parametrization == 'x':
            x_pred = net_out
        elif self.parametrization == 'eps':  # eps
            sigma_t = self.sigma(gamma_t, target_tensor=net_out)
            alpha_t = self.alpha(gamma_t, target_tensor=net_out)
            eps_t = net_out
            x_pred = 1. / alpha_t * (zt - sigma_t * eps_t)
        else:
            raise ValueError(self.parametrization)

        return x_pred

    def compute_error(self, net_out, gamma_t, eps):
        """Computes error, i.e. the most likely prediction of x."""
        eps_t = net_out
        if self.training and self.loss_type == 'l2':
            denom = (self.n_dims + self.in_node_nf) * eps_t.shape[1]
            error = sum_except_batch((eps - eps_t) ** 2) / denom
        else:
            error = sum_except_batch((eps - eps_t) ** 2)
        return error

    def compute_loss(self, x, h, node_mask, edge_mask, context, t0_always):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE)."""

        # This part is about whether to include loss term 0 always.
        # False when training, true others
        if t0_always:
            # loss_term_0 will be computed separately.
            # estimator = loss_0 + loss_t,  where t ~ U({1, ..., T})
            lowest_t = 1
        else:
            # estimator = loss_t,           where t ~ U({0, ..., T})
            lowest_t = 0

        # Sample a timestep t.
        # ~!fp16
        t_int = torch.randint(lowest_t, self.T + 1, size=(x.size(0), 1), device=x.device).float()
        s_int = t_int - 1
        t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0).

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s = s_int / self.T
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        # z_x = utils.sample_center_gravity_zero_gaussian_with_mask(..)
        # z_h = utils.sample_gaussian_with_mask(..)
        # z = torch.cat([z_x, z_h], dim=2)
        eps = self.sample_combined_position_feature_noise(n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)

        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t = alpha_t * xh + sigma_t * eps

        diffusion_utils.assert_mean_zero_with_mask(z_t[:, :, :self.n_dims], node_mask)

        # Neural net prediction.
        net_out = self.phi(z_t, t, node_mask, edge_mask, context)

        # Compute the error.
        error = self.compute_error(net_out, gamma_t, eps)

        if self.training and self.loss_type == 'l2':
            SNR_weight = torch.ones_like(error)
        else:
            # Compute weighting with SNR: (SNR(s-t) - 1) for epsilon parametrization.
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
        assert error.size() == SNR_weight.size()
        loss_t_larger_than_zero = 0.5 * SNR_weight * error
        
        # # ~!wt
        # print(f"=================================================")
        # print(f"SNR_weight              : {SNR_weight.mean().item()}")
        # print(f"error                   : {error.mean().item()}")
        # print(f"loss_t_larger_than_zero : {loss_t_larger_than_zero.mean().item()}")

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_x_given_z0(x, node_mask)

        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros_like(neg_log_constants)

        # The KL between q(z1 | x) and p(z1) = Normal(0, 1). Should be close to zero.
        kl_prior = self.kl_prior(xh, node_mask)

        # Combining the terms
        # false when training, true others
        if t0_always:
            loss_t = loss_t_larger_than_zero
            num_terms = self.T  # Since t=0 is not included here.
            estimator_loss_terms = num_terms * loss_t

            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
            alpha_0 = self.alpha(gamma_0, x)
            sigma_0 = self.sigma(gamma_0, x)

            # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
            eps_0 = self.sample_combined_position_feature_noise(
                n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)
            z_0 = alpha_0 * xh + sigma_0 * eps_0

            net_out = self.phi(z_0, t_zeros, node_mask, edge_mask, context)

            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_0, gamma_0, eps_0, net_out, node_mask)

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()
            assert kl_prior.size() == loss_term_0.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0

            # # ~!wt
            # print(f"[EVAL] kl_prior             : {kl_prior.mean().item()}")
            # print(f"[EVAL] estimator_loss_terms : {estimator_loss_terms.mean().item()}")
            # print(f"[EVAL] neg_log_constants    : {neg_log_constants.mean().item()}")
            # print(f"[EVAL] loss_term_0          : {loss_term_0.mean().item()}")
            # print(f"[EVAL] loss                 : {loss.mean().item()}")

        else:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_t, gamma_t, eps, net_out, node_mask)

            t_is_not_zero = 1 - t_is_zero

            loss_t = loss_term_0 * t_is_zero.squeeze() + t_is_not_zero.squeeze() * loss_t_larger_than_zero

            # Only upweigh estimator if using the vlb objective.
            if self.training and self.loss_type == 'l2':
                estimator_loss_terms = loss_t
            else:
                num_terms = self.T + 1  # Includes t = 0.
                estimator_loss_terms = num_terms * loss_t

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants

            # # ~!wt
            # print(f"[TRAIN] kl_prior             : {kl_prior.mean().item()}")
            # print(f"[TRAIN] estimator_loss_terms : {estimator_loss_terms.mean().item()}")
            # print(f"[TRAIN] neg_log_constants    : {neg_log_constants.mean().item()}")
            # print(f"[TRAIN] loss                 : {loss.mean().item()}")

        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        return loss, \
               {
                't': t_int.squeeze(),         # sampled random timesteps
                'loss_t': loss.squeeze(),     # final loss combined
                'error': error.squeeze()      # sum_except_batch((eps - eps_t) ** 2) / denom
                }
               
    def forward(self, x, h, node_mask=None, edge_mask=None, context=None):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """
        # Normalize data, take into account volume change in x.
        x, h, delta_log_px = self.normalize(x, h, node_mask)

        # Reset delta_log_px if not vlb objective.
        if self.training and self.loss_type == 'l2':
            delta_log_px = torch.zeros_like(delta_log_px)

        if self.training:
            # Only 1 forward pass when t0_always is False.
            loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context, t0_always=False)
        else:
            # Less variance in the estimator, costs two forward passes.
            loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context, t0_always=True)

        neg_log_pxh = loss

        # Correct for normalization on x.
        assert neg_log_pxh.size() == delta_log_px.size()
        neg_log_pxh = neg_log_pxh - delta_log_px

        return neg_log_pxh

    def sample_p_zs_given_zt(self, s, t, zt, node_mask, edge_mask, context, fix_noise=False):
        """
        Samples from zs ~ p(zs | zt). Only used during sampling. 
        NOTE: One sampling step. (NOT final step z0)
        """
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        eps_t = self.phi(zt, t, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        diffusion_utils.assert_mean_zero_with_mask(zt[:, :, :self.n_dims], node_mask)
        diffusion_utils.assert_mean_zero_with_mask(eps_t[:, :, :self.n_dims], node_mask)
        mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        # z_x = utils.sample_center_gravity_zero_gaussian_with_mask(..)
        # z_h = utils.sample_gaussian_with_mask(..)
        # eps = torch.cat([z_x, z_h], dim=2)
        # zs = mu + sigma * eps
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        # Project down to avoid numerical runaway of the center of gravity.
        zs = torch.cat(
            [diffusion_utils.remove_mean_with_mask(zs[:, :, :self.n_dims],
                                                   node_mask),
             zs[:, :, self.n_dims:]], dim=2
        )
        return zs
    
    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context, fix_noise=False):
        """
        Samples x ~ p(x|z0).
        One sampling step. (final step z0)
        NOTE: This method will not run when called from LDM, LDM's sample_p_xh_given_z0()
              will run instead. Refer line 1265.
        """
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        net_out = self.phi(z0, zeros, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        # z_x = utils.sample_center_gravity_zero_gaussian_with_mask(..)
        # z_h = utils.sample_gaussian_with_mask(..)
        # eps = torch.cat([z_x, z_h], dim=2)
        # xh = mu + sigma * eps
        xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)

        x = xh[:, :, :self.n_dims]
        h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        h_cat = z0[:, :, self.n_dims:-1]
        
        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)

        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h_int = torch.round(h_int).long() * node_mask
        h = {'integer': h_int, 'categorical': h_cat}
        return x, h

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = utils.sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
            node_mask=node_mask)
        z_h = utils.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf), device=node_mask.device,
            node_mask=node_mask)
        z = torch.cat([z_x, z_h], dim=2)
        return z





    @torch.no_grad()
    def sample(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False):
        """
        Draw samples from the generative model.
        NOTE: full timesteps T.
        """
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            # from T -> t=1
            z = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context, fix_noise=fix_noise)

        # Final sample z0, t=0
        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise)

        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)

        # remove velocity
        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                  f'the positions down.')
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)

        return x, h



    @torch.no_grad()
    def sample_chain(self, n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=None):
        """
        Draw samples from the generative model, same as sample() above,
        but keeps the intermediate states for visualization purposes.
        """
        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(
                s_array, t_array, z, node_mask, edge_mask, context)

            diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

            # Write to chain tensor.
            write_index = (s * keep_frames) // self.T
            chain[write_index] = self.unnormalize_z(z, node_mask)

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context)

        diffusion_utils.assert_mean_zero_with_mask(x[:, :, :self.n_dims], node_mask)

        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        chain[0] = xh  # Overwrite last frame with the resulting x and h.

        chain_flat = chain.view(n_samples * keep_frames, *z.size()[1:])

        return chain_flat

    def log_info(self):
        """
        Some info logging of the model.
        """
        gamma_0 = self.gamma(torch.zeros(1, device=self.buffer.device))
        gamma_1 = self.gamma(torch.ones(1, device=self.buffer.device))

        log_SNR_max = -gamma_0
        log_SNR_min = -gamma_1

        info = {
            'log_SNR_max': log_SNR_max.item(),
            'log_SNR_min': log_SNR_min.item()}
        print(info)

        return info


class EnHierarchicalVAE(torch.nn.Module):
    """
    The E(n) Hierarchical VAE Module.
    """
    def __init__(
            self,
            encoder: models.EGNN_encoder_QM9,
            decoder: models.EGNN_decoder_QM9,
            in_node_nf: int, 
            n_dims: int, 
            latent_node_nf: int,
            kl_weight: float,
            include_charges=True,
            normalize_x=False,
            normalize_method=None,
            normalize_values=(1., 1., 1.), 
            reweight_coords_loss=None,
            error_x_weight=None,
            reweight_class_loss=None,
            error_h_weight=None,
            class_weights=None,
            atom_decoder=None,
            identifier='VAE'
            ):
        super().__init__()

        self.include_charges = include_charges  # true

        self.encoder = encoder
        self.decoder = decoder

        self.in_node_nf = in_node_nf  # 6
        self.n_dims = n_dims  # 3
        self.latent_node_nf = latent_node_nf  # 1
        self.num_classes = self.in_node_nf - self.include_charges  # 5
        self.kl_weight = kl_weight  # 0.01
        
        self.reweight_coords_loss = reweight_coords_loss
        self.error_x_weight = error_x_weight
        self.reweight_class_loss = reweight_class_loss
        self.error_h_weight = error_h_weight
        self.class_weights = class_weights
        self.atom_decoder = atom_decoder
        self.identifier = identifier

        self.vae_normalize_x = normalize_x
        # self.vae_normalize_x = PARAM_REGISTRY.get('vae_normalize_x', False)
        if self.vae_normalize_x:
            # self.vae_normalize_method = PARAM_REGISTRY.get('vae_normalize_method', None)
            self.vae_normalize_method = normalize_method
            print(f"[{self.identifier}] >> EnHierarchicalVAE self.vae_normalize_method={self.vae_normalize_method}") if self.vae_normalize_x else None
            
            if self.vae_normalize_method == 'scale':
                # norm_values = PARAM_REGISTRY.get('vae_normalize_factors')
                self.norm_values = normalize_values  # (1., 1., 1.)
                print(f"[{self.identifier}] >> EnHierarchicalVAE self.norm_values={self.norm_values}")

            else:
                raise NotImplementedError()
        
        self.register_buffer('buffer', torch.zeros(1))

        # dictionary to store activations
        self.input_activations = {}
        self.output_activations = {}

    def _register_hooks(self):
        self.hook_handles = []
        def hook_fn(module, input, output, name):
            self.input_activations[name] = input[0].clone().to(torch.float32).detach().cpu().numpy()
            self.output_activations[name] = output.clone().to(torch.float32).detach().cpu().numpy()

        # register hooks on all layers to track, i.e. nn.Linear 
        for name, layer in self.named_modules():
            if isinstance(layer, PARAM_REGISTRY.get('vis_activations_instances')):
                handle = layer.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))
                print(name)
                self.hook_handles.append(handle)
        return self.hook_handles

    # ~!norm
    def normalize_x(self, x):
        if self.vae_normalize_method == 'scale':
            print(f"[{self.identifier}] Normalising coords with scale: {self.norm_values[0]}")
            x = x / self.norm_values[0]
        return x
    # ~!norm
    def unnormalize_x(self, x):
        if self.vae_normalize_method == 'scale':
            print(f"[{self.identifier}] Unnormalising coords with scale: {self.norm_values[0]}")
            x = x * self.norm_values[0]
        return x


    def forward(self, x, h, node_mask=None, edge_mask=None, context=None, loss_analysis=False):
        """
        Computes the ELBO if training. And if eval then always computes NLL.
        """

        neg_log_pxh, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context)

        if loss_analysis:
            return neg_log_pxh, loss_dict
        else:
            return neg_log_pxh
        # loss_dict: {
        #     'loss_t': loss.squeeze(), 
        #     'rec_error': loss_recon.squeeze(), 
        #     'recon_loss_dict': recon_loss_dict,  # additional for rec loss analysis and xh saving
        #     'xh_gt': xh_unpack_dict,
        #     'xh_rec': xh_rec_unpack_dict
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


    def compute_loss(self, x, h, node_mask, edge_mask, context):
        """Computes an estimator for the variational lower bound."""

        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        #               x, one-hot,          charges
        #    bs, n_nodes, ?

        # Encoder output.
        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = self.encode(x, h, node_mask, edge_mask, context)
        
        # KL distance.
        # KL for invariant features. h
        zeros, ones = torch.zeros_like(z_h_mu), torch.ones_like(z_h_sigma)
        # --LOSS 01
        loss_kl_h = gaussian_KL(z_h_mu, ones, zeros, ones, node_mask)
        # KL for equivariant features. x
        # ~!fp16
        assert z_x_sigma.mean(dim=(1,2), keepdim=True).expand_as(z_x_sigma).allclose(z_x_sigma, atol=1e-7)
        
        
        zeros, ones = torch.zeros_like(z_x_mu), torch.ones_like(z_x_sigma.mean(dim=(1,2)))
        subspace_d = self.subspace_dimensionality(node_mask)  # (29-1)*3, int
        # --LOSS 02
        loss_kl_x = gaussian_KL_for_dimension(z_x_mu, ones, zeros, ones, subspace_d)
        loss_kl = loss_kl_h + loss_kl_x

        # Infer latent z.
        z_xh_mean = torch.cat([z_x_mu, z_h_mu], dim=2)
        diffusion_utils.assert_correctly_masked(z_xh_mean, node_mask)
        z_xh_sigma = torch.cat([z_x_sigma.expand(-1, -1, 3), z_h_sigma], dim=2)
        
        # RANDOMISER NODE (reparameterization trick): 
        # normal distribution noises: 
        # mu + sigma * eps
        # ----------------
        # z_xh_mean + z_xh_sigma * eps
        # eps = cat([z_x_noise, z_h_noise]):
        #  - z_x_noise = utils.sample_center_gravity_zero_gaussian_with_mask(...)
        #  - z_h_noise = utils.sample_gaussian_with_mask(...)
        z_xh = self.sample_normal(z_xh_mean, z_xh_sigma, node_mask)
        
        # z_xh = z_xh_mean
        diffusion_utils.assert_correctly_masked(z_xh, node_mask)
        diffusion_utils.assert_mean_zero_with_mask(z_xh[:, :, :self.n_dims], node_mask)

        # Decoder output (reconstruction).
        x_recon, h_recon = self.decoder._forward(z_xh, node_mask, edge_mask, context)
        
        # ~!norm
        if self.vae_normalize_x:
            x_recon = self.unnormalize_x(x_recon)
        
        xh_rec = torch.cat([x_recon, h_recon], dim=2)
        # --LOSS 03
        loss_recon = self.compute_reconstruction_error(xh_rec, xh)
        
        # recon loss analysis use
        recon_loss_dict = self.compute_reconstruction_error_components(xh_rec, xh, node_mask)
        # unpack xh
        xh_unpack_dict = self.unpack_xh(xh, argmax_h_cat=False)
        xh_rec_unpack_dict = self.unpack_xh(xh_rec, argmax_h_cat=False)

        # Combining the terms
        assert loss_recon.size() == loss_kl.size()
        loss = loss_recon + self.kl_weight * loss_kl   # 0.01

        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        return loss, {'loss_t': loss.squeeze(), 
                      'rec_error': loss_recon.squeeze(), 
                      'recon_loss_dict': recon_loss_dict,  # additional for rec loss analysis and xh saving
                      'xh_gt': xh_unpack_dict,
                      'xh_rec': xh_rec_unpack_dict
                      }
    
    def encode(self, x, h, node_mask=None, edge_mask=None, context=None):
        """Computes q(z|x)."""

        # ~!norm
        if self.vae_normalize_x:
            x = self.normalize_x(x)

        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)

        diffusion_utils.assert_mean_zero_with_mask(xh[:, :, :self.n_dims], node_mask)

        # Encoder output.
        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = self.encoder._forward(xh, node_mask, edge_mask, context)

        bs, _, _ = z_x_mu.size()
        sigma_0_x = torch.ones(bs, 1, 1).to(z_x_mu) * 0.0032
        sigma_0_h = torch.ones(bs, 1, self.latent_node_nf).to(z_h_mu) * 0.0032

        return z_x_mu, sigma_0_x, z_h_mu, sigma_0_h
    
    def decode(self, z_xh, node_mask=None, edge_mask=None, context=None):
        """Computes p(x|z)."""

        # Decoder output (reconstruction).
        x_recon, h_recon = self.decoder._forward(z_xh, node_mask, edge_mask, context)
        diffusion_utils.assert_mean_zero_with_mask(x_recon, node_mask)

        # ~!norm
        if self.vae_normalize_x:
            x_recon = self.unnormalize_x(x_recon)

        xh = torch.cat([x_recon, h_recon], dim=2)

        x = xh[:, :, :self.n_dims]
        diffusion_utils.assert_correctly_masked(x, node_mask)

        h_int = xh[:, :, -1:] if self.include_charges else torch.zeros(0).to(xh)
        h_cat = xh[:, :, self.n_dims:-1]  # TODO: have issue when include_charges is False
        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h_int = torch.round(h_int).long() * node_mask
        h = {'integer': h_int, 'categorical': h_cat}

        return x, h

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims  # (29-1)*3

    def compute_reconstruction_error(self, xh_rec, xh):
        """Computes reconstruction error."""

        bs, n_nodes, dims = xh.shape

        # Error on positions. / coordinates loss
        x_rec = xh_rec[:, :, :self.n_dims]
        x = xh[:, :, :self.n_dims]
        # if PARAM_REGISTRY.get('reweight_coords_loss') == "inv_class_freq":
        if self.reweight_coords_loss == "inv_class_freq":
            h_cat = xh[:, :, self.n_dims:self.n_dims + self.num_classes]
            # class_weights = PARAM_REGISTRY.get('class_weights').to(h_cat.device, h_cat.dtype)   # sum to 1
            class_weights = self.class_weights.to(h_cat.device, h_cat.dtype)   # sum to 1
            class_weights = class_weights * (bs * n_nodes)                                      # scale back

            atom_classes = h_cat.argmax(dim=2)                      # (batch_size, num_atoms)
            l2_loss_per_atom = torch.sum((x_rec - x) ** 2, dim=-1)  # (batch_size, num_atoms)
            error_x = sum_except_batch(l2_loss_per_atom * class_weights[atom_classes])
        else:
            error_x = sum_except_batch((x_rec - x) ** 2)
        
        # if PARAM_REGISTRY.get('error_x_weight', None) is not None:
        if self.error_x_weight is not None:
            # error_x = error_x * float(PARAM_REGISTRY.get('error_x_weight'))
            error_x = error_x * float(self.error_x_weight)
        
        # Error on classes. / node features (one-hot) loss
        h_cat_rec = xh_rec[:, :, self.n_dims:self.n_dims + self.num_classes]
        h_cat = xh[:, :, self.n_dims:self.n_dims + self.num_classes]
        h_cat_rec = h_cat_rec.reshape(bs * n_nodes, self.num_classes)
        h_cat = h_cat.reshape(bs * n_nodes, self.num_classes)

        # ~!fp16 ~!mp
        # if PARAM_REGISTRY.get('reweight_class_loss') == "inv_class_freq":
        if self.reweight_class_loss == "inv_class_freq":
            # class_weights = PARAM_REGISTRY.get('class_weights').to(h_cat.device, h_cat.dtype)   # sum to 1
            class_weights = self.class_weights.to(h_cat.device, h_cat.dtype)   # sum to 1
            class_weights = class_weights * (bs * n_nodes)                                      # scale back
            error_h_cat = F.cross_entropy(h_cat_rec, h_cat.argmax(dim=1), weight=class_weights, reduction='none')
            print(f"balancing class loss with inv_class_freq={class_weights}")
        else:
            error_h_cat = F.cross_entropy(h_cat_rec, h_cat.argmax(dim=1), reduction='none')

        # if PARAM_REGISTRY.get('error_h_weight', None) is not None:
        if self.error_h_weight is not None:
            # error_h_cat = error_h_cat * float(PARAM_REGISTRY.get('error_h_weight'))
            error_h_cat = error_h_cat * float(self.error_h_weight)

        error_h_cat = error_h_cat.reshape(bs, n_nodes, 1)
        error_h_cat = sum_except_batch(error_h_cat)
        # error_h_cat = sum_except_batch((h_cat_rec - h_cat) ** 2)

        # Error on charges. / periodic table atom charges loss
        if self.include_charges:
            h_int_rec = xh_rec[:, :, -self.include_charges:]
            h_int = xh[:, :, -self.include_charges:]
            error_h_int = sum_except_batch((h_int_rec - h_int) ** 2)
        else:
            error_h_int = 0.
        
        error = error_x + error_h_cat + error_h_int

        if self.training:
            denom = (self.n_dims + self.in_node_nf) * xh.shape[1]
            error = error / denom

        return error


    def compute_reconstruction_error_components(self, xh_rec, xh, node_mask):
        bs, n_nodes, dims = xh.shape
        node_mask = node_mask.view(bs*n_nodes).bool().cpu()

        # Error on positions. / coordinates loss
        x_rec = xh_rec[:, :, :self.n_dims]
        x = xh[:, :, :self.n_dims]
        error_x = sum_except_batch((x_rec - x) ** 2)

        # Error on classes. / node features (one-hot) loss
        h_cat_rec = xh_rec[:, :, self.n_dims:self.n_dims + self.num_classes]
        h_cat = xh[:, :, self.n_dims:self.n_dims + self.num_classes]
        h_cat_rec = h_cat_rec.reshape(bs * n_nodes, self.num_classes)
        h_cat = h_cat.reshape(bs * n_nodes, self.num_classes)

        # ~!fp16 ~!mp
        error_h_cat = F.cross_entropy(h_cat_rec, h_cat.argmax(dim=1), reduction='none')
        
        # accuracy & f1 metrics
        h_cat_rec_idx = h_cat_rec.argmax(dim=-1).view(-1).cpu()
        h_cat_idx = h_cat.argmax(dim=-1).view(-1).cpu()
        
        assert h_cat_rec_idx.shape[-1] == node_mask.shape[-1], f"h_cat_rec_idx: {h_cat_rec_idx.shape}, node_mask: {node_mask.shape}"
        assert h_cat_idx.shape[-1] == node_mask.shape[-1], f"h_cat_idx: {h_cat_idx.shape}, node_mask: {node_mask.shape}"
        
        h_cat_rec_idx = h_cat_rec_idx[node_mask].numpy()
        h_cat_idx = h_cat_idx[node_mask].numpy()
        
        # Overall metrics
        overall_accuracy = accuracy_score(h_cat_idx, h_cat_rec_idx)
        overall_recall = recall_score(h_cat_idx, h_cat_rec_idx, average='macro', zero_division=0)
        overall_f1 = f1_score(h_cat_idx, h_cat_rec_idx, average='macro', zero_division=0)

        # class-wise acurracy
        classwise_accuracy = {}
        # for class_idx, class_char in enumerate(PARAM_REGISTRY.get('atom_decoder')):
        for class_idx, class_char in enumerate(self.atom_decoder):
            # Create a mask for instances belonging to the current class
            class_mask = (h_cat_idx == class_idx)
            true_class = h_cat_idx[class_mask]
            pred_class = h_cat_rec_idx[class_mask]
            if true_class.size > 0:
                acc = accuracy_score(true_class, pred_class)
                classwise_accuracy[class_char] = acc
            else:
                classwise_accuracy[class_char] = float('nan')


        error_h_cat = error_h_cat.reshape(bs, n_nodes, 1)
        error_h_cat = sum_except_batch(error_h_cat)
        # error_h_cat = sum_except_batch((h_cat_rec - h_cat) ** 2)

        # Error on charges. / periodic table atom charges loss
        if self.include_charges:
            h_int_rec = xh_rec[:, :, -self.include_charges:]
            h_int = xh[:, :, -self.include_charges:]
            error_h_int = sum_except_batch((h_int_rec - h_int) ** 2)
        else:
            error_h_int = 0.
        
        error = error_x + error_h_cat + error_h_int

        if self.training:
            denom = (self.n_dims + self.in_node_nf) * xh.shape[1]
            error = error / denom
        else: 
            denom = 0

        return_dict = {
            'error': error,
            'error_x': error_x, 
            'error_h_cat': error_h_cat, 
            'error_h_int': error_h_int,
            'denom': denom,
            'n_dims': self.n_dims,
            'in_node_nf': self.in_node_nf,
            'num_atoms': xh.shape[1],
            'overall_accuracy': overall_accuracy,
            'overall_recall': overall_recall,
            'overall_f1': overall_f1,
            'classwise_accuracy': classwise_accuracy
        }
        return_dict = diffusion_utils.convert_numbers_to_tensors(return_dict, device=xh_rec.device)
        return return_dict


    def unpack_xh(self, xh, argmax_h_cat=False):
        bs, n_nodes, dims = xh.shape

        x = xh[:, :, :self.n_dims]

        h_cat = xh[:, :, self.n_dims:self.n_dims + self.num_classes]
        # h_cat = h_cat.reshape(bs * n_nodes, self.num_classes)
        if argmax_h_cat:
            h_cat = h_cat.argmax(dim=2)

        if self.include_charges:
            h_int = xh[:, :, -self.include_charges:]
        else:
            h_int = None

        return {'x': x, 'h_cat': h_cat, 'h_int': h_int}


    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps
    
    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = utils.sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
            node_mask=node_mask)
        z_h = utils.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.latent_node_nf), device=node_mask.device,
            node_mask=node_mask)
        z = torch.cat([z_x, z_h], dim=2)
        return z

    def custom_sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h,
        and optional z_int for charges
        """
        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)
        
        z_x = z[:, :, :self.n_dims].to(node_mask.device)
        z_h = z[:, :, self.n_dims:].to(node_mask.device)
        z_int = utils.sample_gaussian_with_mask(
                    size=(n_samples, n_nodes, 1), device=node_mask.device,
                    node_mask=node_mask).to(node_mask.device) \
                if self.include_charges else \
                    torch.zeros(0).to(node_mask.device)
        return {
            'x': z_x,
            'categorical': z_h,
            'integer': z_int
        }

    @torch.no_grad()
    def sample(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False):
        """
        Draw samples from the VAE model. Only the decoder will be utilised.
        """

        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.custom_sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.custom_sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        diffusion_utils.assert_mean_zero_with_mask(z['x'], node_mask)

        z_xh = torch.cat([z['x'], z['categorical'], z['integer']], dim=2)
        diffusion_utils.assert_correctly_masked(z_xh, node_mask)
        x, h = self.decode(z_xh, node_mask, edge_mask, context)

        return x, h

    @torch.no_grad()
    def reconstruct(self, x, h, node_mask=None, edge_mask=None, context=None):
        pass

    def log_info(self):
        """
        Some info logging of the model.
        """
        info = None
        print(info)

        return info


def disabled_train(self, mode=True):
    
    """Overwrite model.train with this dummy empty function to make sure train/eval mode
    does not change anymore."""
    return self





class EnLatentDiffusion(EnVariationalDiffusion):
    """
    The E(n) Latent Diffusion Module.
    """
    def __init__(self, **kwargs):
        vae = kwargs.pop('vae')
        trainable_ae_encoder = kwargs.pop('trainable_ae_encoder', False)
        trainable_ae_decoder = kwargs.pop('trainable_ae_decoder', False)
        
        super().__init__(**kwargs)

        # Create self.vae as the first stage model.
        self.trainable_ae_encoder = trainable_ae_encoder
        self.trainable_ae_decoder = trainable_ae_decoder
        
        self.instantiate_first_stage(vae)
        
        # dictionary to store activations
        self.input_activations = {}
        self.output_activations = {}

    def _register_hooks(self):
        self.hook_handles = []
        def hook_fn(module, input, output, name):
            self.input_activations[name] = input[0].clone().to(torch.float32).detach().cpu().numpy()
            self.output_activations[name] = output.clone().to(torch.float32).detach().cpu().numpy()

        # register hooks on all layers to track, i.e. nn.Linear 
        for name, layer in self.named_modules():
            if isinstance(layer, PARAM_REGISTRY.get('vis_activations_instances')):
                handle = layer.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))
                print(name)
                self.hook_handles.append(handle)
        return self.hook_handles
    
    def unnormalize_z(self, z, node_mask):
        # Overwrite the unnormalize_z function to do nothing (for sample_chain). 

        # Parse from z
        x, h_cat = z[:, :, 0:self.n_dims], z[:, :, self.n_dims:self.n_dims+self.num_classes]
        h_int = z[:, :, self.n_dims+self.num_classes:self.n_dims+self.num_classes+1]
        assert h_int.size(2) == self.include_charges

        # Unnormalize
        # x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)
        output = torch.cat([x, h_cat, h_int], dim=2)
        return output
    
    def log_constants_p_h_given_z0(self, h, node_mask):
        """Computes p(h|z0)."""
        batch_size = h.size(0)

        n_nodes = node_mask.squeeze(2).sum(1)  # N has shape [B]
        assert n_nodes.size() == (batch_size,)
        degrees_of_freedom_h = n_nodes * self.n_dims

        zeros = torch.zeros((h.size(0), 1), device=h.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_h * (- log_sigma_x - 0.5 * np.log(2 * np.pi))

    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context, fix_noise=False):
        """Samples x ~ p(x|z0).
        
        NOTE: This overrides the parent class' sample_p_xh_given_z0() function. This child
            funciton will run instead.
        """
        
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        net_out = self.phi(z0, zeros, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)

        x = xh[:, :, :self.n_dims]

        # h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        # x, h_cat, h_int = self.unnormalize(x, z0[:, :, self.n_dims:-1], h_int, node_mask)

        # h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        # h_int = torch.round(h_int).long() * node_mask

        # Make the data structure compatible with the EnVariationalDiffusion sample() and sample_chain().
        h = {'integer': xh[:, :, self.n_dims:], 'categorical': torch.zeros(0).to(xh)}
        
        return x, h
    
    
    # ~!fp16
    def log_pxh_given_z0_without_constants(
            self, x, h, z_t, gamma_0, eps, net_out, node_mask, epsilon=1e-10):

        # Computes the error for the distribution N(latent | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.
        log_pxh_given_z_without_constants = -0.5 * self.compute_error(net_out, gamma_0, eps)

        # Combine log probabilities for x and h.
        log_p_xh_given_z = log_pxh_given_z_without_constants

        return log_p_xh_given_z
    
    def forward(self, x, h, node_mask=None, edge_mask=None, context=None, loss_analysis=False):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """

        """ VAE Encoding """
        # Encode data to latent space.
        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = self.vae.encode(x, h, node_mask, edge_mask, context)

        # ~!fp16
        # self.gamma = self.gamma.to(x.device)

        # Compute fixed sigma values.
        t_zeros = torch.zeros(size=(x.size(0), 1), device=x.device)
        gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
        sigma_0 = self.sigma(gamma_0, x)

        # Infer latent z.
        z_xh_mean = torch.cat([z_x_mu, z_h_mu], dim=2)
        
        # tmp
        # z_xh_mean = z_xh_mean * node_mask
        
        diffusion_utils.assert_correctly_masked(z_xh_mean, node_mask)
        z_xh_sigma = sigma_0
        # z_xh_sigma = torch.cat([z_x_sigma.expand(-1, -1, 3), z_h_sigma], dim=2)
        
        # eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        # z_xh = mu + sigma * eps
        z_xh = self.vae.sample_normal(z_xh_mean, z_xh_sigma, node_mask)
        # z_xh = z_xh_mean
        if not self.trainable_ae_encoder:
            z_xh = z_xh.detach()  # Always keep the VAE's Encoder fixed when training LDM and/or VAE's Decoder
        diffusion_utils.assert_correctly_masked(z_xh, node_mask)

        """ VAE Decoding - required if training VAE too """
        # Compute reconstruction loss.
        if self.trainable_ae_decoder:
            # ground truth
            xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
            # Decoder output (reconstruction).
            x_recon, h_recon = self.vae.decoder._forward(z_xh, node_mask, edge_mask, context)
            
            # ~!norm
            if self.vae.normalize_x:
                x_recon = self.vae.unnormalize_x(x_recon)
            
            xh_rec = torch.cat([x_recon, h_recon], dim=2)
            loss_recon = self.vae.compute_reconstruction_error(xh_rec, xh)
            
            if loss_analysis:
                # recon loss analysis use
                recon_loss_dict = self.vae.compute_reconstruction_error_components(xh_rec, xh, node_mask)

        else:
            loss_recon = 0

        
        """ VAE Encoded features for LDM """
        # LDM
        z_x = z_xh[:, :, :self.n_dims]
        z_h = z_xh[:, :, self.n_dims:]
        diffusion_utils.assert_mean_zero_with_mask(z_x, node_mask)
        # Make the data structure compatible with the EnVariationalDiffusion compute_loss().
        z_h = {'categorical': torch.zeros(0).to(z_h), 'integer': z_h}

        # compute_loss() defined in EnVariationalDiffusion Above
        if self.training:
            # Only 1 forward pass when t0_always is False.
            loss_ld, loss_dict = self.compute_loss(z_x, z_h, node_mask, edge_mask, context, t0_always=False)
        else:
            # Less variance in the estimator, costs two forward passes.
            loss_ld, loss_dict = self.compute_loss(z_x, z_h, node_mask, edge_mask, context, t0_always=True)
        
        if loss_analysis:
            loss_dict['recon_loss_dict'] = recon_loss_dict
        
        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_h_given_z0(
            torch.cat([h['categorical'], h['integer']], dim=2), node_mask)
        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros_like(neg_log_constants)

        neg_log_pxh = loss_ld + loss_recon + neg_log_constants
        
        # # ~!wt
        # print(f"[LDM] loss_ld           : {loss_ld.mean().item()}")
        # print(f"[LDM] loss_recon        : {torch.tensor(float(loss_recon)).mean().item()}")
        # print(f"[LDM] neg_log_constants : {neg_log_constants.mean().item()}")
        # print(f"[LDM] neg_log_pxh       : {neg_log_pxh.mean().item()}")

        if loss_analysis:
            return neg_log_pxh, loss_dict   # negatve log likelihood
        else:
            return neg_log_pxh
    
    
    
    @torch.no_grad()
    def sample(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False):
        """
        Draw samples from the generative model.
        NOTE: full timesteps T.
        """
        # parent, sample LDM model
        z_x, z_h = super().sample(n_samples, n_nodes, node_mask, edge_mask, context, fix_noise)

        z_xh = torch.cat([z_x, z_h['categorical'], z_h['integer']], dim=2)
        diffusion_utils.assert_correctly_masked(z_xh, node_mask)
        x, h = self.vae.decode(z_xh, node_mask, edge_mask, context)

        return x, h
    
    @torch.no_grad()
    def sample_chain(self, n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=None):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """
        chain_flat = super().sample_chain(n_samples, n_nodes, node_mask, edge_mask, context, keep_frames)

        # xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        # chain[0] = xh  # Overwrite last frame with the resulting x and h.

        # chain_flat = chain.view(n_samples * keep_frames, *z.size()[1:])

        chain = chain_flat.view(keep_frames, n_samples, *chain_flat.size()[1:])
        chain_decoded = torch.zeros(
            size=(*chain.size()[:-1], self.vae.in_node_nf + self.vae.n_dims), device=chain.device)

        for i in range(keep_frames):
            z_xh = chain[i]
            diffusion_utils.assert_mean_zero_with_mask(z_xh[:, :, :self.n_dims], node_mask)

            x, h = self.vae.decode(z_xh, node_mask, edge_mask, context)
            xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
            chain_decoded[i] = xh
        
        chain_decoded_flat = chain_decoded.view(n_samples * keep_frames, *chain_decoded.size()[2:])

        return chain_decoded_flat

    def instantiate_first_stage(self, vae: EnHierarchicalVAE):
        self.vae = vae
        if not self.trainable_ae_encoder and not self.trainable_ae_decoder:
            self.vae.eval()
            self.vae.train = disabled_train
            for param in self.vae.parameters():
                param.requires_grad = False
            print(">>> [VAE] (Encoder) (Decoder) requires_grad = False")
        else:
            # set whole model to trainable, but if self.trainable_ae_encoder=False,
            # will detach the VAE Encoder's outputs from the loss computational graph
            # hence weights not updated.

            # update: setting requires_grad part by part - more secure
            self.vae.train()
            # Encoder
            for param in self.vae.encoder.parameters():
                if self.trainable_ae_encoder:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print(f">>> [VAE] (Encoder) requires_grad = {self.trainable_ae_encoder}")
            # Decoder
            for param in self.vae.decoder.parameters():
                if self.trainable_ae_decoder:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print(f">>> [VAE] (Decoder) requires_grad = {self.trainable_ae_decoder}")
