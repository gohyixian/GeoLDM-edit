import torch
import numpy as np


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        
        # ~!fp16
        new_device = new.device
        new = new.to(old.device)
        tmp = old * self.beta + (1 - self.beta) * new
        new = new.to(new_device)
        return tmp
        # return old * self.beta + (1 - self.beta) * new


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)

# ~!fp16
def remove_mean(x):
    # _dtype = x.dtype
    # if _dtype == torch.float16:
    #     x_32 = x.clone().to(torch.float32)
    #     mean = torch.mean(x_32, dim=1, keepdim=True).to(_dtype)
    #     x = x - mean
    #     return x
    # else:
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x

# ~!fp16
def remove_mean_with_mask(x, node_mask):    # node_mask shape: [bs, n_nodes, 1]
    # ~!fp16
    # check if sum of unmasked item passes a threshold
    _dtype = x.dtype
    if _dtype == torch.float16:
        x_32 = x.clone().to(torch.float32)
        masked_max_abs_value = (x_32 * (1 - node_mask)).abs().sum().item()
    else:
        masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    # assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    # ~!fp16
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    
    # calculate mean on masked item only
    N = node_mask.sum(1, keepdims=True)
    # if _dtype == torch.float16:
    #     mean = torch.sum(x_32, dim=1, keepdim=True) / N
    #     mean = mean.to(_dtype)
    #     x = x - mean * node_mask
    #     return x
    # else:
    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x

# ~!fp16
def assert_mean_zero(x):
    _dtype = x.dtype
    if _dtype == torch.float16:
        x_32 = x.clone().to(torch.float32)
        mean = torch.mean(x_32, dim=1, keepdim=True)
    else:
        mean = torch.mean(x, dim=1, keepdim=True)
    assert mean.abs().max().item() < 1e-4


# ~!fp16
# def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)

    _dtype = x.dtype
    if _dtype == torch.float16:
        x_32 = x.clone().to(torch.float32)
        largest_value = x_32.abs().max().item()
        error = torch.sum(x_32, dim=1, keepdim=True).abs().max().item()
    else:
        largest_value = x.abs().max().item()
        error = torch.sum(x, dim=1, keepdim=True).abs().max().item()

    rel_error = error / (largest_value + eps)
    # assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'
    assert rel_error < 1., f'Mean is not zero, relative_error {rel_error}'


# ~!fp16
def assert_correctly_masked(variable, node_mask):
    _dtype = variable.dtype
    if _dtype == torch.float16:
        var_32 = variable.clone().to(torch.float32)
        val = (var_32 * (1 - node_mask)).abs().max().item()
    else:
        val = (variable * (1 - node_mask)).abs().max().item()
    assert val < 1e-4, 'Variables not masked properly.'


def center_gravity_zero_gaussian_log_likelihood(x):
    assert len(x.size()) == 3
    B, N, D = x.size()
    assert_mean_zero(x)

    # r is invariant to a basis change in the relevant hyperplane.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian(size, device):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean(x)
    return x_projected


def center_gravity_zero_gaussian_log_likelihood_with_mask(x, node_mask):
    assert len(x.size()) == 3
    B, N_embedded, D = x.size()
    assert_mean_zero_with_mask(x, node_mask)

    # r is invariant to a basis change in the relevant hyperplane, the masked
    # out values will have zero contribution.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    N = node_mask.squeeze(2).sum(1)  # N has shape [B]
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    x_masked = x * node_mask

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected


def standard_gaussian_log_likelihood(x):
    # Normalizing constant and logpx are computed:
    log_px = sum_except_batch(-0.5 * x * x - 0.5 * np.log(2*np.pi))
    return log_px


def sample_gaussian(size, device):
    x = torch.randn(size, device=device)
    return x


def standard_gaussian_log_likelihood_with_mask(x, node_mask):
    # Normalizing constant and logpx are computed:
    log_px_elementwise = -0.5 * x * x - 0.5 * np.log(2*np.pi)
    log_px = sum_except_batch(log_px_elementwise * node_mask)
    return log_px


def sample_gaussian_with_mask(size, device, node_mask):
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    return x_masked
