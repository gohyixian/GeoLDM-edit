import math
import torch
import numpy as np
import matplotlib.pyplot as plt

def f(t, T=1000):
    return 1 - (t/T)**2

def alpha_t(t, s=1.0e-05):
    return (1 - 2*s)*(f(t)) + s

def sigma_t(t):
    return math.sqrt(1 - alpha_t(t)**2)

def SNR(t):
    return alpha_t(t)**2 / sigma_t(t)**2

def w(t):
    return 1 - (SNR(t-1) / SNR(t))

def gamma(t):
    return -math.log(SNR(t))

# for t in range(1, 1001):
#     print(t, w(t))

t_values = list(range(1, 1000))
w_values = [w(t) for t in t_values]

plt.figure(figsize=(10, 6))
plt.plot(t_values, w_values, label='w(t)', color='b')
plt.xlabel('t')
plt.ylabel('w(t)')
plt.title('Plot of t vs w(t)')
plt.grid(True)
plt.legend()
# plt.show()

print(1000 * w(t=500) * 64)

print(f"log_SNR_min: {-gamma(1)}")
print(f"log_SNR_max: {-gamma(0)}")

t=500
s=t-1
gamma_s = gamma(s)
gamma_t = gamma(t)
gamma_delta = gamma_s - gamma_t
snr_gamma_delta = SNR(gamma_delta)
snr_gamma_delta_m1 = snr_gamma_delta - 1

print(f"gamma_s            : {gamma_s}")
print(f"gamma_t            : {gamma_t}")
print(f"gamma_delta        : {gamma_delta}")
print(f"snr_gamma_delta    : {snr_gamma_delta}")
print(f"snr_gamma_delta_m1 : {snr_gamma_delta_m1}")






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

class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """
    def __init__(self, noise_schedule="polynomial_2", timesteps=1000, precision=1.0e-05):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if 'polynomial' in noise_schedule:  # polynomial_2
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
        print("t_int", t_int)
        return self.gamma[t_int]

def actual_SNR(gamma):
    """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
    return torch.exp(-gamma)


T = 1000
GAMMA = PredefinedNoiseSchedule(noise_schedule="polynomial_2", timesteps=T, precision=1.0e-05)
t = 100
s = t - 1
gamma_s = GAMMA(torch.tensor(s / T))
gamma_t = GAMMA(torch.tensor(t / T))
gamma_delta = gamma_s - gamma_t
snr_gamma_delta = actual_SNR(gamma_delta)
snr_gamma_delta_m1 = snr_gamma_delta - 1

print(f"gamma_s            : {gamma_s}")
print(f"gamma_t            : {gamma_t}")
print(f"gamma_delta        : {gamma_delta}")
print(f"snr_gamma_delta    : {snr_gamma_delta}")
print(f"snr_gamma_delta_m1 : {snr_gamma_delta_m1}")
print(f"x1000 x0.5         : {snr_gamma_delta_m1 * T * 0.5}")