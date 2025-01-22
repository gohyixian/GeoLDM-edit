import numpy as np
import matplotlib.pyplot as plt
import torch

# Constants
s = 10e-5
T = 1000  # Assuming T=1 for simplicity, can be changed if required

# Time points
t = np.linspace(0, T, T)

# Functions
f_t = (1 - (t/T)**2)
alpha_t = ((1 - 2*s) * f_t) + s
sigma_t_squared = 1 - (alpha_t**2)
gamma_t = -(np.log(alpha_t**2) - np.log(sigma_t_squared))
snr_t = (alpha_t**2)/sigma_t_squared

# alpha_t = np.sqrt(np.array(torch.sigmoid(torch.from_numpy(-gamma_t))))
# sigma_t = np.sqrt(np.array(torch.sigmoid(torch.from_numpy(gamma_t))))
# snr_t = np.exp(-gamma_t)

# Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(t, gamma_t, label=r'$\gamma(t)$')
# plt.xlabel('Time (t)')
# plt.ylabel(r'$\gamma(t)$')
# plt.title(r'Plot of $\gamma(t)$')
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(t, alpha_t, label=r'$\alpha(t)$')
# plt.xlabel('Time (t)')
# plt.ylabel(r'$\alpha(t)$')
# plt.title(r'Plot of $\alpha(t)$')
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(t, sigma_t, label=r'$\sigma(t)$')
# plt.xlabel('Time (t)')
# plt.ylabel(r'$\sigma(t)$')
# plt.title(r'Plot of $\sigma(t)$')
# plt.legend()
# plt.grid(True)
# plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t, snr_t, label="SNR(t)")
plt.xlabel('Time (t)')
plt.ylabel("SNR(t)")
plt.title("Plot of SNR(t)")
plt.legend()
plt.grid(True)
plt.show()
