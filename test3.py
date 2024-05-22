import torch

t_int = torch.randint(0, 99 + 1, size=(8, 1)).float()
s_int = t_int - 1
t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0).

print(t_is_zero.dtype)
print(t_int.dtype)

