import torch

dataset_info = {
    'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5, 'S': 6, 'Cl': 7},
    'atom_types': {0: 1577180, 1: 26927525, 2: 6817964, 3: 7638308, 4: 30875, 5: 26331, 6: 308964, 7: 15221},
}

atom_encoder = dataset_info['atom_encoder']
atom_types = dataset_info['atom_types']
index_to_atom = {v: k for k, v in atom_encoder.items()}
sorted_atom_types = sorted(atom_types.items(), key=lambda x: x[1], reverse=True)
sorted_atom_encoder = {index_to_atom[k]: k for k, _ in sorted_atom_types}


class_freq_dict = dataset_info['atom_types']
sorted_keys = sorted(class_freq_dict.keys())
frequencies = torch.tensor([class_freq_dict[key] for key in sorted_keys])
inverse_frequencies = 1.0 / frequencies
class_weights = inverse_frequencies / inverse_frequencies.sum()  # normalize

# Define a smoothing factor (temperature scaling)
smoothing_factor = 0.75  # This value controls the degree of smoothing. Adjust as needed.

# Apply smoothing by scaling the inverse frequencies
smoothed_inverse_frequencies = torch.pow(inverse_frequencies, smoothing_factor)

# Recalculate the normalized class weights
smoothed_class_weights = smoothed_inverse_frequencies / smoothed_inverse_frequencies.sum()

print(f"Smoothing Factor: {smoothing_factor}")
print(f"======================")
for k,idx in sorted_atom_encoder.items():
    print(f"{k:>2}  {dataset_info['atom_types'][idx]:>8}  {inverse_frequencies[idx]:.5e}  {class_weights[idx]:.5e}  {smoothed_class_weights[idx]:.5e}")


# (geoldm) (base) 🎃 gohyixian test % python test_smooth.py
# Smoothing Factor: 0.75
# ======================
#  C  26927525  3.71367e-08  2.64788e-04  1.52112e-03
#  O   7638308  1.30919e-07  9.33464e-04  3.91349e-03
#  N   6817964  1.46671e-07  1.04578e-03  4.26159e-03
#  H   1577180  6.34043e-07  4.52078e-03  1.27762e-02
#  S    308964  3.23662e-06  2.30774e-02  4.33892e-02
#  F     30875  3.23887e-05  2.30934e-01  2.44122e-01
#  P     26331  3.79781e-05  2.70787e-01  2.75082e-01
# Cl     15221  6.56987e-05  4.68437e-01  4.14935e-01
# (geoldm) (base) 🎃 gohyixian test % python test_smooth.py
# Smoothing Factor: 0.5
# ======================
#  C  26927525  3.71367e-08  2.64788e-04  8.20315e-03
#  O   7638308  1.30919e-07  9.33464e-04  1.54021e-02
#  N   6817964  1.46671e-07  1.04578e-03  1.63024e-02
#  H   1577180  6.34043e-07  4.52078e-03  3.38952e-02
#  S    308964  3.23662e-06  2.30774e-02  7.65817e-02
#  F     30875  3.23887e-05  2.30934e-01  2.42257e-01
#  P     26331  3.79781e-05  2.70787e-01  2.62328e-01
# Cl     15221  6.56987e-05  4.68437e-01  3.45030e-01
# (geoldm) (base) 🎃 gohyixian test % python test_smooth.py
# Smoothing Factor: 0.25
# ======================
#  C  26927525  3.71367e-08  2.64788e-04  3.78173e-02
#  O   7638308  1.30919e-07  9.33464e-04  5.18192e-02
#  N   6817964  1.46671e-07  1.04578e-03  5.33122e-02
#  H   1577180  6.34043e-07  4.52078e-03  7.68723e-02
#  S    308964  3.23662e-06  2.30774e-02  1.15548e-01
#  F     30875  3.23887e-05  2.30934e-01  2.05513e-01
#  P     26331  3.79781e-05  2.70787e-01  2.13857e-01
# Cl     15221  6.56987e-05  4.68437e-01  2.45261e-01
# (geoldm) (base) 🎃 gohyixian test % python test_smooth.py
# Smoothing Factor: 0.1
# ======================
#  C  26927525  3.71367e-08  2.64788e-04  8.16116e-02
#  O   7638308  1.30919e-07  9.33464e-04  9.25703e-02
#  N   6817964  1.46671e-07  1.04578e-03  9.36281e-02
#  H   1577180  6.34043e-07  4.52078e-03  1.08389e-01
#  S    308964  3.23662e-06  2.30774e-02  1.27579e-01
#  F     30875  3.23887e-05  2.30934e-01  1.60624e-01
#  P     26331  3.79781e-05  2.70787e-01  1.63202e-01
# Cl     15221  6.56987e-05  4.68437e-01  1.72396e-01