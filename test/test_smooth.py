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
smoothing_factor = 0.1  # This value controls the degree of smoothing. Adjust as needed.

# Apply smoothing by scaling the inverse frequencies
smoothed_inverse_frequencies = torch.pow(inverse_frequencies, smoothing_factor)

# Recalculate the normalized class weights
smoothed_class_weights = smoothed_inverse_frequencies / smoothed_inverse_frequencies.sum()


for k,idx in sorted_atom_encoder.items():
    print(f"{k:>2}  {dataset_info['atom_types'][idx]:>8}  {inverse_frequencies[idx]:.5e}  {class_weights[idx]:.5e}  {smoothed_class_weights[idx]:.5e}")
