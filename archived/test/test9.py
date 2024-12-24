import torch

# Original frequencies with random key order
frequencies_dict = {1: 26927525, 2: 6817964, 3: 7638308, 4: 30875, 5: 26331, 6: 308964, 7: 15221, 0: 1577180}

# Step 1: Sort the keys
sorted_keys = sorted(frequencies_dict.keys())

# Step 2: Reorder the frequencies according to the sorted keys
frequencies = torch.tensor([frequencies_dict[key] for key in sorted_keys], dtype=torch.float32)

# Step 3: Calculate inverse of frequencies
inverse_frequencies = 1.0 / frequencies

# Step 4: Normalize inverse frequencies to get weights
weights = inverse_frequencies / inverse_frequencies.sum()

# Output sorted keys and corresponding weights
print("Sorted Keys:", sorted_keys)
print("Weights:", weights)
print("Inverse Frequencies:", inverse_frequencies)
