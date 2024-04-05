import torch
import numpy as np

atom_mask = torch.from_numpy(np.array([1,2,3]))

edge_mask = atom_mask.unsqueeze(0) * atom_mask.unsqueeze(1)

diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool)  # [1, 29, 29] diagonal
edge_mask *= diag_mask      # remove diagonals / self connections
        
print(edge_mask)