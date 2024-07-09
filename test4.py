import torch


def get_adj_matrix(n_nodes_1, n_nodes_2, batch_size):
    n1, n2 = [], []
    for batch_idx in range(batch_size):
        for i in range(n_nodes_1):
            for j in range(n_nodes_2):
                n1.append(i + batch_idx * n_nodes_1)
                n2.append(j + batch_idx * n_nodes_2)
    # ~!to ~!mp
    edges = [torch.LongTensor(n1),
            torch.LongTensor(n2)]
    return edges
    
    
# Example ligand and pocket atom masks
ligand_atom_mask = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.float32)  # Shape: [2, 3]
pocket_atom_mask = torch.tensor([[1, 0], [1, 1]], dtype=torch.float32)        # Shape: [2, 2]
ligand_atom_mask_batched = torch.tensor([1, 1, 0, 1, 0, 1], dtype=torch.float32)  # Shape: [2, 3]
pocket_atom_mask_batched = torch.tensor([1, 0, 1, 1], dtype=torch.float32)        # Shape: [2, 2]


ligand_batch_size, ligand_n_nodes = ligand_atom_mask.size()
pocket_batch_size, pocket_n_nodes = pocket_atom_mask.size()

assert ligand_batch_size == pocket_batch_size, f"Different batch sizes: Ligand={ligand_batch_size}, Pocket={pocket_batch_size}"

# Creating edge masks
ligand_edge_mask = ligand_atom_mask.unsqueeze(1) * ligand_atom_mask.unsqueeze(2)  # Shape: [batch_size, n_nodes_ligand, n_nodes_ligand]
# joint_edge_mask = ligand_atom_mask.unsqueeze(1) * pocket_atom_mask.unsqueeze(2)  # Shape: [batch_size, n_nodes_ligand, n_nodes_pocket]
joint_edge_mask_2 = pocket_atom_mask.unsqueeze(1) * ligand_atom_mask.unsqueeze(2)  # Shape: [batch_size, n_nodes_ligand, n_nodes_pocket]
# POCKET * LIGAND

ligand_batch = {}
ligand_batch['edge_mask'] = ligand_edge_mask.view(ligand_batch_size * ligand_n_nodes * ligand_n_nodes, 1)
# joint_edge_mask = joint_edge_mask.view(ligand_batch_size * ligand_n_nodes * pocket_n_nodes, 1)
joint_edge_mask_2 = joint_edge_mask_2.view(ligand_batch_size * ligand_n_nodes * pocket_n_nodes, 1)


edge_index = get_adj_matrix(n_nodes_1=3, n_nodes_2=2, batch_size=2)
n1, n2 = edge_index
print(n1, n2)
joint_edge_mask_3 = ligand_atom_mask_batched[n1] * pocket_atom_mask_batched[n2]
print(joint_edge_mask_3)
# print("Ligand Edge Mask:")
# print(ligand_batch['edge_mask'])
# print("Joint Edge Mask:")
# print(joint_edge_mask)
# for i in range(len(joint_edge_mask)):
#     print(f"{int(joint_edge_mask[i][0])} ", end='', flush=True)
# print()
for i in range(len(joint_edge_mask_2)):
    print(f"{int(joint_edge_mask_2[i][0])} ", end='', flush=True)
print()
for i in range(len(joint_edge_mask_3)):
    print(f"{int(joint_edge_mask_3[i])} ", end='', flush=True)
print()
for i in range(len(joint_edge_mask_2)):
    print(f"{int(joint_edge_mask_2[i][0] - joint_edge_mask_3[i])} ", end='')