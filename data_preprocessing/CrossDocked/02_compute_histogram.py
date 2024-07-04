'''
This script computes the necessary stats for the dataset required in configs/datasets_config.py
For CrossDocked, we will compute the stats for the combined [Ligand+Pocket] for VAE training,
and [Ligand] only for LDM & ControlNet training.

All stats listed for [Ligand] will be the stats of [Ligands] only, except for the below ([Ligand+Pocket]):
 - atom_encoder
 - atomic_nb
 - atom_decoder
 - atom_types [This will have the frequency of residues (negative atomic_nb ones) set to 0, for code compatibility]
'''

import os
import pickle
import numpy as np
from tqdm import tqdm
from constants import get_periodictable_list

# path to .npy file containing the combined/processed conformations/molecules,
# all stored in a single array of the format below. Atoms of the same molecule
# will have the same idx number to group them together.
#
# [[idx, atomic_num, x, y, z],
#  [idx, atomic_num, x, y, z],
#  [idx, atomic_num, x, y, z],
#   ...
#  [idx, atomic_num, x, y, z]]



conformation_file = '../../data/d_20240623_CrossDocked_LG_PKT/d_20240623_CrossDocked_LG_PKT__10.0A__CA_Only__no_H.npz'

all_data = np.load(conformation_file)

all_ligand_data = all_data['ligand']  # TODO: fix script for ligand, pocket, ligand+pocket
all_pocket_data = all_data['pocket']

# ligand
ligand_mol_id = all_ligand_data[:, 0].astype(int)
ligand_molecules = all_ligand_data[:, 1:]
ligand_split_indices = np.nonzero(ligand_mol_id[:-1] - ligand_mol_id[1:])[0] + 1
ligand_data_list = np.split(ligand_molecules, ligand_split_indices)

# pocket
pocket_mol_id = all_pocket_data[:, 0].astype(int)
pocket_molecules = all_pocket_data[:, 1:]
pocket_split_indices = np.nonzero(pocket_mol_id[:-1] - pocket_mol_id[1:])[0] + 1
pocket_data_list = np.split(pocket_molecules, pocket_split_indices)

# ligand+pocket
all_data_list = list(ligand_data_list) + list(pocket_data_list)





# [Ligand+Pocket]
# ======================================
all_n_nodes = {}
all_atom_types_raw = {}

for i in tqdm(range(len(all_data_list))):
    # update n_nodes freq count
    n_node = all_data_list[i].shape[0]
    all_n_nodes[n_node] = all_n_nodes.get(n_node, 0) + 1
    
    # update atom_types freq count
    for row in all_data_list[i]:
        atom_type = int(row[0])
        all_atom_types_raw[atom_type] = all_atom_types_raw.get(atom_type, 0) + 1

# maximum number of nodes
all_max_n_nodes = sorted(list(all_n_nodes.keys()))[-1]
all_n_nodes = dict(sorted(all_n_nodes.items()))

# sorted according to atomic number
all_atomic_nb = []
all_tmp_freq = []
for k,v in all_atom_types_raw.items():
    all_atomic_nb.append(k)
    all_tmp_freq.append(v)

# actual atomic_nb
all_atomic_nb, all_tmp_freq = zip(*sorted(zip(all_atomic_nb, all_tmp_freq)))
all_atomic_nb, all_tmp_freq = list(all_atomic_nb), list(all_tmp_freq)

# actual atom_types, atom_encoder, atom_decoder
all_atom_types = {}
all_atom_encoder = {}
all_atom_decoder = []
an2s, s2an = get_periodictable_list(include_aa=True)

for i in range(len(all_atomic_nb)):
    all_atom_types[i] = all_tmp_freq[i]
    
    atom_symbol = str(an2s[all_atomic_nb[i]])
    all_atom_encoder[atom_symbol] = i
    all_atom_decoder.append(atom_symbol)





# [Ligand]
# ======================================
ligand_n_nodes = {}
ligand_atom_types_raw = {}
# include all atoms types avail in VAE
for k in all_atom_decoder:
    ligand_atom_types_raw[s2an[k]] = 0

for i in tqdm(range(len(ligand_data_list))):
    # update n_nodes freq count
    n_node = ligand_data_list[i].shape[0]
    ligand_n_nodes[n_node] = ligand_n_nodes.get(n_node, 0) + 1
    
    # update atom_types freq count
    for row in ligand_data_list[i]:
        atom_type = int(row[0])
        ligand_atom_types_raw[atom_type] = ligand_atom_types_raw.get(atom_type, 0) + 1

# maximum number of nodes
ligand_max_n_nodes = sorted(list(ligand_n_nodes.keys()))[-1]
ligand_n_nodes = dict(sorted(ligand_n_nodes.items()))

# sorted according to atomic number
ligand_atomic_nb = []
ligand_tmp_freq = []
for k,v in ligand_atom_types_raw.items():
    ligand_atomic_nb.append(k)
    ligand_tmp_freq.append(v)

# actual atomic_nb
ligand_atomic_nb, ligand_tmp_freq = zip(*sorted(zip(ligand_atomic_nb, ligand_tmp_freq)))
ligand_atomic_nb, ligand_tmp_freq = list(ligand_atomic_nb), list(ligand_tmp_freq)

# actual atom_types
ligand_atom_types = {}
for i in range(len(ligand_atomic_nb)):
    ligand_atom_types[i] = ligand_tmp_freq[i]








# save to txt file
file_name = conformation_file.split("/")[-1][:-4]
with open(f'stats__{file_name}.txt', 'w') as f:
    print("Ligand+Pocket", file=f)
    print("=============", file=f)
    print(f"'atom_encoder': {all_atom_encoder},", file=f)
    print(f"'atomic_nb': {all_atomic_nb},", file=f)
    print(f"'atom_decoder': {all_atom_decoder},", file=f)
    print(f"'max_n_nodes': {all_max_n_nodes},", file=f)
    print(f"'n_nodes': {all_n_nodes},", file=f)
    print(f"'atom_types': {all_atom_types},", file=f)
    print("\n", file=f)
    print("Ligand", file=f)
    print("======", file=f)
    print(f"'atom_encoder': {all_atom_encoder},", file=f)  # same as [Ligand+Pocket]
    print(f"'atomic_nb': {all_atomic_nb},", file=f)        # same as [Ligand+Pocket]
    print(f"'atom_decoder': {all_atom_decoder},", file=f)  # same as [Ligand+Pocket]
    print(f"'max_n_nodes': {ligand_max_n_nodes},", file=f)
    print(f"'n_nodes': {ligand_n_nodes},", file=f)
    print(f"'atom_types': {ligand_atom_types},", file=f)   # should include Amino Acid types but frequency=0

# # with open('../configs/n_nodes__{file_name}.pkl', 'wb') as file:  # Use 'wb' mode for binary writing
# #     pickle.dump(ligand_n_nodes, file)

print("DONE.")