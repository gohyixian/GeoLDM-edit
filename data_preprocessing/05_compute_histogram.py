'''
This script computes the necessary stats for the dataset required in configs/datasets_config.py
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

conformation_file = '../data/d_20240428_combined_geom_PDBB_full_refined_core_LG_PKT/d_20240428_combined_geom_PDBB_full_refined_core_LG_PKT__geom_1.npy'

all_data = np.load(conformation_file)

mol_id = all_data[:, 0].astype(int)
molecules = all_data[:, 1:]

split_indices = np.nonzero(mol_id[:-1] - mol_id[1:])[0] + 1
data_list = np.split(molecules, split_indices)


n_nodes = {}
atom_types_raw = {}

for i in tqdm(range(len(data_list))):
    # update n_nodes freq count
    n_node = data_list[i].shape[0]
    n_nodes[n_node] = n_nodes.get(n_node, 0) + 1
    
    # update atom_types freq count
    for row in data_list[i]:
        atom_type = int(row[0])
        atom_types_raw[atom_type] = atom_types_raw.get(atom_type, 0) + 1

# maximum number of nodes
max_n_nodes = sorted(list(n_nodes.keys()))[-1]
n_nodes = dict(sorted(n_nodes.items()))

# sorted according to atomic number
atomic_nb = []
tmp_freq = []
for k,v in atom_types_raw.items():
    atomic_nb.append(k)
    tmp_freq.append(v)

# actual atomic_nb
atomic_nb, tmp_freq = zip(*sorted(zip(atomic_nb, tmp_freq)))
atomic_nb, tmp_freq = list(atomic_nb), list(tmp_freq)

# actual atom_types, atom_encoder, atom_decoder
atom_types = {}
atom_encoder = {}
atom_decoder = []
an2s, s2an = get_periodictable_list()

for i in range(len(atomic_nb)):
    atom_types[i] = tmp_freq[i]
    
    atom_symbol = str(an2s[atomic_nb[i]])
    atom_encoder[atom_symbol] = i
    atom_decoder.append(atom_symbol)



# save to txt file
with open('stats__d_20240428_combined_geom_PDBB_full_refined_core_LG_PKT.txt', 'w') as f:
    print(f"'atom_encoder': {atom_encoder},", file=f)
    print(f"'atomic_nb': {atomic_nb},", file=f)
    print(f"'atom_decoder': {atom_decoder},", file=f)
    print(f"'max_n_nodes': {max_n_nodes},", file=f)
    print(f"'n_nodes': {n_nodes},", file=f)
    print(f"'atom_types': {atom_types},", file=f)
    

with open('../configs/n_nodes__d_20240428_combined_geom_PDBB_full_refined_core_LG_PKT.pkl', 'wb') as file:  # Use 'wb' mode for binary writing
    pickle.dump(n_nodes, file)

print("DONE.")



