'''
This script performs stratified sampling to construct our train/test/val split.
In our case, molecules who contain
'''

import os
import random
import pickle
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
from constants import get_periodictable_list
from utils import stratified_sampling_by_atom_distribution, euclidean_distance, plot_combined_boxplot

# path to .npy file containing the combined/processed conformations/molecules,
# all stored in a single array of the format below. Atoms of the same molecule
# will have the same idx number to group them together.
#
# [[idx, atomic_num, x, y, z],
#  [idx, atomic_num, x, y, z],
#  [idx, atomic_num, x, y, z],
#   ...
#  [idx, atomic_num, x, y, z]]



# conformation_file = '../../data/d_20240623_CrossDocked_LG_PKT/d_20240623_CrossDocked_LG_PKT__10.0A__subset_0.01.npz'
# save_splits_file = '../../data/d_20240623_CrossDocked_LG_PKT/d_20240623_CrossDocked_LG_PKT__10.0A__subset_0.01_split_811.npz'
# save_plot_path = '../../data/d_20240623_CrossDocked_LG_PKT/d_20240623_CrossDocked_LG_PKT__10.0A__subset_0.01_split_811'

conformation_file = '../../data/d_20240623_CrossDocked_LG_PKT/d_20240623_CrossDocked_LG_PKT__10.0A.npz'
save_splits_file = '../../data/d_20240623_CrossDocked_LG_PKT/d_20240623_CrossDocked_LG_PKT__10.0A_split_811.npz'
save_plot_path = '../../data/d_20240623_CrossDocked_LG_PKT/d_20240623_CrossDocked_LG_PKT__10.0A_split_811'

SPLIT_BASED_ON = 'ligand'  # ligand | pocket
TRAIN_TEST_VAL = [0.8, 0.1, 0.1]
ATOMS = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl']
MODE = 'presence'

random.seed(42)
np.random.seed(42)

# =========

assert sum(TRAIN_TEST_VAL) == 1.0
an2s, s2an = get_periodictable_list(include_aa=True)
atom_to_class = {}
for i, symbol in enumerate(ATOMS):
    atom_to_class[s2an[symbol]] = i

all_data = np.load(conformation_file)

all_ligand_data = all_data['ligand']  # TODO: fix script for ligand, pocket, ligand+pocket
all_pocket_data = all_data['pocket']

# ligand
ligand_mol_id = all_ligand_data[:, 0].astype(int)
ligand_molecules = all_ligand_data[:, :]
ligand_split_indices = np.nonzero(ligand_mol_id[:-1] - ligand_mol_id[1:])[0] + 1
ligand_data_list = list(np.split(ligand_molecules, ligand_split_indices))

# pocket
pocket_mol_id = all_pocket_data[:, 0].astype(int)
pocket_molecules = all_pocket_data[:, :]
pocket_split_indices = np.nonzero(pocket_mol_id[:-1] - pocket_mol_id[1:])[0] + 1
pocket_data_list = list(np.split(pocket_molecules, pocket_split_indices))

# randomly shuffle in ligand-pocket pairs
paired_list = list(zip(ligand_data_list, pocket_data_list))
random.shuffle(paired_list)
ligand_data_list, pocket_data_list = zip(*paired_list)
ligand_data_list = list(ligand_data_list)
pocket_data_list = list(pocket_data_list)


if SPLIT_BASED_ON.strip().lower() in ['ligand', 'lg']:
    data_list = ligand_data_list
elif SPLIT_BASED_ON.strip().lower() in ['pocket', 'pkt']:
    data_list = pocket_data_list

molecule_data = []  # a 2D list, where each row corresponds to a molecule, and tells us the count of each atom in the molecule
for mol in data_list:
    mol_count = np.zeros(len(atom_to_class), dtype=np.int32)
    for atom in mol:
        atomic_num = atom[1]
        class_idx = atom_to_class.get(atomic_num, None)
            
        assert class_idx is not None, f"class_idx: {class_idx}"
        mol_count[class_idx] += 1  # Count the number of atoms of this type
    molecule_data.append(mol_count.tolist())

molecule_data_np = np.array(molecule_data)

# remove count, keep presence
if MODE == 'presence':
    molecule_data_np = molecule_data_np.astype(np.bool).astype(np.int32)

# stratified sampling
train_idx, test_idx, val_idx = stratified_sampling_by_atom_distribution(molecule_data_np, TRAIN_TEST_VAL)

print(f"len(data_pairs): {len(ligand_data_list)}")
print(f"len(all_idx)   : {len(train_idx + test_idx + val_idx)}")
print(f"len(train_idx) : {len(train_idx)}")
print(f"len(test_idx)  : {len(test_idx)}")
print(f"len(val_idx)   : {len(val_idx)}")

set_train = set(train_idx)
set_test = set(test_idx)
set_val = set(val_idx)
assert len(set_train.intersection(set_test)) == 0
assert len(set_train.intersection(set_val)) == 0
assert len(set_test.intersection(set_val)) == 0



ligand_train = [ligand_data_list[i] for i in train_idx]
pocket_train = [pocket_data_list[i] for i in train_idx]

ligand_test = [ligand_data_list[i] for i in test_idx]
pocket_test = [pocket_data_list[i] for i in test_idx]

ligand_val = [ligand_data_list[i] for i in val_idx]
pocket_val = [pocket_data_list[i] for i in val_idx]



# calculate data distribution between splits
all_data = {
    "ligand_train": ligand_train,
    "ligand_test": ligand_test,
    "ligand_val": ligand_val,
    "pocket_train": pocket_train,
    "pocket_test": pocket_test,
    "pocket_val": pocket_val
}
dist_info = {}

for name, data_splits in all_data.items():
    num_atoms_per_mol = []
    max_radius = []
    atom_type_counts = {}
    
    for mol in data_splits:
        # num atoms
        num_atoms_per_mol.append(len(mol))
        
        # atom type counts
        for atom in mol:
            atom_type = atom[1]
            atom_type_counts[int(atom_type)] = atom_type_counts.get(int(atom_type), 0) + 1
        
        # max radius
        center = np.mean(mol[:, 2:], axis=0, keepdims=True)
        radius = euclidean_distance(mol[:, 2:], center, axis=1)
        max_radius.append(float(np.max(radius)))

    dist_info[name] = {
        'num_atoms_per_mol': num_atoms_per_mol,
        'max_radius': max_radius,
        'atom_type_counts': atom_type_counts
    }


colors = ['skyblue', 'salmon', 'green', 'blue', 'magenta', 'purple']  # first 3 used
if not os.path.exists(save_plot_path):
    os.makedirs(save_plot_path)

for name, y_label in {'num_atoms_per_mol': 'Number of Atoms', 'max_radius': 'Radius in Angstroms'}.items():
    for category in ['ligand', 'pocket']:
        name_list = []
        data_to_plot = []
        for split_name, dist in dist_info.items():
            if category in split_name:
                name_list.append(split_name)
                data_to_plot.append(dist[name])
        
        plot_combined_boxplot(
            data_lists=data_to_plot, 
            colors=colors, 
            dataset_names=name_list, 
            title=name, 
            xlabel='', 
            ylabel=y_label, 
            img_save_path=os.path.join(save_plot_path, f"{name}_{category}.png"), 
            ymin=None, 
            ymax=None
        )


percentage_tbl = PrettyTable()
counts_tbl = PrettyTable()
percentage_tbl.field_names = [''] + ATOMS
counts_tbl.field_names = ['', 'Total Num Atoms'] + ATOMS

for split_name, info_dict in dist_info.items():
    atom_type_count_info_dict = info_dict['atom_type_counts']
    total_num_counts = sum([v for k,v in atom_type_count_info_dict.items()])
    
    percentage_txt = [split_name]
    counts_text = [split_name, total_num_counts]
    
    for a in ATOMS:
        atom_count = atom_type_count_info_dict.get(s2an[a], 0)
        percentage_txt.append(f"{float(atom_count / total_num_counts):3.1} %")
        counts_text.append(atom_count)
    percentage_tbl.add_row(percentage_txt)
    counts_tbl.add_row(counts_text)

for field in percentage_tbl.field_names:
    percentage_tbl.align[field] = 'l'
for field in counts_tbl.field_names:
    counts_tbl.align[field] = 'r'

with open(os.path.join(save_plot_path, 'atom_type_distribution.txt'), 'w') as f:
    print(f"Percentage of each atom class in each split\n{percentage_tbl}\n\n\n", file=f)
    print(f"Frequency of each atom class in each split\n{counts_tbl}\n\n\n", file=f)
    print(f"Logs\n=====", file=f)
    print(f"len(data_pairs): {len(ligand_data_list)}", file=f)
    print(f"len(all_idx)   : {len(train_idx + test_idx + val_idx)}", file=f)
    print(f"len(train_idx) : {len(train_idx)}", file=f)
    print(f"len(test_idx)  : {len(test_idx)}", file=f)
    print(f"len(val_idx)   : {len(val_idx)}", file=f)


ligand_train = np.concatenate(ligand_train, axis=0)
pocket_train = np.concatenate(pocket_train, axis=0)

ligand_test = np.concatenate(ligand_test, axis=0)
pocket_test = np.concatenate(pocket_test, axis=0)

ligand_val = np.concatenate(ligand_val, axis=0)
pocket_val = np.concatenate(pocket_val, axis=0)

np.savez(
    save_splits_file, 
    ligand_train=ligand_train,
    ligand_test=ligand_test,
    ligand_val=ligand_val,
    pocket_train=pocket_train,
    pocket_test=pocket_test,
    pocket_val=pocket_val
)

print("Done.")
