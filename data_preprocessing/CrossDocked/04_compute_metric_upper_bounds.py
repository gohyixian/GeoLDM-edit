'''
This script performs stratified sampling to construct our train/test/val split.
In our case, molecules who contain
'''

import os
import json
import random
import pickle
import torch
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
from data_preprocessing.CrossDocked.constants import get_periodictable_list
from data_preprocessing.CrossDocked.metrics import compute_molecule_metrics
from configs.dataset_configs.datasets_config import get_dataset_info

# path to .npy file containing the combined/processed conformations/molecules,
# all stored in a single array of the format below. Atoms of the same molecule
# will have the same idx number to group them together.
#
# [[idx, atomic_num, x, y, z],
#  [idx, atomic_num, x, y, z],
#  [idx, atomic_num, x, y, z],
#   ...
#  [idx, atomic_num, x, y, z]]


# python -m data_preprocessing.CrossDocked.04_compute_metric_upper_bounds.py

dataset_info = get_dataset_info(
    dataset_name='d_20240623_CrossDocked_LG_PKT__10A__LIGAND',
    remove_h=False
)

save_splits_file = 'data/d_20240623_CrossDocked_LG_PKT/d_20240623_CrossDocked_LG_PKT__10.0A_split_811.npz'
save_metrics_path = 'data_preprocessing/CrossDocked/metric_upper_bounds'

all_data = np.load(save_splits_file)

ligand_data_train = all_data['ligand_train']
ligand_data_test  = all_data['ligand_test']
ligand_data_val   = all_data['ligand_val']

# ligand
ligand_mol_id_train = ligand_data_train[:, 0].astype(int)
ligands_train = ligand_data_train[:, 1:]
ligand_split_indices_train = np.nonzero(ligand_mol_id_train[:-1] - ligand_mol_id_train[1:])[0] + 1
ligand_data_list_train = np.split(ligands_train, ligand_split_indices_train)

ligand_mol_id_test = ligand_data_test[:, 0].astype(int)
ligands_test = ligand_data_test[:, 1:]
ligand_split_indices_test = np.nonzero(ligand_mol_id_test[:-1] - ligand_mol_id_test[1:])[0] + 1
ligand_data_list_test = np.split(ligands_test, ligand_split_indices_test)

ligand_mol_id_val = ligand_data_val[:, 0].astype(int)
ligands_val = ligand_data_val[:, 1:]
ligand_split_indices_val = np.nonzero(ligand_mol_id_val[:-1] - ligand_mol_id_val[1:])[0] + 1
ligand_data_list_val = np.split(ligands_val, ligand_split_indices_val)

an2s, s2an = get_periodictable_list(include_aa=True)


if not os.path.exists(save_metrics_path):
    os.makedirs(save_metrics_path)
master_dict = {}


# compute metrics upper bound: train
x_list, one_hot_list = [], []
for ligand in tqdm(ligand_data_list_train):
    one_hot, x = torch.split(torch.tensor(ligand), [1, 3], dim=1)
    # convert from atomic number -> character symbol -> atom_encoder value
    one_hot = torch.tensor([dataset_info['atom_encoder'][str(an2s[int(i)])] for i in one_hot])
    x_list.append(x)
    one_hot_list.append(one_hot)

metrics_dict_train = compute_molecule_metrics(one_hot_list, x_list, dataset_info)
metrics_dict_train['num_samples'] = len(x_list)
master_dict['train'] = metrics_dict_train
print(master_dict)

with open(os.path.join(save_metrics_path, 'mertrics.json'), 'w') as f:
    json.dump(master_dict, f, indent=4)



# compute metrics upper bound: test
x_list, one_hot_list = [], []
for ligand in tqdm(ligand_data_list_test):
    one_hot, x = torch.split(torch.tensor(ligand), [1, 3], dim=1)
    # convert from atomic number -> character symbol -> atom_encoder value
    one_hot = torch.tensor([dataset_info['atom_encoder'][str(an2s[int(i)])] for i in one_hot])
    x_list.append(x)
    one_hot_list.append(one_hot)

metrics_dict_test = compute_molecule_metrics(one_hot_list, x_list, dataset_info)
metrics_dict_test['num_samples'] = len(x_list)
master_dict['test'] = metrics_dict_test
print(master_dict)

with open(os.path.join(save_metrics_path, 'mertrics.json'), 'w') as f:
    json.dump(master_dict, f, indent=4)



# compute metrics upper bound: val
x_list, one_hot_list = [], []
for ligand in tqdm(ligand_data_list_val):
    one_hot, x = torch.split(torch.tensor(ligand), [1, 3], dim=1)
    # convert from atomic number -> character symbol -> atom_encoder value
    one_hot = torch.tensor([dataset_info['atom_encoder'][str(an2s[int(i)])] for i in one_hot])
    x_list.append(x)
    one_hot_list.append(one_hot)

metrics_dict_val = compute_molecule_metrics(one_hot_list, x_list, dataset_info)
metrics_dict_val['num_samples'] = len(x_list)
master_dict['val'] = metrics_dict_val
print(master_dict)

with open(os.path.join(save_metrics_path, 'mertrics.json'), 'w') as f:
    json.dump(master_dict, f, indent=4)

print('DONE.')