import os
import numpy as np
from constants import get_periodictable_list

an2s, s2an = get_periodictable_list(include_aa=True)


def save_xyz_file(molecule: np.ndarray, filename, is_ca=False):
        
    num_atoms, _ = molecule.shape
    with open(filename, 'w') as f:
        print(f"{num_atoms}", file=f)
        print(f"Generated by script", file=f)
        for atom_i in range(num_atoms):
            atom = molecule[atom_i, :]
            atom_number = atom[0]
            if not is_ca:
                atom_type = an2s[int(atom_number)]
            else:
                atom_type = an2s[1]  # arbitrarily use hydrogens
            x, y, z = float(atom[1]), float(atom[2]), float(atom[3])
            f.write("%s %.9f %.9f %.9f\n" % (atom_type, x, y, z))


processed_dataset_file = \
    '/Users/gohyixian/Documents/GitHub/FYP/GeoLDM-edit/data/d_20241203_CrossDocked_LG_PKT_MMseq2_split_CA_only/d_20241203_CrossDocked_LG_PKT_MMseq2_split__10.0A__CA_Only.npz'

# partition = 'train'
partition = 'test'

save_path = f"validate_processed_dataset/{partition}"
if not os.path.exists(save_path):
    os.makedirs(save_path)

all_data = np.load(processed_dataset_file)
ligand_data = all_data[f'ligand_{partition}']
pocket_data = all_data[f'pocket_{partition}']

# ligand
ligand_mol_id = ligand_data[:, 0].astype(int)
ligand_molecules = ligand_data[:, 1:]
ligand_split_indices = np.nonzero(ligand_mol_id[:-1] - ligand_mol_id[1:])[0] + 1
ligand_data_list = np.split(ligand_molecules, ligand_split_indices)
ligand_data_list = list(ligand_data_list)

# pocket
pocket_mol_id = pocket_data[:, 0].astype(int)
pocket_molecules = pocket_data[:, 1:]
pocket_split_indices = np.nonzero(pocket_mol_id[:-1] - pocket_mol_id[1:])[0] + 1
pocket_data_list = np.split(pocket_molecules, pocket_split_indices)
pocket_data_list = list(pocket_data_list)


# IDs = [0, 100, 234, 876, 1001, 9999]
IDs = [0, 5, 30, 78, 52]
for ID in IDs:
    save_xyz_file(
        np.concatenate((ligand_data_list[ID], pocket_data_list[ID]), axis=0), 
        str(os.path.join(save_path, f"{str(ID).zfill(7)}_ligand+pocket.xyz"))
    )

