import numpy as np
from build_geom_dataset import process_splitted_pair_data


conformation_file = "data/d_20241203_CrossDocked_LG_PKT_MMseq2_split_CA_only/d_20241203_CrossDocked_LG_PKT_MMseq2_split__10.0A__CA_Only.npz"

filter_ligand_size = 100
filter_pocket_size = 80

all_data = np.load(conformation_file)  # 2d array: num_atoms x 5

ligand_data_list_train, ligand_data_list_test, ligand_data_list_val, \
    pocket_data_list_train, pocket_data_list_test, pocket_data_list_val \
        = process_splitted_pair_data(all_data, filter_ligand_size, filter_pocket_size)


print(f"Ligand Size: {filter_ligand_size}")
print(f"Pocket Size: {filter_pocket_size}")
print(f"====================")
print(f"Train: {len(ligand_data_list_train)}")
print(f"Test : {len(ligand_data_list_test)}")
print(f"Val  : {len(ligand_data_list_val)}")


# (geoldm) (base) 🎃 gohyixian GeoLDM-edit % python filter_dataset_size.py
# Ligand Size: 100
# Pocket Size: 1000
# ====================
# Train: 99928
# Test : 100
# Val  : 100
# (geoldm) (base) 🎃 gohyixian GeoLDM-edit % python filter_dataset_size.py
# Ligand Size: 100
# Pocket Size: 200
# ====================
# Train: 99928
# Test : 100
# Val  : 100
# (geoldm) (base) 🎃 gohyixian GeoLDM-edit % python filter_dataset_size.py
# Ligand Size: 100
# Pocket Size: 100
# ====================
# Train: 99910
# Test : 100
# Val  : 100
# (geoldm) (base) 🎃 gohyixian GeoLDM-edit % python filter_dataset_size.py
# Ligand Size: 100
# Pocket Size: 80
# ====================
# Train: 99188
# Test : 100
# Val  : 100