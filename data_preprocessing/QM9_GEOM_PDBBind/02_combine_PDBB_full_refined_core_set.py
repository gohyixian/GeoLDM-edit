# Each folder in refined-set has only 1 ligand position, 1 pocket position, 1 protein position
# Each folder in core-set has 1 ligand position, 1 pocket position, 1 protein position

# There are folders that exists in  both core-set and refined-set. This script combines both
# core-set and refined-set by eliminating the overlapping folders in refined-set and instead using
# the ones from core-set as they contain more possible positions about the ligands for that specific
# pocket & protein.

import os
import shutil
from tqdm import tqdm

# full_set = '/Users/gohyixian/Documents/Documents/3.2_FYP_1/data/PDBBind2020/v2020-other-PL'
full_set = '/mnt/c/Users/PC/Desktop/yixian/data/PDBBind2020/v2020-other-PL'

# refined_set = '/Users/gohyixian/Documents/Documents/3.2_FYP_1/data/PDBBind2020/refined-set'
refined_set = '/mnt/c/Users/PC/Desktop/yixian/data/PDBBind2020/refined-set'

# core_set = '/Users/gohyixian/Documents/Documents/3.2_FYP_1/data/PDBBind2020/CASF-2016/coreset'
core_set = '/mnt/c/Users/PC/Desktop/yixian/data/PDBBind2020/CASF-2016/coreset'

# to be created
# combined_set = '/Users/gohyixian/Documents/Documents/3.2_FYP_1/data/PDBBind2020/00-combined-full-refined-and-core-set'
combined_set = '/mnt/c/Users/PC/Desktop/yixian/data/PDBBind2020/00-combined-full-refined-and-core-set'

ligand = 'ligand'      # possible ones: ligand.mols, ligand_opt.mol2
ligand_ext = '.mol2'

pocket = 'pocket'
pocket_ext = '.pdb'

protein = 'protein'
protein_ext = '.pdb'

to_remove = ['.DS_Store', 'index', 'readme']

full_set_folders = sorted(os.listdir(full_set))
core_set_folders = sorted(os.listdir(core_set))
refined_set_folders = sorted(os.listdir(refined_set))

full_set_folders = [f for f in full_set_folders if f not in to_remove]
core_set_folders = [f for f in core_set_folders if f not in to_remove]
refined_set_folders = [f for f in refined_set_folders if f not in to_remove]

# refined & core
intersecting_folders = set(core_set_folders).intersection(set(refined_set_folders))
print("Number of intersecting folders (refined vs core):", len(list(intersecting_folders)))
# all & refined & core
intersecting_folders_full_n_refined = set(full_set_folders).intersection(set(refined_set_folders))
intersecting_folders_full_n_core = set(full_set_folders).intersection(set(core_set_folders))
to_be_removed = set(list(intersecting_folders_full_n_core) + list(intersecting_folders_full_n_refined))
print("Number of intersecting folders (full vs refined & core):", len(list(to_be_removed)))


full_set_folders_with_intersection_removed = sorted(list(set(full_set_folders).difference(set(to_be_removed))))
refined_set_folders_with_intersection_removed = sorted(list(set(refined_set_folders).difference(set(intersecting_folders))))



if not os.path.exists(combined_set):
    os.makedirs(combined_set)


# copy full set first
for folder in tqdm(full_set_folders_with_intersection_removed):
    all_files = os.listdir(os.path.join(full_set, folder))
    
    ligand_file = [f for f in all_files if ligand in f and ligand_ext in f]  # check filename and extension
    assert len(ligand_file) == 1, print(folder)
    ligand_file_name = ligand_file[0]
    
    pocket_file = [f for f in all_files if pocket in f and pocket_ext in f]
    assert len(pocket_file) == 1, print(folder)
    pocket_file_name = pocket_file[0]
    
    protein_file = [f for f in all_files if protein in f and protein_ext in f]
    assert len(protein_file) == 1, print(folder)
    protein_file_name = protein_file[0]
    
    target_folder = os.path.join(combined_set, folder)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    shutil.copy(os.path.join(full_set, folder, ligand_file_name), os.path.join(target_folder, ligand_file_name))
    shutil.copy(os.path.join(full_set, folder, pocket_file_name), os.path.join(target_folder, pocket_file_name))
    shutil.copy(os.path.join(full_set, folder, protein_file_name), os.path.join(target_folder, protein_file_name))



# then copy refined set
for folder in tqdm(refined_set_folders_with_intersection_removed):
    all_files = os.listdir(os.path.join(refined_set, folder))
    
    ligand_file = [f for f in all_files if ligand in f and ligand_ext in f]  # check filename and extension
    assert len(ligand_file) == 1, print(folder)
    ligand_file_name = ligand_file[0]
    
    pocket_file = [f for f in all_files if pocket in f and pocket_ext in f]
    assert len(pocket_file) == 1, print(folder)
    pocket_file_name = pocket_file[0]
    
    protein_file = [f for f in all_files if protein in f and protein_ext in f]
    assert len(protein_file) == 1, print(folder)
    protein_file_name = protein_file[0]
    
    target_folder = os.path.join(combined_set, folder)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    shutil.copy(os.path.join(refined_set, folder, ligand_file_name), os.path.join(target_folder, ligand_file_name))
    shutil.copy(os.path.join(refined_set, folder, pocket_file_name), os.path.join(target_folder, pocket_file_name))
    shutil.copy(os.path.join(refined_set, folder, protein_file_name), os.path.join(target_folder, protein_file_name))



# now copy core set
for folder in tqdm(core_set_folders):
    all_files = os.listdir(os.path.join(core_set, folder))
    
    # a few ligands: 2
    ligand_files = [f for f in all_files if ligand in f and ligand_ext in f]  # check filename and extension
    assert len(ligand_files) == 2, print(folder)
    ligand_files_append_names = [f.split(ligand)[1].split(ligand_ext)[0] for f in ligand_files]   # ['', '_opt']
    
    # 1 pocket
    pocket_file = [f for f in all_files if pocket in f and pocket_ext in f]
    assert len(pocket_file) == 1, print(folder)
    pocket_file_name = pocket_file[0]
    
    # 1 protein
    protein_file = [f for f in all_files if protein in f and protein_ext in f]
    assert len(protein_file) == 1, print(folder)
    protein_file_name = protein_file[0]
    
    
    for i in range(len(ligand_files)):
        target_folder = os.path.join(combined_set, folder+ligand_files_append_names[i])

        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        shutil.copy(os.path.join(core_set, folder, ligand_files[i]), os.path.join(target_folder, ligand_files[i]))
        shutil.copy(os.path.join(core_set, folder, pocket_file_name), os.path.join(target_folder, pocket_file_name))
        shutil.copy(os.path.join(core_set, folder, protein_file_name), os.path.join(target_folder, protein_file_name))

