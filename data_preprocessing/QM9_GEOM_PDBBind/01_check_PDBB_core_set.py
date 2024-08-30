import os
from tqdm import tqdm

# pth = '/Users/gohyixian/Documents/Documents/3.2_FYP_1/data/PDBBind2020/CASF-2016/coreset'
pth = '/mnt/c/Users/PC/Desktop/yixian/data/PDBBind2020/CASF-2016/coreset'

folders = sorted(os.listdir(pth))
folders.remove('.DS_Store')

filename = 'pocket'
extension = '.pdb'

multiple_pockets = []
for folder in tqdm(folders):
    pocket = [f for f in os.listdir(os.path.join(pth, folder)) if filename in f and extension in f]
    if len(pocket) > 1:
        multiple_pockets.append(folder)
    assert len(pocket) > 0, print(folder)

print(f"Folders in coreset with multiple pocket positions: {multiple_pockets}")   # None



filename = 'ligand'
extension = '.mol2'

multiple_ligands = dict()
for folder in tqdm(folders):
    ligands = [f for f in os.listdir(os.path.join(pth, folder)) if filename in f and extension in f]
    key = len(ligands)
    multiple_ligands[key] = multiple_ligands.get(key, []) + [folder]

print('Number of possible ligand positions : number of folders with that number of possible ligand positions')
for k,v in multiple_ligands.items():
    print(k,':',len(v))     # 2 : 285   <-- all folders only have exactly 2 ligand positions


# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 285/285 [00:00<00:00, 13644.29it/s]
# Folders in coreset with multiple pocket positions: []
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 285/285 [00:00<00:00, 23783.38it/s]
# Number of possible ligand positions : number of folders with that number of possible ligand positions
# 2 : 285
