import os
import shutil
from tqdm import tqdm

in_dir = "/mnt/c/Users/PC/Desktop/yixian/geoldm-edit/data/d_20241115_CrossDocked_LG_PKT_MMseq2_split/test_val_paired_files/val_pocket"
out_dir = "/mnt/c/Users/PC/Desktop/yixian/geoldm-edit/data/d_20241115_CrossDocked_LG_PKT_MMseq2_split/test_val_paired_files/val_pocket_ori"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for file in tqdm(os.listdir(in_dir)):
    in_file = os.path.join(in_dir, file)
    out_file = os.path.join(out_dir, file[8:])
    
    shutil.copy(in_file, out_file)

print("Pocket DONE.")

in_dir = "/mnt/c/Users/PC/Desktop/yixian/geoldm-edit/data/d_20241115_CrossDocked_LG_PKT_MMseq2_split/test_val_paired_files/val_ligand"
out_dir = "/mnt/c/Users/PC/Desktop/yixian/geoldm-edit/data/d_20241115_CrossDocked_LG_PKT_MMseq2_split/test_val_paired_files/val_ligand_ori"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for file in tqdm(os.listdir(in_dir)):
    in_file = os.path.join(in_dir, file)
    out_file = os.path.join(out_dir, file[8:])
    
    shutil.copy(in_file, out_file)

print("Ligand DONE.")