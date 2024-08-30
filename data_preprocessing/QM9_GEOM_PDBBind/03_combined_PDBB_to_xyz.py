"""
This script converts .sdf, .mol2, .pdb and any other supported file formats listed below,
from those formats to .xyz for easy loading. open-babel is required for the conversion,
installation as below. 

Supported Formats: https://open-babel.readthedocs.io/en/latest/FileFormats/Overview.html#file-formats
"""


# sudo apt-get install openbabel
# brew install open-babel


# # Convert .sdf files to .xyz
# babel input_file.sdf output_file.xyz

# # Convert .mol2 files to .xyz
# babel input_file.mol2 output_file.xyz

# # Convert .pdb files to .xyz
# babel input_file.pdb output_file.xyz


# # Convert all .sdf files in a directory to .xyz
# babel *.sdf -o xyz

# # Convert all .mol2 files in a directory to .xyz
# babel *.mol2 -o xyz

# # Convert all .pdb files in a directory to .xyz
# babel *.pdb -o xyz


import os
import subprocess
from tqdm import tqdm

# input_dir = '/Users/gohyixian/Documents/Documents/3.2_FYP_1/data/PDBBind2020/00-combined-full-refined-and-core-set'
input_dir = '/mnt/c/Users/PC/Desktop/yixian/data/PDBBind2020/00-combined-full-refined-and-core-set'

# output_dir = '/Users/gohyixian/Documents/Documents/3.2_FYP_1/data/PDBBind2020/01-combined-full-refined-and-core-set-xyz'
output_dir = '/mnt/c/Users/PC/Desktop/yixian/data/PDBBind2020/01-combined-full-refined-and-core-set-xyz'

allowed_files = ['mol2', 'pdb']
out_ext = 'xyz'
to_remove = ['.DS_store'] # mac
# usage: obabel -i pdb /Users/gohyixian/Downloads/1a1e/1a1e_pocket.pdb -o xyz -O /Users/gohyixian/Downloads/1a1e/1a1e_pocket.xyz

all_protein_ligand = sorted(os.listdir(input_dir))
all_protein_ligand = [f for f in all_protein_ligand if f not in to_remove]

for folder in tqdm(all_protein_ligand):
    print(f">>> {folder}")
    files = sorted(os.listdir(os.path.join(input_dir, folder)))
    for f in files:
        ext = f.split('.')[-1]
        if ext in allowed_files:
            
            out_dir = os.path.join(output_dir, folder)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                
            in_file = os.path.join(input_dir, folder, f)
            out_file = os.path.join(output_dir, folder, f.split('.')[0] + '.' + out_ext)
            command = f"obabel -i {ext} {in_file} -o {out_ext} -O {out_file}"
            subprocess.run(command, shell=True)
            print(f"{in_file} -> {out_file}")
print("DONE")