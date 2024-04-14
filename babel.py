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

input_dir = "/Users/gohyixian/Downloads/PDBBind2020/refined-set"
output_dir = "/Users/gohyixian/Downloads/PDBBind2020/refined-set-xyz"
allowed_files = ['sdf', 'pdb']   # drop mol2
out_ext = 'xyz'
# babel_path = '/opt/homebrew/Cellar/open-babel/3.1.1_2/bin/obabel'
# /opt/homebrew/Cellar/open-babel/3.1.1_2/bin/obabel -i pdb /Users/gohyixian/Downloads/1a1e/1a1e_pocket.pdb -o xyz -O /Users/gohyixian/Downloads/1a1e/1a1e_pocket.xyz

all_protein_ligand = sorted(os.listdir(input_dir))
all_protein_ligand.remove('.DS_Store') # mac

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
            # command = f"{babel_path} -i {ext} {in_file} -o {out_ext} -O {out_file}"
            command = f"obabel -i {ext} {in_file} -o {out_ext} -O {out_file}"
            subprocess.run(command, shell=True)
            print(f"{in_file} -> {out_file}")
print("DONE")