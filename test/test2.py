import os
from tqdm import tqdm
import sys


def get_unique_amino_acids(pdbb_folder, to_remove, target_file, target_ext):
    all_folders = sorted([f for f in os.listdir(pdbb_folder) if f not in to_remove])

    unique_animo_acids = []

    for folder in tqdm(all_folders):
        all_files = sorted(os.listdir(os.path.join(pdbb_folder, folder)))
        for f in all_files:
            if target_file in f and target_ext in f:
                print(f)
                # read file
                file_path = os.path.join(pdbb_folder, folder, f)
                with open(file_path, 'r') as openfile:
                    lines = [l for l in openfile]
                
                for line in lines:
                    if line.split(' ')[0] == 'ATOM':
                        splits = line.split(' ')
                        splits = [s for s in splits if s.strip()]
                        unique_animo_acids.append(splits[3])
                
                unique_animo_acids = sorted(list(set(list(unique_animo_acids))))
    return unique_animo_acids


pdbb_folder = '/Users/gohyixian/Documents/Documents/3.2_FYP_1/data/PDBBind2020/v2020-other-PL'
refined_folder = '/Users/gohyixian/Documents/Documents/3.2_FYP_1/data/PDBBind2020/refined-set'
core_folder = '/Users/gohyixian/Documents/Documents/3.2_FYP_1/data/PDBBind2020/CASF-2016/coreset'

to_remove = ['.DS_Store', 'readme', 'index']
target_file = 'pocket'
target_ext = '.pdb'

pdbb_list = get_unique_amino_acids(pdbb_folder, to_remove, target_file, target_ext)
refined_list = get_unique_amino_acids(refined_folder, to_remove, target_file, target_ext)
core_list = get_unique_amino_acids(core_folder, to_remove, target_file, target_ext)

combined_list = list(set(list(pdbb_list) + list(refined_list) + list(core_list)))

with open('test2.txt', 'w') as writefile:
    print(f"PDBB:     {pdbb_list}", file=writefile)
    print(f"refined:  {refined_list}", file=writefile)
    print(f"core:     {core_list}", file=writefile)
    print(f"combined: {core_list}", file=writefile)
    
    
# twenty types of amino acids:
# https://www.cryst.bbk.ac.uk/education/AminoAcid/the_twenty.html