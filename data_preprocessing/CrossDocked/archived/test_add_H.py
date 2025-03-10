import os
import copy
import shutil
import random
import argparse
from pathlib import Path

import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

from constants import get_periodictable_list


# Function to calculate 3D Euclidean distance
def calculate_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 +
                     (coord1[1] - coord2[1]) ** 2 +
                     (coord1[2] - coord2[2]) ** 2)


def process_ligand_and_pocket(pdbfile, sdffile, add_H=True, save_dir='test_add_H'):
    pdb_struct = PDBParser(QUIET=True).get_structure('', pdbfile)

    try:
        ligand = Chem.SDMolSupplier(str(sdffile))[0]
    except:
        raise Exception(f'cannot read sdf mol ({sdffile})')

    an2s, s2an = get_periodictable_list(include_aa=True)

    # original paths
    sdffile_path = Path(sdffile)
    sdffile_save_path       = Path(save_dir, f"{sdffile_path.stem}.sdf")
    sdffile_add_H_save_path = Path(save_dir, f"{sdffile_path.stem}_add_H.sdf")
    

    # Embed the original molecule (before adding hydrogens)
    ligand_with_h = copy.deepcopy(ligand)
    AllChem.EmbedMolecule(ligand_with_h, randomSeed=42)  # Embed in 3D space

    # Add hydrogens to the molecule while preserving the original heavy atom coordinates
    ligand_with_h = Chem.AddHs(ligand_with_h, addCoords=True)

    # Copy original heavy atom coordinates to the new molecule with hydrogens
    conf = ligand.GetConformer(0)
    new_conf = ligand_with_h.GetConformer(0)
    for i, atom in enumerate(ligand.GetAtoms()):
        if atom.GetAtomicNum() > 1:  # Skip hydrogens
            pos = conf.GetAtomPosition(i)
            new_conf.SetAtomPosition(i, pos)

    # Optimize only hydrogen positions
    ff = AllChem.MMFFGetMoleculeForceField(ligand_with_h, AllChem.MMFFGetMoleculeProperties(ligand_with_h))
    if ff is None:
        raise Exception(f"Cannot optimise molecule ({sdffile})")
    for i, atom in enumerate(ligand_with_h.GetAtoms()):
        if atom.GetAtomicNum() > 1:  # Freeze heavy atoms
            ff.AddFixedPoint(i)
    ff.Minimize()


    # Save the original ligand to an SDF file
    original_writer = Chem.SDWriter(sdffile_save_path)
    original_writer.write(ligand)
    original_writer.close()
    print(f"Original ligand saved to {sdffile_save_path}.")

    # Save the updated molecule with hydrogens to a new SDF file
    hydrogens_writer = Chem.SDWriter(sdffile_add_H_save_path)
    hydrogens_writer.write(ligand_with_h)
    hydrogens_writer.close()

    print(f"Ligand with hydrogens saved to {sdffile_add_H_save_path}.")


    # calculate euclidean distance to see of atoms originally present in ligand have been shifted in ligand_with_h
    conf_ligand = ligand.GetConformer(0)
    conf_ligand_with_h = ligand_with_h.GetConformer(0)

    # Sum of 3D distances for all heavy atoms
    total_distance = 0.0

    for atom in ligand.GetAtoms():
        if atom.GetAtomicNum() > 1:  # Exclude hydrogens
            atom_idx = atom.GetIdx()  # Index in the original molecule
            coord_ligand = list(conf_ligand.GetAtomPosition(atom_idx))
            coord_ligand_with_h = list(conf_ligand_with_h.GetAtomPosition(atom_idx))
            print(f"{coord_ligand}    {coord_ligand_with_h}")
            # Compute distance
            distance = calculate_distance(coord_ligand, coord_ligand_with_h)
            total_distance += distance

    print(f"Total sum of distances for all heavy atoms: {total_distance:.4f}")

    return total_distance, sdffile_path.stem

    # # LIGAND
    # ligand_atom_charge_positions = []
    # if no_H:
    #     for idx,atom in enumerate(ligand.GetAtoms()):
    #         if atom.GetSymbol().capitalize() != 'H':
    #             x, y, z = list(ligand.GetConformer(0).GetAtomPosition(idx))
    #             ligand_atom_charge_positions.append([float(mol_id), float(s2an[atom.GetSymbol().capitalize()]), float(x), float(y), float(z)])
    # else:
    #     for idx,atom in enumerate(ligand.GetAtoms()):
    #         x, y, z = list(ligand.GetConformer(0).GetAtomPosition(idx))
    #         ligand_atom_charge_positions.append([float(mol_id), float(s2an[atom.GetSymbol().capitalize()]), float(x), float(y), float(z)])
    # ligand_atom_charge_positions = np.array(ligand_atom_charge_positions)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_crossd_basedir', type=Path)
    parser.add_argument("--save_dir", type=str, default="data/CrossDocked_LG_PKT")
    args = parser.parse_args()

    # python test_add_H.py --raw_crossd_basedir /mnt/c/Users/PC/Desktop/yixian/data/CrossDocked --save_dir /mnt/c/Users/PC/Desktop/yixian/CrossDocked_add_H_test


    datadir = args.raw_crossd_basedir / 'crossdocked_pocket10/'

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Read data split
    split_path = Path(args.raw_crossd_basedir, 'split_by_name.pt')
    data_split = torch.load(split_path)
    
    # TODO: update this. temporary split test to test and val by 50:50 ratio (this will affect benchmarking with other models running on this split)
    # data_split_test = copy.deepcopy(data_split['test'])
    # data_split_test = random.shuffle(data_split_test)  # randomise
    # num_test = len(data_split_test)
    # half_num = int(num_test // 2)
    # data_split['test'] = []  # reset to empty
    # data_split['val'] = []
    # for i in range(num_test):
    #     if i > half_num:
    #         data_split['val'].append(data_split_test[i])
    #     else:
    #         data_split['test'].append(data_split_test[i])
    data_split['val'] = copy.deepcopy(data_split['test'])
    
    print(f"Num Pairs Train: {len(data_split['train'])}")
    print(f"Num Pairs Test : {len(data_split['test'])}")
    print(f"Num Pairs Val  : {len(data_split['val'])}")
    
    
    ligand_dataset = dict()
    pocket_dataset = dict()
    mol_id = 0


    num_failed = 0
    failed_save = []
    total_distance_result = {
        'id': [], 'dist': []
    }
    for split in data_split.keys():
        ligand_dataset[split] = []
        pocket_dataset[split] = []
        
        pbar = tqdm(data_split[split])
        pbar.set_description(f'#failed: {num_failed}')
        for pocket_fn, ligand_fn in pbar:

            sdffile = datadir / f'{ligand_fn}'
            pdbfile = datadir / f'{pocket_fn}'
            

            try:
                struct_copy = PDBParser(QUIET=True).get_structure('', pdbfile)
            except:
                num_failed += 1
                failed_save.append((pocket_fn, ligand_fn))
                print(failed_save[-1])
                pbar.set_description(f'#failed: {num_failed}')
                continue

            try:
                total_distance, ligand_id = process_ligand_and_pocket(
                    pdbfile, sdffile, add_H=True, save_dir=args.save_dir
                )
                total_distance_result['id'].append(str(ligand_id))
                total_distance_result['dist'].append(float(total_distance))
                
            except (Exception, KeyError, AssertionError, FileNotFoundError, IndexError,
                    ValueError) as e:
                print(type(e).__name__, e, pocket_fn, ligand_fn)
                num_failed += 1
                pbar.set_description(f'#failed: {num_failed}')
                continue
            
            # print("===============")
            # print(ligand_fn)
            # print(pocket_fn)

    df = pd.DataFrame.from_dict(total_distance_result)
    df.to_csv(Path(args.save_dir, 'total_distances.csv'))
    
    print("Dataset processed.")


# failed: 67: 100%|| 100100/100100 [40:13<00:00, 41.47it/s]

# --dist_cutoff 10.0 --no_H --ca_only 

# [LG] Total Atom Num  : 2361427
# [LG] Ave. Atom Num   : 23.606951844928073
# [PKT] Total Atom Num : 5056527
# [PKT] Ave. Atom Num  : 50.549599624116524
# Total number of Ligand-Pocket pairs: 100031
# Dataset processed.



# --dist_cutoff 10.0

# [LG] Total Atom Num  : 2361445
# [LG] Ave. Atom Num   : 23.607131789145367
# [PKT] Total Atom Num : 40980923
# [PKT] Ave. Atom Num  : 409.68222850916214
# Total number of Ligand-Pocket pairs: 100031
# Dataset processed.