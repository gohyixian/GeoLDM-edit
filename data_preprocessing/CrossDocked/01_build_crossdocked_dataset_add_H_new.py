import os
import copy
import math
import shutil
import random
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

from constants import get_periodictable_list


# euclidean distance between 2 points in 3D space
def get_euclidean_distance(coord1: list, coord2: list):
    assert len(coord1) == 3, f"Error on coordinates: len(coord1)={len(coord1)}"
    assert len(coord2) == 3, f"Error on coordinates: len(coord2)={len(coord2)}"
    return math.sqrt((coord1[0] - coord2[0]) ** 2 +
                     (coord1[1] - coord2[1]) ** 2 +
                     (coord1[2] - coord2[2]) ** 2)


def process_ligand_and_pocket(pdbfile, sdffile, dist_cutoff, ca_only, no_H, add_H, mol_id, determine_distance_by_ca=False):
    pdb_struct = PDBParser(QUIET=True).get_structure('', pdbfile)

    try:
        ligand = Chem.SDMolSupplier(str(sdffile))[0]
    except:
        raise Exception(f'cannot read sdf mol ({sdffile})')

    an2s, s2an = get_periodictable_list(include_aa=True)
    
    
    # LIGAND
    ligand_atom_charge_positions = []
    
    # extracts all atoms except for hydrogens
    if no_H:
        for idx,atom in enumerate(ligand.GetAtoms()):
            if atom.GetSymbol().capitalize() != 'H':
                x, y, z = list(ligand.GetConformer(0).GetAtomPosition(idx))
                ligand_atom_charge_positions.append([float(mol_id), float(s2an[atom.GetSymbol().capitalize()]), float(x), float(y), float(z)])
    
    # adds hydrogens to ligands using RDKit
    elif add_H:
        ligand_with_h = copy.deepcopy(ligand)
        
        # Embed in 3D space
        AllChem.EmbedMolecule(ligand_with_h, randomSeed=42)
        
        # add Hydrogens
        ligand_with_h = Chem.AddHs(ligand_with_h, addCoords=True)
        
        # Copy original heavy atom coordinates to the new molecule with hydrogens
        id_atoms_in_ligand = []
        conf = ligand.GetConformer(0)
        new_conf = ligand_with_h.GetConformer(0)
        for i, atom in enumerate(ligand.GetAtoms()):
            # if atom.GetAtomicNum() > 1:  # Skip hydrogens
            pos = conf.GetAtomPosition(i)
            new_conf.SetAtomPosition(i, pos)
            id_atoms_in_ligand.append(i)
        
        # Optimize only hydrogen positions while keeping original heavy atom coordinates fixed
        ff = AllChem.MMFFGetMoleculeForceField(ligand_with_h, AllChem.MMFFGetMoleculeProperties(ligand_with_h))
        if ff is None:
            raise Exception(f"Cannot optimise molecule ({sdffile})")
        for i, atom in enumerate(ligand_with_h.GetAtoms()):
            # if atom.GetAtomicNum() > 1:  # Freeze heavy atoms
            if i in id_atoms_in_ligand:
                ff.AddFixedPoint(i)
        ff.Minimize()

        # Sanity Check: calculate euclidean distance to see of atoms originally 
        # present in ligand have been shifted in ligand_with_h
        conf_ligand = ligand.GetConformer(0)
        conf_ligand_with_h = ligand_with_h.GetConformer(0)

        # TO REMOVE
        # sanity check: verify atom mapping same
        for i, atom in enumerate(ligand.GetAtoms()):
            ligand_with_h_atom = ligand_with_h.GetAtomWithIdx(i)
            assert atom.GetAtomicNum() == ligand_with_h_atom.GetAtomicNum(), f"Atom mismatch at index {i}"

        # Sum of 3D distances for all heavy atoms
        total_distance = 0.0
        for atom in ligand.GetAtoms():
            if atom.GetAtomicNum() > 1:  # Exclude hydrogens
                atom_idx = atom.GetIdx()  # Index in the original molecule
                coord_ligand = list(conf_ligand.GetAtomPosition(atom_idx))
                coord_ligand_with_h = list(conf_ligand_with_h.GetAtomPosition(atom_idx))
                # Compute distance
                distance = get_euclidean_distance(coord_ligand, coord_ligand_with_h)
                total_distance += distance
        
        # if atoms originally present in ligand are not shifted, proceed
        if total_distance == 0:
            for idx,atom in enumerate(ligand_with_h.GetAtoms()):
                x, y, z = list(ligand_with_h.GetConformer(0).GetAtomPosition(idx))
                ligand_atom_charge_positions.append([float(mol_id), float(s2an[atom.GetSymbol().capitalize()]), float(x), float(y), float(z)])
        # if shifted, end process here
        else:
            print(f">> Total sum of distances for all heavy atoms: {total_distance:.4f}")
            return None, None, total_distance, ligand, ligand_with_h
    
    # regular processing of atoms in ligands
    else:
        for idx,atom in enumerate(ligand.GetAtoms()):
            x, y, z = list(ligand.GetConformer(0).GetAtomPosition(idx))
            ligand_atom_charge_positions.append([float(mol_id), float(s2an[atom.GetSymbol().capitalize()]), float(x), float(y), float(z)])
    ligand_atom_charge_positions = np.array(ligand_atom_charge_positions)


    # POCKET
    # Find interacting pocket residues based on distance cutoff
    pocket_residues = []
    lig_coords = np.array([list(ligand.GetConformer(0).GetAtomPosition(idx))
                           for idx in range(ligand.GetNumAtoms())])
    for residue in pdb_struct[0].get_residues():
        if determine_distance_by_ca:
            res_ca_coord = np.array([residue['CA'].get_coord()])
            if is_aa(residue.get_resname(), standard=True) and \
                    (((res_ca_coord - lig_coords) ** 2).sum(-1) ** 0.5).min() < dist_cutoff:
                pocket_residues.append(residue)
        else:
            res_coords = np.array([a.get_coord() for a in residue.get_atoms()])
            if is_aa(residue.get_resname(), standard=True) and \
                    (((res_coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(-1) ** 0.5).min() < dist_cutoff:
                pocket_residues.append(residue)

    pocket_atom_charge_positions = []
    if ca_only:
        for res in pocket_residues:
            x, y, z = res['CA'].get_coord()
            pocket_atom_charge_positions.append([float(mol_id), float(s2an[str(res.get_resname()).upper()]), float(x), float(y), float(z)])
    else:
        for res in pocket_residues:
            all_res_atoms = res.get_atoms()
            if no_H:
                all_res_atoms = [a for a in all_res_atoms if a.element != 'H']
            for atom in all_res_atoms:
                x, y, z = atom.get_coord()
                pocket_atom_charge_positions.append([float(mol_id), float(s2an[atom.element]), float(x), float(y), float(z)])
    pocket_atom_charge_positions = np.array(pocket_atom_charge_positions)

    if add_H:
        return ligand_atom_charge_positions, pocket_atom_charge_positions, total_distance, ligand, ligand_with_h
    else:
        return ligand_atom_charge_positions, pocket_atom_charge_positions



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_crossd_basedir', type=Path)
    parser.add_argument("--save_dir", type=str, default="data/CrossDocked_LG_PKT")
    parser.add_argument("--save_dataset_name", type=str, default="CrossDocked_LG_PKT")
    parser.add_argument("--copy_files_dir", type=str, default="data/CrossDocked_LG_PKT/test_val_paired_files")
    
    parser.add_argument('--dist_cutoff', type=float, default=10.0)  # 8.0
    parser.add_argument('--ca_only', action='store_true')
    parser.add_argument('--add_H', action='store_true')
    parser.add_argument('--no_H', action='store_true')
    parser.add_argument('--determine_distance_by_ca', action='store_true')  # wrong method, do not use
    args = parser.parse_args()

    # python 01_build_crossdocked_dataset_add_H.py --raw_crossd_basedir /mnt/c/Users/PC/Desktop/yixian/data/CrossDocked --dist_cutoff 10.0 --add_H --save_dir /mnt/c/Users/PC/Desktop/yixian/GeoLDM-edit/data/d_20241126_CrossDocked_LG_PKT_MMseq2_split_add_H --save_dataset_name d_20241126_CrossDocked_LG_PKT_MMseq2_split_add_H --copy_files_dir /mnt/c/Users/PC/Desktop/yixian/GeoLDM-edit/data/d_20241126_CrossDocked_LG_PKT_MMseq2_split_add_H/test_val_paired_files

    # python 01_build_crossdocked_dataset_add_H_new.py --raw_crossd_basedir /mnt/c/Users/PC/Desktop/yixian/data/CrossDocked --dist_cutoff 10.0 --add_H --save_dir /mnt/c/Users/PC/Desktop/yixian/GeoLDM-edit/data/d_20241126_CrossDocked_LG_PKT_MMseq2_split_add_H_new --save_dataset_name d_20241126_CrossDocked_LG_PKT_MMseq2_split_add_H_new --copy_files_dir /mnt/c/Users/PC/Desktop/yixian/GeoLDM-edit/data/d_20241126_CrossDocked_LG_PKT_MMseq2_split_add_H_new/test_val_paired_files

    datadir = args.raw_crossd_basedir / 'crossdocked_pocket10/'


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
    dataset_size = dict()
    mol_id = 0


    num_failed = 0
    failed_save = []
    
    if args.add_H:
        args.add_H_save_dir = Path(args.save_dir, "add_H_sanity_check")
        total_distance_result = {
            'split': [], 'id': [], 'dist': []
        }
    
    for split in data_split.keys():
        ligand_dataset[split] = []
        pocket_dataset[split] = []
        
        if args.add_H:
            add_H_save_path = Path(args.add_H_save_dir, split)
            if not os.path.exists(str(add_H_save_path)):
                os.makedirs(str(add_H_save_path))
        
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
                if args.add_H:
                    ligand_data, pocket_data, total_distance, ligand, ligand_with_h = \
                        process_ligand_and_pocket(
                            pdbfile, sdffile, dist_cutoff=args.dist_cutoff, 
                            ca_only=args.ca_only, no_H=args.no_H, add_H=True, 
                            mol_id=mol_id,
                            determine_distance_by_ca=args.determine_distance_by_ca
                        )
                else:
                    ligand_data, pocket_data = process_ligand_and_pocket(
                        pdbfile, sdffile, dist_cutoff=args.dist_cutoff, 
                        ca_only=args.ca_only, no_H=args.no_H, add_H=False, 
                        mol_id=mol_id,
                        determine_distance_by_ca=args.determine_distance_by_ca
                    )
                
                # skip ligand-pocket pair if atoms originally present in ligand 
                # are shifted after adding hydrogens (total distance > 0)
                if (ligand_data is None) or (pocket_data is None):
                    continue
                
                if len(list(ligand_data.shape)) == 2 and len(list(pocket_data.shape)) == 2:
                    if ligand_data.shape[0] > 0 and pocket_data.shape[0] > 0 and \
                        ligand_data.shape[1] == 5 and pocket_data.shape[1] == 5:
                        ligand_dataset[split].append(ligand_data)
                        pocket_dataset[split].append(pocket_data)
                        
                        # save for visualisation
                        if args.add_H:
                            ligand_file_path = Path(sdffile)
                            id = ligand_file_path.stem.replace('_', '-')
                            filename = f"{str(mol_id).zfill(7)}_{id}"
                            
                            total_distance_result['split'].append(split)
                            total_distance_result['id'].append(filename)
                            total_distance_result['dist'].append(total_distance)
                            
                            # Save the original ligand to an SDF file
                            original_writer = Chem.SDWriter(str(Path(add_H_save_path, f"{filename}.sdf")))
                            original_writer.write(ligand)
                            original_writer.close()
                            # Save the updated molecule with hydrogens to a new SDF file
                            hydrogens_writer = Chem.SDWriter(str(Path(add_H_save_path, f"{filename}_add_H.sdf")))
                            hydrogens_writer.write(ligand_with_h)
                            hydrogens_writer.close()
                        
                        if split in ['val']:
                            # Copy PDB file
                            pdb_file_dir = os.path.join(args.copy_files_dir, f"{split}_pocket")
                            if not os.path.exists(pdb_file_dir):
                                os.makedirs(pdb_file_dir, exist_ok=True)
                            new_rec_name = Path(pdbfile).stem.replace('_', '-')
                            pdb_file_out = Path(pdb_file_dir, f"{str(mol_id).zfill(7)}_{new_rec_name}.pdb")
                            shutil.copy(pdbfile, pdb_file_out)

                            # Copy SDF file
                            sdf_file_dir = os.path.join(args.copy_files_dir, f"{split}_ligand")
                            if not os.path.exists(sdf_file_dir):
                                os.makedirs(sdf_file_dir, exist_ok=True)
                            new_lig_name = new_rec_name + '_' + Path(sdffile).stem.replace('_', '-')
                            sdf_file_out = Path(sdf_file_dir, f'{str(mol_id).zfill(7)}_{new_lig_name}.sdf')
                            shutil.copy(sdffile, sdf_file_out)
                        
                        mol_id += 1
                    else:
                        print(f">> Skipped due to ligand {ligand_data.shape}, or pocket {pocket_data.shape}")
                        num_failed += 1
                else:
                    print(f">> Skipped due to ligand {ligand_data.shape}, or pocket {pocket_data.shape}")
                    num_failed += 1
                
            # except (KeyError, AssertionError, FileNotFoundError, IndexError,
            #         ValueError) as e:
            except Exception as e:
                print(type(e).__name__, e, pocket_fn, ligand_fn)
                num_failed += 1
                pbar.set_description(f'#failed: {num_failed}')
                continue
            
            # print("===============")
            # print(ligand_fn)
            # print(pocket_fn)

        assert len(ligand_dataset[split]) == len(pocket_dataset[split]), f"len(ligand): {len(ligand_dataset[split])}, len(pocket): {len(pocket_dataset[split])}"
        dataset_size[split] = len(ligand_dataset[split])

        ligand_dataset[split] = np.vstack(ligand_dataset[split])
        pocket_dataset[split] = np.vstack(pocket_dataset[split])
        
        # final checks
        if args.no_H:
            ligand_atom_name = ligand_dataset[split][:, 1]
            pocket_atom_name = pocket_dataset[split][:, 1]
            
            assert np.float64(1.0) not in ligand_atom_name
            assert np.float64(1.0) not in pocket_atom_name
        
        if args.ca_only:
            pocket_atom_name = pocket_dataset[:, 1]
            assert np.all(pocket_atom_name < 0.) == True
    
    
    # saving dataset
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_file = f"{args.save_dataset_name}__{args.dist_cutoff}A{'__CA_Only' if args.ca_only else ''}{'__no_H' if args.no_H else ''}.npz"
    
    np.savez(
        os.path.join(args.save_dir, save_file), 
        
        ligand_train=ligand_dataset['train'], 
        ligand_test=ligand_dataset['test'], 
        ligand_val=ligand_dataset['val'], 
        
        pocket_train=pocket_dataset['train'], 
        pocket_test=pocket_dataset['test'], 
        pocket_val=pocket_dataset['val'], 
    )
    
    total_distance_result['split'].append("")
    total_distance_result['id'].append("SUM(dist)")
    total_distance_result['dist'].append(sum(total_distance_result['dist']))
    
    df = pd.DataFrame.from_dict(total_distance_result)
    df.to_csv(Path(str(args.add_H_save_dir), 'total_distances.csv'))
    
    
    ligand_total_num_atoms = ligand_dataset['train'].shape[0] + ligand_dataset['test'].shape[0] + ligand_dataset['val'].shape[0]
    pocket_total_num_atoms = pocket_dataset['train'].shape[0] + pocket_dataset['test'].shape[0] + pocket_dataset['val'].shape[0]
    
    print()
    print(f"Num Pairs Train: {len(data_split['train'])}")
    print(f"Num Pairs Test : {len(data_split['test'])}")
    print(f"Num Pairs Val  : {len(data_split['val'])}")
    print()
    print(f"Actual Num Pairs Train: {dataset_size['train']}")
    print(f"Actual Num Pairs Test : {dataset_size['test']}")
    print(f"Actual Num Pairs Val  : {dataset_size['val']}")
    print()
    print(f"[LG] Total Atom Num  : {ligand_total_num_atoms}", )
    print(f"[LG] Ave. Atom Num   : {ligand_total_num_atoms / (mol_id+1)}")
    print(f"[PKT] Total Atom Num : {pocket_total_num_atoms}", )
    print(f"[PKT] Ave. Atom Num  : {pocket_total_num_atoms / (mol_id+1)}")
    print("Total number of Ligand-Pocket pairs:", mol_id+1)
    print("Dataset processed.")


# Num Pairs Train: 100000
# Num Pairs Test : 100
# Num Pairs Val  : 100

# Actual Num Pairs Train: 99560
# Actual Num Pairs Test : 100
# Actual Num Pairs Val  : 100

# [LG] Total Atom Num  : 4198399
# [LG] Ave. Atom Num   : 42.08457212738445
# [PKT] Total Atom Num : 40870756
# [PKT] Ave. Atom Num  : 409.68671123986326
# Total number of Ligand-Pocket pairs: 99761
# Dataset processed.