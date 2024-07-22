import os
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

from constants import get_periodictable_list


def process_ligand_and_pocket(pdbfile, sdffile, dist_cutoff, ca_only, no_H, mol_id, determine_distance_by_ca=False):
    pdb_struct = PDBParser(QUIET=True).get_structure('', pdbfile)

    try:
        ligand = Chem.SDMolSupplier(str(sdffile))[0]
    except:
        raise Exception(f'cannot read sdf mol ({sdffile})')

    an2s, s2an = get_periodictable_list(include_aa=True)
    
    # LIGAND
    ligand_atom_charge_positions = []
    if no_H:
        for idx,atom in enumerate(ligand.GetAtoms()):
            if atom.GetSymbol().capitalize() != 'H':
                x, y, z = list(ligand.GetConformer(0).GetAtomPosition(idx))
                ligand_atom_charge_positions.append([float(mol_id), float(s2an[atom.GetSymbol().capitalize()]), float(x), float(y), float(z)])
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

    return ligand_atom_charge_positions, pocket_atom_charge_positions



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_crossd_basedir', type=Path)
    parser.add_argument("--save_dir", type=str, default="data/CrossDocked_LG_PKT")
    parser.add_argument("--save_dataset_name", type=str, default="CrossDocked_LG_PKT")
    
    parser.add_argument('--dist_cutoff', type=float, default=10.0)  # 8.0
    parser.add_argument('--ca_only', action='store_true')
    parser.add_argument('--no_H', action='store_true')
    parser.add_argument('--determine_distance_by_ca', action='store_true')  # wrong method, do not use
    args = parser.parse_args()

    # python 01_build_crossdocked_dataset.py --raw_crossd_basedir /Users/gohyixian/Documents/Documents/3.2_FYP_1/data/CrossDocked --dist_cutoff 10.0 --no_H --ca_only --save_dir /Users/gohyixian/Documents/GitHub/FYP/GeoLDM-edit/data/d_20240623_CrossDocked_LG_PKT --save_dataset_name d_20240623_CrossDocked_LG_PKT
    
    # python 01_build_crossdocked_dataset.py --raw_crossd_basedir /mnt/c/Users/PC/Desktop/yixian/data/CrossDocked --dist_cutoff 10.0 --no_H --ca_only --save_dir /mnt/c/Users/PC/Desktop/yixian/GeoLDM-edit/data/d_20240623_CrossDocked_LG_PKT --save_dataset_name d_20240623_CrossDocked_LG_PKT
    


    datadir = args.raw_crossd_basedir / 'crossdocked_pocket10/'


    # Read data split
    split_path = Path(args.raw_crossd_basedir, 'split_by_name.pt')
    data_split = torch.load(split_path)
    
    # combined train and test set = all
    all_data = data_split['train'] + data_split['test']  # list


    ligand_dataset = []
    pocket_dataset = []
    mol_id = 0


    num_failed = 0
    failed_save = []
    pbar = tqdm(all_data)
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
            ligand_data, pocket_data = process_ligand_and_pocket(
                pdbfile, sdffile, dist_cutoff=args.dist_cutoff,
                ca_only=args.ca_only, no_H=args.no_H, mol_id=mol_id,
                determine_distance_by_ca=args.determine_distance_by_ca)
            
            # print("\nLIGAND DATA")
            # print(ligand_data.shape, '\n', ligand_data)
            # print("\nPOCKET DATA")
            # print(pocket_data.shape, '\n', pocket_data)
            
            if len(list(ligand_data.shape)) == 2 and len(list(pocket_data.shape)) == 2:
                if ligand_data.shape[0] > 0 and pocket_data.shape[0] > 0 and \
                    ligand_data.shape[1] == 5 and pocket_data.shape[1] == 5:
                    ligand_dataset.append(ligand_data)
                    pocket_dataset.append(pocket_data)
                    mol_id += 1
                else:
                    print(f">> Skipped due to ligand {ligand_data.shape}, or pocket {pocket_data.shape}")
                    num_failed += 1
            else:
                print(f">> Skipped due to ligand {ligand_data.shape}, or pocket {pocket_data.shape}")
                num_failed += 1
            
        except (KeyError, AssertionError, FileNotFoundError, IndexError,
                ValueError) as e:
            print(type(e).__name__, e, pocket_fn, ligand_fn)
            num_failed += 1
            pbar.set_description(f'#failed: {num_failed}')
            continue
        
        # print("===============")
        # print(ligand_fn)
        # print(pocket_fn)


    ligand_dataset = np.vstack(ligand_dataset)
    pocket_dataset = np.vstack(pocket_dataset)
    
    # final checks
    if args.no_H:
        ligand_atom_name = ligand_dataset[:, 1]
        pocket_atom_name = pocket_dataset[:, 1]
        
        assert np.float64(1.0) not in ligand_atom_name
        assert np.float64(1.0) not in pocket_atom_name
    
    if args.ca_only:
        pocket_atom_name = pocket_dataset[:, 1]
        assert np.all(pocket_atom_name < 0.) == True
    
    
    # saving dataset
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_file = f"{args.save_dataset_name}__{args.dist_cutoff}A{'__CA_Only' if args.ca_only else ''}{'__no_H' if args.no_H else ''}.npz"
    
    np.savez(os.path.join(args.save_dir, save_file), ligand=ligand_dataset, pocket=pocket_dataset)
    
    print()
    print(f"[LG] Total Atom Num  : {ligand_dataset.shape[0]}", )
    print(f"[LG] Ave. Atom Num   : {ligand_dataset.shape[0] / (mol_id+1)}")
    print(f"[PKT] Total Atom Num : {pocket_dataset.shape[0]}", )
    print(f"[PKT] Ave. Atom Num  : {pocket_dataset.shape[0] / (mol_id+1)}")
    print("Total number of Ligand-Pocket pairs:", mol_id+1)
    print("Dataset processed.")


# failed: 67: 100%|| 100100/100100 [40:13<00:00, 41.47it/s]
#
# [LG] Total Atom Num  : 2361427
# [LG] Ave. Atom Num   : 23.606951844928073
# [PKT] Total Atom Num : 5056527
# [PKT] Ave. Atom Num  : 50.549599624116524
# Total number of Ligand-Pocket pairs: 100031
# Dataset processed.