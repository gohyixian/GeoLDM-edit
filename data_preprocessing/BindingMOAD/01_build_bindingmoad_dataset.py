import os
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

from constants import get_periodictable_list



def read_label_file(csv_path):
    """
    Read BindingMOAD's label file
    Args:
        csv_path: path to 'every.csv'
    Returns:
        Nested dictionary with all ligands. First level: EC number,
            Second level: PDB ID, Third level: list of ligands. Each ligand is
            represented as a tuple (ligand name, validity, SMILES string)
    """
    ligand_dict = {}

    with open(csv_path, 'r') as f:
        for line in f.readlines():
            row = line.split(',')

            # new protein class
            if len(row[0]) > 0:
                curr_class = row[0]
                ligand_dict[curr_class] = {}
                continue

            # new protein
            if len(row[2]) > 0:
                curr_prot = row[2]
                ligand_dict[curr_class][curr_prot] = []
                continue

            # new small molecule
            if len(row[3]) > 0:
                ligand_dict[curr_class][curr_prot].append(
                    # (ligand name, validity, SMILES string)
                    [row[3], row[4], row[9]]
                )

    return ligand_dict


def filter_and_flatten(ligand_dict, max_occurences=-1):

    filtered_examples = []
    all_examples = [i for i in ligand_dict]

    ligand_name_counter = defaultdict(int)
    print("Filtering examples...")
    for c, p, m in tqdm(all_examples):
        ligand_name, ligand_chain, ligand_resi = m[0].split(':')
        if m[1] == 'valid':
            if max_occurences == -1:
                filtered_examples.append(
                    (c, p, m)
                )
                ligand_name_counter[ligand_name] += 1
            else:
                if ligand_name_counter[ligand_name] < max_occurences:
                    filtered_examples.append(
                        (c, p, m)
                    )
                    ligand_name_counter[ligand_name] += 1

    return filtered_examples



def ligand_list_to_dict(ligand_list):
    out_dict = defaultdict(list)
    for _, p, m in ligand_list:
        out_dict[p].append(m)
    return out_dict


def process_ligand_and_pocket(pdb_struct, ligand_name, ligand_chain,
                              ligand_resi, dist_cutoff, ca_only,
                              no_H, mol_id, determine_distance_by_ca=False):
    try:
        residues = {obj.id[1]: obj for obj in
                    pdb_struct[0][ligand_chain].get_residues()}
    except KeyError as e:
        raise KeyError(f'Chain {e} not found ({pdbfile}, '
                       f'{ligand_name}:{ligand_chain}:{ligand_resi})')
    ligand = residues[ligand_resi]
    assert ligand.get_resname() == ligand_name, \
        f"{ligand.get_resname()} != {ligand_name}"

    an2s, s2an = get_periodictable_list(include_aa=True)

    # LIGAND
    if no_H:
        lig_atoms = [a for a in ligand.get_atoms() if a.element != 'H']
    else:
        lig_atoms = [a for a in ligand.get_atoms()]
    
    
    ligand_atom_charge_positions = []
    for atom in lig_atoms:
        x, y, z = atom.get_coord()
        ligand_atom_charge_positions.append([float(mol_id), float(s2an[atom.element]), float(x), float(y), float(z)])
    ligand_atom_charge_positions = np.array(ligand_atom_charge_positions)

    # POCKET
    # Find interacting pocket residues based on distance cutoff
    pocket_residues = []
    lig_coords = np.array([a.get_coord() for a in lig_atoms])
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
    parser.add_argument('--raw_moad_basedir', type=Path)
    parser.add_argument("--save_dir", type=str, default="data/BindingMOAD_LG_PKT")
    parser.add_argument("--save_dataset_name", type=str, default="BindingMOAD_LG_PKT")
    
    parser.add_argument('--dist_cutoff', type=float, default=10.0)  # 8.0
    parser.add_argument('--max_occurences', type=int, default=-1)  # 50
    parser.add_argument('--ca_only', action='store_true')
    parser.add_argument('--no_H', action='store_true')
    parser.add_argument('--determine_distance_by_ca', action='store_true')  # wrong method, do not use
    args = parser.parse_args()
    
    # python -W ignore 01_build_bindingmoad_dataset.py --raw_moad_basedir /Users/gohyixian/Documents/Documents/3.2_FYP_1/data/BindingMOAD --dist_cutoff 10.0 --max_occurences 50 --no_H --ca_only --save_dir /Users/gohyixian/Documents/GitHub/FYP/GeoLDM-edit/data/d_20240623_BindingMOAD_LG_PKT --save_dataset_name d_20240623_BindingMOAD_LG_PKT
    
    # python -W ignore 01_build_bindingmoad_dataset.py --raw_moad_basedir /mnt/c/Users/PC/Desktop/yixian/data/BindingMOAD --dist_cutoff 10.0 --max_occurences 50 --no_H --ca_only --save_dir /mnt/c/Users/PC/Desktop/yixian/GeoLDM-edit/data/d_20240623_BindingMOAD_LG_PKT --save_dataset_name d_20240623_BindingMOAD_LG_PKT
    
    # python -W ignore 01_build_bindingmoad_dataset.py --raw_moad_basedir /mnt/c/Users/PC/Desktop/yixian/data/BindingMOAD --dist_cutoff 10.0 --max_occurences 50 --save_dir /mnt/c/Users/PC/Desktop/yixian/GeoLDM-edit/data/d_20240623_BindingMOAD_LG_PKT --save_dataset_name d_20240623_BindingMOAD_LG_PKT


    pdbdir = args.raw_moad_basedir / 'BindingMOAD_2020/'

    # Process the label file
    csv_path = args.raw_moad_basedir / 'every.csv'
    ligand_dict = read_label_file(csv_path)
    all_data = [(c, p, m) for c in ligand_dict for p in ligand_dict[c]
                for m in ligand_dict[c][p]]
    all_data = filter_and_flatten(all_data, max_occurences=args.max_occurences)
    print(f'{len(all_data)} examples after filtering')


    ligand_dataset = []
    pocket_dataset = []
    mol_id = 0
    
    
    n_tot = len(all_data)
    # 4Y0D: [(RW2:A:502,), (RW2:B:501,), (RW2:C:503,), (RW2:D:501,)]
    pair_dict = ligand_list_to_dict(all_data)

    num_failed = 0
    with tqdm(total=n_tot) as pbar:
        # 1A0J
        for p in pair_dict:
            # print("=============")
            # print(f"P: {p}")

            pdb_successful = set()

            # try all available .bio files
            # look for all files with name '1a0j.bio?'
            for pdbfile in sorted(pdbdir.glob(f"{p.lower()}.bio*")):

                # Skip if all ligands have been processed already
                if len(pair_dict[p]) == len(pdb_successful):
                    continue

                try:
                    pdb_struct = PDBParser(QUIET=True).get_structure('', pdbfile)
                except:
                    print("SKIPPED")
                    continue

                struct_copy = pdb_struct.copy()

                n_bio_successful = 0
                # (RW2:A:502,)
                for m in pair_dict[p]:
                    # print(f"M: {m}")

                    # Skip already processed ligand (duplicates: ligands with >1 optimal docking positions)
                    # RW2:A:502
                    if m[0] in pdb_successful:
                        continue

                    # RW2        A             502
                    ligand_name, ligand_chain, ligand_resi = m[0].split(':')
                    ligand_resi = int(ligand_resi)

                    try:
                        # print(ligand_name, ligand_chain, ligand_resi)
                        ligand_data, pocket_data = process_ligand_and_pocket(
                            pdb_struct, ligand_name, ligand_chain, ligand_resi,
                            dist_cutoff=args.dist_cutoff, ca_only=args.ca_only, 
                            no_H=args.no_H, mol_id=mol_id, 
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
                            
                        
                    except (KeyError, AssertionError, FileNotFoundError,
                            IndexError, ValueError) as e:
                        # print(type(e).__name__, e)
                        continue

                    pdb_successful.add(m[0])
                    n_bio_successful += 1
                    

            pbar.update(len(pair_dict[p]))
            num_failed += (len(pair_dict[p]) - len(pdb_successful))
            pbar.set_description(f'#failed: {num_failed}')

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
    save_file = f"{args.save_dataset_name}__{args.dist_cutoff}A__MaxOcc{args.max_occurences}{'__CA_Only' if args.ca_only else ''}{'__no_H' if args.no_H else ''}.npz"
    
    np.savez(os.path.join(args.save_dir, save_file), ligand=ligand_dataset, pocket=pocket_dataset)
    
    print()
    print(f"[LG] Total Atom Num  : {ligand_dataset.shape[0]}", )
    print(f"[LG] Ave. Atom Num   : {ligand_dataset.shape[0] / (mol_id+1)}")
    print(f"[PKT] Total Atom Num : {pocket_dataset.shape[0]}", )
    print(f"[PKT] Ave. Atom Num  : {pocket_dataset.shape[0] / (mol_id+1)}")
    print("Total number of Ligand-Pocket pairs:", mol_id+1)
    print("Dataset processed.")


#failed: 15273: 100% || 66031/66031 [2:07:53<00:00, 8.60it/s]
#
# [LG] Total Atom Num : 1200758
# [LG] Ave. Atom Num  : 23.65606099410942
# [PT] Total Atom Num : 3390278
# [PKT] Ave. Atom Num : 66.79166256230423
# Total number of Ligand-Pocket pairs: 50759
# Dataset processed.




