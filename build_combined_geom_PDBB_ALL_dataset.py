import msgpack
import os
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, SequentialSampler
import argparse
from qm9.data import collate as qm9_collate
from utils import get_periodictable_list


def extract_conformers(args):
    save_file = f"{args.save_dataset_name}__{'no_h_' if args.remove_h else ''}geom_{args.geom_conformations}"
    # smiles_list_file = 'geom_drugs_smiles.txt'
    # number_atoms_file = f"geom_drugs_n_{'no_h_' if args.remove_h else ''}{args.geom_conformations}"

    # all_smiles = []
    # all_number_atoms = []
    dataset_conformers = []
    mol_id = 0
    geom_count = 0
    PDBB_count = 0
    
    
    # GEOM
    # ===================
    geom_drugs_file = args.geom_data_file
    unpacker = msgpack.Unpacker(open(geom_drugs_file, "rb"))

    for i, drugs_1k in enumerate(unpacker):
        print(f"Unpacking file {i}...")
        for smiles, all_info in drugs_1k.items():
            # all_smiles.append(smiles)
            conformers = all_info['conformers']
            # Get the energy of each conformer. Keep only the lowest values
            all_energies = []
            for conformer in conformers:
                all_energies.append(conformer['totalenergy'])
            all_energies = np.array(all_energies)
            argsort = np.argsort(all_energies)
            lowest_energies = argsort[:args.geom_conformations]
            for id in lowest_energies:
                conformer = conformers[id]
                coords = np.array(conformer['xyz']).astype(float)        # n x 4   [atomic_num, x, y, z]
                if args.remove_h:
                    mask = coords[:, 0] != 1.0    # hydrogen's atomic_num = 1
                    coords = coords[mask]
                n = coords.shape[0]
                # all_number_atoms.append(n)
                mol_id_arr = mol_id * np.ones((n, 1), dtype=float)
                id_coords = np.hstack((mol_id_arr, coords))

                dataset_conformers.append(id_coords)
                mol_id += 1
                geom_count += 1
    
    # PDBB_refined_core
    # ===================
    LG_PKT_pair_folders = sorted(os.listdir(args.PDBB_refined_core_folder))
    
    to_remove = ['.DS_Store']
    for r in to_remove:
        if r in LG_PKT_pair_folders:
            LG_PKT_pair_folders.remove(r)
    
    target_files = ['ligand', 'protein']   # omit protein
    
    for i, folder in enumerate(LG_PKT_pair_folders):
        print(f"Processing folder {i}...")
        files = os.listdir(os.path.join(args.PDBB_refined_core_folder, folder))
        
        filtered_files = []
        for t in target_files:
            filtered_files += [f for f in files if t in f]
        
        for f in filtered_files:
            with open(os.path.join(args.PDBB_refined_core_folder, folder, f), 'r') as datafile:
                # xyz_lines = [line.encode('UTF-8') for line in datafile.readlines()]
                xyz_lines = [line for line in datafile]
                
                an2s, s2an = get_periodictable_list()

                num_atoms = int(xyz_lines[0])
                mol_xyz = xyz_lines[2:num_atoms+2]

                atom_charge_positions = []
                for line in mol_xyz:
                    atom, posx, posy, posz = line.replace('*^', 'e').split()
                    atom_charge_positions.append([float(s2an[atom]), float(posx), float(posy), float(posz)])
                coords = np.array(atom_charge_positions).astype(float)        # n x 4   [atomic_num, x, y, z]
                
                if args.remove_h:
                    mask = coords[:, 0] != 1.0    # hydrogen's atomic_num = 1
                    coords = coords[mask]

                n = coords.shape[0]
                # all_number_atoms.append(n)
                mol_id_arr = mol_id * np.ones((n, 1), dtype=float)
                id_coords = np.hstack((mol_id_arr, coords))

                dataset_conformers.append(id_coords)
                mol_id += 1
                PDBB_count += 1



    print("Total number of molecules (geom-conformers, PDBB-LG, PDBB-PKT) saved", mol_id)
    # all_number_atoms = np.array(all_number_atoms)
    dataset = np.vstack(dataset_conformers)

    print("Total number of atoms in the dataset", dataset.shape[0])
    print("Average number of atoms per molecule", dataset.shape[0] / mol_id)
    print("Geom count", geom_count)
    print("PDBB count", PDBB_count)

    # Save conformations
    np.save(os.path.join(args.save_dir, save_file), dataset)
    # Save SMILES
    # with open(os.path.join(args.data_dir, smiles_list_file), 'w') as f:
    #     for s in all_smiles:
    #         f.write(s)
    #         f.write('\n')

    # Save number of atoms per conformation
    # np.save(os.path.join(args.data_dir, number_atoms_file), all_number_atoms)
    print("Dataset processed.")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # quick note: conformations = possible 3D arrangements of a molecule due to factors like
    #             single bonds are rotateble bonds.
    parser.add_argument("--geom_conformations", type=int, default=30, help="Max number of conformations kept for each molecule.")
    parser.add_argument("--geom_data_file", type=str, default="drugs_crude.msgpack")
    parser.add_argument("--PDBB_refined_core_folder", type=str, default="01-combined-refined-and-core-set-xyz")
    
    parser.add_argument("--remove_h", action='store_true', help="Remove hydrogens from the dataset.")
    
    parser.add_argument("--save_dir", type=str, default="data/combined_geom_PDBB_refined_core_LG_PKT")
    parser.add_argument("--save_dataset_name", type=str, default="combined_geom_PDBB_refined_core_LG_PKT")
    
    # python build_combined_geom_PDBB_ALL_dataset.py --geom_conformations 30 --geom_data_file /Users/gohyixian/Documents/Documents/3.2_FYP_1/data/GEOM/drugs_crude.msgpack --PDBB_refined_core_folder /Users/gohyixian/Documents/Documents/3.2_FYP_1/data/PDBBind2020/01-combined-refined-and-core-set-xyz --save_dir /Users/gohyixian/Documents/GitHub/FYP/GeoLDM-edit/data/combined_geom_PDBB_refined_core_LG_PKT --save_dataset_name combined_geom_PDBB_refined_core_LG_PKT
    
    # proportion of PDBB-LG & PDBB-PKT (5620*2) occupies too less in total of ((5,620*2)+6,922,516) 
    # samples if we set geom_conformations to 30, hence the minimum value of 1. With 1, we get
    # around (5620*2) / ((5,620*2)+(6,922,516/30)) = 0.04644809797 for PDBB-LG & PDBB-PKT
    # python build_combined_geom_PDBB_ALL_dataset.py --geom_conformations 1 --geom_data_file /Users/gohyixian/Documents/Documents/3.2_FYP_1/data/GEOM/drugs_crude.msgpack --PDBB_refined_core_folder /Users/gohyixian/Documents/Documents/3.2_FYP_1/data/PDBBind2020/01-combined-refined-and-core-set-xyz --save_dir /Users/gohyixian/Documents/GitHub/FYP/GeoLDM-edit/data/combined_geom_PDBB_refined_core_LG_PKT --save_dataset_name combined_geom_PDBB_refined_core_LG_PKT
    
    
    args = parser.parse_args()
    extract_conformers(args)
    print("DONE.")
