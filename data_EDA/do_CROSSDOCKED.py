from pathlib import Path
from time import time
import argparse
import pickle

from tqdm import tqdm
import numpy as np
from Bio.PDB import PDBParser
# from Bio.PDB.Polypeptide import three_to_one, is_aa
from Bio.PDB.Polypeptide import three_to_index, index_to_one, is_aa
from rdkit import Chem
import torch

from constants import dataset_params, get_periodictable_list
from utils import euclidean_distance


def process_ligand_and_pocket(pdbfile, sdffile, atom_dict, dist_cutoff, 
                              ca_only, determine_distance_by_ca=False):
    pdb_struct = PDBParser(QUIET=True).get_structure('', pdbfile)

    try:
        ligand = Chem.SDMolSupplier(str(sdffile))[0]
    except:
        raise Exception(f'cannot read sdf mol ({sdffile})')

    # remove H atoms if not in atom_dict, other atom types that aren't allowed
    # should stay so that the entire ligand can be removed from the dataset
    lig_atoms = [a.GetSymbol() for a in ligand.GetAtoms()
                 if (a.GetSymbol().capitalize() in atom_dict)]
    lig_coords = np.array([list(ligand.GetConformer(0).GetAtomPosition(idx))
                           for idx in range(ligand.GetNumAtoms())])

    lig_num_atoms = len(lig_atoms)
    lig_num_atoms_no_H = len([a for a in lig_atoms if a != 'H'])

    lig_coord_center = np.mean(lig_coords)
    lig_radius = euclidean_distance(lig_coords, lig_coord_center, axis=0)
    lig_radius_mean = float(np.mean(lig_radius))
    lig_radius_min = float(np.min(lig_radius))
    lig_radius_max = float(np.max(lig_radius))

    an2s, s2an = get_periodictable_list()
    lig_atomic_num_freq = dict()
    for a in lig_atoms:
        atomic_num = s2an[a]
        lig_atomic_num_freq[atomic_num] = lig_atomic_num_freq.get(atomic_num, 0) + 1

    # Find interacting pocket residues based on distance cutoff
    pocket_residues = []
    num_resi = 0
    resi_num_atoms = 0
    resi_num_atoms_no_H = 0
    for residue in pdb_struct[0].get_residues():
        if determine_distance_by_ca:
            res_ca_coord = np.array([residue['CA'].get_coord()])
            if is_aa(residue.get_resname(), standard=True) and \
                    (((res_ca_coord - lig_coords) ** 2).sum(-1) ** 0.5).min() < dist_cutoff:
                pocket_residues.append(residue)
                num_resi += 1
                resi_num_atoms += len([a for a in residue.get_atoms()])
                resi_num_atoms_no_H += len([a for a in residue.get_atoms() if a.element != 'H'])
        else: 
            res_coords = np.array([a.get_coord() for a in residue.get_atoms()])
            if is_aa(residue.get_resname(), standard=True) and \
                    (((res_coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(
                        -1) ** 0.5).min() < dist_cutoff:
                pocket_residues.append(residue)
                num_resi += 1
                resi_num_atoms += len([a for a in residue.get_atoms()])
                resi_num_atoms_no_H += len([a for a in residue.get_atoms() if a.element != 'H'])


    pocket_atomic_num_freq = dict()

    if ca_only:
        try:
            full_coords = []
            for res in pocket_residues:
                for atom in res.get_atoms():
                    if atom.name == 'CA':
                        full_coords.append(atom.coord)
            full_coords = np.stack(full_coords)
            pocket_coords_center = np.mean(full_coords)
            pocket_radius = euclidean_distance(full_coords, pocket_coords_center, axis=0)
            pocket_radius_mean = float(np.mean(pocket_radius))
            pocket_radius_min = float(np.min(pocket_radius))
            pocket_radius_max = float(np.max(pocket_radius))
            pocket_atomic_num_freq[s2an['C']] = len(full_coords)
        except KeyError as e:
            raise KeyError(
                f'{e} not in amino acid dict ({pdbfile}, {sdffile})')
    else:
        full_atoms = np.concatenate(
            [np.array([atom.element for atom in res.get_atoms()])
             for res in pocket_residues], axis=0)
        full_coords = np.concatenate(
            [np.array([atom.coord for atom in res.get_atoms()])
             for res in pocket_residues], axis=0)

        pocket_coords_center = np.mean(full_coords)
        pocket_radius = euclidean_distance(full_coords, pocket_coords_center, axis=0)
        pocket_radius_mean = float(np.mean(pocket_radius))
        pocket_radius_min = float(np.min(pocket_radius))
        pocket_radius_max = float(np.max(pocket_radius))
        
        for a in full_atoms:
            atomic_num = s2an[str(a)]
            pocket_atomic_num_freq[atomic_num] = pocket_atomic_num_freq.get(atomic_num, 0) + 1


    ligand_data = {
        'lig_num_atoms': lig_num_atoms,
        'lig_num_atoms_no_H': lig_num_atoms_no_H,
        'lig_radius_mean': lig_radius_mean,
        'lig_radius_min': lig_radius_min,
        'lig_radius_max': lig_radius_max,
        'lig_atomic_num_freq': lig_atomic_num_freq
    }
    pocket_data = {
        'pocket_num_resi': num_resi,
        'pocket_num_atoms': resi_num_atoms,
        'pocket_num_atoms_no_H': resi_num_atoms_no_H,
        'pocket_radius_mean': pocket_radius_mean,
        'pocket_radius_min': pocket_radius_min,
        'pocket_radius_max': pocket_radius_max,
        'pocket_atomic_num_freq': pocket_atomic_num_freq
    }
    return ligand_data, pocket_data


class CROSSDOCKED:
    def __init__(self):
        self.atomic_num_freq = dict()   # atomic_num: freq
        self.num_atoms = []
        self.num_atoms_no_H = []
        self.mol_count = 0
        
        # molecule radius in Angstroms from center
        self.radius_mean = []
        self.radius_min = []
        self.radius_max = []
        self.num_resi = []  # pocket only



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir', type=Path)
    parser.add_argument('--ca_only', action='store_true')
    parser.add_argument('--dist_cutoff', type=float, default=10.0)  # 8.0
    parser.add_argument('--determine_distance_by_ca', action='store_true')  # wrong method, do not use
    args = parser.parse_args()

    datadir = args.basedir / 'crossdocked_pocket10/'

    ligand_data_obj = CROSSDOCKED()
    pocket_data_obj = CROSSDOCKED()

    if args.ca_only:
        dataset_info = dataset_params['crossdock']
    else:
        dataset_info = dataset_params['crossdock_full']
    atom_dict = dataset_info['atom_encoder']

    # Read data split
    split_path = Path(args.basedir, 'split_by_name.pt')
    data_split = torch.load(split_path)
    
    # combined train and test set = all
    all_data = data_split['train'] + data_split['test']  # list

    failed_save = []

    tic = time()
    num_failed = 0
    pbar = tqdm(all_data)
    pbar.set_description(f'#failed: {num_failed}')
    for pocket_fn, ligand_fn in pbar:

        sdffile = datadir / f'{ligand_fn}'
        pdbfile = datadir / f'{pocket_fn}'
        
        # print("===============")
        # print(ligand_fn)
        # print(pocket_fn)

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
                pdbfile, sdffile, atom_dict=atom_dict, 
                dist_cutoff=args.dist_cutoff,
                ca_only=args.ca_only, 
                determine_distance_by_ca=args.determine_distance_by_ca)
        except (KeyError, AssertionError, FileNotFoundError, IndexError,
                ValueError) as e:
            print(type(e).__name__, e, pocket_fn, ligand_fn)
            num_failed += 1
            pbar.set_description(f'#failed: {num_failed}')
            continue

        # filter data the same way as in dataset preparation script
        if ligand_data['lig_num_atoms'] > 0 and ligand_data['lig_num_atoms_no_H'] > 0 and \
            pocket_data['pocket_num_atoms'] > 0 and pocket_data['pocket_num_atoms_no_H'] > 0:

            ligand_data_obj.num_atoms.append(ligand_data['lig_num_atoms'])
            ligand_data_obj.num_atoms_no_H.append(ligand_data['lig_num_atoms_no_H'])
            ligand_data_obj.radius_mean.append(ligand_data['lig_radius_mean'])
            ligand_data_obj.radius_min.append(ligand_data['lig_radius_min'])
            ligand_data_obj.radius_max.append(ligand_data['lig_radius_max'])
            for k,v in ligand_data['lig_atomic_num_freq'].items():
                ligand_data_obj.atomic_num_freq[k] = ligand_data_obj.atomic_num_freq.get(k, 0) + v
            ligand_data_obj.mol_count += 1
            
            pocket_data_obj.num_atoms.append(pocket_data['pocket_num_atoms'])
            pocket_data_obj.num_atoms_no_H.append(pocket_data['pocket_num_atoms_no_H'])
            pocket_data_obj.radius_mean.append(pocket_data['pocket_radius_mean'])
            pocket_data_obj.radius_min.append(pocket_data['pocket_radius_min'])
            pocket_data_obj.radius_max.append(pocket_data['pocket_radius_max'])
            pocket_data_obj.num_resi.append(pocket_data['pocket_num_resi'])
            for k,v in pocket_data['pocket_atomic_num_freq'].items():
                pocket_data_obj.atomic_num_freq[k] = pocket_data_obj.atomic_num_freq.get(k, 0) + v
            pocket_data_obj.mol_count += 1

    print(f"Processing took {(time() - tic) / 60.0:.2f} minutes")


    print("\nLIGAND")
    print("=====================")
    # LIGAND
    # Store the object to a file using pickle
    distance_by_ca = "ca_dist_" if args.determine_distance_by_ca else ""
    # crossdocked_ligand_data_object_pkl = f"/mnt/c/Users/PC/Desktop/yixian/GeoLDM-edit/data_EDA/data_object_cache/CROSSDOCKED_LIGAND_{distance_by_ca}data_object.pkl"
    crossdocked_ligand_data_object_pkl = f'/Users/gohyixian/Documents/GitHub/FYP/GeoLDM-edit/data_EDA/data_object_cache/CROSSDOCKED_LIGAND_{distance_by_ca}data_object.pkl'
    with open(crossdocked_ligand_data_object_pkl, 'wb') as file:  # Use 'wb' mode for binary writing
        pickle.dump(ligand_data_obj, file)
    del ligand_data_obj

    # Load the object back from the file
    with open(crossdocked_ligand_data_object_pkl, 'rb') as file:  # Use 'rb' mode for binary reading
        ligand_data_obj = pickle.load(file)
    
    attributes_values = vars(ligand_data_obj)
    for attr, value in attributes_values.items():
        print(f"{attr}: {value}")

    print("\nPOCKET")
    print("=====================")
    # POCKET
    # Store the object to a file using pickle
    ca_only = 'CA_ONLY_' if args.ca_only else ''
    # crossdocked_pocket_data_object_pkl = f"/mnt/c/Users/PC/Desktop/yixian/GeoLDM-edit/data_EDA/data_object_cache/CROSSDOCKED_POCKET_{distance_by_ca}{args.dist_cutoff}A_{ca_only}data_object.pkl"
    crossdocked_pocket_data_object_pkl = f'/Users/gohyixian/Documents/GitHub/FYP/GeoLDM-edit/data_EDA/data_object_cache/CROSSDOCKED_POCKET_{distance_by_ca}{args.dist_cutoff}A_{ca_only}data_object.pkl'
    with open(crossdocked_pocket_data_object_pkl, 'wb') as file:  # Use 'wb' mode for binary writing
        pickle.dump(pocket_data_obj, file)
    del pocket_data_obj

    # Load the object back from the file
    with open(crossdocked_pocket_data_object_pkl, 'rb') as file:  # Use 'rb' mode for binary reading
        pocket_data_obj = pickle.load(file)
    
    attributes_values = vars(pocket_data_obj)
    for attr, value in attributes_values.items():
        print(f"{attr}: {value}")

# python do_CROSSDOCKED.py /Users/gohyixian/Documents/Documents/3.2_FYP_1/data/CrossDocked --dist_cutoff 10.0

# python do_CROSSDOCKED.py /Users/gohyixian/Documents/Documents/3.2_FYP_1/data/CrossDocked --dist_cutoff 10.0 --determine_distance_by_ca
