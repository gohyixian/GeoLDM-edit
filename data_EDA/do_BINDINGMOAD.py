from pathlib import Path
from time import time
from collections import defaultdict
import argparse
import pickle

from tqdm import tqdm
import numpy as np
from Bio.PDB import PDBParser
# from Bio.PDB.Polypeptide import three_to_one, is_aa
from Bio.PDB.Polypeptide import three_to_index, index_to_one, is_aa

from constants import dataset_params, get_periodictable_list
from utils import euclidean_distance

dataset_info = dataset_params['bindingmoad']
# amino_acid_dict = dataset_info['aa_encoder']
atom_dict = dataset_info['atom_encoder']
# atom_decoder = dataset_info['atom_decoder']


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


def process_ligand_and_pocket(pdb_struct, ligand_name, ligand_chain, ligand_resi, 
                              dist_cutoff, ca_only, determine_distance_by_ca=False):
    try:
        residues = {obj.id[1]: obj for obj in
                    pdb_struct[0][ligand_chain].get_residues()}
    except KeyError as e:
        raise KeyError(f'Chain {e} not found ({pdbfile}, '
                       f'{ligand_name}:{ligand_chain}:{ligand_resi})')
    ligand = residues[ligand_resi]
    assert ligand.get_resname() == ligand_name, \
        f"{ligand.get_resname()} != {ligand_name}"

    # remove H atoms if not in atom_dict, other atom types that aren't allowed
    # should stay so that the entire ligand can be removed from the dataset
    # lig_atoms = [a for a in ligand.get_atoms()
    #              if (a.element.capitalize() in atom_dict or a.element != 'H')]
    lig_atoms = [a for a in ligand.get_atoms()
                 if (a.element.capitalize() in atom_dict)]
    lig_coords = np.array([a.get_coord() for a in lig_atoms])
    
    lig_num_atoms = len(lig_atoms)
    lig_num_atoms_no_H = len([a for a in lig_atoms if a.element != 'H'])
    
    lig_coord_center = np.mean(lig_coords, axis=0)
    print(f"LG: {lig_coord_center.shape}")
    lig_radius = euclidean_distance(lig_coords, lig_coord_center, axis=1)
    print(f"LG: {lig_radius.shape}")
    lig_radius_mean = float(np.mean(lig_radius))
    lig_radius_min = float(np.min(lig_radius))
    lig_radius_max = float(np.max(lig_radius))
    
    an2s, s2an = get_periodictable_list()
    lig_atomic_num_freq = dict()
    for a in lig_atoms:
        atomic_num = s2an[a.element]
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
                    (((res_coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(-1) ** 0.5).min() < dist_cutoff:
                pocket_residues.append(residue)
                num_resi += 1
                resi_num_atoms += len([a for a in residue.get_atoms()])
                resi_num_atoms_no_H += len([a for a in residue.get_atoms() if a.element != 'H'])

    # Compute transform of the canonical reference frame
    n_xyz = np.array([res['N'].get_coord() for res in pocket_residues])
    ca_xyz = np.array([res['CA'].get_coord() for res in pocket_residues])
    c_xyz = np.array([res['C'].get_coord() for res in pocket_residues])

    c_alpha = ca_xyz
    pocket_atomic_num_freq = dict()

    if ca_only:
        pocket_coords = c_alpha
        pocket_coords_center = np.mean(pocket_coords, axis=0)
        print(f"PKT: {pocket_coords_center.shape}")
        pocket_radius = euclidean_distance(pocket_coords, pocket_coords_center, axis=1)
        print(f"PKT: {pocket_radius.shape}")
        pocket_radius_mean = float(np.mean(pocket_radius))
        pocket_radius_min = float(np.min(pocket_radius))
        pocket_radius_max = float(np.max(pocket_radius))
        pocket_atomic_num_freq[s2an['C']] = len(pocket_coords)
    else:
        pocket_atoms = [a for res in pocket_residues for a in res.get_atoms()
                        if (a.element.capitalize() in atom_dict or a.element != 'H')]
        pocket_coords = np.array([a.get_coord() for a in pocket_atoms])
        pocket_coords_center = np.mean(pocket_coords, axis=0)
        print(f"PKT: {pocket_coords_center.shape}")
        pocket_radius = euclidean_distance(pocket_coords, pocket_coords_center, axis=1)
        print(f"PKT: {pocket_radius.shape}")
        pocket_radius_mean = float(np.mean(pocket_radius))
        pocket_radius_min = float(np.min(pocket_radius))
        pocket_radius_max = float(np.max(pocket_radius))
        
        for a in pocket_atoms:
            atomic_num = s2an[a.element]
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


class BINDINGMOAD:
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
    parser.add_argument('--dist_cutoff', type=float, default=10.0)  # 8.0
    parser.add_argument('--max_occurences', type=int, default=-1)  # 50
    parser.add_argument('--ca_only', action='store_true')
    parser.add_argument('--determine_distance_by_ca', action='store_true')  # wrong method, do not use
    args = parser.parse_args()

    pdbdir = args.basedir / 'BindingMOAD_2020/'

    ligand_data_obj = BINDINGMOAD()
    pocket_data_obj = BINDINGMOAD()

    # Process the label file
    csv_path = args.basedir / 'every.csv'
    ligand_dict = read_label_file(csv_path)
    all_data = [(c, p, m) for c in ligand_dict for p in ligand_dict[c]
                for m in ligand_dict[c][p]]
    all_data = filter_and_flatten(all_data, max_occurences=args.max_occurences)
    print(f'{len(all_data)} examples after filtering')


    lig_coords = []
    lig_one_hot = []
    pocket_coords = []
    pocket_one_hot = []

    n_tot = len(all_data)
    # 4Y0D: [(RW2:A:502,), (RW2:B:501,), (RW2:C:503,), (RW2:D:501,)]
    pair_dict = ligand_list_to_dict(all_data)

    tic = time()
    num_failed = 0
    with tqdm(total=n_tot) as pbar:
        # 1A0J
        for p in pair_dict:
            print("=============")
            print(f"P: {p}")

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
                    print(f"M: {m}")

                    # Skip already processed ligand (duplicates: ligands with >1 optimal docking positions)
                    # RW2:A:502
                    if m[0] in pdb_successful:
                        continue

                    # RW2        A             502
                    ligand_name, ligand_chain, ligand_resi = m[0].split(':')
                    ligand_resi = int(ligand_resi)

                    try:
                        print(ligand_name, ligand_chain, ligand_resi)
                        ligand_data, pocket_data = process_ligand_and_pocket(
                            pdb_struct, ligand_name, ligand_chain, ligand_resi,
                            dist_cutoff=args.dist_cutoff, ca_only=args.ca_only,
                            determine_distance_by_ca=args.determine_distance_by_ca)
                    except (KeyError, AssertionError, FileNotFoundError,
                            IndexError, ValueError) as e:
                        # print(type(e).__name__, e)
                        continue

                    # filter data the same way as in dataset preparation script
                    if ligand_data['lig_num_atoms'] > 0 and ligand_data['lig_num_atoms_no_H'] > 0 and \
                        pocket_data['pocket_num_atoms'] > 0 and pocket_data['pocket_num_atoms_no_H'] > 0:

                        pdb_successful.add(m[0])
                        n_bio_successful += 1
                        
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

            pbar.update(len(pair_dict[p]))
            num_failed += (len(pair_dict[p]) - len(pdb_successful))
            pbar.set_description(f'#failed: {num_failed}')


    print("\nLIGAND")
    print("=====================")
    # LIGAND
    # Store the object to a file using pickle
    distance_by_ca = "ca_dist_" if args.determine_distance_by_ca else ""
    moad_ligand_data_object_pkl = f'/Users/gohyixian/Documents/GitHub/FYP/GeoLDM-edit/data_EDA/data_object_cache/BINDINGMOAD_{distance_by_ca}LIGAND_data_object.pkl'
    with open(moad_ligand_data_object_pkl, 'wb') as file:  # Use 'wb' mode for binary writing
        pickle.dump(ligand_data_obj, file)
    del ligand_data_obj

    # Load the object back from the file
    with open(moad_ligand_data_object_pkl, 'rb') as file:  # Use 'rb' mode for binary reading
        ligand_data_obj = pickle.load(file)
    
    attributes_values = vars(ligand_data_obj)
    for attr, value in attributes_values.items():
        print(f"{attr}: {value}")

    print("\nPOCKET")
    print("=====================")
    # POCKET
    # Store the object to a file using pickle
    ca_only = 'CA_ONLY_' if args.ca_only else ''
    moad_pocket_data_object_pkl = f'/Users/gohyixian/Documents/GitHub/FYP/GeoLDM-edit/data_EDA/data_object_cache/BINDINGMOAD_POCKET_{distance_by_ca}{args.dist_cutoff}A_{ca_only}data_object.pkl'
    with open(moad_pocket_data_object_pkl, 'wb') as file:  # Use 'wb' mode for binary writing
        pickle.dump(pocket_data_obj, file)
    del pocket_data_obj

    # Load the object back from the file
    with open(moad_pocket_data_object_pkl, 'rb') as file:  # Use 'rb' mode for binary reading
        pocket_data_obj = pickle.load(file)
    
    attributes_values = vars(pocket_data_obj)
    for attr, value in attributes_values.items():
        print(f"{attr}: {value}")


# python -W ignore do_BINDINGMOAD.py /Users/gohyixian/Documents/Documents/3.2_FYP_1/data/BindingMOAD --dist_cutoff 10.0 --max_occurences 50

# python -W ignore do_BINDINGMOAD.py /Users/gohyixian/Documents/Documents/3.2_FYP_1/data/BindingMOAD --dist_cutoff 10.0 --max_occurences 50 --determine_distance_by_ca

