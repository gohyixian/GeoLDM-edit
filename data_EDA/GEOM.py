from tqdm import tqdm
import numpy as np
import os
import pickle
import msgpack
from constants import get_periodictable_list
from utils import euclidean_distance


class GEOM:
    def __init__(self):
        self.atomic_num_freq = dict()   # atomic_num: freq
        self.num_atoms = []
        self.num_atoms_no_H = []
        self.mol_count = 0
        
        # molecule radius in Angstroms from center
        self.radius_mean = []
        self.radius_min = []
        self.radius_max = []



    def extract_conformers(self, unpacker, max_conformers_per_molecule=30):

        for i, drugs_1k in enumerate(unpacker):
            print(f"Unpacking file {i}/292...")
            for smiles, all_info in tqdm(drugs_1k.items()):
                conformers = all_info['conformers']
                # Get the energy of each conformer. Keep only the lowest values
                all_energies = []
                for conformer in conformers:
                    all_energies.append(conformer['totalenergy'])
                all_energies = np.array(all_energies)
                argsort = np.argsort(all_energies)
                lowest_energies = argsort[:max_conformers_per_molecule]
                for id in lowest_energies:
                    
                    conformer = conformers[id]
                    coords = np.array(conformer['xyz']).astype(float)        # n x 4   [atomic_num, x, y, z]
                    atomic_nums = coords[:, 0].squeeze().astype(int)
                    
                    # if args.remove_h:
                    #     mask = coords[:, 0] != 1.0    # hydrogen's atomic_num = 1
                    #     coords = coords[mask]
                    
                    self.num_atoms.append(len(atomic_nums))
                    self.num_atoms_no_H.append(len(atomic_nums[atomic_nums > 1]))
                    
                    for ac in atomic_nums:
                        self.atomic_num_freq[int(ac)] = int(self.atomic_num_freq.get(int(ac), 0) + 1)
                    
                    xyz = coords[:, 1:]
                    center_of_coordinates = np.mean(xyz, axis=0)
                    distances_to_center = np.array([euclidean_distance(atom, center_of_coordinates) for atom in xyz])
                    self.radius_mean.append(float(np.mean(distances_to_center)))
                    self.radius_min.append(float(np.min(distances_to_center)))
                    self.radius_max.append(float(np.max(distances_to_center)))

                    self.mol_count += 1


if __name__ == "__main__":
    path_to_GEOM_msgpack = '/Users/gohyixian/Documents/Documents/3.2_FYP_1/data/GEOM/drugs_crude.msgpack'
    unpacker = msgpack.Unpacker(open(path_to_GEOM_msgpack, "rb"))
    
    geom_data_obj = GEOM()
    
    # for now, work with the one conformer with lowest total energy only (in Hartree)
    # geom_data_obj.extract_conformers(unpacker, max_conformers_per_molecule=1)
    geom_data_obj.extract_conformers(unpacker, max_conformers_per_molecule=30)   # 30 to align with default geom dataset configuration
    
    
    # Store the object to a file using pickle
    # geom_data_object_pkl = f'/Users/gohyixian/Documents/GitHub/FYP/GeoLDM-edit/data_EDA/data_object_cache/GEOM_data_object.pkl'
    geom_data_object_pkl = f'/Users/gohyixian/Documents/GitHub/FYP/GeoLDM-edit/data_EDA/data_object_cache/GEOM_C30_data_object.pkl'
    
    with open(geom_data_object_pkl, 'wb') as file:  # Use 'wb' mode for binary writing
        pickle.dump(geom_data_obj, file)
    
    del geom_data_obj

    # Load the object back from the file
    with open(geom_data_object_pkl, 'rb') as file:  # Use 'rb' mode for binary reading
        geom_data_obj = pickle.load(file)
    
    attributes_values = vars(geom_data_obj)
    for attr, value in attributes_values.items():
        print(f"{attr}: {value}")