from tqdm import tqdm
import numpy as np
import os
import pickle
from constants import get_periodictable_list
from utils import euclidean_distance



class PDBBIND:
    def __init__(self):
        self.atomic_num_freq = dict()   # atomic_num: freq
        self.num_atoms = []
        self.num_atoms_no_H = []
        self.mol_count = 0
        
        # molecule radius in Angstroms from center
        self.radius_mean = []
        self.radius_min = []
        self.radius_max = []


    def process_xyz_gdb9(self, datafile):
        """
        Parameters
        ----------
        datafile : python file object
            File object containing the molecular data in the MD17 dataset.
        """
        # xyz_lines = [line.encode('UTF-8') for line in datafile.readlines()]
        xyz_lines = [line for line in datafile]
        
        an2s, s2an = get_periodictable_list()

        num_atoms = int(xyz_lines[0])
        self.num_atoms.append(num_atoms)
        self.mol_count += 1
        
        mol_xyz = xyz_lines[2:num_atoms+2]

        atom_charges, atom_positions = [], []
        for line in mol_xyz:
            atom, posx, posy, posz = line.replace('*^', 'e').split()
            
            atom_charges.append(s2an[atom])
            atom_positions.append([float(posx), float(posy), float(posz)])
        
        atom_charges_np = np.array(atom_charges)
        num_atoms_no_H = len(atom_charges_np[atom_charges_np > 1])
        self.num_atoms_no_H.append(num_atoms_no_H)
        
        for ac in atom_charges:
            self.atomic_num_freq[int(ac)] = int(self.atomic_num_freq.get(int(ac), 0) + 1)
        
        # molecule's coordinate center
        xyz = np.array(atom_positions)
        center_of_coordinates = np.mean(xyz, axis=0)
        distances_to_center = np.array([euclidean_distance(atom, center_of_coordinates) for atom in xyz])
        self.radius_mean.append(float(np.mean(distances_to_center)))
        self.radius_min.append(float(np.min(distances_to_center)))
        self.radius_max.append(float(np.max(distances_to_center)))




if __name__ == "__main__":
    path_to_pdbbind = "/Users/gohyixian/Documents/Documents/3.2_FYP_1/data/PDBBind2020/refined-set-xyz"
    files_to_omit = ['.DS_Store']   # mac
    ligand_pocket_pair_folders = sorted([os.path.join(path_to_pdbbind, f) for f in os.listdir(path_to_pdbbind) if f not in files_to_omit])
    
    
    # mode = 'ligand'
    # mode = 'pocket'
    mode = 'protein'
    pdbbind_data_obj = PDBBIND()
    for folder in tqdm(ligand_pocket_pair_folders):
        files = [s for s in sorted(os.listdir(folder)) if mode in s]
        f = os.path.join(folder, files[0])
        with open(f, 'r') as openfile:
            pdbbind_data_obj.process_xyz_gdb9(openfile)
    
    # Store the object to a file using pickle
    pdbbind_data_object_pkl = f'/Users/gohyixian/Documents/GitHub/FYP/GeoLDM-edit/data_EDA/data_object_cache/PDBBIND_{mode.upper()}_data_object.pkl'
    with open(pdbbind_data_object_pkl, 'wb') as file:  # Use 'wb' mode for binary writing
        pickle.dump(pdbbind_data_obj, file)
    del pdbbind_data_obj

    # Load the object back from the file
    with open(pdbbind_data_object_pkl, 'rb') as file:  # Use 'rb' mode for binary reading
        qm9_data_obj = pickle.load(file)
    
    attributes_values = vars(qm9_data_obj)
    for attr, value in attributes_values.items():
        print(f"{attr}: {value}")
    
