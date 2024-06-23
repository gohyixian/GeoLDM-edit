from tqdm import tqdm
import numpy as np
import os
import pickle
from constants import get_periodictable_list
from utils import euclidean_distance



class QM9:
    def __init__(self):
        self.atomic_num_freq = dict()   # atomic_num: freq
        self.num_atoms = []
        self.num_atoms_no_H = []
        self.mol_count = 0
        
        # molecule radius in Angstroms from center
        self.radius_mean = []
        self.radius_min = []
        self.radius_max = []
        
        # properties
        #  1  tag       -            "gdb9"; string constant to ease extraction via grep
        #  2  index     -            Consecutive, 1-based integer identifier of molecule
        #  3  A         GHz          Rotational constant A
        #  4  B         GHz          Rotational constant B
        #  5  C         GHz          Rotational constant C
        self.mu = []      # 6   Debye        Dipole moment
        self.alpha = []   # 7   Bohr^3       Isotropic polarizability
        self.homo = []    # 8   Hartree      Energy of Highest occupied molecular orbital (HOMO)
        self.lumo = []    # 9   Hartree      Energy of Lowest occupied molecular orbital (LUMO)
        self.gap = []     # 10  Hartree      Gap, difference between LUMO and HOMO
        self.r2 = []      # 11  Bohr^2       Electronic spatial extent
        self.zpve = []    # 12  Hartree      Zero point vibrational energy
        self.U0 = []      # 13  Hartree      Internal energy at 0 K
        self.U = []       # 14  Hartree      Internal energy at 298.15 K
        self.H = []       # 15  Hartree      Enthalpy at 298.15 K
        self.G = []       # 16  Hartree      Free energy at 298.15 K
        self.Cv = []      # 17  cal/(mol K)  Heat capacity at 298.15 K



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
        
        mol_props = xyz_lines[1].split()
        mol_xyz = xyz_lines[2:num_atoms+2]

        atom_charges, atom_positions = [], []
        for line in mol_xyz:
            atom, posx, posy, posz, _ = line.replace('*^', 'e').split()
            
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


        # properties
        prop_strings = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
        prop_strings = prop_strings[1:]
        mol_props = [int(mol_props[1])] + [float(x) for x in mol_props[2:]]
        mol_props = dict(zip(prop_strings, mol_props))
        self.mu.append(mol_props['mu'])
        self.alpha.append(mol_props['alpha'])
        self.homo.append(mol_props['homo'])
        self.lumo.append(mol_props['lumo'])
        self.gap.append(mol_props['gap'])
        self.r2.append(mol_props['r2'])
        self.zpve.append(mol_props['zpve'])
        self.U0.append(mol_props['U0'])
        self.U.append(mol_props['U'])
        self.H.append(mol_props['H'])
        self.G.append(mol_props['G'])
        self.Cv.append(mol_props['Cv'])




if __name__ == "__main__":
    path_to_qm9 = "/Users/gohyixian/Documents/Documents/3.2_FYP_1/data/QM9/QM9_113885_GDB-9_molecules_xyz"
    files_to_omit = ['.DS_Store']   # mac
    files = sorted([os.path.join(path_to_qm9, f) for f in os.listdir(path_to_qm9) if f not in files_to_omit])
    
    qm9_data_obj = QM9()
    
    for f in tqdm(files):
        with open(f, 'r') as openfile:
            qm9_data_obj.process_xyz_gdb9(openfile)
    
    # Store the object to a file using pickle
    qm9_data_object_pkl = '/Users/gohyixian/Documents/GitHub/FYP/GeoLDM-edit/data_EDA/data_object_cache/QM9_data_object.pkl'
    with open(qm9_data_object_pkl, 'wb') as file:  # Use 'wb' mode for binary writing
        pickle.dump(qm9_data_obj, file)
    
    del qm9_data_obj

    # Load the object back from the file
    with open(qm9_data_object_pkl, 'rb') as file:  # Use 'rb' mode for binary reading
        qm9_data_obj = pickle.load(file)
    
    attributes_values = vars(qm9_data_obj)
    for attr, value in attributes_values.items():
        print(f"{attr}: {value}")
    
