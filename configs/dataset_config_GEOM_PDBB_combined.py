import pickle
from configs.constants_colors_radius import get_radius, get_colors


# too large to put in script
with open('configs/n_nodes__d_20240428_combined_geom_PDBB_full_refined_core_LG_PKT.pkl', 'rb') as file:
    d_20240428_combined_geom_PDBB_full_refined_core_LG_PKT_n_nodes_dict = pickle.load(file)

d_20240428_combined_geom_PDBB_full_refined_core_LG_PKT_atom_decoder = \
    ['H', 'Li', 'Be',  'B',  'C',  'N',  'O',  'F', 'Na', 'Mg',  'Al', 'Si', 'P',  'S',  'Cl', 'K',  'Ca', 'V',  
     'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',  'As', 'Se',  'Br', 'Rb', 'Sr', 'Ru', 'Rh', 'Cd', 'In', 'Sb', 'Te',  
     'I',  'Cs', 'Ba', 'Re', 'Os', 'Ir', 'Pt', 'Au',  'Hg',  'Bi', 'Og']

d_20240428_combined_geom_PDBB_full_refined_core_LG_PKT = {
    'name': 'combined_geom_PDBB_full_refined_core_LG_PKT',
    'atom_encoder': {'H': 0, 'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Na': 8, 'Mg': 9, 'Al': 10, 'Si': 11, 
                     'P': 12, 'S': 13, 'Cl': 14, 'K': 15, 'Ca': 16, 'V': 17, 'Mn': 18, 'Fe': 19, 'Co': 20, 'Ni': 21, 'Cu': 22, 
                     'Zn': 23, 'Ga': 24, 'As': 25, 'Se': 26, 'Br': 27, 'Rb': 28, 'Sr': 29, 'Ru': 30, 'Rh': 31, 'Cd': 32, 'In': 33, 
                     'Sb': 34, 'Te': 35, 'I': 36, 'Cs': 37, 'Ba': 38, 'Re': 39, 'Os': 40, 'Ir': 41, 'Pt': 42, 'Au': 43, 'Hg': 44, 
                     'Bi': 45, 'Og': 46},
    'atomic_nb': [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 23, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 37, 38, 
                  44, 45, 48, 49, 51, 52, 53, 55, 56, 75, 76, 77, 78, 79, 80, 83, 118],
    'atom_decoder': d_20240428_combined_geom_PDBB_full_refined_core_LG_PKT_atom_decoder,
    'max_n_nodes': 75944,
    'n_nodes': d_20240428_combined_geom_PDBB_full_refined_core_LG_PKT_n_nodes_dict,
    'atom_types': {0: 53453229, 1: 14, 2: 1, 3: 278, 4: 50498410, 5: 12954304, 6: 18703098, 7: 69713, 8: 1398, 9: 3795, 10: 1, 
                   11: 17, 12: 6188, 13: 515574, 14: 56434, 15: 474, 16: 3759, 17: 2, 18: 1126, 19: 276, 20: 218, 21: 278, 22: 65, 
                   23: 5321, 24: 1, 25: 6, 26: 1482, 27: 14585, 28: 3, 29: 2, 30: 16, 31: 2, 32: 506, 33: 1, 34: 1, 35: 1, 36: 417, 
                   37: 15, 38: 1, 39: 1, 40: 1, 41: 6, 42: 2, 43: 6, 44: 95, 45: 3, 46: 2},
    'colors_dic': [get_colors(f) for f in d_20240428_combined_geom_PDBB_full_refined_core_LG_PKT_atom_decoder],
    'radius_dic': [get_radius(f) for f in d_20240428_combined_geom_PDBB_full_refined_core_LG_PKT_atom_decoder],
    'with_h': True
}