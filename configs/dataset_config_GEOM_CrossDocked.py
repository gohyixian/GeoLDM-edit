import pickle
from configs.constants_colors_radius import get_radius, get_colors



# for LDM, ControlNet
# This config is specially curated for the ControlNet trained with pretrained GeoLDM weights provided by the authors
# All configs are essentially the same as CrossDocked, except for the below, that are borrowed from the GEOM_with_h dataset
#  - name
#  - atom_encoder
#  - atom_decoder
#  - atomic_nb
#  - atom_types

d_20241115_GEOM_LDM_CrossDocked_LG_PKT_MMseq2_split__10A__LIGAND_atom_decoder = \
    ['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi']
    # ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl']

d_20241115_GEOM_LDM_CrossDocked_LG_PKT_MMseq2_split__10A__LIGAND = {
    'name': 'd_20241115_GEOM_LDM_CrossDocked_LG_PKT_MMseq2_split__10A__LIGAND',
    # 'name': 'd_20240623_CrossDocked_LG_PKT__10A__LIGAND',
    
    'atom_encoder': {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7, 'P': 8, 'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15},
    # 'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5, 'S': 6, 'Cl': 7},
    
    'atomic_nb': [1,  5,  6,  7,  8,  9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83],
    # 'atomic_nb': [1, 6, 7, 8, 9, 15, 16, 17],
    
    'atom_decoder': d_20241115_GEOM_LDM_CrossDocked_LG_PKT_MMseq2_split__10A__LIGAND_atom_decoder,
    'max_n_nodes': 106,
    'n_nodes': {3: 4, 4: 45, 5: 111, 6: 709, 7: 655, 8: 1043, 9: 1861, 10: 2884, 11: 3104, 12: 2910, 13: 2167, 14: 2439, 15: 2301, 16: 3117, 17: 2384, 18: 2611, 19: 3118, 20: 4890, 21: 4488, 22: 3356, 23: 4446, 24: 3814, 25: 4593, 26: 4770, 27: 5496, 28: 4772, 29: 3143, 30: 3582, 31: 3383, 32: 3312, 33: 2658, 34: 2162, 35: 1852, 36: 1357, 37: 927, 38: 919, 39: 739, 40: 674, 41: 607, 42: 487, 43: 360, 44: 673, 45: 184, 46: 212, 47: 105, 48: 186, 49: 73, 50: 38, 51: 20, 52: 42, 53: 3, 54: 93, 55: 35, 56: 12, 57: 9, 58: 18, 59: 5, 61: 9, 62: 1, 63: 5, 65: 2, 66: 20, 67: 28, 68: 1, 71: 1, 77: 1, 86: 1, 98: 1, 106: 2},
    
    'atom_types': {0: 18, 2: 1585352, 3: 276495, 4: 400601, 5: 30875, 8: 26331, 9: 26552, 10: 15221,
                   1:0, 6:0, 7:0, 11:0, 12:0, 13:0, 14:0, 15:0},
    # 'atom_types': {0: 18, 1: 1585352, 2: 276495, 3: 400601, 4: 30875, 5: 26331, 6: 26552, 7: 15221},
    
    'colors_dic': [get_colors(f) for f in d_20241115_GEOM_LDM_CrossDocked_LG_PKT_MMseq2_split__10A__LIGAND_atom_decoder],
    'radius_dic': [get_radius(f) for f in d_20241115_GEOM_LDM_CrossDocked_LG_PKT_MMseq2_split__10A__LIGAND_atom_decoder],
    'with_h': True
}


# index (idx) according to GEOM
# count according to CrossDocked
# 
#      idx  count
# H  - 0  - 18
# C  - 2  - 1585352
# N  - 3  - 276495
# O  - 4  - 400601
# F  - 5  - 30875
# P  - 8  - 26331
# S  - 9  - 26552
# Cl - 10 - 15221

