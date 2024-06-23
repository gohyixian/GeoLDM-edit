import periodictable
# https://periodictable.readthedocs.io/en/latest/guide/using.html
# https://www.geeksforgeeks.org/get-the-details-of-an-element-by-atomic-number-using-python/
from Bio.Data import IUPACData
from Bio.PDB.Polypeptide import three_to_index, index_to_one, is_aa


def get_periodictable_list(include_aa=False):
    atom_num = []
    symbol = []
    
    if include_aa:
        for three, one in IUPACData.protein_letters_3to1.items():
            atom_num.append(int(three_to_index(three.upper())) - 20)
            symbol.append(three.upper())

    for element in periodictable.elements:
        atom_num.append(int(element.number))
        symbol.append(str(element.symbol))

    an2s = dict(zip(atom_num, symbol))
    s2an = dict(zip(symbol, atom_num))
    
    return an2s, s2an



# BindingMOAD & Cross-Docked (DiffSBDD)
# https://en.wikipedia.org/wiki/Covalent_radius#Radii_for_multiple_bonds
# (2022/08/14)
covalent_radii = {'H': 32, 'C': 60, 'N': 54, 'O': 53, 'F': 53, 'B': 73,
                  'Al': 111, 'Si': 102, 'P': 94, 'S': 94, 'Cl': 93, 'As': 106,
                  'Br': 109, 'I': 125, 'Hg': 133, 'Bi': 135}

# ------------------------------------------------------------------------------
# Dataset-specific constants
# ------------------------------------------------------------------------------
dataset_params = {}
dataset_params['bindingmoad'] = {  # 'H' included
    'atom_encoder': {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9, 'H':10},
    'atom_decoder': ['C', 'N', 'O', 'S', 'B', 'Br', 'Cl', 'P', 'I', 'F', 'H'],
    'aa_encoder': {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19},
    'aa_decoder': ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
}

dataset_params['crossdock_full'] = {  # 'H' included
      'atom_encoder': {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9, 'others': 10, 'H':11},
      'atom_decoder': ['C', 'N', 'O', 'S', 'B', 'Br', 'Cl', 'P', 'I', 'F', 'others', 'H'],
      'aa_encoder': {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9, 'others': 10, 'H':11},
      'aa_decoder': ['C', 'N', 'O', 'S', 'B', 'Br', 'Cl', 'P', 'I', 'F', 'others', 'H'],
}

dataset_params['crossdock'] = {  # 'H' included
      'atom_encoder': {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9, 'H':10},
      'atom_decoder': ['C', 'N', 'O', 'S', 'B', 'Br', 'Cl', 'P', 'I', 'F', 'H'],
      'aa_encoder': {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19},
      'aa_decoder': ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
}
