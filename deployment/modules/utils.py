# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass

import os
import torch
import periodictable
from Bio.Data import IUPACData
from Bio.PDB.Polypeptide import three_to_index

from analysis.molecule_builder import build_molecule


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


def save_mols_to_sdf(
    molecule_list, 
    dataset_info, 
    filenames
) -> list[str]:
    
    one_hot = molecule_list['one_hot']
    x = molecule_list['x']
    node_mask = molecule_list['node_mask']

    if isinstance(node_mask, torch.Tensor):
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [torch.sum(m) for m in node_mask]

    n_samples = len(x)

    processed_list = []
    for i in range(n_samples):
        atom_type = one_hot[i].argmax(1).cpu().detach()
        pos = x[i].cpu().detach()

        atom_type = atom_type[0:int(atomsxmol[i])]
        pos = pos[0:int(atomsxmol[i])]
        processed_list.append((pos, atom_type))
    
    assert len(processed_list) == len(filenames)
    
    saved_mol_filenames = []
    for i, (pos, atom_type) in enumerate(processed_list):
        try:
            # build RDKit molecule
            mol = build_molecule(pos, atom_type, dataset_info)
        except Exception as e:
            print(f"Failed to build molecule: {e}")
            continue
        
        if mol is not None:
            save_path = str(filenames[i])
            folder_path = os.path.dirname(save_path)
            
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            with Chem.SDWriter(save_path) as writer:
                writer.write(mol)
            saved_mol_filenames.append(save_path)
    
    return saved_mol_filenames