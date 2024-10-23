try:
    from rdkit import Chem
    from qm9.rdkit_functions import BasicMolecularMetrics
    use_rdkit = True
except ModuleNotFoundError:
    use_rdkit = False
import torch
import numpy as np

# code borrowed from DiffSBDD
# https://github.com/arneschneuing/DiffSBDD/tree/main/analysis
from analysis.metrics import BasicMolecularMetrics as DiffSBDD_MolecularMetrics
from analysis.metrics import MoleculeProperties
from analysis.molecule_builder import build_molecule

from qm9.analyze import check_stability


def compute_molecule_metrics(one_hot, x, dataset_info):

    n_samples = len(x)

    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0

    processed_list = []

    for i in range(n_samples):
        atom_type = one_hot[i].cpu().detach()
        pos = x[i].cpu().detach()

        # atom_type = atom_type[0:int(atomsxmol[i])]
        # pos = pos[0:int(atomsxmol[i])]
        processed_list.append((pos, atom_type))

    for mol in processed_list:
        pos, atom_type = mol
        validity_results = check_stability(pos, atom_type, dataset_info)

        molecule_stable += int(validity_results[0])
        nr_stable_bonds += int(validity_results[1])
        n_atoms += int(validity_results[2])

    # Stability
    fraction_mol_stable = molecule_stable / float(n_samples)
    fraction_atm_stable = nr_stable_bonds / float(n_atoms)
    
    if use_rdkit:
        # validity, uniquness, novelty
        metrics = BasicMolecularMetrics(dataset_info)
        rdkit_metrics = metrics.evaluate(processed_list)
    else:
        rdkit_metrics = None
    
    # Other metrics referenced from DiffSBDD
    # convert into rdmols
    rdmols = [build_molecule(pos, atom_type, dataset_info) \
              for (pos, atom_type) in processed_list]
    # won't be computing novelty & uniqueness with 
    # this, no need for dataset SMILES list.
    ligand_metrics = DiffSBDD_MolecularMetrics(dataset_info, dataset_smiles_list=None)
    molecule_properties = MoleculeProperties()
    
    # filter valid molecules
    valid_mols, _ = ligand_metrics.compute_validity(rdmols)
    
    # compute connectivity
    connected_mols, connectivity, _ = \
            ligand_metrics.compute_connectivity(valid_mols)
    
    # other basic metrics
    # qed, sa, logp, lipinski, diversity = \
    #     molecule_properties.evaluate_mean(connected_mols)

    if len(connected_mols) < 1:
        qed, sa, logp, lipinski = 0.0, 0.0, 0.0, 0.0
    else:
        for mol in connected_mols:
            Chem.SanitizeMol(mol)
            assert mol is not None, "only evaluate valid molecules"

        qed = np.mean([molecule_properties.calculate_qed(mol) for mol in connected_mols])
        sa = np.mean([molecule_properties.calculate_sa(mol) for mol in connected_mols])
        logp = np.mean([molecule_properties.calculate_logp(mol) for mol in connected_mols])
        lipinski = np.mean([molecule_properties.calculate_lipinski(mol) for mol in connected_mols])

    metrics_dict = {
        'validity': rdkit_metrics[0][0] if use_rdkit else None,
        'uniqueness': rdkit_metrics[0][1] if use_rdkit else None,
        'novelty': rdkit_metrics[0][2] if use_rdkit else None,
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
        'connectivity': connectivity,
        'QED': qed,
        'SA': sa,
        'logP': logp,
        'lipinski': lipinski
    }
    
    return metrics_dict