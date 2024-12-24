try:
    from rdkit import Chem
    from qm9.rdkit_functions import BasicMolecularMetrics
    use_rdkit = True
except ModuleNotFoundError:
    use_rdkit = False

import os
import copy
import glob
import torch
import subprocess
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sp_stats
from Bio.PDB import PDBParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qm9 import bond_analyze
import qm9.dataset as dataset
from analysis.metrics import rdmol_to_smiles

# code borrowed from DiffSBDD
# https://github.com/arneschneuing/DiffSBDD/tree/main/analysis
from analysis.metrics import BasicMolecularMetrics as DiffSBDD_MolecularMetrics
from analysis.metrics import MoleculeProperties
from analysis.molecule_builder import build_molecule



def get_pocket_center(pdb_file):
    """Calculate the geometric center of a pocket from a PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pocket", pdb_file)
    coordinates = []

    # Extract atomic coordinates
    for atom in structure.get_atoms():
        coordinates.append(atom.coord)

    # Calculate the geometric center
    coordinates = np.array(coordinates)
    center = np.mean(coordinates, axis=0)
    print(f">>> Pocket center: {center}")
    
    return center

def center_ligand(ligand):
    """Center the ligand coordinates around (0, 0, 0)."""
    conf = ligand.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(ligand.GetNumAtoms())])

    # Calculate ligand's center
    ligand_center = np.mean(coords, axis=0)
    print(f">>> Ligand center beforehand: {ligand_center}")

    # Translate coordinates to center the ligand at (0, 0, 0)
    for i in range(ligand.GetNumAtoms()):
        conf.SetAtomPosition(i, coords[i] - ligand_center)
    
    return ligand, ligand_center

def translate_ligand_to_pocket_center(ligand, pocket_center):
    """Translate the ligand to align its center with the pocket center and display the new center."""
    # Get the conformer (3D coordinates) of the ligand
    conf = ligand.GetConformer()
    
    # Iterate over each atom in the ligand
    for i in range(ligand.GetNumAtoms()):
        # Get the current position of the atom
        current_pos = np.array(conf.GetAtomPosition(i))
        
        # Translate the atom position by adding the pocket center (move the ligand)
        conf.SetAtomPosition(i, current_pos + pocket_center)

    # Calculate the new center of the ligand after translation
    new_center = np.mean([np.array(conf.GetAtomPosition(i)) for i in range(ligand.GetNumAtoms())], axis=0)
    
    # Display the new center (xyz coordinates)
    print(f">>> Ligand center after: {new_center}")
    
    # Return the modified ligand with updated positions
    return ligand


def compute_qvina2_score(
        dir: str = "",
        pocket_pdb_dir="", 
        output_dir="",
        mgltools_env_name="mgltools-python2",
        ligand_add_H=False,
        receptor_add_H=False,
        remove_nonstd_resi=False,
        size=20,
        exhaustiveness=16,
        seed=42,
        cleanup_files=True,
        save_csv=True
    ):
    
    sdf_files = sorted([f for f in os.listdir(dir) if f.endswith(".sdf")])
    
    rdmols = []
    rdmols_ids = []
    
    for sdf_file in sdf_files:
        file = Path(dir, sdf_file)
        # Load the single molecule from the SDF file
        tmp_mol = Chem.SDMolSupplier(str(file), sanitize=False)[0]

        # Build new molecule. This is a workaround to remove radicals.
        mol = Chem.RWMol()
        for atom in tmp_mol.GetAtoms():
            mol.AddAtom(Chem.Atom(atom.GetSymbol()))
        mol.AddConformer(tmp_mol.GetConformer(0))

        for bond in tmp_mol.GetBonds():
            mol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                        bond.GetBondType())

        rdmols.append(mol)
        rdmols_ids.append(int(Path(sdf_file).stem))
    
    # compute qvina scores for each pocket-ligand pair
    scores = []
    results = {
        'receptor': [],
        'ligands': [],
        'scores': []
    }
    for i in tqdm(range(len(rdmols))):
        
        id = str(rdmols_ids[i]).zfill(7)
        
        pattern = os.path.join(pocket_pdb_dir, f'{id}*.pdb')
        pkt_file = glob.glob(pattern)
        assert len(pkt_file) <= 1
        
        if len(pkt_file) == 0:
            print(f">>> [qm9.analyze.compute_qvina_score] Pocket ID: {id} not found")
            continue
        
        lg_sdf_file   = Path(output_dir, f"{id}.sdf")
        lg_pdb_file   = Path(output_dir, f"{id}.pdb")
        lg_pdbqt_file = Path(output_dir, f"{id}.pdbqt")
        pkt_pdb_file  = Path(pkt_file[0])
        pkt_pdbqt_file = Path(output_dir, f"{pkt_pdb_file.stem}.pdbqt")
        qvina_out_file = Path(output_dir, f"{id}_qvina.txt")
        
        
        # Move ligand's center (x,y,z) to pocket's center (x,y,z) so that it falls in 
        # the bounding box search space in order for qvina2.1 to work
        pocket_center = get_pocket_center(str(pkt_pdb_file))
        # Center ligand around (0, 0, 0) first
        mol_trans = copy.deepcopy(rdmols[i])
        mol_trans, ligand_original_center = center_ligand(mol_trans)
        # Translate ligand to the pocket center
        mol_trans = translate_ligand_to_pocket_center(mol_trans, pocket_center)
        pocket_center = pocket_center.tolist()
        cx, cy, cz = float(pocket_center[0]), float(pocket_center[1]), float(pocket_center[2])
        
        
        # LG: .sdf
        with Chem.SDWriter(str(lg_sdf_file)) as writer:
            writer.write(mol_trans)
            # writer.write(rdmols[i])
        
        # LG: .pdb
        os.popen(f'obabel {lg_sdf_file} -O {lg_pdb_file}').read()
        print(f">>> {os.path.exists(lg_pdb_file)}")
        print(lg_pdb_file)
        
        # LG: .pdbqt (add charges and torsions)
        cd_cmd = f"cd {os.path.dirname(lg_pdb_file)}"
        prep_lg_cmd = f"{cd_cmd} && conda run -n {mgltools_env_name} prepare_ligand4.py -l {os.path.basename(lg_pdb_file)} -o {os.path.basename(lg_pdbqt_file)}"
        prep_lg_cmd += " -A hydrogens" if ligand_add_H else ""
        subprocess.run(prep_lg_cmd, shell=True)
        
        # PKT: .pdbqt
        prep_pkt_cmd = f"conda run -n {mgltools_env_name} prepare_receptor4.py -r {pkt_pdb_file} -o {pkt_pdbqt_file}"
        prep_pkt_cmd += " -A checkhydrogens" if receptor_add_H else ""
        prep_pkt_cmd += " -e" if remove_nonstd_resi else ""
        subprocess.run(prep_pkt_cmd, shell=True)

        # # center box at ligand's center of mass 
        # # only applicable to DiffSBDD as the model directly positions the ligands in the pockets
        # cx, cy, cz = rdmols[i].GetConformer().GetPositions().mean(0)

        # run QuickVina 2
        out = os.popen(
            f'./analysis/qvina/qvina2.1 --receptor {pkt_pdbqt_file} '
            f'--ligand {lg_pdbqt_file} '
            f'--center_x {cx:.4f} --center_y {cy:.4f} --center_z {cz:.4f} '
            f'--size_x {size} --size_y {size} --size_z {size} '
            f'--exhaustiveness {exhaustiveness}'
            f'--seed {seed}'
        ).read()
        print(out)
        with open(str(qvina_out_file), 'w') as f:
            print(out, file=f)

        if '-----+------------+----------+----------' not in out:
            scores.append(np.nan)
            continue

        out_split = out.splitlines()
        best_idx = out_split.index('-----+------------+----------+----------') + 1
        best_line = out_split[best_idx].split()
        assert best_line[0] == '1'
        scores.append(float(best_line[1]))

        results['receptor'].append(str(pkt_pdb_file))
        results['ligands'].append(str(lg_sdf_file))
        results['scores'].append(scores)
        
        # clean up
        if cleanup_files:
            lg_pdb_file.unlink()
            lg_pdbqt_file.unlink()
            pkt_pdbqt_file.unlink()
    
    filtered_scores = [score for score in scores if not np.isnan(score)]
    scores_average = sum(filtered_scores) / len(filtered_scores) if filtered_scores else float('nan')
    
    if save_csv:
        results_save = copy.deepcopy(results)
        results_save['receptor'].append("")
        results_save['ligands'].append("Mean Score")
        results_save['scores'].append([scores_average])
        df = pd.DataFrame.from_dict(results_save)
        df.to_csv(Path(output_dir, 'qvina2_scores.csv'))
    
    return {
        'mean': scores_average,
        'all': scores,
        'results': results
    }



if __name__ == '__main__':
    
    ligand_sdf_files_dir = "/mnt/c/Users/PC/Desktop/yixian/epoch_1_iter_3122/epoch_1_iter_3122"
    pocket_pdb_files_dir = "./data/d_20241203_CrossDocked_LG_PKT_MMseq2_split_CA_only/test_val_paired_files/val_pocket"
    output_dir = "/mnt/c/Users/PC/Desktop/yixian/epoch_1_iter_3122/test_output"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    results = compute_qvina2_score(
        dir=ligand_sdf_files_dir,
        pocket_pdb_dir=pocket_pdb_files_dir, 
        output_dir=output_dir,
        mgltools_env_name="mgltools-python2",
        ligand_add_H=True,
        receptor_add_H=True,
        remove_nonstd_resi=False,
        size=20,
        exhaustiveness=16,
        seed=42,
        cleanup_files=False,
        save_csv=True
    )
    
    print(results)

    not_nan = [n for n in results['all'] if not np.isnan(n)]
    print(not_nan)
    print(len(not_nan))