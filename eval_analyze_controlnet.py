# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass

import os
import re
import time
import math
import copy
import torch
import pickle
import random
import argparse
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import periodictable
from Bio.PDB import PDBParser
from Bio.Data import IUPACData
from Bio.PDB.Polypeptide import three_to_index, is_aa

import utils
import build_geom_dataset
from analysis.metrics import rdmol_to_smiles
from analysis.molecule_builder import build_molecule
from configs.dataset_configs.datasets_config import get_dataset_info

from qm9.sampling import sample_controlnet
from qm9.models import get_controlled_latent_diffusion
from qm9.analyze import compute_molecule_metrics, get_pocket_center, translate_ligand_to_pocket_center, center_ligand

from global_registry import PARAM_REGISTRY, Config




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



def compute_qvina2_score(
        molecule_list, 
        dataset_info, 
        pocket_filenames=[], 
        pocket_pdb_dir="", 
        output_dir="",
        mgltools_env_name="mgltools-python2",
        connectivity_thres=1.,
        ligand_add_H=False,
        receptor_add_H=False,
        remove_nonstd_resi=False,
        size=20,
        exhaustiveness=16,
        seed=42,
        cleanup_files=True,
        save_csv=True
    ):
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
    
    # filter molecules
    rdmols = []
    rdmols_filenames = []
    for i, (pos, atom_type) in enumerate(processed_list):
        try:
            # build RDKit molecule
            mol = build_molecule(pos, atom_type, dataset_info)
        except Exception as e:
            print(f"Failed to build molecule: {e}")
            continue
        
        if mol is not None:
            # filter valid molecules
            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                continue
            
            if mol is not None:
                # filter connected molecules
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                largest_mol = \
                    max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                if largest_mol.GetNumAtoms() / mol.GetNumAtoms() >= connectivity_thres:
                    smiles = rdmol_to_smiles(largest_mol)
                    if smiles is not None:
                        rdmols.append(largest_mol)
                        rdmols_filenames.append(pocket_filenames[i])
    
    # compute qvina scores for each pocket-ligand pair
    scores = []
    results = {
        'receptor': [],
        'ligands': [],
        'scores': []
    }
    for i in tqdm(range(len(rdmols))):
        
        id = str(rdmols_filenames[i])
        
        pkt_file = os.path.join(pocket_pdb_dir, f'{id}.pdb')
        
        if not os.path.exists(pkt_file):
            print(f">>> [qm9.analyze.compute_qvina_score] Pocket File: {pkt_file} not found")
            continue
        
        unique_id = str(time.time_ns())
        lg_sdf_file    = Path(output_dir, id, f"{unique_id}_LG.sdf")
        lg_pdb_file    = Path(output_dir, id, f"{unique_id}_LG.pdb")
        lg_pdbqt_file  = Path(output_dir, id, f"{unique_id}_LG.pdbqt")
        pkt_pdb_file   = Path(pkt_file)
        pkt_pdbqt_file = Path(output_dir, id, f"{unique_id}_PKT.pdbqt")
        qvina_out_file = Path(output_dir, id, f"{unique_id}_Qvina.txt")
        
        
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

        # run QuickVina 2
        qvina_cmd = \
            f'./analysis/qvina/qvina2.1 --receptor {pkt_pdbqt_file} ' + \
            f'--ligand {lg_pdbqt_file} ' + \
            f'--center_x {cx:.4f} --center_y {cy:.4f} --center_z {cz:.4f} ' + \
            f'--size_x {size} --size_y {size} --size_z {size} ' + \
            f'--exhaustiveness {exhaustiveness} ' + \
            f'--seed {seed}'
        print(qvina_cmd)
        out = os.popen(qvina_cmd).read()
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
        best_score = float(best_line[1])
        scores.append(best_score)

        results['receptor'].append(str(pkt_pdb_file))
        results['ligands'].append(str(lg_sdf_file))
        results['scores'].append(best_score)
        
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


def analyze_and_save_controlnet(
    model, 
    nodes_dist, 
    nodes_dist_delta,
    args, 
    device, 
    dataset_info,
    batch_size=100, 
    pocket_data_list=[], 
    pocket_id_lists=[],
    pocket_pdb_dir="",
    output_dir="",
    disable_qvina=False,
    mgltools_env_name="mgltools-python2",
    connectivity_thres=1.,
    ligand_add_H=False,
    receptor_add_H=False,
    remove_nonstd_resi=False,
    qvina_size=20,
    qvina_exhaustiveness=16,
    qvina_seed=42,
    qvina_cleanup_files=False
):
    n_samples = len(pocket_data_list)
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    batch_id = 0
    for _ in tqdm(range(math.ceil(n_samples/batch_size))):
        
        current_batch_size = min(batch_size, n_samples - batch_id)
        
        # this returns the number of nodes. i.e. n_samples=3, return=tensor([16, 17, 15]) / tensor([14, 15, 19]) / tensor([17, 27, 18])
        nodesxsample = nodes_dist.sample(current_batch_size)
        
        # apply adjustment to the sampled number of atoms
        nodesxsample += nodes_dist_delta

        pocket_dict_list = []
        for j in range(current_batch_size):
            pocket_dict_list.append(pocket_data_list[j + batch_id])

        one_hot, charges, x, node_mask = \
            sample_controlnet(
                args, 
                device, 
                model, 
                dataset_info,
                nodesxsample=nodesxsample, 
                context=None, 
                fix_noise=False, 
                pocket_dict_list=pocket_dict_list
            )

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())
        batch_id += current_batch_size

    assert len(pocket_id_lists) == batch_id
    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}

    metrics_dict = compute_molecule_metrics(molecules, dataset_info)
    metrics_dict['num_samples_generated'] = batch_id
    
    if not disable_qvina:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        qvina_scores_dict = compute_qvina2_score(
            molecules, 
            dataset_info, 
            pocket_filenames=pocket_id_lists, 
            pocket_pdb_dir=pocket_pdb_dir, 
            output_dir=output_dir,
            mgltools_env_name=mgltools_env_name,
            connectivity_thres=connectivity_thres,
            ligand_add_H=ligand_add_H,
            receptor_add_H=receptor_add_H,
            remove_nonstd_resi=remove_nonstd_resi,
            size=qvina_size,
            exhaustiveness=qvina_exhaustiveness,
            seed=qvina_seed,
            cleanup_files=qvina_cleanup_files,
            save_csv=True
        )
        metrics_dict['Qvina2'] = qvina_scores_dict['mean']
        metrics_dict['Qvina2_Num_Docked'] = len([n for n in qvina_scores_dict['all'] if not np.isnan(n)])
        print(f"Qvina over {len(qvina_scores_dict['all'])} molecules: {qvina_scores_dict['mean']}")
    
    return metrics_dict


# python eval_analyze_controlnet.py --model_path outputs_selected/controlnet/03_latent2_nf256_ds1k_fusReplace_CA__epoch1k_bs60_lr1e-4_NoEMA__20241203__10A --load_last --pocket_pdb_dir /mnt/c/Users/PC/Desktop/test_dataset/test_val_paired_files/val_pocket --num_samples_per_pocket 1 --delta_num_atoms 5 --batch_size 2
# python eval_analyze_controlnet.py --model_path outputs_selected/controlnet/03_latent2_nf256_ds1k_fusReplace_CA__epoch1k_bs60_lr1e-4_NoEMA__20241203__10A --load_last --pocket_pdb_dir /mnt/c/Users/PC/Desktop/test_dataset/test_val_paired_files/val_pocket --num_samples_per_pocket 3 --delta_num_atoms 5 --batch_size 3 --connectivity_thres 0.7

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="outputs/edm_1",
                        help='Specify model path')
    parser.add_argument('--load_last', action='store_true', 
                        help='Load weights of model of last epoch')
    parser.add_argument('--pocket_pdb_dir', type=str, default='data/test_val_paired_files/val_pocket',
                        help='Directody of pocket pdb files')
    parser.add_argument('--num_samples_per_pocket', type=int, default=1,
                        help='Number of samples to generate & evaluate per pocket')
    parser.add_argument('--delta_num_atoms', type=int, default=0,
                        help='Correction for the sampled number of atoms to generate for a ligand')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Sampling batch size')
    parser.add_argument('--save_path', type=str, default='eval_controlnet',
                        help='Path to save results')
    
    parser.add_argument('--disable_qvina', action='store_true',
                        help="Don't compute Qvina2.1 docking score")
    parser.add_argument('--mgltools_env_name', type=str, default="mgltools-python2",
                        help='Name of Python 2 Conda Environment with MGL-Tools installed')
    parser.add_argument('--connectivity_thres', type=float, default=1.,
                        help='Size of largest fragment in molecule for the molecule to be considered connected')
    parser.add_argument('--ligand_add_H', action='store_true',
                        help='Adds Hydrogen atoms to ligands before docking with Qvina2.1')
    parser.add_argument('--receptor_add_H', action='store_true',
                        help='Adds Hydrogen atoms to pocket / receptor before docking with Qvina2.1')
    parser.add_argument('--remove_nonstd_resi', action='store_true',
                        help='Removes any non-standard pocket residues')
    parser.add_argument('--size', type=int, default=20,
                        help='Qvina2.1 docking space search size (all 3 axes) in Angstroms around ligand center')
    parser.add_argument('--exhaustiveness', type=int, default=16,
                        help='Qvina2.1 docking search exhaustiveness')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for PyTorch, Numpy, Random & Qvina2.1')
    parser.add_argument('--cleanup_files', action='store_true',
                        help='Cleanup Qvina2.1 temporary docking files')
    eval_args = parser.parse_args()


    assert os.path.exists(eval_args.model_path)
    assert os.path.exists(eval_args.pocket_pdb_dir)
    assert eval_args.num_samples_per_pocket > 0



    # Load pickled Config object
    with open(os.path.join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    # Create output dir
    output_dir = Path(eval_args.save_path, args.exp_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    eval_args.save_path = str(output_dir)

    # Set random seed
    torch.manual_seed(eval_args.seed)
    random.seed(eval_args.seed)
    np.random.seed(eval_args.seed)

    # Load pre-computed dataset configs
    ligand_dataset_info = get_dataset_info(dataset_name=args.dataset, remove_h=args.remove_h)
    pocket_dataset_info = get_dataset_info(dataset_name=args.pocket_vae.dataset, remove_h=args.pocket_vae.remove_h)

    # Set device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    args.device_ = "cuda" if args.cuda else "cpu"

    # Set dtype
    torch.set_default_dtype(args.dtype)

    # Add missing configs with default values
    args = utils.add_missing_configs_controlnet(args, args.dtype, ligand_dataset_info, pocket_dataset_info)

    # Create params global registry for easy access
    PARAM_REGISTRY.update_from_config(args)




    # Init model & load weights
    if args.training_mode == "ControlNet":
        model, nodes_dist, _ = get_controlled_latent_diffusion(args, args.device, ligand_dataset_info, pocket_dataset_info)
        model.to(device)
    else:
        raise NotImplementedError()

    if eval_args.load_last:
        if args.ema_decay > 0:
            pattern_str = r'generative_model_ema_(\d+)_iter_(\d+)\.npy'
        else:
            pattern_str = r'generative_model_(\d+)_iter_(\d+)\.npy'
            
        filtered_files = [f for f in os.listdir(eval_args.model_path) if re.compile(pattern_str).match(f)]
        filtered_files.sort(key=lambda x: (
            int(re.search(pattern_str, x).group(1)),
            int(re.search(pattern_str, x).group(2))
        ))
        fn = filtered_files[-1]
    else:
        fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    
    print(f">> Loading model weights from {os.path.join(eval_args.model_path, fn)}")
    state_dict = torch.load(os.path.join(eval_args.model_path, fn), map_location=device)
    model.load_state_dict(state_dict)


    # Model details logging
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes
    mem_mb, mem_gb = mem/(1024**2), mem/(1024**3)
    print(f"Model running on device  : {args.device}")
    print(f"Model running on dtype   : {args.dtype}")
    print(f"Model Size               : {mem_gb} GB  /  {mem_mb} MB  /  {mem} Bytes")
    print(f"Model Training Mode      : {args.training_mode}")
    print(f"Training Dataset Name    : {args.dataset}")
    print(f"Pocket CA Only           : {args.pocket_vae.ca_only}")
    print(f"Pocket Remove H          : {args.pocket_vae.remove_h}")
    print(f"================================")
    print(model)




    # Read pocket files to construct conditional pocket data
    # TODO: this script doesn't take into account filtering residues that are say 10 Angstroms away from the ligands
    pocket_data_list = []
    pocket_filename_list = []
    error_filename_list = []
    pdb_files = sorted([i for i in os.listdir(eval_args.pocket_pdb_dir) if i.endswith('.pdb')])

    for pdb_file in tqdm(pdb_files):

        full_path = os.path.join(eval_args.pocket_pdb_dir, pdb_file)
        pdb_struct = PDBParser(QUIET=True).get_structure('', full_path)

        an2s, s2an = get_periodictable_list(include_aa=True)

        pocket_residues = []
        for residue in pdb_struct[0].get_residues():
            if is_aa(residue.get_resname(), standard=True):
                pocket_residues.append(residue)

        pocket_atom_charge_positions = []
        if args.pocket_vae.ca_only:
            for res in pocket_residues:
                x, y, z = res['CA'].get_coord()
                pocket_atom_charge_positions.append([float(s2an[str(res.get_resname()).upper()]), float(x), float(y), float(z)])
        else:
            for res in pocket_residues:
                all_res_atoms = res.get_atoms()
                if args.pocket_vae.remove_h:
                    all_res_atoms = [a for a in all_res_atoms if a.element != 'H']
                for atom in all_res_atoms:
                    x, y, z = atom.get_coord()
                    pocket_atom_charge_positions.append([float(s2an[atom.element]), float(x), float(y), float(z)])
        pocket_atom_charge_positions = np.array(pocket_atom_charge_positions)

        processed = False
        if len(list(pocket_atom_charge_positions.shape)) == 2:
            if pocket_atom_charge_positions.shape[0] > 0 and pocket_atom_charge_positions.shape[1] == 4:
                pocket_data_list.append(pocket_atom_charge_positions)
                pocket_filename_list.append(Path(pdb_file).stem)
                processed = True
        if not processed:
            print(f"Error pocessing pocket file: {pdb_file}")
            error_filename_list.append(pdb_file)

    print(f">> Read {len(pdb_files)} pocket files, got {len(error_filename_list)} errors while procesing, resulting in {len(pocket_data_list)} processed pockets.")


    # ['positions'], ['one_hot'], ['charges'], ['atom_mask'] are added here
    pocket_transform = build_geom_dataset.GeomDrugsTransform(pocket_dataset_info, args.pocket_vae.include_charges, args.device, args.sequential)


    # Further processing of pocket data for compatible data structure
    processed_pocket_id = []
    processed_pocket_data_list = []
    for i in range(len(pocket_data_list)):
        pocket_data = pocket_data_list[i]
        pocket_filename = pocket_filename_list[i]

        for _ in range(eval_args.num_samples_per_pocket):
            # apply pocket transform
            transformed_pocket_data = pocket_transform(pocket_data)
            processed_pocket_id.append(pocket_filename)
            processed_pocket_data_list.append(transformed_pocket_data)
    
    print(f">> {len(pocket_data_list)} processed pockets x {eval_args.num_samples_per_pocket} samples per pocket = {len(processed_pocket_data_list)} samples to generate.")


    
    
    start  = time.time()
    print(">> Entering Analyze & Save")
    
    metrics_dict = analyze_and_save_controlnet(
        model=model,
        nodes_dist=nodes_dist, 
        nodes_dist_delta=eval_args.delta_num_atoms,
        args=args, 
        device=device, 
        dataset_info=ligand_dataset_info,
        batch_size=eval_args.batch_size, 
        pocket_data_list=processed_pocket_data_list, 
        pocket_id_lists=processed_pocket_id,
        pocket_pdb_dir=eval_args.pocket_pdb_dir,
        output_dir=output_dir,
        disable_qvina=eval_args.disable_qvina,
        mgltools_env_name=eval_args.mgltools_env_name,
        connectivity_thres=eval_args.connectivity_thres,
        ligand_add_H=eval_args.ligand_add_H,
        receptor_add_H=eval_args.receptor_add_H,
        remove_nonstd_resi=eval_args.remove_nonstd_resi,
        qvina_size=eval_args.size,
        qvina_exhaustiveness=eval_args.exhaustiveness,
        qvina_seed=eval_args.seed,
        qvina_cleanup_files=eval_args.cleanup_files
    )
    print(f">> Analyze & Save took {time.time() - start:.1f} seconds.")

    # print metrics to log file and console
    with open(os.path.join(output_dir, 'eval_log.txt'), 'w') as f:
        
        def print_multi(*args, **kwargs):
            print(*args, **kwargs)          # print to console
            print(*args, file=f, **kwargs)  # pint to file
        
        print_multi(f"No. Samples Total: {metrics_dict['num_samples_generated']}")
        print_multi(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print_multi(f"Mol Stability    : {metrics_dict['mol_stable']}")
        print_multi(f"Atom Stability   : {metrics_dict['atm_stable']}")
        print_multi(f"──────────────────────────────────────────")
        print_multi(f"Validity         : {metrics_dict['validity']}")
        print_multi(f"Uniqueness       : {metrics_dict['uniqueness']}")
        print_multi(f"Diversity        : {metrics_dict['diversity']}")
        print_multi(f"Novelty          : {metrics_dict['novelty']}")
        print_multi(f"──────────────────────────────────────────")
        print_multi(f"Connectivity     : {metrics_dict['connectivity']}")
        print_multi(f"QED              : {metrics_dict['QED']}")
        print_multi(f"SA               : {metrics_dict['SA']}")
        print_multi(f"LogP             : {metrics_dict['logP']}")
        print_multi(f"Lipinski         : {metrics_dict['lipinski']}")
        print_multi(f"──────────────────────────────────────────")
        print_multi(f"Qvina            : {metrics_dict['Qvina2']}")
        print_multi(f"Qvina No. Docked : {metrics_dict['Qvina2_Num_Docked']}")
        print_multi(f"──────────────────────────────────────────")
        print_multi(f"\n\n")
        print_multi(f"Pocket Data Dir:")
        print_multi(f"{eval_args.pocket_pdb_dir}")
        print_multi(f"")
        print_multi(f"Pocket files with issues: {len(error_filename_list)}")
        for file in error_filename_list:
            print_multi(f"  - {file}")
        print_multi(f"\n\n")
        print_multi(f"Sampling Configs")
        print_multi(f"──────────────────────────────────────────")
        for key, value in vars(eval_args).items():
            print_multi(f"{key:<22} : {value}")

if __name__ == "__main__":
    main()
