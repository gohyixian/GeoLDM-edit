"""
This script is used to read the index.pkl file provided in the CrossDocked dataset.
It will then attempt to read all the .sdf files. If success, it would create a copy
of the file, and create a mapping between the original index and the name of the 
file copied.

The copied .sdf files would then be read into Chimera X to add hydrogens for all 
ligands (batch), and save as .pdb files, which would then be read into the actual
dataset processing script.
"""

import os
import json
import torch
import argparse
from tqdm import tqdm
from rdkit import Chem
from pathlib import Path



def process_ligand_and_pocket(sdffile, savefile, id):

    try:
        ligand = Chem.SDMolSupplier(str(sdffile))[0]
    except:
        raise Exception(f'cannot read sdf mol ({sdffile})')

    try:
        # get only the first conformer
        conformer = ligand.GetConformer(0)
        
        # for attr, value in conformer.__dict__.items():
        #     print(f"{attr}: {value}")

        # Create a new molecule to save only the first conformer
        ligand_with_first_conformer = Chem.Mol(ligand)
        new_conformer = Chem.Conformer(conformer)
        ligand_with_first_conformer.RemoveAllConformers()
        ligand_with_first_conformer.AddConformer(new_conformer, assignId=True)

        ligand_with_first_conformer.SetProp(key='_Name', val=id)
        
        # Write to a new SDF file
        writer = Chem.SDWriter(savefile)
        writer.write(ligand_with_first_conformer)
        writer.close()
    except:
        raise Exception(f'Error saving first conformer ({sdffile})')


# python 01_index_sdf_files.py --raw_crossd_basedir /Users/gohyixian/Downloads/Crossdocked_Pocket10 --copy_files_dir /Users/gohyixian/Downloads/Crossdocked_Pocket10_ToAddH/data --mapping_file /Users/gohyixian/Downloads/Crossdocked_Pocket10_ToAddH/mapping.json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_crossd_basedir', type=Path)
    parser.add_argument("--copy_files_dir", type=str, default="data/CrossDocked_LG_PKT/test_val_paired_files")
    parser.add_argument("--mapping_file", type=str, default="mapping.json")
    args = parser.parse_args()
    
    if not os.path.exists(args.copy_files_dir):
        os.makedirs(args.copy_files_dir)
    
    map_dir = os.path.dirname(args.mapping_file)
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)
    
    
    datadir = args.raw_crossd_basedir / 'crossdocked_pocket10/'
    
    # Read data split
    split_path = Path(args.raw_crossd_basedir, 'split_by_name.pt')
    data_split = torch.load(split_path)  # only has keys: train, test
    
    mol_id = 0
    ligand_id_mapping = dict()
    
    num_failed = 0
    failed_save = []
    
    
    for split in data_split.keys():
        
        pbar = tqdm(data_split[split])
        pbar.set_description(f'#failed: {num_failed}')
        for pocket_fn, ligand_fn in pbar:

            sdffile = datadir / f'{ligand_fn}'

            try:
                id = str(mol_id).zfill(8)
                savefile = os.path.join(args.copy_files_dir, f"{id}.sdf")
                process_ligand_and_pocket(sdffile, savefile, id)
                
                ligand_id_mapping[ligand_fn] = id
                mol_id += 1
                
            except (Exception) as e:
                print(type(e).__name__, e, pocket_fn, ligand_fn)
                pbar.set_description(f'#failed: {num_failed}')
                
                num_failed += 1
                failed_save.append(ligand_fn)
                continue
    
    with open(args.mapping_file, 'w') as f:
        json.dump(
            {
                'mapping': ligand_id_mapping,
                'failed': failed_save
            },
            f,
            indent=4
        )
    
    print(f"Success: {len(list(ligand_id_mapping.keys()))}")
    print(f"Failed : {len(failed_save)}")
