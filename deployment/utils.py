import os
import re
import math
import yaml
import torch
import zipfile
import subprocess
import pandas as pd
from pathlib import Path

from configs.dataset_configs.datasets_config import get_dataset_info


def get_available_models(
    model_path: str = "deployment/models/controlnet"
) -> dict[str, dict[str, str]]:
    model_folders = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))]
    
    patterns = {
        "raw_config"    : r'.*\.yaml$',
        "model_weights" : r'generative_model.*\.npy$',
        "pickled_config": r'args.*\.pickle$',
    }
    models = {}
    for folder in model_folders:
        files = os.listdir(Path(model_path, folder))
        
        # sanity check if all 3 files are present
        if all(any(re.match(pattern, f) for f in files) for pattern in list(patterns.values())):
            raw_config     = sorted([f for f in files if re.match(patterns['raw_config'], f)])[0]
            model_weights  = sorted([f for f in files if re.match(patterns['model_weights'], f)])[0]
            pickled_config = sorted([f for f in files if re.match(patterns['pickled_config'], f)])[0]
            
            models[str(folder)] = {
                'raw_config'    : str(Path(model_path, folder, raw_config)),
                'model_weights' : str(Path(model_path, folder, model_weights)),
                'pickled_config': str(Path(model_path, folder, pickled_config))
            }
    return models


def get_nvidia_smi_usage(
    smi_txt: str
) -> dict[int, dict[str, int]]:
    gpu_id_pattern = r'^\|\s*(\d+)\s+'
    memory_pattern = r'(\d+)MiB\s*/\s*(\d+)MiB'

    gpu_memory_usage = {}
    gpu_ids = re.findall(gpu_id_pattern, smi_txt, re.MULTILINE)
    memories = re.findall(memory_pattern, smi_txt)

    # assuming GPU IDs and memories are in the same order
    for gpu_id, (used_memory, total_memory) in zip(gpu_ids, memories):
        gpu_memory_usage[int(gpu_id)] = {
            'used_mem': int(used_memory),
            'total_mem': int(total_memory)
        }
    return gpu_memory_usage


def approximate_max_batch_size(
    available_models: dict[str, str]
) -> dict[str, dict]:
    
    # if nvidia gpu is available
    if torch.cuda.is_available():
        smi_txt = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8')
        gpu_info = get_nvidia_smi_usage(smi_txt)
        
        available_mem = min([gpu_info[d]['total_mem'] for d in gpu_info.keys()])
    # cpu
    else:
        available_mem = 10000  # assume 10GB (in MiB)
    
    models = {}
    for model, model_dict in available_models.items():
        with open(model_dict['raw_config'], 'r') as f:
            raw_config = yaml.safe_load(f)
        
        try:
            ca_only = raw_config['pocket_vae']['ca_only']
        except:
            ca_only = False
        
        if ca_only:
            # bs = (7/950)*gpu - (160/19)
            DELTA = 5    # gap for safety
            max_bs = math.floor((7 / 950)*available_mem - (160 / 19)) - DELTA
        else:
            # bs = (7/9000)*gpu - (4/3)
            DELTA = 2    # gap for safety
            max_bs = math.floor((7 / 9000)*available_mem - (4 / 3)) - DELTA
        
        model_dict['max_bs'] = max_bs
        models[model] = model_dict
    return models



"""
Equations below are used to model the GPU usage of our CA-only and ALL-atoms models.

For CA-only:
bs = (7/950)*gpu - (160/19)

For ALL-atoms:
bs = (7/9000)*gpu - (4/3)

These equations are approximated using the below data's lower bounds. 
The data are collected from running the models on CrossDocked's test set.

CA-only
==============
BS   GPU (MiB)
--------------
10    2,500
20    3,500
30    5,000
40    6,000
50    7,000
60    8,000
70    9,500
80   12,000


ALL-atoms
==============
BS   GPU (MiB)
--------------
1     3,000
2     4,000
3     5,000
4     6,000
5     7,000
6     8,000
7     9,500
8    12,000
"""


def get_model_n_nodes_distribution(
    available_models: dict[str, str]
) -> dict[str, dict]:

    models = {}
    for model, model_dict in available_models.items():
        with open(model_dict['raw_config'], 'r') as f:
            raw_config = yaml.safe_load(f)
        
        model_dataset_name = raw_config.get('dataset', 'd_20241203_CrossDocked_LG_PKT_MMseq2_split_CA_only__10A__LIGAND')
        model_dataset_remove_h = raw_config.get('remove_h', False)
        
        model_dataset_info = get_dataset_info(model_dataset_name, model_dataset_remove_h)
        model_dataset_n_nodes = model_dataset_info['n_nodes']
        
        model_dict['n_nodes'] = model_dataset_n_nodes
        models[model] = model_dict
    return models


def zip_folder(folder_path, zip_filename):
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Create a relative path for the file within the zip archive
                # This ensures the folder structure is preserved inside the zip
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)


def get_empty_metrics_df():
    df = pd.DataFrame(
        {
            "Metrics": [
                "No. Samples Total",
                "Atom Stability",
                "Mol Stability",
                "Validity",
                "Uniqueness",
                "Diversity",
                "Connectivity",
                "QED",
                "SA",
                "LogP",
                "Lipinski",
                "Qvina",
                "Qvina No. Docked Samples"
            ],
            "Values": ["-"] * 13
        }
    )
    return df