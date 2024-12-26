# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass

import time
import math
import copy
import yaml
import wandb
import torch
import pickle
import random
import argparse
import numpy as np
from os.path import join

import utils
import train_test
import build_geom_dataset
from configs.dataset_configs.datasets_config import get_dataset_info
from equivariant_diffusion import utils as diffusion_utils
from equivariant_diffusion import en_diffusion, control_en_diffusion
from qm9.models import get_optim, get_controlled_latent_diffusion
from global_registry import PARAM_REGISTRY, Config



MMSEQ2_SPLIT = "MMseq2_split"

def main():
    parser = argparse.ArgumentParser(description='e3_diffusion')
    parser.add_argument('--config_file', type=str, default='configs/model_configs/CrossDocked/20240623__10A/full/AMP/controlnet/03_latent2_nf256_ds1k_fusReplace_CA__epoch1k_bs60_lr1e-4_NoEMA__20241203__10A.yaml')
    opt = parser.parse_args()

    with open(opt.config_file, 'r') as file:
        args_dict = yaml.safe_load(file)
    
    # args = Config(**args_dict)
    args = Config(
        **{k: 
            Config(**v) if isinstance(v, dict) \
            else v \
            for k, v in args_dict.items()
        }
    )

    # set random seed
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # load pre-computed dataset configs
    ligand_dataset_info = get_dataset_info(dataset_name=args.dataset, remove_h=args.remove_h)
    pocket_dataset_info = get_dataset_info(dataset_name=args.pocket_vae.dataset, remove_h=args.pocket_vae.remove_h)

    # device settings
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    args.device_ = "cuda" if args.cuda else "cpu"

    # dtype settings
    _, dtype_name = args.dtype.split('.')
    dtype = getattr(torch, dtype_name)
    args.dtype = dtype
    torch.set_default_dtype(dtype)

    # add missing configs with default values
    args = utils.add_missing_configs_controlnet(args, dtype, ligand_dataset_info, pocket_dataset_info)

    with open('./args.pickle', 'wb') as f:
        pickle.dump(args, f)


if __name__ == "__main__":
    main()