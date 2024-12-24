from configs.dataset_configs.datasets_config import geom_with_h, get_dataset_info
import yaml
import argparse
from os.path import join
from qm9.models import get_optim, get_model, get_autoencoder, get_latent_diffusion, get_controlled_latent_diffusion
import torch
from global_registry import PARAM_REGISTRY, Config


CONTROLNET = 'ControlNet'
LDM = "LDM"
VAE = "VAE"
# CONTROLNET_WEIGHTS_PATH = '/Users/gohyixian/Downloads/test/generative_model_ema.npy'
CONTROLNET_WEIGHTS_PATH = '/Users/gohyixian/Downloads/CrossDocked_base_03_CONTROL_20240623__10A__CA_Only__no_H/generative_model_ema.npy'

def main():
    parser = argparse.ArgumentParser(description='e3_diffusion')
    parser.add_argument('--config_file', type=str, default='configs/model_configs/base_geom_config.yaml')
    opt = parser.parse_args()

    with open(opt.config_file, 'r') as file:
        args_dict = yaml.safe_load(file)
    args = Config(**args_dict)


    # # priority check goes here
    # if args.remove_h:
    #     raise NotImplementedError()
    # else:
    #     dataset_info = geom_with_h
    dataset_info = get_dataset_info(dataset_name=args.dataset, remove_h=args.remove_h)


    # device settings
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    args.device_ = "cuda" if args.cuda else "cpu"


    # dtype settings
    module_name, dtype_name = args.dtype.split('.')
    dtype = getattr(torch, dtype_name)
    args.dtype = dtype
    torch.set_default_dtype(dtype)


    # params global registry for easy access
    PARAM_REGISTRY.update_from_config(args)


    atom_encoder = dataset_info['atom_encoder']
    atom_decoder = dataset_info['atom_decoder']


    context_node_nf = 0
    property_norms = None

    args.context_node_nf = context_node_nf


    # Create Control-LDM, Latent Diffusion Model or Autoencoder
    if args.training_mode == CONTROLNET:
        model, nodes_dist, prop_dist = get_controlled_latent_diffusion(args, args.device, dataset_info, None)
    elif args.training_mode == LDM:
        model, nodes_dist, prop_dist = get_latent_diffusion(args, args.device, dataset_info, None)
    elif args.training_mode == VAE:
        model, nodes_dist, prop_dist = get_autoencoder(args, args.device, dataset_info, None)
    else:
        raise NotImplementedError()

    model = model.to(args.device)
    flow_state_dict = torch.load(CONTROLNET_WEIGHTS_PATH, map_location=args.device)

    model.load_state_dict(flow_state_dict)
    
    for name, param in model.named_parameters():
        # print(f"{name}    {param.shape}    {param.sum().item()} ")
        # if param.sum().item() == 0.0:
            # print(name)
        if param.requires_grad == True:
            print(name)
            print(param)
            # print(param)
            # print()
            # print()



if __name__ == "__main__":
    main()


# python check_weights.py --config CrossDocked_base_03_CONTROL_20240623__10A__CA_Only__no_H__CHECK_WEIGHTS.yaml
