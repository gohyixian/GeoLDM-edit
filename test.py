import yaml
import argparse

# config object for yaml
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


parser = argparse.ArgumentParser(description='E3Diffusion')
parser.add_argument('--config_file', type=str, default='custom_config/base_qm9_config.yaml')

opt = parser.parse_args()

with open(opt.config_file, 'r') as file:
    args_dict = yaml.safe_load(file)

args = Config(**args_dict)

print(args.ode_regularization)
print(type(args.ode_regularization))
