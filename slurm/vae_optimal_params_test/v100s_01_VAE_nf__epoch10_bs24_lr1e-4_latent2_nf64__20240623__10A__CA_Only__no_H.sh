#! /bin/bash -l

#SBATCH --partition=gpu-v100s
#SBATCH --ntasks=48
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --job-name=01_VAE_nf__epoch10_bs24_lr1e-4_latent2_nf64__20240623__10A__CA_Only__no_H
#SBATCH --output=slurm_out/vae_optimal_params_test/01_VAE_nf__epoch10_bs24_lr1e-4_latent2_nf64__20240623__10A__CA_Only__no_H.out
#SBATCH --error=slurm_err/vae_optimal_params_test/01_VAE_nf__epoch10_bs24_lr1e-4_latent2_nf64__20240623__10A__CA_Only__no_H.err
#SBATCH --qos=long
#SBATCH --hint=multithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 89263
# module load cuda/12.1       # gpu-a100
# module load miniconda/24.1.2
# conda activate geoldm-a100


module load cuda/cuda-11.8  # gpu-v100s
module load miniconda/miniconda3
conda activate geoldm


cd /home/user/yixian.goh/geoldm-edit
python check_gpu.py
python main_geom_drugs.py --config_file custom_config/CrossDocked/01_VAE_nf__epoch10_bs24_lr1e-4_latent2_nf64__20240623__10A__CA_Only__no_H.yaml