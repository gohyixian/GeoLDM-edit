#! /bin/bash -l

#SBATCH --partition=gpu-v100s
#SBATCH --ntasks=60
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --gpus=2
#SBATCH --job-name=01_VAE_nf__latent2_nf128_epoch60_bs4_lr1e-5__20240623__10A
#SBATCH --output=slurm_out/vae_optimal_params_test/20240623__10A/01_VAE_nf__latent2_nf128_epoch60_bs4_lr1e-5__20240623__10A.out
#SBATCH --error=slurm_err/vae_optimal_params_test/20240623__10A/01_VAE_nf__latent2_nf128_epoch60_bs4_lr1e-5__20240623__10A.err
#SBATCH --qos=long
#SBATCH --hint=multithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 96396
# module load cuda/12.1       # gpu-a100
# module load miniconda/24.1.2
# conda activate geoldm-a100


module load cuda/cuda-11.8  # gpu-v100s
module load miniconda/miniconda3
conda activate geoldm


cd /home/user/yixian.goh/geoldm-edit
python check_gpu.py
python main_geom_drugs.py --config_file custom_config/CrossDocked/20240623__10A/01_VAE_nf__latent2_nf128_epoch60_bs4_lr1e-5__20240623__10A.yaml