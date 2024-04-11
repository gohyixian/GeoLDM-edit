#! /bin/bash -l

#SBATCH --partition=gpu-a100
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --job-name=train_base_qm9_vae
#SBATCH --output=slurm_out/train_base_qm9_vae.out
#SBATCH --error=slurm_err/train_base_qm9_vae.err
#SBATCH --qos=long
#SBATCH --hint=multithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 80461
# module los cuda/12.4
module load cuda/12.1
module load miniconda/24.1.2
conda activate geoldm
python check_gpu.py
python main_qm9.py --config_file custom_config/base_qm9_vae_config.yaml