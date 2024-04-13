#! /bin/bash -l

#SBATCH --partition=gpu-v100s
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

# 80559
module load cuda/cuda-11.8
module load miniconda/miniconda3
conda activate geoldm
cd /home/user/yixian.goh/geoldm-edit
python check_gpu.py
python main_qm9.py --config_file custom_config/base_qm9_vae_config.yaml