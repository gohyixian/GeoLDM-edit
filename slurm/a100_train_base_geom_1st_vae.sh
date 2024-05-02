#! /bin/bash -l

#SBATCH --partition=gpu-a100
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --job-name=train_base_geom_1st_vae
#SBATCH --output=slurm_out/train_base_geom_1st_vae_a100.out
#SBATCH --error=slurm_err/train_base_geom_1st_vae_a100.err
#SBATCH --qos=long
#SBATCH --hint=multithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 81722
module load cuda/12.1       # gpu-a100
module load miniconda/24.1.2
conda activate geoldm-a100

cd /home/user/yixian.goh/geoldm-edit
python check_gpu.py
python main_geom_drugs.py --config_file custom_config/base_geom_vae_config.yaml