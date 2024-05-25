#! /bin/bash -l

#SBATCH --partition=gpu-a100
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --job-name=train_base_geom_amp
#SBATCH --output=slurm_out/train_base_geom_a100_amp.out
#SBATCH --error=slurm_err/train_base_geom_a100_amp.err
#SBATCH --qos=long
#SBATCH --hint=multithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 85796
module load cuda/12.1       # gpu-a100
module load miniconda/24.1.2
conda activate geoldm-a100

cd /home/user/yixian.goh/geoldm-edit
python check_gpu.py
python main_geom_drugs.py --config_file custom_config/base_geom_config.yaml