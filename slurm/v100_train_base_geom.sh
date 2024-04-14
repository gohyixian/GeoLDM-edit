#! /bin/bash -l

#SBATCH --partition=gpu-v100s
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --gpus=2
#SBATCH --job-name=train_base_geom
#SBATCH --output=slurm_out/train_base_geom.out
#SBATCH --error=slurm_err/train_base_geom.err
#SBATCH --qos=long
#SBATCH --hint=multithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 80568
module load cuda/cuda-11.8
module load miniconda/miniconda3
conda activate geoldm
cd /home/user/yixian.goh/geoldm-edit
python check_gpu.py
python main_geom_drugs.py --config_file custom_config/base_geom_config.yaml