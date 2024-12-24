#! /bin/bash -l

#SBATCH --partition=gpu-a100
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --job-name=train_base_geomPDBB_frc_LG_PKT_vae_config_AMP___20240528
#SBATCH --output=slurm_out/train_base_geomPDBB_frc_LG_PKT_vae_config_AMP___20240528.out
#SBATCH --error=slurm_err/train_base_geomPDBB_frc_LG_PKT_vae_config_AMP___20240528.err
#SBATCH --qos=long
#SBATCH --hint=multithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 88203
module load cuda/12.1       # gpu-a100
module load miniconda/24.1.2
conda activate geoldm-a100

cd /home/user/yixian.goh/geoldm-edit
python check_gpu.py
python main_geom_drugs.py --config_file configs/model_configs/geomPDBB/base_geomPDBB_frc_LG_PKT_vae_config_AMP__.yaml