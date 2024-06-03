#! /bin/bash -l

#SBATCH --partition=gpu-v100s
#SBATCH --ntasks=48
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --gpus=2
#SBATCH --job-name=train_base_geomPDBB_frc_LG_PKT_vae_config__noAMP_bs2_nf64__20240603
#SBATCH --output=slurm_out/train_base_geomPDBB_frc_LG_PKT_vae_config__noAMP_bs2_nf64__20240603.out
#SBATCH --error=slurm_err/train_base_geomPDBB_frc_LG_PKT_vae_config__noAMP_bs2_nf64__20240603.err
#SBATCH --qos=long
#SBATCH --hint=multithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 89262
# module load cuda/12.1       # gpu-a100
# module load miniconda/24.1.2
# conda activate geoldm-a100


module load cuda/cuda-11.8  # gpu-v100s
module load miniconda/miniconda3
conda activate geoldm

cd /home/user/yixian.goh/geoldm-edit
python check_gpu.py
python main_geom_drugs.py --config_file custom_config/geomPDBB/base_geomPDBB_frc_LG_PKT_vae_config_AMP__.yaml