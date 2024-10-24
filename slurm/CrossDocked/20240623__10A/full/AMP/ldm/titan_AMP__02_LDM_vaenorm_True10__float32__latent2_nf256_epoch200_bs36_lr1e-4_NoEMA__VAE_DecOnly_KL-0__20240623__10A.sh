#! /bin/bash -l

#SBATCH --partition=gpu-titan
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --mem=60G
#SBATCH --gpus=1
#SBATCH --job-name=AMP__02_LDM_vaenorm_True10__float32__latent2_nf256_epoch200_bs36_lr1e-4_NoEMA__VAE_DecOnly_KL-0__20240623__10A
#SBATCH --output=slurm_out/AMP__02_LDM_vaenorm_True10__float32__latent2_nf256_epoch200_bs36_lr1e-4_NoEMA__VAE_DecOnly_KL-0__20240623__10A.out
#SBATCH --error=slurm_err/AMP__02_LDM_vaenorm_True10__float32__latent2_nf256_epoch200_bs36_lr1e-4_NoEMA__VAE_DecOnly_KL-0__20240623__10A.err
#SBATCH --qos=long
#SBATCH --hint=multithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 117132
# module load cuda/12.1       # gpu-a100
# module load miniconda/24.1.2
# conda activate geoldm-a100


# module load cuda/cuda-11.8  # gpu-v100s
# module load miniconda/miniconda3
# conda activate geoldm


module load cuda/cuda-11.8  # gpu-titan
module load miniconda/24.1.2
conda activate geoldm

cd /home/user/yixian.goh/geoldm-edit
python check_gpu.py
python main_geom_drugs.py --config_file custom_config/CrossDocked/20240623__10A/full/AMP/ldm/best/AMP__02_LDM_vaenorm_True10__float32__latent2_nf256_epoch200_bs36_lr1e-4_NoEMA__VAE_DecOnly_KL-0__20240623__10A.yaml