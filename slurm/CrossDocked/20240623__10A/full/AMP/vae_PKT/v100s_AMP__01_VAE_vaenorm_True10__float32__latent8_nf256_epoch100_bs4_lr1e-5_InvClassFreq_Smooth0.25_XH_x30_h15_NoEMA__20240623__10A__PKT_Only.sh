#! /bin/bash -l

#SBATCH --partition=gpu-v100s
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --mem=160G
#SBATCH --gpus=2
#SBATCH --job-name=AMP__01_VAE_vaenorm_True10__float32__latent8_nf256_epoch100_bs4_lr1e-5_InvClassFreq_Smooth0.25_XH_x30_h15_NoEMA__20240623__10A__PKT_Only
#SBATCH --output=slurm_out/AMP__01_VAE_vaenorm_True10__float32__latent8_nf256_epoch100_bs4_lr1e-5_InvClassFreq_Smooth0.25_XH_x30_h15_NoEMA__20240623__10A__PKT_Only.out
#SBATCH --error=slurm_err/AMP__01_VAE_vaenorm_True10__float32__latent8_nf256_epoch100_bs4_lr1e-5_InvClassFreq_Smooth0.25_XH_x30_h15_NoEMA__20240623__10A__PKT_Only.err
#SBATCH --qos=long
#SBATCH --hint=multithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 122326
# module load cuda/12.1       # gpu-a100
# module load miniconda/24.1.2
# conda activate geoldm-a100


module load cuda/cuda-11.8  # gpu-v100s
module load miniconda/miniconda3
conda activate geoldm


cd /home/user/yixian.goh/geoldm-edit
python check_gpu.py
python main_geom_drugs.py --config_file configs/model_configs/CrossDocked/20240623__10A/full/AMP/PKT_Only/latest/DICC/best/AMP__01_VAE_vaenorm_True10__float32__latent8_nf256_epoch100_bs4_lr1e-5_InvClassFreq_Smooth0.25_XH_x30_h15_NoEMA__20240623__10A__PKT_Only.yaml