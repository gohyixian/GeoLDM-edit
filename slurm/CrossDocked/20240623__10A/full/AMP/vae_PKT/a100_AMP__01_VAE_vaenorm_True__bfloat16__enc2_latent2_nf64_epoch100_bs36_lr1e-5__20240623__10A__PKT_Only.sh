#! /bin/bash -l

#SBATCH --partition=gpu-a100
#SBATCH --ntasks=60
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --gpus=3
#SBATCH --job-name=AMP__01_VAE_vaenorm_True__bfloat16__enc2_latent2_nf64_epoch100_bs36_lr1e-5__20240623__10A__PKT_Only
#SBATCH --output=slurm_out/AMP__01_VAE_vaenorm_True__bfloat16__enc2_latent2_nf64_epoch100_bs36_lr1e-5__20240623__10A__PKT_Only.out
#SBATCH --error=slurm_err/AMP__01_VAE_vaenorm_True__bfloat16__enc2_latent2_nf64_epoch100_bs36_lr1e-5__20240623__10A__PKT_Only.err
#SBATCH --qos=long
#SBATCH --hint=multithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 103794
module load cuda/12.1       # gpu-a100
module load miniconda/24.1.2
conda activate geoldm-a100


# module load cuda/cuda-11.8  # gpu-v100s
# module load miniconda/miniconda3
# conda activate geoldm


cd /home/user/yixian.goh/geoldm-edit
python check_gpu.py
python main_geom_drugs.py --config_file configs/model_configs/CrossDocked/20240623__10A/full/AMP/PKT_Only/AMP__01_VAE_vaenorm_True__bfloat16__enc2_latent2_nf64_epoch100_bs36_lr1e-5__20240623__10A__PKT_Only.yaml