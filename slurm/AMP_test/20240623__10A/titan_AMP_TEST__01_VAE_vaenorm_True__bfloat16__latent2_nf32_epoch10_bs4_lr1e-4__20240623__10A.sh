#! /bin/bash -l

#SBATCH --partition=gpu-titan
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --gpus=2
#SBATCH --job-name=AMP_TEST__01_VAE_vaenorm_True__bfloat16__latent2_nf32_epoch10_bs4_lr1e-4__20240623__10A
#SBATCH --output=slurm_out/AMP_test/20240623__10A/AMP_TEST__01_VAE_vaenorm_True__bfloat16__latent2_nf32_epoch10_bs4_lr1e-4__20240623__10A.out
#SBATCH --error=slurm_err/AMP_test/20240623__10A/AMP_TEST__01_VAE_vaenorm_True__bfloat16__latent2_nf32_epoch10_bs4_lr1e-4__20240623__10A.err
#SBATCH --qos=long
#SBATCH --hint=multithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 101224
# module load cuda/12.1       # gpu-a100
# module load miniconda/24.1.2
# conda activate geoldm-a100


module load cuda/cuda-11.8  # gpu-v100s or others
module load miniconda/24.1.2
conda activate geoldm


cd /home/user/yixian.goh/geoldm-edit
python check_gpu.py
python main_geom_drugs.py --config_file custom_config/CrossDocked/20240623__10A/AMP/AMP_TEST__01_VAE_vaenorm_True__bfloat16__latent2_nf32_epoch10_bs4_lr1e-4__20240623__10A.yaml