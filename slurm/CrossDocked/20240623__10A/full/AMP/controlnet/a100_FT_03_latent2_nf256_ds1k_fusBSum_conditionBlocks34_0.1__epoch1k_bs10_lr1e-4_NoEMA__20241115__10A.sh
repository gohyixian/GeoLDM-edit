#! /bin/bash -l

#SBATCH --partition=gpu-a100
#SBATCH --ntasks=60
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --gpus=2
#SBATCH --job-name=FT_03_latent2_nf256_ds1k_fusBSum_conditionBlocks34_0.1__epoch1k_bs10_lr1e-4_NoEMA__20241115__10A
#SBATCH --output=slurm_out/FT_03_latent2_nf256_ds1k_fusBSum_conditionBlocks34_0.1__epoch1k_bs10_lr1e-4_NoEMA__20241115__10A.out
#SBATCH --error=slurm_err/FT_03_latent2_nf256_ds1k_fusBSum_conditionBlocks34_0.1__epoch1k_bs10_lr1e-4_NoEMA__20241115__10A.err
#SBATCH --qos=long
#SBATCH --hint=multithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 139215
module load cuda/12.1       # gpu-a100
module load miniconda/24.1.2
conda activate geoldm-a100


# module load cuda/cuda-11.8  # gpu-v100s
# module load miniconda/miniconda3
# conda activate geoldm


cd /home/user/yixian.goh/geoldm-edit
chmod +x analysis/qvina/qvina2.1
python check_gpu.py
python main_geom_drugs_control.py --config_file configs/model_configs/CrossDocked/20240623__10A/full/AMP/controlnet/FT_03_latent2_nf256_ds1k_fusBSum_conditionBlocks34_0.1__epoch1k_bs10_lr1e-4_NoEMA__20241115__10A.yaml