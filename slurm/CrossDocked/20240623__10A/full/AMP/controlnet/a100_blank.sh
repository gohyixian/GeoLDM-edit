#! /bin/bash -l

#SBATCH --partition=gpu-a100
#SBATCH --ntasks=60
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --gpus=2
#SBATCH --job-name=AMP__03_CONTROL_test
#SBATCH --output=slurm_out/AMP__03_CONTROL_test.out
#SBATCH --error=slurm_err/AMP__03_CONTROL_test.err
#SBATCH --qos=long
#SBATCH --hint=multithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 122320
module load cuda/12.1       # gpu-a100
module load miniconda/24.1.2
conda activate geoldm-a100


# module load cuda/cuda-11.8  # gpu-v100s
# module load miniconda/miniconda3
# conda activate geoldm


cd /home/user/yixian.goh/geoldm-edit
chmod +x analysis/qvina/qvina2.1
python check_gpu.py
python main_geom_drugs_control.py --config_file custom_config/CrossDocked/20240623__10A/full/AMP/controlnet/AMP__03_CONTROL_latent8_nf128_ds500_fusBalancedSum__ConFus_epoch200_bs14_lr1e-4_NoEMA__20240623__10A.yaml