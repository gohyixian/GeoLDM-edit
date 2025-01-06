#! /bin/bash -l

#SBATCH --partition=gpu-a100
#SBATCH --ntasks=60
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --job-name=eval_20250106
#SBATCH --output=slurm_out/eval_20250106.out
#SBATCH --error=slurm_err/eval_20250106.err
#SBATCH --qos=long
#SBATCH --hint=multithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 131127
module load cuda/12.1       # gpu-a100
module load miniconda/24.1.2
conda activate geoldm-a100


# module load cuda/cuda-11.8  # gpu-v100s
# module load miniconda/miniconda3
# conda activate geoldm


cd /home/user/yixian.goh/geoldm-edit
chmod +x analysis/qvina/qvina2.1
python check_gpu.py


# FT CA
python eval_analyze_controlnet.py --model_path outputs_selected/FT_03_latent2_nf256_ds1k_fusBSum_CA_conditionAll_0.1__epoch1k_bs60_lr1e-4_NoEMA__20241203__10A_5x_resume --load_last --pocket_pdb_dir data/d_20241203_CrossDocked_LG_PKT_MMseq2_split_CA_only/test_val_paired_files/val_pocket --num_samples_per_pocket 10 --delta_num_atoms 5 --batch_size 512 --seed 42 --cleanup_files --ligand_add_H --receptor_add_H
python eval_analyze_controlnet.py --model_path outputs_selected/FT_03_latent2_nf256_ds1k_fusBSum_CA_conditionAll_0.5__epoch1k_bs60_lr1e-4_NoEMA__20241203__10A_5x_resume --load_last --pocket_pdb_dir data/d_20241203_CrossDocked_LG_PKT_MMseq2_split_CA_only/test_val_paired_files/val_pocket --num_samples_per_pocket 10 --delta_num_atoms 5 --batch_size 512 --seed 42 --cleanup_files --ligand_add_H --receptor_add_H
python eval_analyze_controlnet.py --model_path outputs_selected/FT_03_latent2_nf256_ds1k_fusBSum_CA_conditionBlocks34_0.1__epoch1k_bs60_lr1e-4_NoEMA__20241203__10A_5x_resume --load_last --pocket_pdb_dir data/d_20241203_CrossDocked_LG_PKT_MMseq2_split_CA_only/test_val_paired_files/val_pocket --num_samples_per_pocket 10 --delta_num_atoms 5 --batch_size 512 --seed 42 --cleanup_files --ligand_add_H --receptor_add_H
python eval_analyze_controlnet.py --model_path outputs_selected/FT_03_latent2_nf256_ds1k_fusBSum_CA_conditionBlocks34_0.5__epoch1k_bs60_lr1e-4_NoEMA__20241203__10A_5x_resume --load_last --pocket_pdb_dir data/d_20241203_CrossDocked_LG_PKT_MMseq2_split_CA_only/test_val_paired_files/val_pocket --num_samples_per_pocket 10 --delta_num_atoms 5 --batch_size 512 --seed 42 --cleanup_files --ligand_add_H --receptor_add_H
python eval_analyze_controlnet.py --model_path outputs_selected/FT_03_latent2_nf256_ds1k_fusReplace_CA__epoch1k_bs60_lr1e-4_NoEMA__20241203__10A_2x_resume --load_last --pocket_pdb_dir data/d_20241203_CrossDocked_LG_PKT_MMseq2_split_CA_only/test_val_paired_files/val_pocket --num_samples_per_pocket 10 --delta_num_atoms 5 --batch_size 512 --seed 42 --cleanup_files --ligand_add_H --receptor_add_H

# CA conditionAll 0.9
python eval_analyze_controlnet.py --model_path outputs_selected/03_latent2_nf256_ds1k_fusBSum_CA_conditionAll_0.9__epoch1k_bs60_lr1e-4_NoEMA__20241203__10A_3x_resume --load_last --pocket_pdb_dir data/d_20241203_CrossDocked_LG_PKT_MMseq2_split_CA_only/test_val_paired_files/val_pocket --num_samples_per_pocket 10 --delta_num_atoms 5 --batch_size 512 --seed 42 --cleanup_files --ligand_add_H --receptor_add_H

# ALL conditionBlocks34 0.1
python eval_analyze_controlnet.py --model_path outputs_selected/controlnet/03_latent2_nf256_ds1k_fusBSum_conditionBlocks34_0.1__epoch1k_bs10_lr1e-4_NoEMA__20241115__10A_4x_resume --load_last --pocket_pdb_dir data/d_20241115_CrossDocked_LG_PKT_MMseq2_split/test_val_paired_files/val_pocket --num_samples_per_pocket 10 --delta_num_atoms 5 --batch_size 5 --seed 42 --cleanup_files --ligand_add_H --receptor_add_H

# ALL GEOM-LDM conditionAll 0.5
python eval_analyze_controlnet.py --model_path outputs_selected/controlnet/03_latent2_nf256_ds1k_fusBSum_conditionAll_0.5__GEOM_LDM_base__epoch1k_bs10_lr1e-4_NoEMA__20241115__10A_4x_resume --load_last --pocket_pdb_dir data/d_20241115_CrossDocked_LG_PKT_MMseq2_split/test_val_paired_files/val_pocket --num_samples_per_pocket 10 --delta_num_atoms 5 --batch_size 5 --seed 42 --cleanup_files --ligand_add_H --receptor_add_H
