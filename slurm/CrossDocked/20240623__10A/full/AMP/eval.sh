#! /bin/bash -l

#SBATCH --partition=gpu-a100
#SBATCH --ntasks=30
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --gpus=2
#SBATCH --job-name=eval
#SBATCH --output=slurm_out/eval.out
#SBATCH --error=slurm_err/eval.err
#SBATCH --qos=long
#SBATCH --hint=multithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 123487
module load cuda/12.1       # gpu-a100
module load miniconda/24.1.2
conda activate geoldm-a100


# module load cuda/cuda-11.8  # gpu-v100s
# module load miniconda/miniconda3
# conda activate geoldm


cd /home/user/yixian.goh/geoldm-edit
chmod +x analysis/qvina/qvina2.1
python check_gpu.py

python check_vae_recon_loss.py --model_path outputs_selected/vae_pockets/AMP__01_VAE_vaenorm_True10__float32__latent2_nf256_epoch100_bs4_lr1e-5_InvClassFreq_Smooth0.25_XH_x30_h15_NoEMA__20240623__10A__PKT_Only --load_last --data_size 0.1
python check_vae_recon_loss.py --model_path outputs_selected/vae_pockets/AMP__01_VAE_vaenorm_True10__float32__latent2_nf256_epoch100_bs4_lr1e-5_InvClassFreq_Smooth0.25_XH_x30_h15_NoEMA__20240623__10A__PKT_Only_1x_resume --load_last --data_size 0.1
python check_vae_recon_loss.py --model_path outputs_selected/vae_pockets/AMP__01_VAE_vaenorm_True10__float32__latent2_nf256_epoch100_bs4_lr1e-5_InvClassFreq_Smooth0.25_XH_x30_h15_NoEMA__20240623__10A__PKT_Only_2x_resume --load_last --data_size 0.1
python check_vae_recon_loss.py --model_path outputs_selected/vae_pockets/AMP__01_VAE_vaenorm_True10__float32__latent8_nf256_epoch100_bs4_lr1e-5_InvClassFreq_Smooth0.25_XH_x30_h15_NoEMA__20240623__10A__PKT_Only --load_last --data_size 0.1
python check_vae_recon_loss.py --model_path outputs_selected/vae_pockets/AMP__01_VAE_vaenorm_True10__float32__latent8_nf256_epoch100_bs4_lr1e-5_InvClassFreq_Smooth0.25_XH_x30_h15_NoEMA__20240623__10A__PKT_Only_1x_resume --load_last --data_size 0.1
python check_vae_recon_loss.py --model_path outputs_selected/vae_pockets/AMP__01_VAE_vaenorm_True10__float32__latent8_nf256_epoch100_bs4_lr1e-5_InvClassFreq_Smooth0.25_XH_x30_h15_NoEMA__20240623__10A__PKT_Only_2x_resume --load_last --data_size 0.1


python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent2_nf256_epoch200_bs36_lr1e-4_EMA-0.99__VAE_DecOnly_KL-0__20240623__10A_7x_resume
python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent2_nf256_epoch200_bs36_lr1e-4_EMA-0.99__VAE_DecOnly_KL-0__20240623__10A_8x_resume

python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent2_nf256_epoch200_bs36_lr1e-4_NoEMA__VAE_DecOnly_KL-0__20240623__10A_7x_resume
python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent2_nf256_epoch200_bs36_lr1e-4_NoEMA__VAE_DecOnly_KL-0__20240623__10A_8x_resume
python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent2_nf256_epoch200_bs36_lr1e-4_NoEMA__VAE_DecOnly_KL-0__20240623__10A_9x_resume

python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent8_nf128_epoch200_bs64_lr1e-4_EMA-0.99__20240623__10A_3x_resume
python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent8_nf128_epoch200_bs64_lr1e-4_EMA-0.99__20240623__10A_4x_resume
python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent8_nf128_epoch200_bs64_lr1e-4_EMA-0.99__20240623__10A_5x_resume
python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent8_nf128_epoch200_bs64_lr1e-4_EMA-0.99__20240623__10A_6x_resume
python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent8_nf128_epoch200_bs64_lr1e-4_EMA-0.99__20240623__10A_7x_resume

python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent8_nf128_epoch200_bs64_lr1e-4_NoEMA__20240623__10A_3x_resume
python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent8_nf128_epoch200_bs64_lr1e-4_NoEMA__20240623__10A_4x_resume
python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent8_nf128_epoch200_bs64_lr1e-4_NoEMA__20240623__10A_5x_resume
python eval_analyze_ldm.py --load_last --batch_size_gen 20 --n_samples 200 --model_path outputs_selected/ldm/AMP__02_LDM_vaenorm_True10__float32__latent8_nf128_epoch200_bs64_lr1e-4_NoEMA__20240623__10A_6x_resume
