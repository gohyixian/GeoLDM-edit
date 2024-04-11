#! /bin/bash -l

#SBATCH --partition=cpu-epyc
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --mem=150G
#SBATCH --job-name=build_geom_dataset
#SBATCH --output=build_geom_dataset.out
#SBATCH --error=build_geom_dataset.err
#SBATCH --qos=normal
#SBATCH --hint=nomultithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 80463
module load miniconda/24.1.2
conda activate geoldm
python build_geom_dataset.py --data_dir /scr/user/yixian.goh/geoldm-edit-data/geom --data_file drugs_crude.msgpack