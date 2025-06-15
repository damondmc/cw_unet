#!/bin/bash
#SBATCH --job-name=r2unet_500Hz_14400
#SBATCH --output=/scratch/kriles_root/kriles0/damoncht/unet_f/log/r2unet_500Hz_th50_leakynormtanho_unet_10d_s3kstep3n1k_D20_Tsft14400_a1b1_old_nomask.out
#SBATCH --error=/scratch/kriles_root/kriles0/damoncht/unet_f/log/r2unet_500Hz_th50_leakynormtanho_unet_10d_s3kstep3n1k_D20_Tsft14400_a1b1_old_nomask.err
#SBATCH --time=1-20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=10GB
#SBATCH --mail-user=damoncht@umich.edu
#SBATCH --mail-type=END
#SBATCH --account=kriles0
#SBATCH --partition=gpu

source /home/damoncht/.bashrc
source activate ml
python -u train_r2unet.py --f0 500 --n_step 3 --n_data 3 --alpha 1 --beta 1 --freq_size 512 --Tsft 14400
