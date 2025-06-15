#!/bin/bash
#SBATCH --job-name=tunet_500Hz_D20_7200
#SBATCH --output=/scratch/kriles_root/kriles0/damoncht/unet_f/log/f500Hz_th50_leakynormtanho_unet_10d_s5step1_D20_Tsft7200_a1b1.out
#SBATCH --error=/scratch/kriles_root/kriles0/damoncht/unet_f/log/f500Hz_th50_leakynormtanho_unet_10d_s5step1_D20_Tsft7200_a1b1.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=10GB
#SBATCH --mail-user=damoncht@umich.edu
#SBATCH --mail-type=END
#SBATCH --account=kriles0
#SBATCH --partition=gpu

source /home/damoncht/.bashrc
source activate ml
python -u train.py --f0 500 --n_step 1 --n_data 5 --alpha 1 --beta 1 --freq_size 256 --Tsft 7200
