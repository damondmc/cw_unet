#!/bin/bash
#SBATCH --job-name=500Hz_10d
#SBATCH --output=/scratch/kriles_root/kriles0/damoncht/unet_f/log/500Hz_10d_n7_T14400.out
#SBATCH --error=/scratch/kriles_root/kriles0/damoncht/unet_f/log/500Hz_10d_n7_T14400.err
#SBATCH --time=8-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --mail-user=damoncht@umich.edu
#SBATCH --mail-type=END
#SBATCH --account=kriles0
#SBATCH --partition=gpu

source /home/damoncht/.bashrc
source activate ml
python -u train.py --f0 500 --n_data 7 --n_noise 7 --freq_size 512 --Tsft 14400 --latent_channels 64 --lr 0.0001 --obsTime 921600
