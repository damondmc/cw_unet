#!/bin/bash
#SBATCH --job-name=20Hz_10d
#SBATCH --output=/scratch/kriles_root/kriles0/damoncht/unet_f/log/20Hz_10d_n10_T14400_l128.out
#SBATCH --error=/scratch/kriles_root/kriles0/damoncht/unet_f/log/20Hz_10d_n10_T14400_l128.err
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=10GB
#SBATCH --mail-user=damoncht@umich.edu
#SBATCH --mail-type=END
#SBATCH --account=kriles0
#SBATCH --partition=spgpu

source /home/damoncht/.bashrc
source activate ml
python -u train.py --f0 20 --n_data 10 --n_noise 10 --freq_size 512 --Tsft 14400 --latent_channels 128 --lr 0.0001 --obsTime 921600
