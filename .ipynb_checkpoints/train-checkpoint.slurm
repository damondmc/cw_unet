#!/bin/bash
#SBATCH --job-name=unetx
#SBATCH --output=/scratch/kriles_root/kriles0/damoncht/unet_f/log/500Hz_th50_leakynormtanho_unet_10d_s1kstep2n0.5k_D20_Tsft7200.out
#SBATCH --error=/scratch/kriles_root/kriles0/damoncht/unet_f/log/500Hz_th50_leakynormtanho_unet_10d_s1kstep2n0.5k_D20_Tsft7200.err
#SBATCH --time=1-20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --mail-user=damoncht@umich.edu
#SBATCH --mail-type=END
#SBATCH --account=kriles0
#SBATCH --partition=gpu

source /home/damoncht/.bashrc
source activate ml
python train.py --f0 500
