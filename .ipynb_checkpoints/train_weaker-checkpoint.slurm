#!/bin/bash
#SBATCH --job-name=unet_200Hz_2035
#SBATCH --output=/scratch/kriles_root/kriles0/damoncht/unet_f/log/weaker_200Hz_th50_leakynormtanho_unet_10d_s1kstep3n1k_D2035_Tsft7200_a1b1.out
#SBATCH --error=/scratch/kriles_root/kriles0/damoncht/unet_f/log/weaker_200Hz_th50_leakynormtanho_unet_10d_s1kstep3n1k_D2035_Tsft7200_a1b1.err
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
python train_weaker.py --f0 200 --n_step 3 --alpha 1 --beta 1
