#!/bin/bash
#SBATCH --job-name=unet_class
#SBATCH --output=/scratch/kriles_root/kriles0/damoncht/unet_f/log/200Hz_th50_leakynormtanho_class_unet_10d_s1kstep3n1k_D20_Tsft7200v2.out
#SBATCH --error=/scratch/kriles_root/kriles0/damoncht/unet_f/log/200Hz_th50_leakynormtanho_class_unet_10d_s1kstep3n1k_D20_Tsft7200v2.err
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
python train_classification_only_entropy.py --f0 200
