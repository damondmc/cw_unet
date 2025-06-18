#!/bin/bash
#SBATCH --job-name=att_leaky_14400_D20
#SBATCH --output=/scratch/kriles_root/kriles0/damoncht/unet_f/log/att_leaky_500Hz_leakytanho_10d_s5_D20_T14400_a1b1.out
#SBATCH --error=/scratch/kriles_root/kriles0/damoncht/unet_f/log/att_leaky_500Hz_leakytanho_10d_s5_D20_T14400_a1b1.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5GB
#SBATCH --mail-user=damoncht@umich.edu
#SBATCH --mail-type=END
#SBATCH --account=kriles0
#SBATCH --partition=gpu

source /home/damoncht/.bashrc
source activate ml
python -u train_new.py --f0 500 --n_step 1 --n_data 5 --alpha 1 --beta 1 --freq_size 512 --Tsft 14400
