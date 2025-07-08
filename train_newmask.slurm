#!/bin/bash
#SBATCH --job-name=5_mask2sigma
#SBATCH --output=/scratch/kriles_root/kriles0/damoncht/unet_f/log/2sd_500Hz_10d_s5n5_D20_T14400_a1b1.out
#SBATCH --error=/scratch/kriles_root/kriles0/damoncht/unet_f/log/2sd_500Hz_10d_s5n5_D20_T14400_a1b1.err
#SBATCH --time=3-00:00:00
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
python -u train_newmask.py --f0 500 --n_step 1 --n_data 5 --n_noise 5 --alpha 1 --beta 1 --freq_size 512 --Tsft 14400 --latent_channels 64 --batch_size 8 --lr 0.0001
