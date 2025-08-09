#!/bin/bash
#SBATCH --job-name=sky0
#SBATCH --output=/scratch/kriles_root/kriles0/damoncht/unet_f/log/sky0Hz.out
#SBATCH --error=/scratch/kriles_root/kriles0/damoncht/unet_f/log/sky0Hz.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --mail-user=damoncht@umich.edu
#SBATCH --mail-type=END
#SBATCH --account=kriles0
#SBATCH --partition=gpu

source /home/damoncht/.bashrc
source activate ml
python -u skymapv2.py --f0 0
