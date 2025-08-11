#!/bin/bash
#SBATCH --job-name=sky500
#SBATCH --output=/scratch/kriles_root/kriles0/damoncht/unet_f/log/sky500Hz.out
#SBATCH --error=/scratch/kriles_root/kriles0/damoncht/unet_f/log/sky500Hz.err
#SBATCH --time=2:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --mail-user=damoncht@umich.edu
#SBATCH --mail-type=END
#SBATCH --account=kriles0
#SBATCH --partition=spgpu

source /home/damoncht/.bashrc
source activate ml
python -u skymapv2.py --f0 500
