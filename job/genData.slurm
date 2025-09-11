#!/bin/bash
#SBATCH --job-name=gen_data
#SBATCH --output=/scratch/kriles_root/kriles0/damoncht/unet_f/log/gen_data15d.out
#SBATCH --error=/scratch/kriles_root/kriles0/damoncht/unet_f/log/gen_data15d.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=5GB
#SBATCH --mail-user=damoncht@umich.edu
#SBATCH --mail-type=END
#SBATCH --account=lsa1
#SBATCH --partition=standard

source /home/damoncht/.bashrc
source activate ml
python /scratch/kriles_root/kriles0/damoncht/unet_f/makeTrainData.py --num_cpus 20 --f0 1 --Tsft 14400 --freq_size 512 --obsTime 1382400 --v
