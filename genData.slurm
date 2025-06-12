#!/bin/bash
#SBATCH --job-name=gen_four
#SBATCH --output=/scratch/kriles_root/kriles0/damoncht/unet_f/log/gen_data_7200.out
#SBATCH --error=/scratch/kriles_root/kriles0/damoncht/unet_f/log/gen_data_7200.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=5GB
#SBATCH --mail-user=damoncht@umich.edu
#SBATCH --mail-type=END
#SBATCH --account=kriles0
#SBATCH --partition=standard

source /home/damoncht/.bashrc
source activate ml
python makeTrainData.py --num_cpus 20 --f0 500 --Tsft 7200 --freq_size 256 
 
