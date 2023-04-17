#!/bin/sh
#SBATCH --job-name=ProcessDataset
#SBATCH --output=process_dataset%j.out
#SBATCH --time=02:00:00
#SBATCH --account=igf@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10

source ~/modules.pytorch.sh
data_path="/gpfswork/rech/igf/commun/sc2023-heatpde-validation/"
python3 process_dataset.py $data_path 1000 1000 --num_workers $1 --dataset_type $2
