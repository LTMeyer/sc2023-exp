#!/bin/sh
#SBATCH --job-name=offline_training
#SBATCH --output=offline_training_sc_heat.%j.out
#SBATCH --time=20:00:00
#SBATCH --account=igf@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --hint=nomultithread

source ~/modules.pytorch.sh

set +x
set +e
data_dir="/gpfsscratch/rech/igf/commun/"
srun python3 run_offline_study.py --num_workers 8 --train 1 --mesh-size 1000 --data_dir $data_dir --epochs 100
