#!/bin/sh
#SBATCH --job-name=SC-Heat
#SBATCH --output=sc-hpde.out
#SBATCH --time=01:00:00
#SBATCH --account=igf@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=50
#SBATCH --ntasks-per-node=40

module load pytorch-gpu/py3/1.13.0
source $ALL_CCFRWORK/$USER/melissa/melissa_set_env.sh

exec melissa-launcher --config_name config_offline
