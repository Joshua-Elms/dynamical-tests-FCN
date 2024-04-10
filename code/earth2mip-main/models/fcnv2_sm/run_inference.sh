#!/bin/bash

#SBATCH -J fcnv2_sm_inference
#SBATCH -p general
#SBATCH -o output.out
#SBATCH -e log.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmelms@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --mem=200GB
#SBATCH -A r00389

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB
export MASTER_ADDR=$(hostname)

set -x

source activate ~/envs/earth2mip # environment for FCN

echo "Beginning inference script"

python run_inference.py

echo "Inference script complete"