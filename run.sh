#!/bin/bash
# ===== SBATCH SETTINGS =====
#SBATCH --job-name=dsmil_loc
#SBATCH --partition=V100-32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=/home/nmammadli/WSSED/DSMIL-Loc/slurm-%x-%j.out
#SBATCH --chdir=/home/nmammadli/WSSED/DSMIL-Loc

# Adjust defaults through environment variables when submitting, e.g.:
#   sbatch --export=ALL,PARTITIONS="A100-40GB,H100" run.sh

srun -K \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
  --container-workdir=/home/nmammadli/WSSED/DSMIL-Loc \
  --container-mounts=/home/nmammadli:/home/nmammadli,/ds-iml:/ds-iml:ro \
  --gpus=1 \
  --task-prolog=/home/nmammadli/WSSED/DSMIL-Loc/install.sh \
  python /home/nmammadli/WSSED/DSMIL-Loc/main.py "$@"
