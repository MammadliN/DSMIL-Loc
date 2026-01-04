#!/bin/bash
# ===== SBATCH SETTINGS =====
#SBATCH --job-name=dsmil_loc
#SBATCH --partition=${PARTITIONS:-A100-40GB,A100-80GB,A100-IML,A100-IN,A100-PBR,A100-PCI,A100-RP,A100-SDS,B200,B200-TRU,H100,H100-PCI,H100-RP,H100-SEE,H100-SLT,H100-SLT-NP,H100-Trails,H200,H200-AV,H200-DA,H200-PCI,H200-SDS,L40S,L40S-AV,L40S-DSA,RTXA6000,RTXA6000-AV,RTXA6000-EI,RTXA6000-MLT,RTXA6000-SDS,RTXA6000-SLT,V100-32GB,V100-32GB-SDS}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=/home/nmammadli/WSSED/slurm-%x-%j.out
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
