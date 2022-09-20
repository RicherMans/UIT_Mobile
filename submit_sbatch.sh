#!/bin/bash
#SBATCH --job-name=train_superkws_transformer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --output=logs/%x-%j.log

python3 run.py run $@
