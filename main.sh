#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --partition=hgx

#SBATCH --job-name=TEST_CB.out
#SBATCH -o SLURM.%N.TEST_CB.out
#SBATCH -e SLURM.%N.TEST_CB.err

#SBATCH --gres=gpu:hgx

hostname
date

module add ANACONDA/2020.11
module add CUDA/11.2.2
CUDA_VISIBLE_DEVICES=2 python -u cb_recommender.py > TEST_CB.out