#!/bin/bash

#SBATCH -n 4
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --gres=gpumem:20g
#SBATCH --job-name=upp_training
#SBATCH --output=/cluster/home/%u/%x.out                                                                         
#SBATCH --error=/cluster/home/%u/%x.err
#SBATCH --gpus=1
#SBATCH --mail-type=NONE

module load gcc/8.2.0 python_gpu/3.10.4

python3.10 $HOME/DiffusionRoads/main.py \
    --device cuda \
    --data_path /cluster/scratch/$USER/data.nosync/processed \
    --datasets cil \
    --metadata $HOME/DiffusionRoads/metadata.json \
    --num_workers 4 \
    --pin_memory \
    --model unet++ \
    --batch_size 8 \
    --epochs 100 \
    --lr 3e-4 \
    --miou_weight 0.0 \
    --bce_weight 1.0 \
    --mse_weight 0.0 \
    --focal_weight 0.0 \
    --vec_weight 0.0 \
    --topo_weight 0.1 \
    --log_dir /cluster/scratch/$USER/logs \
    --log_to_file \
