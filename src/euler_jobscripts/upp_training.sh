#!/bin/bash

#SBATCH -n 4
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=upp_training
#SBATCH --output=/cluster/home/%u/%x.out                                                                         
#SBATCH --error=/cluster/home/%u/%x.err
#SBATCH --gpus=1
#SBATCH --mail-type=NONE

module load gcc/8.2.0 python_gpu/3.10.4
pip install albumentations

# $HOME/cil_dataset
# alternative path: /cluster/scratch/$USER/data.nosync/processed
# euler runs out of memory for batch size 8

python3.10 $HOME/DiffusionRoads/main.py \
    --device cuda \
    --data_path /cluster/scratch/$USER/data.nosync/processed \
    --datasets all \
    --metadata $HOME/DiffusionRoads/metadata.json \
    --model unet++ \
    --batch_size 4 \
    --epochs 5 \
    --lr 3e-4 \
    --log_dir $HOME/DiffusionRoads/logs \
    --log_to_file \
    --batches_per_epoch 2000

