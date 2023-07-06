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

# default euler gpu runs out of memory if batch size is 8
# can request higher memory gpu with --gres=gpumem:15G, but might result in longer waiting time

module load gcc/8.2.0 python_gpu/3.10.4
pip install albumentations

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
    --log_dir $HOME/DiffusionRoads/logs \
    --log_to_file \
    
#--batches_per_epoch 2000
