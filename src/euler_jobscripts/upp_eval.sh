#!/bin/bash

#SBATCH -n 4
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=upp_eval
#SBATCH --output=/cluster/home/%u/%x.out                                                                         
#SBATCH --error=/cluster/home/%u/%x.err
#SBATCH --gpus=1
#SBATCH --mail-type=NONE

module load gcc/8.2.0 python_gpu/3.10.4
pip install albumentations

python3.10 $HOME/DiffusionRoads/main.py \
    --device cuda \
    --eval \
    --model_path $HOME/DiffusionRoads/logs/2023-06-27_14-35-45/best_model.pt \
    --data_path /cluster/scratch/$USER/data.nosync/processed \
    --datasets cil \
    --metadata $HOME/DiffusionRoads/metadata.json \
    --num_workers 4 \
    --pin_memory \
    --model unet++ \
    --batch_size 8 \
    --log_dir $HOME/DiffusionRoads/logs \
    --log_to_file \

