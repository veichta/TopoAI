#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10g
#SBATCH --time=04:00:00
#SBATCH --job-name="train-unet++"
#SBATCH --mem-per-cpu=3g
#SBATCH --output="slurm.txt"
#SBATCH --open-mode=append
#SBATCH --mail-type=NONE
#SBATCH --mail-user=rdanis@ethz.ch

module load gcc/8.2.0 python_gpu/3.10.4

python main.py \
    --data_path /cluster/scratch/$USER/data.nosync/processed \
    --datasets cil \
    --device cuda \
    --log_dir /cluster/scratch/rdanis/logs \
    --epochs 300 \
    --lr 0.001 \
    --model unet++ \
    --num_workers 0 \
    --wandb \
    --wandb_dir /cluster/scratch/rdanis/ \
    --patience 40 \
    --topo_weight $1 \
    --topo_k0 $2 \
    --topo_k1 $3