#!/bin/bash
#SBATCH --job-name=vit_finetune
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --partition=mb-a30          # GPU partition (A30 GPUs)
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --cpus-per-task=4           # Number of CPU cores
#SBATCH --mem=32G                   # RAM
#SBATCH --time=12:00:00             # Walltime (12 hours max for A30 queue)

#SBATCH --account=vitftune          # ARCC project account
#SBATCH --qos=normal                # Queue (normal = <=3 days)


#Load Python / Conda
#module load miniconda3/24.3.0
conda activate vit                 # conda environment name

echo "Using Python: $(which python)"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"


#Project folder
cd /project/vitftune/kgiri/ViT-Research/vit-finetune

# Training
python train.py \
    --path datasets/collections/ucmerced \
    --image_size 224 \
    --num_classes 21 \
    --split 0.3 \
    --model_name vit_b_16 \
    --model_path models/collections \
    --batch_size 100 \
    --lr 0.00003 \
    --epochs 30
