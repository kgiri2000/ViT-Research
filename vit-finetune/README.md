# 1. Installation

### Create and activate a Conda environment:

```bash
module load miniconda3/24.3.0
conda create -n vit
conda activate vit
conda install python=3.12.4
pip install - r requirements.txt
```

# 2. Download dataset

```bash
python datasets/download_dataset.py \
    --dataset ucmerced \
    --output datasets/collections \
    --cache datasets/cache

```

# 3. Training in ARCC using SLURM

```bash
sbatch slurm_train.sh

```
### SLURM File
```
#!/bin/bash
#SBATCH --job-name=vit_finetune
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --partition=mb-a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

#SBATCH --account=vitftune

module load miniconda
conda activate vit

cd /project/vitftune/kgiri/VIT-RESEARCH/vit-finetune

python train.py \
    --path datasets/collections/ucmerced \
    --image_size 224 \
    --split 0.7 \
    --num_classes 21 \
    --model_name vit_b_16 \
    --model_path models/collections \
    --batch_size 64 \
    --lr 0.0003 \
    --epochs 20
```
### Output will be logged in
```
logs/vit_finetune_<jobid>.out
logs/vit_finetune_<jobid>.err

```
