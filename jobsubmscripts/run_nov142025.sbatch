#!/bin/bash
#SBATCH --partition=notchpeak-gpu
#SBATCH --account=notchpeak-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=7G
#SBATCH --time=04:00:00
#SBATCH --job-name=icecube-128
#SBATCH --output=logs/icecube-128-%j.out

module purge
source ~/icecube-ml/venv/bin/activate

cd ~/icecube-ml

python icecube_conditional_decoder_film.py \
  --csv data/camsim/meta/train.csv \
  --val_csv data/camsim/meta/val.csv \
  --img_size 128 \
  --standardize_params \
  --batch_size 156 \
  --epochs 50 \
  --num_workers 4 --pin_memory \
  --amp \
  --ckpt_every 5 \
  --out_dir runs/exp25
