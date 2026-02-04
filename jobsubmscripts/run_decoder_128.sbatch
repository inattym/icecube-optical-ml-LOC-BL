#!/bin/bash
#SBATCH --job-name=icecube_train
#SBATCH --partition=notchpeak-gpu
#SBATCH --account=notchpeak-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module purge
# module load cuda/12.1     # only if your venv needs CHPC CUDA module
# module load python/3.11.3 # only if your venv was built on this module
source ~/.venvs/icecube-ml/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,expandable_segments:True
export TORCH_NCCL_BLOCKING_WAIT=1

python icecube_conditional_decoder_film.py \
  --csv data/camsim/meta/train.csv \
  --val_csv data/camsim/meta/val.csv \
  --img_size 128 \
  --standardize_params \
  --batch_size 877 \
  --epochs 50 \
  --num_workers 8 --pin_memory \
  --amp \
  --ckpt_every 5 \
  --out_dir runs/exp_gpu128
  