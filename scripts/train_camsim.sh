#!/bin/bash
#SBATCH --job-name=icecube_train
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -euo pipefail

######## 1) Environment ########
module load miniconda3/23.11.0
source /uufs/chpc.utah.edu/sys/installdir/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate /uufs/chpc.utah.edu/common/home/u1494626/.conda/envs/icecube

# Be conservative with threading (helps dataloader stability on shared nodes)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

######## 2) Paths ########
PROJ=/uufs/chpc.utah.edu/common/home/u1494626/icecube-ml
cd "$PROJ"
mkdir -p logs runs
RUN_ID="exp_gpu_$(date +%Y%m%d_%H%M)"
OUT_DIR="runs/${RUN_ID}"

TRAIN_CSV="data/camsim/meta/train.csv"
VAL_CSV="data/camsim/meta/val.csv"

######## 3) Model / train defaults (override via CLI or env) ########
IMG_SIZE=${IMG_SIZE:-128}
START_RES=${START_RES:-4}
BASE_CH=${BASE_CH:-256}
COND_DIM=${COND_DIM:-256}
MLP_HIDDEN=${MLP_HIDDEN:-256}
CH_MULTS=${CH_MULTS:-"1.0,1.0,0.75,0.5,0.5,0.25"}
N_BLOCKS_PER_UP=${N_BLOCKS_PER_UP:-2}
UP_MODE=${UP_MODE:-bilinear}

EPOCHS=${EPOCHS:-200}
LR=${LR:-3e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
GRAD_CLIP=${GRAD_CLIP:-1.0}

# Safer defaults for stability on this cluster
BATCH_SIZE=${BATCH_SIZE:-16}   # 2080 Ti (11GB). Bump to 32 if you see plenty of free mem.
NUM_WORKERS=${NUM_WORKERS:-0}  # 0 avoids occasional dataloader hangs on shared nodes
USE_AMP=${USE_AMP:-0}          # 0 by default; set USE_AMP=1 to enable

######## 4) Build ARG string ########
ARGS=(
  --csv "$TRAIN_CSV"
  --val_csv "$VAL_CSV"
  --out_dir "$OUT_DIR"

  --img_size "$IMG_SIZE"
  --start_res "$START_RES"
  --base_ch "$BASE_CH"
  --cond_dim "$COND_DIM"
  --mlp_hidden "$MLP_HIDDEN"
  --ch_mults "$CH_MULTS"
  --n_blocks_per_up "$N_BLOCKS_PER_UP"
  --up_mode "$UP_MODE"

  --standardize_params
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --grad_clip "$GRAD_CLIP"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --ckpt_every 10
)

# AMP toggle (only add --amp when explicitly enabled)
if [[ "$USE_AMP" -eq 1 ]]; then
  ARGS+=("--amp")
fi

# Allow extra flags at submit time to override any of the above
ARGS+=("$@")

######## 5) Log environment ########
echo "===== ENV ====="
echo "JOB : ${SLURM_JOB_ID:-N/A}"
echo "HOST: $(hostname)"
echo "PWD : $(pwd)"
echo "PY  : $(which python)"
python -V
echo "CUDA/:"
nvidia-smi || true
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda build:", getattr(torch.version, "cuda", None))
print("cuda available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
echo "===== ARGS ====="
printf '%q ' "${ARGS[@]}"; echo
echo "================"

######## 6) Train ########
python icecube_conditional_decoder_film.py "${ARGS[@]}"

