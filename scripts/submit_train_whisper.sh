#!/bin/bash
# Submit Whisper training with all caches/logs under /project (not $HOME).
#
# Example:
#   SLURM_TIME=12:00:00 ./scripts/submit_train_whisper.sh
#   SLURM_PARTITION=preempt SLURM_TIME=12:00:00 ./scripts/submit_train_whisper.sh --resume
#
# Extra args after the script are passed to train.py (e.g. --resume --curriculum).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-config/whisper_config.yaml}"

# --- Slurm (override via env) ---
SLURM_PARTITION="${SLURM_PARTITION:-general}"
SLURM_TIME="${SLURM_TIME:-12:00:00}"
SLURM_MEM="${SLURM_MEM:-128G}"
SLURM_CPUS="${SLURM_CPUS:-16}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-whisper-train}"

# --- Project-local paths (nothing in $HOME) ---
PROJECT_ROOT="${PROJECT_ROOT:-/project/community/rmwisene}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/pipeline_outputs/whisper_runs/kin-only-sunbird-e10}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/hf_cache}"
WANDB_DIR="${WANDB_DIR:-$RUN_ROOT/wandb}"
# Do NOT inherit shell TMPDIR (/mnt/tmp on orchard) — always use project path
JOB_TMPDIR="${JOB_TMPDIR:-$RUN_ROOT/tmp}"
TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/torch_cache}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/xdg_cache}"

mkdir -p "$RUN_ROOT" "$HF_HOME/hub" "$WANDB_DIR" "$JOB_TMPDIR" "$TORCH_HOME" "$XDG_CACHE_HOME"

# Remaining CLI args → train.py
TRAIN_ARGS=("$@")
if [[ ${#TRAIN_ARGS[@]} -eq 0 ]]; then
  TRAIN_ARGS=(--resume)
fi

sbatch <<EOF
#!/bin/bash
#SBATCH -p ${SLURM_PARTITION}
#SBATCH --gres=gpu:${SLURM_GPUS}
#SBATCH --time=${SLURM_TIME}
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH -J ${SLURM_JOB_NAME}
#SBATCH -o ${RUN_ROOT}/train_%j.log

set -euo pipefail
source "\${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate healthasr

export HF_HOME="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${TORCH_HOME}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME}"
export TMPDIR="${JOB_TMPDIR}"
export WANDB_DIR="${WANDB_DIR}"
export WANDB_CACHE_DIR="${WANDB_DIR}/cache"
mkdir -p "\$HUGGINGFACE_HUB_CACHE" "\$HF_DATASETS_CACHE" "\$WANDB_CACHE_DIR" "\$TMPDIR"

cd "${REPO_ROOT}"
set -a
[[ -f .env ]] && source .env
set +a

python training/whisper/train.py --config "${CONFIG}" ${TRAIN_ARGS[*]}
EOF
