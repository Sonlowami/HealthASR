#!/bin/bash
# Submit Whisper FFN pruning + recovery finetune (torch-pruning).
#
# Example:
#   CONFIG=config/whisper_prune.yaml \
#   RUN_ROOT=/project/community/rmwisene/pipeline_outputs/compression/whisper_kin_dav_prune \
#   MODEL_PATH=/project/community/rmwisene/pipeline_outputs/whisper_runs/kin-dav-balanced-27h-curriculum-e15/final \
#   SLURM_TIME=12:00:00 \
#   ./scripts/submit_prune_whisper.sh --ratio 0.1 --ratio 0.2 --ratio 0.5

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-config/whisper_prune.yaml}"

SLURM_PARTITION="${SLURM_PARTITION:-general}"
SLURM_TIME="${SLURM_TIME:-12:00:00}"
SLURM_MEM="${SLURM_MEM:-128G}"
SLURM_CPUS="${SLURM_CPUS:-32}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-whisper-prune}"

PROJECT_ROOT="${PROJECT_ROOT:-/project/community/rmwisene}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/pipeline_outputs/compression/whisper_prune}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/hf_cache}"
TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/torch_cache}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/xdg_cache}"
MODEL_PATH="${MODEL_PATH:-}"

mkdir -p "$RUN_ROOT" "$HF_HOME/hub" "$TORCH_HOME" "$XDG_CACHE_HOME"

PY_ARGS=(--config "$CONFIG" --output_dir "$RUN_ROOT")
if [[ -n "$MODEL_PATH" ]]; then
  PY_ARGS+=(--model_path "$MODEL_PATH")
fi
PY_ARGS+=("$@")

PY_ARGS_STR=$(printf '%q ' "${PY_ARGS[@]}")

sbatch <<EOF
#!/bin/bash
#SBATCH -p ${SLURM_PARTITION}
#SBATCH --gres=gpu:${SLURM_GPUS}
#SBATCH --time=${SLURM_TIME}
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH -J ${SLURM_JOB_NAME}
#SBATCH -o ${RUN_ROOT}/prune_%j.log

set -euo pipefail
source "\${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate healthasr

export HF_HOME="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${TORCH_HOME}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME}"
export TMPDIR="/tmp/w_\${SLURM_JOB_ID}"
mkdir -p "\$HUGGINGFACE_HUB_CACHE" "\$HF_DATASETS_CACHE" "\$TMPDIR"

cd "${REPO_ROOT}"
set -a
[[ -f .env ]] && source .env
set +a

python -c "import torch_pruning" 2>/dev/null || pip install -q torch-pruning
python -c "import thop" 2>/dev/null || pip install -q thop

python training/whisper/prune_finetune.py ${PY_ARGS_STR}
EOF
