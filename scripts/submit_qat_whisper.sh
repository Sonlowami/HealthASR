#!/bin/bash
# Submit Whisper QAT + short finetune (torchao). Model-agnostic via MODEL_PATH / --model_path.
#
# Example:
#   CONFIG=config/whisper_qat.yaml \
#   RUN_ROOT=/project/community/rmwisene/pipeline_outputs/compression/whisper_kin_dav_qat \
#   MODEL_PATH=/project/community/rmwisene/pipeline_outputs/whisper_runs/kin-dav-balanced-27h-curriculum-e15/final \
#   SLURM_TIME=2:00:00 \
#   ./scripts/submit_qat_whisper.sh --quant int8_weight_qat

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-config/whisper_qat.yaml}"

SLURM_PARTITION="${SLURM_PARTITION:-general}"
SLURM_TIME="${SLURM_TIME:-2:00:00}"
SLURM_MEM="${SLURM_MEM:-128G}"
SLURM_CPUS="${SLURM_CPUS:-32}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-whisper-qat}"

PROJECT_ROOT="${PROJECT_ROOT:-/project/community/rmwisene}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/pipeline_outputs/compression/whisper_qat}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/hf_cache}"
TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/torch_cache}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/xdg_cache}"
MODEL_PATH="${MODEL_PATH:-}"

mkdir -p "$RUN_ROOT" "$HF_HOME/hub" "$TORCH_HOME" "$XDG_CACHE_HOME"

# Build argv for the python job (stored for the batch script)
PY_ARGS=(--config "$CONFIG" --output_dir "$RUN_ROOT")
if [[ -n "$MODEL_PATH" ]]; then
  PY_ARGS+=(--model_path "$MODEL_PATH")
fi
PY_ARGS+=("$@")

# Serialize args safely for the heredoc
PY_ARGS_STR=$(printf '%q ' "${PY_ARGS[@]}")

sbatch <<EOF
#!/bin/bash
#SBATCH -p ${SLURM_PARTITION}
#SBATCH --gres=gpu:${SLURM_GPUS}
#SBATCH --time=${SLURM_TIME}
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH -J ${SLURM_JOB_NAME}
#SBATCH -o ${RUN_ROOT}/qat_%j.log

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

python -c "import torchao" 2>/dev/null || pip install -q torchao

python training/whisper/qat_finetune.py ${PY_ARGS_STR}
EOF
