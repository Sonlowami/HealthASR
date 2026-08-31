#!/bin/bash
# Int8 QAT Whisper inference → Kin test CSV + WER metrics.
#
# Example:
#   CONFIG=config/whisper_int8_kin_test.yaml \
#   RUN_ROOT=/project/community/rmwisene/pipeline_outputs/whisper_predictions \
#   SLURM_TIME=3:00:00 \
#   ./scripts/submit_infer_int8_csv.sh --keep_long_audio

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-config/whisper_int8_kin_test.yaml}"

SLURM_PARTITION="${SLURM_PARTITION:-general}"
SLURM_TIME="${SLURM_TIME:-3:00:00}"
SLURM_MEM="${SLURM_MEM:-128G}"
SLURM_CPUS="${SLURM_CPUS:-32}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-whisper-int8-test}"

PROJECT_ROOT="${PROJECT_ROOT:-/project/community/rmwisene}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/pipeline_outputs/whisper_predictions}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/hf_cache}"
TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/torch_cache}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/xdg_cache}"

mkdir -p "$RUN_ROOT" "$HF_HOME/hub" "$TORCH_HOME" "$XDG_CACHE_HOME"

PY_ARGS=(--config "$CONFIG" --output_dir "$RUN_ROOT")
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
#SBATCH -o ${RUN_ROOT}/infer_int8_%j.log

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

python training/whisper/infer_int8_csv.py ${PY_ARGS_STR}
EOF
