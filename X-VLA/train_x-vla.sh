#!/usr/bin/env bash
#SBATCH --job-name=xvla_factory_conveyor
#SBATCH --output=/projects/haitian003ssd/ICRA2027-baseline/X-VLA/logs/xvla_factory_conveyor_%j.out
#SBATCH --error=/projects/haitian003ssd/ICRA2027-baseline/X-VLA/logs/xvla_factory_conveyor_%j.err
#SBATCH --gpus=pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

BASE_DIR=/projects/haitian003ssd/ICRA2027-baseline
X_VLA_DIR=${BASE_DIR}/X-VLA
CONDA_ENV=${BASE_DIR}/.conda/lerobot-baselines
PRETRAINED_PATH=${PRETRAINED_PATH:-${X_VLA_DIR}/pretrained}

module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
cd "${X_VLA_DIR}"

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-${BASE_DIR}/.cache/hf}
export WANDB_DIR=${WANDB_DIR:-${X_VLA_DIR}/wandb}
export WANDB_MODE=${WANDB_MODE:-online}

# Keep the credential outside this script. WANDB_API_KEY takes precedence; a
# single-line key in WANDB_API_KEY_FILE is also supported for Slurm jobs.
WANDB_API_KEY_FILE=${WANDB_API_KEY_FILE:-${X_VLA_DIR}/.wandb_api_key}
if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_API_KEY_FILE}" ]]; then
    export WANDB_API_KEY
    WANDB_API_KEY=$(<"${WANDB_API_KEY_FILE}")
fi

# LeRobot v3 factory-conveyor dataset. Its logical local id follows the Pi05 recipe.
DATASET_ROOT=${DATASET_ROOT:-/projects/haitian003ssd/Dataset/factory_conveyor_level2_seeded}
DATASET_REPO_ID=${DATASET_REPO_ID:-local/factory_conveyor_level2_seeded}
RUN_NAME=${RUN_NAME:-xvla_factory_conveyor_${SLURM_JOB_ID:-local}}
OUTPUT_DIR=${OUTPUT_DIR:-${X_VLA_DIR}/outputs/train/${RUN_NAME}}

if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
    echo "DATASET_ROOT is not a LeRobot dataset: ${DATASET_ROOT}" >&2
    exit 1
fi
DATASET_VERSION=$(sed -n 's/.*"codebase_version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "${DATASET_ROOT}/meta/info.json" | head -n 1)
if [[ "${DATASET_VERSION}" != "v3.0" ]]; then
    echo "${DATASET_ROOT} is LeRobot ${DATASET_VERSION:-unknown}; this X-VLA environment requires v3.0." >&2
    echo "Use the converted sibling dataset or explicitly convert a copy before training." >&2
    exit 1
fi
if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY is required for online logging. Export it before sbatch, or put it in ${WANDB_API_KEY_FILE}." >&2
    exit 1
fi

# LeRobot creates OUTPUT_DIR itself and refuses to overwrite an existing run.
mkdir -p "${X_VLA_DIR}/logs" "${WANDB_DIR}"
if [[ ! -f "${PRETRAINED_PATH}/config.json" ]]; then
    echo "X-VLA pretrained checkpoint is missing: ${PRETRAINED_PATH}" >&2
    exit 1
fi
command -v lerobot-train >/dev/null
python -c 'import wandb; print(f"W&B client: {wandb.__version__}")'

# X-VLA's recommended Phase-II configuration: tune the VLM, policy transformer
# and soft prompts in bfloat16. `auto` pads/trims the dataset action dimension.
lerobot-train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.video_backend=pyav \
    --rename_map='{"observation.images.front":"observation.images.image","observation.images.overview":"observation.images.image2","observation.images.wrist":"observation.images.image3"}' \
    --policy.path="${PRETRAINED_PATH}" \
    --policy.push_to_hub=false \
    --policy.dtype=bfloat16 \
    --policy.action_mode=auto \
    --policy.max_action_dim=20 \
    --policy.device=cuda \
    --policy.freeze_vision_encoder=false \
    --policy.freeze_language_encoder=false \
    --policy.train_policy_transformer=true \
    --policy.train_soft_prompts=true \
    --batch_size="${BATCH_SIZE:-32}" \
    --steps="${STEPS:-80000}" \
    --num_workers="${NUM_WORKERS:-4}" \
    --log_freq="${LOG_FREQ:-50}" \
    --save_freq="${SAVE_FREQ:-40000}" \
    --env_eval_freq=0 \
    --eval_steps=0 \
    --seed="${SEED:-1000}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${RUN_NAME}" \
    --wandb.enable=true \
    --wandb.mode="${WANDB_MODE}" \
    --wandb.project="${WANDB_PROJECT:-xvla_factory_conveyor}"
