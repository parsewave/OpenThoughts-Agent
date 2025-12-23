#!/bin/bash
#SBATCH -p capella
#SBATCH --account p_agents_finetuning
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --mem=188000
#SBATCH --job-name=data-cat-ws-befe330h-dcagent-checkpoints-GLM-4.6-stackexchange-overflow-sandboxes-32eps-65k-reas
#SBATCH --output=experiments/logs/data-cat-ws-befe330h-dcagent-checkpoints-GLM-4.6-stackexchange-overflow-sandboxes-32eps-65k-reas_%j.out
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user=bf996@nyu.edu
#SBATCH --exclude=c66,c68

set -euo pipefail

if [ -n "${DCFT:-}" ] && [ -f "$DCFT/hpc/dotenv/zih_capella.env" ]; then
  source "$DCFT/hpc/dotenv/zih_capella.env"
elif [ -n "${DC_AGENT:-}" ] && [ -f "$DC_AGENT/hpc/dotenv/zih_capella.env" ]; then
  source "$DC_AGENT/hpc/dotenv/zih_capella.env"
fi

module load CUDA/12.8.0

if [ -n "${DCFT_ACTIVATE_ENV:-}" ]; then
  set +u
  eval "$DCFT_ACTIVATE_ENV"
  set -u
fi

SECRET_FILE="${DC_AGENT_SECRET_ENV:-${KEYS:-}}"
if [[ -n "${SECRET_FILE}" && -f "${SECRET_FILE}" ]]; then
  set -a
  source "${SECRET_FILE}"
  set +a
fi

export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1

if [ -z "${DCFT:-}" ]; then
  if [ -n "${DC_AGENT:-}" ]; then
    export DCFT="$DC_AGENT"
  else
    export DCFT="$PWD"
  fi
fi

mkdir -p "$DCFT/experiments"
mkdir -p "$DCFT/experiments/logs"

export CONSOLIDATE_INPUT="/data/cat/ws/befe330h-dcagent/checkpoints/GLM-4.6-stackexchange-overflow-sandboxes-32eps-65k-reasoning_Qwen3-32B"
export CONSOLIDATE_INPUT_KIND="local"
export CONSOLIDATE_OUTPUT_REPO="laion/GLM-4.6-stackexchange-overflow-sandboxes-32eps-65k-reasoning_Qwen3-32B"
export CONSOLIDATE_BASE_REPO="Qwen/Qwen3-32B"
export CONSOLIDATE_WORKDIR="/data/cat/ws/befe330h-dcagent/consolidate/qwen3-32b"
export CONSOLIDATE_COMMIT_MESSAGE="Merge ZeRO shards into safetensors"

python3 /data/cat/ws/befe330h-dcagent/OpenThoughts-Agent/hpc/sbatch_consolidate/consolidate.py \
  --input "${CONSOLIDATE_INPUT}" \
  --input-kind "${CONSOLIDATE_INPUT_KIND}" \
  --output-repo "${CONSOLIDATE_OUTPUT_REPO}" \
  --base-repo "${CONSOLIDATE_BASE_REPO}" \
  --workdir "${CONSOLIDATE_WORKDIR}" \
  --commit-message "${CONSOLIDATE_COMMIT_MESSAGE}" \
  --project-root "/data/cat/ws/befe330h-dcagent/OpenThoughts-Agent"
