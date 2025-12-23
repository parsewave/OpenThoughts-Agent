#!/bin/bash
#SBATCH --account=p_agents_finetuning
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=710GB
#SBATCH --time=47:59:00
#SBATCH --exclusive
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --job-name=stackexchange-tezos-sandboxes_glm_4.6_traces_together_Qwen3-8B
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user=bf996@nyu.edu
#SBATCH --exclude=c66,c68

export DISABLE_VERSION_CHECK=1
set -euo pipefail

export CONDA_BACKUP_CXX="${CONDA_BACKUP_CXX:-}"
export CONDA_BACKUP_CC="${CONDA_BACKUP_CC:-}"
export CONDA_BACKUP_FC="${CONDA_BACKUP_FC:-}"

if [ -n "${DCFT:-}" ] && [ -f "$DCFT/hpc/dotenv/zih_capella.env" ]; then
  # shellcheck disable=SC1090
  source "$DCFT/hpc/dotenv/zih_capella.env"
elif [ -n "${DC_AGENT:-}" ] && [ -f "$DC_AGENT/hpc/dotenv/zih_capella.env" ]; then
  # shellcheck disable=SC1090
  source "$DC_AGENT/hpc/dotenv/zih_capella.env"
fi

module load CUDA/12.8.0

if [ -n "${DCFT_ACTIVATE_ENV:-}" ]; then
  set +u
  eval "$DCFT_ACTIVATE_ENV"
  set -u
fi

SECRET_FILE="${DC_AGENT_SECRET_ENV:-${KEYS:-}}"
if [[ -n "${SECRET_FILE}" ]]; then
    if [[ -f "${SECRET_FILE}" ]]; then
        echo "Sourcing secrets from ${SECRET_FILE}"
        set -a
        # shellcheck disable=SC1090
        source "${SECRET_FILE}"
        set +a
    else
        echo "Warning: secrets file ${SECRET_FILE} not found; database registration may fail." >&2
    fi
else
    echo "Warning: DC_AGENT_SECRET_ENV is not set; database registration may fail." >&2
fi

for _supabase_var in SUPABASE_URL SUPABASE_ANON_KEY SUPABASE_SERVICE_ROLE_KEY; do
    if [[ -n "${!_supabase_var:-}" ]]; then
        export "${_supabase_var}=${!_supabase_var}"
    else
        echo "Warning: ${_supabase_var} is not set; Supabase registration may fail." >&2
    fi
done

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_PROTO=simple
export NCCL_IB_TIMEOUT=23
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_NET_GDR_LEVEL="SYS"
export NCCL_NET_GDR_READ=1
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
TMP_DIR="$DCFT/experiments/tmp"
mkdir -p "$TMP_DIR"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-12802}
export NUM_NODES=$SLURM_JOB_NUM_NODES
export NUM_GPUS_PER_NODE=4
export NUM_GPUS=$((NUM_GPUS_PER_NODE*SLURM_NNODES))
export HF_HOME=${HF_HOME:-$HF_HUB_CACHE}
export WANDB_DIR=${DCFT_WANDB_DIR:-$DCFT/experiments/wandb}

CONFIG="$DCFT/experiments/configs/stackexchange-tezos-sandboxes_glm_4.6_traces_together_Qwen3-8B_train_config.yaml"
OUTPUT_DIR="$DCFT/experiments"
echo -e "CONFIG: $CONFIG\nOUTPUT_DIR: $OUTPUT_DIR"

DEEPSPEED_CONFIG_FILE=sft/llamafactory/examples/deepspeed/ds_z3_config.json

CMD="torchrun \
    --nproc-per-node $NUM_GPUS_PER_NODE \
    --nnodes $NUM_NODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    sft/llamafactory/src/train.py $CONFIG"

SRUN_ARGS="
    --nodes=$NUM_NODES \
    --gpus-per-node=$NUM_GPUS_PER_NODE \
    --cpus-per-task=$SLURM_CPUS_PER_TASK \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --label \
    --jobid $SLURM_JOBID"

export PYTHONPATH="$DCFT:${PYTHONPATH:-}"

cd "$DCFT"
srun $SRUN_ARGS bash -c "$CMD"
