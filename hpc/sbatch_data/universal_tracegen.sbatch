#!/bin/bash
#SBATCH --time={time_limit}
#SBATCH --nodes {num_nodes}
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --output={experiments_dir}/logs/%x_%j.out
#SBATCH --job-name={job_name}
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user={email_address}
{sbatch_extra_directives}

# ==============================================================================
# Universal Trace Generation SBATCH Template
# ==============================================================================
# This template replaces the 600+ line cluster-specific scripts by delegating
# all logic to the TracegenJobRunner Python class.
#
# Usage: The launcher writes a JSON config file and substitutes {config_path}
# ==============================================================================

set -euo pipefail
ulimit -c 0  # Disable core dumps to avoid filling disk space

# Handle bash completion scripts that use BASH_COMPLETION_DEBUG
if [ -z "${BASH_COMPLETION_DEBUG+x}" ]; then
  export BASH_COMPLETION_DEBUG=""
fi

# --- Module and Conda Setup ---
# Disable unbound variable check for module loading and conda activation.
# Module reloads can trigger conda deactivation scripts that reference unset variables.
set +u

# --- Module loading (cluster-specific, substituted by launcher) ---
{module_commands}

# Some module command templates toggle set -u; ensure we stay in +u mode
# through conda activation and env sourcing to avoid unbound variable errors.
set +u

# --- Environment setup ---
if [ -n "${DCFT_PRIVATE:-}" ]; then
  WORKDIR="$DCFT_PRIVATE"
elif [ -n "${DCFT:-}" ]; then
  WORKDIR="$DCFT"
else
  WORKDIR="$PWD"
fi
cd "$WORKDIR"

if [ -z "${DCFT:-}" ]; then
  export DCFT="$WORKDIR"
fi

# --- Conda activation (cluster-specific, substituted by launcher) ---
{conda_activate}

# --- Source environment files ---
# Keep set +u active - dotenv files and DCFT_ACTIVATE_ENV may trigger module reloads
# which can trigger conda deactivation scripts with unset variables
if [ -n "${DCFT:-}" ] && [ -f "$DCFT/hpc/dotenv/{cluster_env_file}" ]; then
  source "$DCFT/hpc/dotenv/{cluster_env_file}"
fi
if [ -n "${DC_AGENT_SECRET_ENV:-}" ] && [ -f "$DC_AGENT_SECRET_ENV" ]; then
  source "$DC_AGENT_SECRET_ENV"
fi
if [ -n "${DCFT_ACTIVATE_ENV:-}" ]; then
  eval "$DCFT_ACTIVATE_ENV"
fi

# Re-enable strict unbound variable checking after all env setup is complete
set -u

# --- Standard environment variables ---
export HF_HOME="/tmp/hf_home"
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=0
export NCCL_TIMEOUT=1800
export NCCL_IB_TIMEOUT=23
export PYTORCH_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"

# --- Ray/vLLM GPU visibility fix ---
# Prevent Ray from modifying CUDA_VISIBLE_DEVICES per-actor.
# Without these, Ray sets CVD to a single physical GPU index (e.g. "3") per actor,
# but vLLM uses Ray-reported indices as torch ordinals, causing "invalid device ordinal".
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_NOSET_CUDA_VISIBLE_DEVICES=1

# --- Triton/TorchInductor cache settings (node-local to avoid shared FS issues) ---
export TRITON_CACHE_VERBOSE=1
source "$WORKDIR/hpc/shell_utils/triton_cache.sh"

# --- Create experiment directories ---
# Note: {experiments_dir} is an absolute path substituted by the launcher
mkdir -p "{experiments_dir}"
mkdir -p "{experiments_dir}/logs"
mkdir -p "{experiments_dir}/ray_logs"

# --- Cleanup trap to preserve Ray logs ---
# Ray uses /tmp by default for logs/sockets. This trap copies them to the
# experiments directory for debugging after the job terminates.
cleanup_ray_logs() {
  echo "Preserving Ray logs to {experiments_dir}/ray_logs/"
  # Find and copy Ray session directories from /tmp on head node
  if [[ -d /tmp/ray ]]; then
    rsync -a --ignore-errors /tmp/ray/ "{experiments_dir}/ray_logs/" 2>/dev/null || true
  fi
  # Also try common Ray log locations
  for ray_dir in /tmp/ray_logs /tmp/ray_tmp; do
    if [[ -d "$ray_dir" ]]; then
      rsync -a --ignore-errors "$ray_dir/" "{experiments_dir}/ray_logs/$(basename $ray_dir)/" 2>/dev/null || true
    fi
  done
  echo "Ray log preservation complete"
}
trap cleanup_ray_logs EXIT

# --- Docker/Podman Runtime Setup (for Harbor Docker backend) ---
# Only run if harbor_env is "docker" - other backends (daytona, modal) don't need this.
if [ "{harbor_env}" = "docker" ]; then
  if [ -f "$WORKDIR/docker/setup_docker_runtime.sh" ]; then
    echo "[sbatch] Setting up Docker/Podman runtime..."
    # Source the setup script to detect and configure runtime
    # shellcheck source=docker/setup_docker_runtime.sh
    source "$WORKDIR/docker/setup_docker_runtime.sh"

    # Ensure a docker-compatible CLI exists when only podman-hpc is available.
    if ! command -v docker &>/dev/null && command -v podman-hpc &>/dev/null; then
      mkdir -p "$WORKDIR/.bin"
      cat > "$WORKDIR/.bin/docker" <<'EOF'
#!/usr/bin/env bash
exec podman-hpc "$@"
EOF
      chmod +x "$WORKDIR/.bin/docker"
      export PATH="$WORKDIR/.bin:$PATH"
      echo "[sbatch] Installed podman-hpc docker shim at $WORKDIR/.bin/docker"
    fi

    # Verify connectivity if DOCKER_HOST is set
    if [ -n "${DOCKER_HOST:-}" ]; then
      echo "[sbatch] DOCKER_HOST=$DOCKER_HOST"
      echo "[sbatch] CONTAINER_RUNTIME=${CONTAINER_RUNTIME:-unknown}"

      # Quick connectivity check (timeout after 10s)
      # Try docker first, then podman-hpc (Perlmutter may not have docker CLI)
      if command -v docker &>/dev/null && timeout 10 docker info &>/dev/null; then
        echo "[sbatch] Docker runtime verified successfully (docker info)"
      elif command -v podman-hpc &>/dev/null && timeout 10 podman-hpc info &>/dev/null; then
        echo "[sbatch] Docker runtime verified successfully (podman-hpc info)"
      elif command -v podman &>/dev/null && timeout 10 podman info &>/dev/null; then
        echo "[sbatch] Docker runtime verified successfully (podman info)"
      else
        echo "[sbatch] ERROR: Docker daemon not responding at DOCKER_HOST=$DOCKER_HOST"
        echo "[sbatch] HINT: Check 'podman-hpc info' or 'docker info' manually"
        exit 4  # EXIT_CONNECTIVITY_FAILED
      fi
    else
      echo "[sbatch] WARNING: DOCKER_HOST not set, Harbor Docker backend may fail"
    fi
  fi
else
  echo "[sbatch] Harbor env: {harbor_env} (skipping Docker runtime setup)"
fi

# --- Run the trace generation job via Python runner ---
echo "=== Universal Trace Generation Runner ==="
echo "Config: {config_path}"
echo "Working directory: $WORKDIR"
echo "========================================="

python -m hpc.datagen_launch_utils --mode tracegen --config "{config_path}"
