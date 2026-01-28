#!/bin/bash
# =============================================================================
# RL Environment Setup Script
# =============================================================================
# Creates a standalone Python 3.12 virtual environment for SkyRL training.
# This environment is separate from the main project environment due to
# dependency conflicts between RL (torch 2.8, vllm 0.11.0) and datagen
# (torch 2.9, vllm 0.11.2).
#
# Usage:
#   ./hpc/setup_rl_env.sh [--force] [--rocm]
#
# Options:
#   --force    Remove existing RL environment and recreate
#   --rocm     Install ROCm/AMD GPU dependencies instead of CUDA (for Frontier)
#
# The environment is created at: $DCFT/envs/rl or ./envs/rl
# The RL launcher (hpc/launch.py --job_type rl) will automatically use this.
# =============================================================================

set -euo pipefail

# Configuration
RL_ENV_NAME="rl"
PYTHON_VERSION="3.12"
USE_ROCM=false

# Detect architecture (aarch64 for ARM-based systems like GH200)
ARCH=$(uname -m)
IS_AARCH64=false
if [[ "$ARCH" == "aarch64" ]] || [[ "$ARCH" == "arm64" ]]; then
    IS_AARCH64=true
fi

# ROCm configuration (for OLCF Frontier with AMD MI250X GPUs)
# See: https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html
# Note: vLLM wheels are available for ROCm 7.0.0 - try that first, fallback to 6.4.1
ROCM_VERSION="7.0.2"
ROCM_VERSION_FALLBACK="6.4.1"
ROCM_MODULES=(
    "PrgEnv-gnu/8.6.0"
    "cray-mpich/9.0.0"
    "gcc-native/14.2"
    "miniforge3/23.11.0-0"
    "rocm/7.0.2"
    "craype-accel-amd-gfx90a"
)
ROCM_MODULES_FALLBACK=(
    "PrgEnv-gnu/8.6.0"
    "cray-mpich/9.0.0"
    "gcc-native/14.2"
    "miniforge3/23.11.0-0"
    "rocm/6.4.1"
    "craype-accel-amd-gfx90a"
)

# Determine base directory (project root with pyproject.toml)
# DCFT_PRIVATE is the project dir on clusters where DCFT is a parent scratch dir
if [[ -n "${DCFT_PRIVATE:-}" ]]; then
    BASE_DIR="$DCFT_PRIVATE"
elif [[ -n "${DCFT:-}" ]]; then
    BASE_DIR="$DCFT"
else
    BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

RL_ENV_DIR="$BASE_DIR/envs/$RL_ENV_NAME"
RL_REQUIREMENTS="$BASE_DIR/hpc/rl_requirements.txt"

# Parse arguments
FORCE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift
            ;;
        --rocm)
            USE_ROCM=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--force] [--rocm]"
            exit 1
            ;;
    esac
done

echo "=== RL Environment Setup ==="
echo "Base directory: $BASE_DIR"
echo "Environment directory: $RL_ENV_DIR"
echo "Python version: $PYTHON_VERSION"
echo "Architecture: $ARCH"
if [[ "$USE_ROCM" == "true" ]]; then
    echo "GPU Backend: ROCm $ROCM_VERSION (AMD)"
elif [[ "$IS_AARCH64" == "true" ]]; then
    echo "GPU Backend: CUDA (NVIDIA) - aarch64/ARM"
else
    echo "GPU Backend: CUDA (NVIDIA) - x86_64"
fi
echo ""

# Ensure ~/.local/bin is in PATH (for user-installed tools like uv)
export PATH="$HOME/.local/bin:$PATH"

# Check for uv and install if not available BEFORE loading modules
# (module loads can deactivate conda and change PATH)
if ! command -v uv &> /dev/null; then
    echo "'uv' not found. Installing uv to ~/.local/bin..."
    # Prefer curl installer - it's more reliable and always installs to ~/.local/bin
    if command -v curl &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    elif command -v pip &> /dev/null; then
        pip install --user uv --quiet 2>/dev/null || pip install --user uv
    elif command -v pip3 &> /dev/null; then
        pip3 install --user uv --quiet 2>/dev/null || pip3 install --user uv
    fi
    # Verify installation
    if ! command -v uv &> /dev/null; then
        echo "Error: Failed to install uv. Please install manually:"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    echo "uv installed successfully."
fi

# Load ROCm modules if requested (for Frontier/AMD systems)
# This is done AFTER uv install since module loads can change PATH
if [[ "$USE_ROCM" == "true" ]]; then
    echo "Loading ROCm modules for AMD GPU support..."
    # Check if module command exists (HPC systems)
    if command -v module &> /dev/null; then
        # Try ROCm 7.0.0 first (required for vLLM wheel), fall back to 6.4.1
        ROCM_LOADED=false
        for mod in "${ROCM_MODULES[@]}"; do
            echo "  module load $mod"
            if module load "$mod" 2>/dev/null; then
                ROCM_LOADED=true
            else
                echo "    Warning: Could not load $mod"
                # If ROCm 7.0.0 failed, try fallback
                if [[ "$mod" == "rocm/7.0.2" ]]; then
                    echo "    Trying fallback: rocm/${ROCM_VERSION_FALLBACK}..."
                    if module load "rocm/${ROCM_VERSION_FALLBACK}" 2>/dev/null; then
                        ROCM_VERSION="$ROCM_VERSION_FALLBACK"
                        ROCM_LOADED=true
                        echo "    Loaded ROCm ${ROCM_VERSION_FALLBACK} (fallback)"
                    fi
                fi
            fi
        done
        echo ""
        echo "Using ROCm version: $ROCM_VERSION"
    else
        echo "Warning: 'module' command not found. Skipping module loads."
        echo "Make sure ROCm is available in your PATH."
        echo ""
    fi
    # Re-add ~/.local/bin to PATH after module loads (they can reset PATH)
    export PATH="$HOME/.local/bin:$PATH"

    # Check for uv again after module loads (conda deactivation may have removed it)
    if ! command -v uv &> /dev/null; then
        echo "'uv' not found after module loads. Installing uv..."
        # Use curl installer - most reliable, always installs to ~/.local/bin
        if command -v curl &> /dev/null; then
            echo "Using curl installer for uv..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            # Add both possible install locations to PATH
            export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        elif command -v pip &> /dev/null; then
            # Fallback to pip, but find the actual install location
            pip install --user uv
            # Try to find where pip installed the uv script
            UV_SCRIPT=$(python -c "import site; print(site.USER_BASE)" 2>/dev/null)/bin
            if [[ -d "$UV_SCRIPT" ]]; then
                export PATH="$UV_SCRIPT:$PATH"
            fi
        fi
        if ! command -v uv &> /dev/null; then
            echo "Error: Failed to install uv after module loads."
            echo "Please install manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
            exit 1
        fi
        echo "uv installed successfully."
    fi
fi

# Check Python 3.12 availability
if ! uv python find "$PYTHON_VERSION" &> /dev/null; then
    echo "Python $PYTHON_VERSION not found. Attempting to install..."
    uv python install "$PYTHON_VERSION"
fi

# Handle existing environment
if [[ -d "$RL_ENV_DIR" ]]; then
    if [[ "$FORCE" == "true" ]]; then
        echo "Removing existing environment (--force)..."
        rm -rf "$RL_ENV_DIR"
    else
        echo "RL environment already exists at: $RL_ENV_DIR"
        echo "Use --force to recreate, or activate with:"
        echo "  source $RL_ENV_DIR/bin/activate"
        exit 0
    fi
fi

# Create environment directory
mkdir -p "$(dirname "$RL_ENV_DIR")"

echo "Creating Python $PYTHON_VERSION virtual environment..."
uv venv "$RL_ENV_DIR" --python "$PYTHON_VERSION"

echo "Activating environment..."
source "$RL_ENV_DIR/bin/activate"

echo "Installing RL dependencies..."

# =============================================================================
# IMPORTANT: Install PyTorch FIRST
# =============================================================================
# flash-attn (a dependency of skyrl-train) requires torch to be present during
# its build phase. We install torch first to avoid build failures.
# Version must match skyrl-train[vllm] requirement (torch==2.8.0)
echo "Installing PyTorch first (required for flash-attn build)..."
if [[ "$USE_ROCM" == "true" ]]; then
    # ROCm/AMD version for Frontier (MI250X GPUs)
    # See: https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html
    echo "Installing PyTorch with ROCm $ROCM_VERSION support..."
    uv pip install "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0" \
        --index-url https://download.pytorch.org/whl/rocm6.4
elif [[ "$IS_AARCH64" == "true" ]]; then
    # aarch64/ARM (e.g., GH200 Grace-Hopper)
    # The cu128-specific wheels don't have aarch64 builds.
    # Use standard PyPI wheels which include aarch64 CUDA builds.
    # Note: PyTorch 2.8.0 aarch64 wheels have CUDA 12.8 support built-in.
    echo "Installing PyTorch for aarch64 (using standard PyPI wheels)..."
    uv pip install "torch==2.8.0" "torchvision" "torchaudio"
else
    # CUDA/NVIDIA x86_64 version (default)
    uv pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
fi

# =============================================================================
# Install build dependencies (needed for flash-attn and harbor)
# =============================================================================
echo "Installing build dependencies (packaging, uv_build)..."
uv pip install packaging "uv_build>=0.8.4,<0.9.0" || true

# =============================================================================
# Try to install flash-attn (optional but recommended) - CUDA only
# =============================================================================
FLASH_ATTN_INSTALLED=false
if [[ "$USE_ROCM" == "true" ]]; then
    echo ""
    echo "=== Skipping Flash Attention 2 (ROCm) ==="
    echo "flash-attn is CUDA-specific and not available for ROCm/AMD GPUs."
    echo "PyTorch will use its built-in attention implementation instead."
    echo ""
else
    echo ""
    echo "=== Installing Flash Attention 2 (optional) ==="
    echo "Note: flash-attn can be difficult to build on some systems."
    echo "If installation fails, training will still work (just slower)."
    echo ""

    # Try to install flash-attn with --no-build-isolation (uses installed torch)
    if uv pip install "flash-attn>=2.8.3" --no-build-isolation 2>&1; then
        echo "flash-attn installed successfully!"
        FLASH_ATTN_INSTALLED=true
    else
        echo ""
        echo "========================================================================"
        echo "WARNING: flash-attn installation failed."
        echo "This is common on systems without CUDA or with incompatible compilers."
        echo "Training will still work, but attention computation may be slower."
        echo ""
        echo "To try installing manually later:"
        echo "  pip install flash-attn --no-build-isolation"
        echo "========================================================================"
        echo ""
    fi
fi

# =============================================================================
# Install remaining dependencies
# =============================================================================
if [[ "$USE_ROCM" == "true" ]]; then
    # ROCm: Skip requirements file (contains skyrl-train[vllm] which pulls CUDA deps)
    # We'll install dependencies manually in the SkyRL section below
    echo "ROCm mode: Skipping requirements file (will install deps manually)"

    # Install Harbor and other non-CUDA deps from requirements
    echo "Installing Harbor and utilities..."
    uv pip install \
        "harbor @ git+https://github.com/laude-institute/harbor.git@penfever/temp-override" \
        "Jinja2" \
        "pyyaml" \
        || true
else
    # CUDA: Use requirements file as normal
    if [[ -f "$RL_REQUIREMENTS" ]]; then
        echo "Using requirements file: $RL_REQUIREMENTS"
        # Use --no-build-isolation so packages use our installed torch/flash-attn
        uv pip install -r "$RL_REQUIREMENTS" --no-build-isolation || {
            echo "Standard install failed, trying without flash-attn..."
            # Create a temp requirements file excluding skyrl-train (we'll install it separately)
            grep -v "skyrl-train" "$RL_REQUIREMENTS" > /tmp/rl_requirements_no_skyrl.txt || true
            uv pip install -r /tmp/rl_requirements_no_skyrl.txt --no-build-isolation || true
        }
    else
        echo "Using pyproject.toml [rl] extra..."
        # Install the project with rl extra
        # Note: We install in editable mode so changes to hpc/ are reflected
        uv pip install -e "$BASE_DIR[rl]" --no-build-isolation || true
    fi
fi

# =============================================================================
# Clone SkyRL repository (required for examples/terminal_bench entrypoints)
# =============================================================================
# The terminal_bench entrypoint lives in examples/, which is NOT part of the
# installed skyrl-train package. We need the full repo cloned.

SKYRL_REPO="https://github.com/penfever/SkyRL.git"
SKYRL_BRANCH="penfever/working"

# Determine SKYRL_HOME location
if [[ -n "${SKYRL_HOME:-}" ]]; then
    SKYRL_DIR="$SKYRL_HOME"
elif [[ -n "${SCRATCH:-}" ]]; then
    SKYRL_DIR="$SCRATCH/SkyRL"
else
    SKYRL_DIR="$BASE_DIR/SkyRL"
fi

echo ""
echo "=== SkyRL Repository Setup ==="
echo "Target directory: $SKYRL_DIR"

if [[ -d "$SKYRL_DIR" ]]; then
    echo "SkyRL repo already exists at: $SKYRL_DIR"
    echo "Updating to latest $SKYRL_BRANCH..."
    pushd "$SKYRL_DIR" > /dev/null
    git fetch origin
    git checkout "$SKYRL_BRANCH" 2>/dev/null || git checkout -b "$SKYRL_BRANCH" "origin/$SKYRL_BRANCH"
    git pull origin "$SKYRL_BRANCH" || echo "Warning: Could not pull latest changes"
    popd > /dev/null
else
    echo "Cloning SkyRL from $SKYRL_REPO..."
    git clone --branch "$SKYRL_BRANCH" "$SKYRL_REPO" "$SKYRL_DIR"
fi

echo "SkyRL repo ready at: $SKYRL_DIR"

# =============================================================================
# Install SkyRL packages (skyrl-train and skyrl-gym)
# =============================================================================
# skyrl-train: Core RL training framework
# skyrl-gym: Environment implementations (GSM8K, AIME, etc.) - required by skyrl-train
# Note: torch and flash-attn were already installed at the beginning of the script

echo ""
echo "=== Installing SkyRL Packages ==="

echo "Installing skyrl-gym (environment implementations)..."
uv pip install -e "$SKYRL_DIR/skyrl-gym" --no-build-isolation || uv pip install -e "$SKYRL_DIR/skyrl-gym"

echo "Installing skyrl-train (RL training framework)..."

if [[ "$USE_ROCM" == "true" ]]; then
    # ==========================================================================
    # ROCm-specific installation path
    # ==========================================================================
    # flash-attn is CUDA-only - we skip it and use PyTorch native attention
    # vLLM now has official ROCm wheels (as of Jan 2025)!
    echo "Using ROCm-compatible installation (skipping flash-attn)..."

    # Install non-CUDA dependencies manually
    echo "Installing ROCm-compatible dependencies..."
    uv pip install \
        "ray>=2.50.0" \
        "transformers>=4.51.0" \
        "accelerate" \
        "datasets>=4.0.0" \
        "omegaconf" \
        "hydra-core==1.3.2" \
        "loguru" \
        "wandb" \
        "peft" \
        "tensorboard" \
        "tqdm" \
        "polars" \
        "fastapi" \
        "uvicorn" \
        "jaxtyping" \
        "tensordict" \
        || true

    # Install skyrl-train without dependencies (to avoid flash-attn)
    echo "Installing skyrl-train (--no-deps to skip flash-attn)..."
    uv pip install -e "$SKYRL_DIR/skyrl-train" --no-deps

    # Install vLLM ROCm wheel (available for ROCm 7.0.x)
    # See: https://www.phoronix.com/news/AMD-ROCm-vLLM-Wheel
    # Note: vLLM ROCm requires pre-release aiter dependency, so we use --prerelease=allow
    if [[ "$ROCM_VERSION" == 7.0.* ]]; then
        echo "Installing vLLM with ROCm 7.0.0 wheel..."
        uv pip install "vllm==0.14.0+rocm700" \
            --extra-index-url https://wheels.vllm.ai/rocm/0.14.0/rocm700 \
            --prerelease=allow \
            || echo "Warning: vLLM ROCm wheel installation failed"
    else
        echo ""
        echo "NOTE: vLLM ROCm wheel requires ROCm 7.0.0, but ROCm $ROCM_VERSION is loaded."
        echo "vLLM will not be installed. For manual installation, see:"
        echo "  https://docs.vllm.ai/en/latest/getting_started/amd-installation.html"
    fi
else
    # ==========================================================================
    # CUDA installation path (original)
    # ==========================================================================
    # Use --no-build-isolation so it uses our pre-installed torch/flash-attn
    uv pip install -e "$SKYRL_DIR/skyrl-train" --no-build-isolation || {
        echo "Trying fallback installation..."
        # Install deps first, then editable package with --no-deps
        uv pip install ray transformers accelerate datasets omegaconf hydra-core loguru wandb vllm || true
        uv pip install -e "$SKYRL_DIR/skyrl-train" --no-deps
    }
fi

echo ""
echo "IMPORTANT: Add this to your shell or source your cluster's dotenv file:"
echo "  export SKYRL_HOME=\"$SKYRL_DIR\""
echo "  export PYTHONPATH=\"\$SKYRL_HOME/skyrl-train:\$PYTHONPATH\""

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check GPU backend
if [[ "$USE_ROCM" == "true" ]]; then
    python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()} (AMD uses CUDA API)')"
    python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')" 2>/dev/null || true
    echo "flash-attn: N/A (ROCm uses PyTorch native attention)"
else
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}')" 2>/dev/null || echo "flash-attn: NOT installed (optional - training will be slower)"
fi

python -c "import vllm; print(f'vLLM: {vllm.__version__}')" || echo "Warning: vLLM not installed (may be CPU-only system or ROCm)"
python -c "import ray; print(f'Ray: {ray.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import skyrl_gym; print(f'skyrl-gym: installed')"
python -c "import skyrl_train; print(f'skyrl-train: installed')"

echo ""
echo "=== RL Environment Setup Complete ==="
echo ""
echo "Environment location: $RL_ENV_DIR"
echo ""
echo "To activate manually:"
echo "  source $RL_ENV_DIR/bin/activate"
echo ""
echo "The RL launcher will use this environment automatically when you run:"
echo "  python -m hpc.launch --job_type rl --rl_config terminal_bench.yaml ..."
echo ""

# Create activation helper script
ACTIVATE_SCRIPT="$BASE_DIR/hpc/activate_rl_env.sh"
if [[ "$USE_ROCM" == "true" ]]; then
    cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
# Source this script to activate the RL environment (ROCm/AMD)
# Usage: source hpc/activate_rl_env.sh

# Load ROCm modules (for Frontier/AMD systems)
if command -v module &> /dev/null; then
    module load PrgEnv-gnu/8.6.0 2>/dev/null || true
    # Try ROCm 7.0.0 first (for vLLM), fall back to 6.4.1
    module load rocm/7.0.2 2>/dev/null || module load rocm/6.4.1 2>/dev/null || true
    module load craype-accel-amd-gfx90a 2>/dev/null || true
fi

export RL_ENV_DIR="$RL_ENV_DIR"
source "\$RL_ENV_DIR/bin/activate"
echo "Activated RL environment (ROCm): \$RL_ENV_DIR"
EOF
else
    cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
# Source this script to activate the RL environment
# Usage: source hpc/activate_rl_env.sh

export RL_ENV_DIR="$RL_ENV_DIR"
source "\$RL_ENV_DIR/bin/activate"
echo "Activated RL environment: \$RL_ENV_DIR"
EOF
fi
chmod +x "$ACTIVATE_SCRIPT"
echo "Created activation helper: $ACTIVATE_SCRIPT"
