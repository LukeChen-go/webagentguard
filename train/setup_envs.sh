#!/bin/bash
# Create the two conda environments used for training:
#   pia-sft : LLaMA-Factory (SFT)   torch 2.4.1 + CUDA 12.1
#   pia-rl  : verl (GRPO RL)        torch 2.8.0 + vLLM 0.11.0
#
# Usage:
#   bash setup_envs.sh          # set up both environments
#   bash setup_envs.sh sft      # only the SFT environment
#   bash setup_envs.sh rl       # only the RL environment
#
# Environment names can be overridden: SFT_ENV=my-sft RL_ENV=my-rl bash setup_envs.sh
# flash-attn is compiled from source if no matching prebuilt wheel exists, which
# needs nvcc (CUDA toolkit) on PATH and can take a while. Set MAX_JOBS to limit memory.

set -euo pipefail

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$TRAIN_DIR/sft/LLaMA-Factory"
RL_DIR="$TRAIN_DIR/rl/verl"
SFT_ENV="${SFT_ENV:-pia-sft}"
RL_ENV="${RL_ENV:-pia-rl}"
PYTHON_VERSION="3.10"
TARGET="${1:-all}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Install Miniconda/Anaconda first." >&2
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

create_env() {
    local name=$1
    if conda env list | awk '{print $1}' | grep -qx "$name"; then
        echo ">>> conda env '$name' already exists, reusing it"
    else
        conda create -y -n "$name" python="$PYTHON_VERSION"
    fi
    conda activate "$name"
    python -m pip install --upgrade pip setuptools wheel ninja packaging
}

setup_sft() {
    echo ">>> Setting up SFT environment: $SFT_ENV"
    create_env "$SFT_ENV"

    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu121

    # LLaMA-Factory in editable mode (provides `llamafactory-cli`)
    pip install -e "$SFT_DIR[metrics]"
    pip install \
        transformers==4.57.1 \
        accelerate==1.7.0 \
        datasets==3.6.0 \
        peft==0.15.2 \
        trl==0.9.6 \
        deepspeed==0.15.4 \
        "numpy<2.0.0" \
        av \
        wandb
    pip install flash-attn==2.8.2 --no-build-isolation

    python -c "import torch, transformers, llamafactory; print('torch', torch.__version__, '| transformers', transformers.__version__, '| llamafactory', llamafactory.__version__)"
    conda deactivate
}

setup_rl() {
    echo ">>> Setting up RL environment: $RL_ENV"
    create_env "$RL_ENV"

    # vLLM pins torch==2.8.0, torchvision==0.23.0 and xformers
    pip install vllm==0.11.0

    # verl in editable mode with the vision-language extras (mathruler, qwen_vl_utils)
    pip install -e "$RL_DIR[geo,math]"
    pip install \
        transformers==4.57.1 \
        ray[default]==2.50.1 \
        liger-kernel==0.6.2 \
        qwen_vl_utils==0.0.14 \
        wandb
    pip install flash-attn==2.8.3 --no-build-isolation

    python -c "import torch, vllm, verl, transformers; print('torch', torch.__version__, '| vllm', vllm.__version__, '| transformers', transformers.__version__)"
    conda deactivate
}

case "$TARGET" in
    sft) setup_sft ;;
    rl)  setup_rl ;;
    all) setup_sft; setup_rl ;;
    *)   echo "Unknown target '$TARGET'. Use: sft | rl | all" >&2; exit 1 ;;
esac

echo ">>> Done. Activate with: conda activate $SFT_ENV  (SFT)  /  conda activate $RL_ENV  (RL)"
