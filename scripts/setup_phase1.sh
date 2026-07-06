#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Phase 1 setup ==="
echo "Repo root: $REPO_ROOT"

if ! command -v conda &>/dev/null; then
    echo "Error: conda not found."
    echo "Install Miniconda, then re-run this script."
    exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_NAME="semi_seg_ecg"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda env '$ENV_NAME' already exists."
else
    echo "Creating conda env '$ENV_NAME' (python 3.9)..."
    conda create -n "$ENV_NAME" python=3.9 -y
fi

conda activate "$ENV_NAME"

echo "Installing Python dependencies..."
pip install -r semi-seg-ecg/requirements.txt
pip install matplotlib gdown

echo "Checking CUDA / PyTorch..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo "Downloading LUDB data (if needed)..."
python data/data.py --download

echo "Running split audit..."
python data/audit.py || true

mkdir -p baseline/exps

echo ""
echo "Setup complete."
echo "Activate env: conda activate $ENV_NAME"
echo "Next: bash scripts/run_phase1.sh --gpus 0"
