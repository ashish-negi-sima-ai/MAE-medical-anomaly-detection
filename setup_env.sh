#!/bin/bash
# Setup script for MAE Medical Anomaly Detection
# This script creates a conda environment and installs all required dependencies
#
# USAGE:
#   source ./setup_env.sh    (recommended - activates environment in current shell)
#   OR
#   ./setup_env.sh           (installs but requires manual activation after)

set -e  # Exit on error

echo "=========================================="
echo "MAE Medical Anomaly Detection - Setup"
echo "=========================================="
echo ""

# Detect if script is being sourced or executed
SOURCED=0
if [ -n "$ZSH_VERSION" ]; then
    case $ZSH_EVAL_CONTEXT in *:file) SOURCED=1;; esac
elif [ -n "$BASH_VERSION" ]; then
    (return 0 2>/dev/null) && SOURCED=1
else
    case ${0##*/} in bash|dash|sh) SOURCED=1;; esac
fi

if [ $SOURCED -eq 0 ]; then
    echo "⚠️  Note: Script is being executed (not sourced)"
    echo "   The conda environment will be installed but not activated"
    echo "   in the current shell."
    echo ""
    echo "   To have the environment activated automatically, run:"
    echo "   source ./setup_env.sh"
    echo ""
    read -p "Continue anyway? (Y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo "Exiting. Please run: source ./setup_env.sh"
        exit 0
    fi
fi
echo ""

# Configuration
ENV_NAME="mae-medical"
PYTHON_VERSION="3.8"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda found: $(conda --version)"
echo ""

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Environment '${ENV_NAME}' already exists"
    read -p "Do you want to remove it and create a fresh environment? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
        echo "✓ Environment removed"
    else
        echo "Keeping existing environment and updating packages..."
    fi
fi

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "Step 1: Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
    echo "✓ Environment created"
fi

echo ""
echo "Step 2: Detecting CUDA availability..."

# Try to detect CUDA version
CUDA_VERSION=""
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1
    
    # Try to get CUDA version
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
        echo "✓ CUDA version detected: ${CUDA_VERSION}"
    else
        echo "⚠️  nvcc not found, will prompt for CUDA version"
    fi
else
    echo "⚠️  No NVIDIA GPU detected (nvidia-smi not available)"
fi

echo ""
echo "Step 3: Installing PyTorch..."
echo ""
echo "Select your CUDA version (or CPU-only):"
echo "  1) CUDA 11.3"
echo "  2) CUDA 11.7"
echo "  3) CUDA 11.8"
echo "  4) CUDA 12.1"
echo "  5) CUDA 12.2 (or 12.x)"
echo "  6) CPU only (not recommended for training)"
echo "  7) Skip PyTorch installation (already installed)"
echo ""

# Auto-select based on detected CUDA version
DEFAULT_CHOICE=1
if [[ $CUDA_VERSION == "11.3"* ]]; then
    DEFAULT_CHOICE=1
elif [[ $CUDA_VERSION == "11.7"* ]]; then
    DEFAULT_CHOICE=2
elif [[ $CUDA_VERSION == "11.8"* ]]; then
    DEFAULT_CHOICE=3
elif [[ $CUDA_VERSION == "12.1"* ]]; then
    DEFAULT_CHOICE=4
elif [[ $CUDA_VERSION == "12.2"* ]] || [[ $CUDA_VERSION == "12."* ]]; then
    DEFAULT_CHOICE=5
elif [ -z "$CUDA_VERSION" ]; then
    DEFAULT_CHOICE=6
fi

read -p "Enter choice [${DEFAULT_CHOICE}]: " PYTORCH_CHOICE
PYTORCH_CHOICE=${PYTORCH_CHOICE:-$DEFAULT_CHOICE}

# Activate environment for package installation
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

case $PYTORCH_CHOICE in
    1)
        echo "Installing PyTorch with CUDA 11.3..."
        conda install pytorch torchvision torchaudio pytorch-cuda=11.3 -c pytorch -c nvidia -y
        ;;
    2)
        echo "Installing PyTorch with CUDA 11.7..."
        conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
        ;;
    3)
        echo "Installing PyTorch with CUDA 11.8..."
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
        ;;
    4)
        echo "Installing PyTorch with CUDA 12.1..."
        conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
        ;;
    5)
        echo "Installing PyTorch with CUDA 12.2 (using pytorch-cuda=12.1, compatible with 12.2)..."
        conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
        ;;
    6)
        echo "Installing PyTorch (CPU only)..."
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
        ;;
    7)
        echo "Skipping PyTorch installation..."
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo "✓ PyTorch installation complete"

echo ""
echo "Step 4: Installing required packages from requirements.txt..."
pip install -r requirements.txt
echo "✓ Required packages installed"

echo ""
echo "Step 5: Verifying installation..."
echo ""

# Verify PyTorch
echo "Testing PyTorch..."
python -c "import torch; print(f'  PyTorch version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Verify timm version
echo ""
echo "Testing timm..."
python -c "import timm; assert timm.__version__ == '0.3.2', f'Wrong timm version: {timm.__version__}'; print(f'  timm version: {timm.__version__} ✓')"

# Verify other dependencies
echo ""
echo "Testing other dependencies..."
python -c "
import h5py
import nibabel
import cv2
import sklearn
import matplotlib
import albumentations
import numpy as np
from PIL import Image
print('  ✓ h5py')
print('  ✓ nibabel')
print('  ✓ opencv-python')
print('  ✓ scikit-learn')
print('  ✓ matplotlib')
print('  ✓ albumentations')
print('  ✓ numpy')
print('  ✓ Pillow')
"

# Test model creation
echo ""
echo "Testing model creation..."
python -c "
import torch
import models_mae

model = models_mae.mae_vit_base_patch16(img_size=224, patch_size=16)
print('  ✓ Model created successfully')

# Test forward pass
x = torch.randn(1, 1, 224, 224)
with torch.no_grad():
    loss, pred, mask = model(x, mask_ratio=0.75)
print(f'  ✓ Forward pass successful')
"

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""

# Activate environment if sourced, otherwise provide instructions
if [ $SOURCED -eq 1 ]; then
    echo "✓ Environment '${ENV_NAME}' is now active in your current shell!"
    echo ""
else
    echo "To activate the environment, run:"
    echo "  conda activate ${ENV_NAME}"
    echo ""
    echo "Or to avoid this step next time, source the script:"
    echo "  source ./setup_env.sh"
    echo ""
fi
echo "To verify GPU availability:"
echo "  python -c \"import torch; print(torch.cuda.is_available())\""
echo ""
echo "Next steps:"
echo "  1. Download datasets (BraTS2020, LUNA16)"
echo "  2. Follow instructions in README.md for training"
echo ""
echo "Optional: Install ONNX support for model export:"
echo "  pip install onnx onnxruntime"
echo ""
echo "Optional: Install Jupyter for interactive development:"
echo "  pip install jupyter ipython"
echo ""
echo "=========================================="


