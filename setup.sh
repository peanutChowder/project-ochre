#!/bin/bash
# Installation script for Vast.ai / Linux CUDA environments
# Run with: bash setup.sh
set -e  # Exit on error

echo "Installing vqvae checkpoint..."

pip install gdown
cd project-ochre
mkdir checkpoints && cd checkpoints
gdown 1hpBa3d-JX3vmHtH-e1FkSEdvyBtzcN6z # vqvae v2.1.6
cd ../..

echo "Installing dataset..."
gdown 104I8PGlrUdshuI4LPT5TXDwjWvaxVu2i
unzip preprocessedv4.zip
ls preprocessedv4

echo "🚀 Installing dependencies"

# Detect CUDA version and GPU architecture
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "✅ Detected CUDA version: $CUDA_VERSION"
else
    echo "⚠️  nvcc not found. Will install PyTorch with auto-detected CUDA."
    CUDA_VERSION="auto"
fi

# Detect if RTX 5090 or other Blackwell GPUs present (requires PyTorch 2.7.0+)
GPU_NEEDS_SM120=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi "50[0-9][0-9]"; then
        echo "✅ Detected RTX 50-series GPU (Blackwell architecture)"
        GPU_NEEDS_SM120=true
    fi
fi

# Install PyTorch based on CUDA version and GPU architecture
# RTX 5090 (sm_120) requires PyTorch 2.7.0+
if [[ "$GPU_NEEDS_SM120" == "true" ]]; then
    echo "📦 Installing PyTorch 2.7.0+ for Blackwell architecture (RTX 50-series)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
elif [[ "$CUDA_VERSION" == "11.8"* ]]; then
    echo "📦 Installing PyTorch 2.1.0 for CUDA 11.8..."
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
elif [[ "$CUDA_VERSION" == "12."* ]]; then
    echo "📦 Installing PyTorch 2.7.0+ for CUDA 12.x (compatible with $CUDA_VERSION)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
else
    echo "📦 Installing PyTorch 2.7.0+ (auto-detect CUDA)..."
    pip install torch torchvision torchaudio
fi

# Install other dependencies (matching Kaggle setup)
echo "📦 Installing core dependencies..."
pip install "numpy<2.0" webdataset tqdm pillow lpips

echo "📦 Installing wandb..."
pip install wandb==0.22.3

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}'); print(f'✅ CUDA available: {torch.cuda.is_available()}'); print(f'✅ CUDA version: {torch.version.cuda}'); print(f'✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

python -c "import wandb, lpips, webdataset, tqdm, numpy, PIL; print('✅ All packages imported successfully')"

echo ""
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Set WANDB_API_KEY: export WANDB_API_KEY=your_key"
echo "2. Update paths in train.py (DATA_DIR, VQVAE_PATH, RESUME_PATH)"
echo "3. Upload dataset and checkpoints to /workspace/"
echo "4. Run: python train.py"
