#!/bin/bash
# Installation script for Vast.ai / Linux CUDA environments
# Run with: bash setup.sh (from project-ochre directory or parent)
set -e  # Exit on error

echo "Installing vqvae checkpoint..."

pip install gdown

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints
cd checkpoints
gdown 1hpBa3d-JX3vmHtH-e1FkSEdvyBtzcN6z # vqvae v2.1.6
cd ..

echo "Installing dataset..."
# Download to parent directory or current directory based on location
if [[ $(basename "$PWD") == "project-ochre" ]]; then
    # Already in project-ochre, go up one level for dataset
    cd ..
    gdown 104I8PGlrUdshuI4LPT5TXDwjWvaxVu2i
    unzip -q preprocessedv4.zip
    ls preprocessedv4
    cd project-ochre
else
    # In parent directory
    gdown 104I8PGlrUdshuI4LPT5TXDwjWvaxVu2i
    unzip -q preprocessedv4.zip
    ls preprocessedv4
fi

echo "🚀 Installing dependencies"

# Detect CUDA version and GPU architecture
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "✅ Detected CUDA version: $CUDA_VERSION"
else
    echo "⚠️  nvcc not found. Will install PyTorch with auto-detected CUDA."
    CUDA_VERSION="auto"
fi

# Detect if RTX 5090 or other Blackwell GPUs present (requires PyTorch nightly)
GPU_NEEDS_SM120=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi "50[0-9][0-9]"; then
        echo "⚠️  Detected RTX 50-series GPU (Blackwell architecture)"
        echo "⚠️  RTX 5090 is not yet supported by stable PyTorch releases"
        echo "⚠️  Please use a different GPU (RTX 4090, H100, A100) or wait for PyTorch 2.11+ stable"
        GPU_NEEDS_SM120=true
    fi
fi

# Install PyTorch based on CUDA version and GPU architecture
# RTX 5090 (sm_120) NOT YET SUPPORTED - warn and skip
if [[ "$GPU_NEEDS_SM120" == "true" ]]; then
    echo "❌ Skipping PyTorch installation - RTX 5090 requires sm_120 support"
    echo "   Current PyTorch releases only support up to sm_90 (H100, RTX 4090)"
    echo "   Manually install nightly build: pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126"
    echo "   Note: Even nightly builds may not have sm_120 support yet as of 2025-12-31"
    exit 1
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
