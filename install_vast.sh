#!/bin/bash
# Installation script for Vast.ai / Linux CUDA environments
# Run with: bash install_vast.sh

set -e  # Exit on error

echo "🚀 Installing dependencies"

# Detect CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "✅ Detected CUDA version: $CUDA_VERSION"
else
    echo "⚠️  nvcc not found. Will install PyTorch with auto-detected CUDA."
    CUDA_VERSION="auto"
fi

# Install PyTorch based on CUDA version
# CUDA 12.x requires PyTorch 2.2.0+
if [[ "$CUDA_VERSION" == "11.8"* ]]; then
    echo "📦 Installing PyTorch 2.1.0 for CUDA 11.8..."
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
elif [[ "$CUDA_VERSION" == "12."* ]]; then
    echo "📦 Installing PyTorch 2.2.0 for CUDA 12.x (compatible with $CUDA_VERSION)..."
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
else
    echo "📦 Installing PyTorch 2.2.0 (auto-detect CUDA)..."
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
fi

# Install other dependencies (matching Kaggle setup)
echo "📦 Installing core dependencies..."
pip install webdataset tqdm numpy pillow lpips

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
