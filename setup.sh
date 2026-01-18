#!/bin/bash
# Installation script for Vast.ai / Linux CUDA environments
# Run with: bash setup.sh (from project-ochre directory or parent)
set -e  # Exit on error

echo "🚀 Starting setup..."

# Install gdown if needed
if ! command -v gdown &> /dev/null; then
    echo "📦 Installing gdown..."
    pip install gdown
else
    echo "✅ gdown already installed"
fi

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

# Download VQ-VAE checkpoint if not present
VQVAE_CHECKPOINT="checkpoints/vqvae_v2.1.6.pt"
if [[ -f "$VQVAE_CHECKPOINT" ]] || [[ -f "checkpoints/vqvae.pt" ]]; then
    echo "✅ VQ-VAE checkpoint already exists, skipping download"
else
    echo "📦 Downloading VQ-VAE checkpoint..."
    cd checkpoints
    gdown 1hpBa3d-JX3vmHtH-e1FkSEdvyBtzcN6z # vqvae v2.1.6
    cd ..
fi

echo "📦 Checking dataset..."
# Download to parent directory or current directory based on location
if [[ $(basename "$PWD") == "project-ochre" ]]; then
    # Already in project-ochre, check parent directory for dataset
    if [[ -d "../preprocessedv5_plains_clear" ]]; then
        echo "✅ Dataset already exists at ../preprocessedv5_plains_clear, skipping download"
    else
        echo "📦 Downloading dataset to parent directory..."
        cd ..
        gdown 1sqxDK2jHQu--pWH343l9gTzGnJozJLRX
        tar -xvf preprocessedv5_plains_clear.tar
        echo "✅ Dataset extracted to $(pwd)/preprocessedv5"
        ls preprocessedv5_plains_clear | head -5
        cd project-ochre
    fi
else
    # In parent directory
    if [[ -d "preprocessedv5_plains_clear" ]]; then
        echo "✅ Dataset already exists at ./preprocessedv5_plains_clear, skipping download"
    else
        echo "📦 Downloading dataset..."
        gdown 1sqxDK2jHQu--pWH343l9gTzGnJozJLRX
        unzip -q preprocessedv5_plains_clear.zip
        echo "✅ Dataset extracted to $(pwd)/preprocessedv5_plains_clear"
        ls preprocessedv5_plains_clear | head -5
    fi
fi

echo "🚀 Installing dependencies"

# Check if PyTorch is already installed
PYTORCH_INSTALLED=false
if python -c "import torch" 2>/dev/null; then
    PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
    echo "✅ PyTorch $PYTORCH_VERSION already installed"
    PYTORCH_INSTALLED=true
fi

# Detect CUDA version and GPU architecture
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "✅ Detected CUDA version: $CUDA_VERSION"
else
    echo "⚠️  nvcc not found. Will install PyTorch with auto-detected CUDA."
    CUDA_VERSION="auto"
fi

# Detect if RTX 50-series (Blackwell) GPUs present
GPU_IS_BLACKWELL=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi "50[0-9][0-9]"; then
        echo "✅ Detected RTX 50-series GPU (Blackwell architecture)"
        GPU_IS_BLACKWELL=true
    fi
fi

# Install PyTorch based on CUDA version and GPU architecture (skip if already installed)
if [[ "$PYTORCH_INSTALLED" == "true" ]]; then
    echo "⏭️  Skipping PyTorch installation (already installed: $PYTORCH_VERSION)"
    echo "   To reinstall, run: pip uninstall torch torchvision torchaudio && bash setup.sh"
else
    # Blackwell (RTX 50-series) support:
    # - Prefer stable CUDA 12.6 wheels from the official PyTorch index.
    # - If that fails (temporary packaging gaps), fall back to nightly.
    if [[ "$GPU_IS_BLACKWELL" == "true" ]]; then
        echo "📦 Installing PyTorch for Blackwell (prefer stable cu126 wheels; fallback to nightly if needed)..."
        set +e
        pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
        status=$?
        if [[ $status -ne 0 ]]; then
            echo "⚠️  Stable cu126 install failed; retrying with nightly cu126..."
            pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
            status=$?
        fi
        set -e
        if [[ $status -ne 0 ]]; then
            echo "❌ Failed to install PyTorch for Blackwell GPU."
            echo "   Try running manually:"
            echo "   pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126"
            echo "   or nightly:"
            echo "   pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126"
            exit 1
        fi
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
fi

# Install other dependencies (matching Kaggle setup)
echo "📦 Checking core dependencies..."
MISSING_DEPS=()

# Check each dependency
python -c "import numpy" 2>/dev/null || MISSING_DEPS+=("numpy<2.0")
python -c "import webdataset" 2>/dev/null || MISSING_DEPS+=("webdataset")
python -c "import tqdm" 2>/dev/null || MISSING_DEPS+=("tqdm")
python -c "import PIL" 2>/dev/null || MISSING_DEPS+=("pillow")
python -c "import lpips" 2>/dev/null || MISSING_DEPS+=("lpips")

if [ ${#MISSING_DEPS[@]} -eq 0 ]; then
    echo "✅ All core dependencies already installed"
else
    echo "📦 Installing missing dependencies: ${MISSING_DEPS[*]}"
    pip install "${MISSING_DEPS[@]}"
fi

# Check wandb
if python -c "import wandb" 2>/dev/null; then
    WANDB_VERSION=$(python -c "import wandb; print(wandb.__version__)" 2>/dev/null)
    echo "✅ wandb already installed (version $WANDB_VERSION)"
else
    echo "📦 Installing wandb..."
    pip install wandb==0.22.3
fi

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA version: {torch.version.cuda}')
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    compute_cap = torch.cuda.get_device_capability(0)
    print(f'✅ GPU compute capability: {compute_cap[0]}.{compute_cap[1]}')
else:
    print('⚠️  No GPU available!')
"

python -c "import wandb, lpips, webdataset, tqdm, numpy, PIL; print('✅ All packages imported successfully')"

echo ""
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Set WANDB_API_KEY: export WANDB_API_KEY=your_key"
echo "2. Update paths in train.py (DATA_DIR, VQVAE_PATH, RESUME_PATH)"
echo "3. Upload dataset and checkpoints to /workspace/"
echo "4. Run: python train.py"
