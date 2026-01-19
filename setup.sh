#!/bin/bash
# Installation script for Vast.ai / Linux CUDA environments
# Run with: bash setup.sh (from project-ochre directory or parent)
set -e  # Exit on error

echo "🚀 Starting setup..."

# Prefer python3/pip3 in these environments, and use module form for consistency.
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_CMD="${PIP_CMD:-$PYTHON_BIN -m pip}"

# Install gdown if needed
if ! command -v gdown &> /dev/null; then
    echo "📦 Installing gdown..."
    $PIP_CMD install gdown
else
    echo "✅ gdown already installed"
fi

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

# Download VQ-VAE checkpoint if not present
VQVAE_CHECKPOINT="checkpoints/vqvae_v2.1.6__epoch100.pt"
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
        echo "✅ Dataset extracted to $(pwd)/preprocessedv5_plains_clear"
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
if $PYTHON_BIN -c "import torch" 2>/dev/null; then
    PYTORCH_VERSION=$($PYTHON_BIN -c "import torch; print(torch.__version__)" 2>/dev/null)
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

# Detect NVIDIA Blackwell (typically GeForce RTX 50-series). Best-effort based on GPU name.
GPU_IS_BLACKWELL=false
if command -v nvidia-smi &> /dev/null; then
    GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ';')
    # Match common naming patterns without accidentally matching older workstation parts (e.g., RTX A5000).
    if echo "$GPU_NAMES" | grep -Eqi "(GeForce[[:space:]]+RTX[[:space:]]+50|RTX[[:space:]]+50[0-9]{2}|RTX[[:space:]]+5090|RTX[[:space:]]+5080|RTX[[:space:]]+5070|Blackwell)"; then
        echo "✅ Detected Blackwell-class GPU (likely RTX 50-series): $GPU_NAMES"
        GPU_IS_BLACKWELL=true
    fi
fi

# If torch is installed, detect whether this build supports the GPU.
# Blackwell GPUs (compute capability 12.x / sm_120) typically require recent CUDA 12.8+ nightly wheels today.
PYTORCH_NEEDS_REINSTALL=false
if [[ "$PYTORCH_INSTALLED" == "true" ]] && command -v nvidia-smi &> /dev/null; then
    # This check is best-effort; it will still work even if torch emits a warning during import.
    read -r TORCH_CUDA_VERSION GPU_CC_MAJOR GPU_CC_MINOR <<< "$($PYTHON_BIN - <<'PY' 2>/dev/null || true
import torch
cuda_ver = torch.version.cuda or "none"
cc = None
try:
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(0)
except Exception:
    cc = None
if cc is None:
    print(cuda_ver, -1, -1)
else:
    print(cuda_ver, cc[0], cc[1])
PY
)"
    if [[ "$GPU_CC_MAJOR" -ge 12 ]]; then
        # Best-effort: if torch reports CUDA build not matching 12.8*, it may lack sm_120 kernels.
        if [[ "$TORCH_CUDA_VERSION" != 12.8* ]]; then
            echo "⚠️  Installed PyTorch CUDA build ($TORCH_CUDA_VERSION) likely does not support compute capability ${GPU_CC_MAJOR}.${GPU_CC_MINOR} (Blackwell sm_120)."
            PYTORCH_NEEDS_REINSTALL=true
        fi
    fi
fi

# Install PyTorch based on CUDA version and GPU architecture (skip if already installed)
if [[ "$PYTORCH_INSTALLED" == "true" ]] && [[ "$PYTORCH_NEEDS_REINSTALL" != "true" ]]; then
    echo "⏭️  Skipping PyTorch installation (already installed: $PYTORCH_VERSION)"
    echo "   To reinstall, run: pip uninstall torch torchvision torchaudio && bash setup.sh"
else
    if [[ "$PYTORCH_INSTALLED" == "true" ]] && [[ "$PYTORCH_NEEDS_REINSTALL" == "true" ]]; then
        echo "🧹 Uninstalling incompatible PyTorch build..."
        set +e
        $PIP_CMD uninstall -y torch torchvision torchaudio
        set -e
    fi

    # CUDA 12.8 / Blackwell (RTX 50-series) support:
    # - PyTorch 2.7+ ships official CUDA 12.8 (cu128) wheels and includes Blackwell support.
    #   See PyTorch 2.7 release notes/blog for the recommended install command.
    # - Nightly cu128 can be used as a fallback if the stable wheel set is temporarily missing for your Python/ABI.
    if [[ "$GPU_IS_BLACKWELL" == "true" ]]; then
        echo "📦 Installing PyTorch for Blackwell (stable cu128)..."
        set +e
        $PIP_CMD install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
        INSTALL_RC=$?
        set -e
        if [[ $INSTALL_RC -ne 0 ]]; then
            echo "⚠️  Stable cu128 wheel install failed; falling back to nightly cu128 (can happen if today's nightly wheels are incomplete)."
            $PIP_CMD install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
        fi
    elif [[ "$CUDA_VERSION" == "11.8"* ]]; then
        echo "📦 Installing PyTorch 2.1.0 for CUDA 11.8..."
        $PIP_CMD install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
    elif [[ "$CUDA_VERSION" == "12.8"* ]]; then
        echo "📦 Installing PyTorch for CUDA 12.8 (stable cu128)..."
        set +e
        $PIP_CMD install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
        INSTALL_RC=$?
        set -e
        if [[ $INSTALL_RC -ne 0 ]]; then
            echo "⚠️  Stable cu128 wheel install failed; falling back to nightly cu128."
            $PIP_CMD install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
        fi
    elif [[ "$CUDA_VERSION" == "12."* ]]; then
        echo "📦 Installing PyTorch for CUDA 12.x..."
        # Prefer cu126 stable for pre-Blackwell GPUs.
        $PIP_CMD install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    else
        echo "📦 Installing PyTorch 2.7.0+ (auto-detect CUDA)..."
        $PIP_CMD install --upgrade torch torchvision torchaudio
    fi
fi

# Install other dependencies (matching Kaggle setup)
echo "📦 Checking core dependencies..."
MISSING_DEPS=()

# Check each dependency
$PYTHON_BIN -c "import numpy" 2>/dev/null || MISSING_DEPS+=("numpy<2.0")
$PYTHON_BIN -c "import webdataset" 2>/dev/null || MISSING_DEPS+=("webdataset")
$PYTHON_BIN -c "import tqdm" 2>/dev/null || MISSING_DEPS+=("tqdm")
$PYTHON_BIN -c "import PIL" 2>/dev/null || MISSING_DEPS+=("pillow")
$PYTHON_BIN -c "import lpips" 2>/dev/null || MISSING_DEPS+=("lpips")

if [ ${#MISSING_DEPS[@]} -eq 0 ]; then
    echo "✅ All core dependencies already installed"
else
    echo "📦 Installing missing dependencies: ${MISSING_DEPS[*]}"
    $PIP_CMD install "${MISSING_DEPS[@]}"
fi

# Check wandb
if $PYTHON_BIN -c "import wandb" 2>/dev/null; then
    WANDB_VERSION=$($PYTHON_BIN -c "import wandb; print(wandb.__version__)" 2>/dev/null)
    echo "✅ wandb already installed (version $WANDB_VERSION)"
else
    echo "📦 Installing wandb..."
    $PIP_CMD install wandb==0.22.3
fi

# Verify installation
echo ""
echo "🔍 Verifying installation..."
$PYTHON_BIN -c "
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

$PYTHON_BIN -c "import wandb, lpips, webdataset, tqdm, numpy, PIL; print('✅ All packages imported successfully')"

echo ""
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Set WANDB_API_KEY: export WANDB_API_KEY=your_key"
echo "2. Update paths in train.py (DATA_DIR, VQVAE_PATH, RESUME_PATH)"
echo "3. Upload dataset and checkpoints to /workspace/"
echo "4. Run: python train.py"
