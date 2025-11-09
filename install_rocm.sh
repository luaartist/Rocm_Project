#!/bin/bash
# ROCm GPU Enablement Script
# Simple installation for AMD GPUs on Ubuntu

echo "🚀 Installing ROCm for AMD GPU support..."

# Update package list
sudo apt update

# Install ROCm packages
echo "📦 Installing ROCm packages..."
sudo apt install -y \
    rocminfo \
    rocm-smi \
    rocm-cmake \
    rocm-device-libs

# Add user to GPU access groups
echo "👤 Adding user to GPU groups..."
sudo usermod -a -G render,video $USER

echo "✅ ROCm installation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Log out and log back in (or reboot)"
echo "2. Run: rocminfo"
echo "3. Run: rocm-smi"
echo ""
echo "🔧 For PyTorch GPU support, install with:"
echo "pip install torch --index-url https://download.pytorch.org/whl/rocm5.6"