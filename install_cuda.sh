#!/bin/bash
# CUDA GPU Enablement Script
# Simple installation for NVIDIA GPUs on Ubuntu

echo "🚀 Installing CUDA for NVIDIA GPU support..."

# Install NVIDIA drivers
echo "📦 Installing NVIDIA drivers..."
sudo ubuntu-drivers autoinstall

# Install CUDA toolkit
echo "📦 Installing CUDA toolkit..."
sudo apt install -y nvidia-cuda-toolkit

echo "✅ CUDA installation complete!"
echo ""
echo "📋 Verification:"
echo "Run: nvidia-smi"
echo "Run: nvcc --version"
echo ""
echo "🔧 For PyTorch GPU support, install with:"
echo "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"