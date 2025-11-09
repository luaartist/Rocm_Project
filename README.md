# ROCm GPU Enablement Guide

This guide shows how to enable GPU acceleration on Ubuntu systems using ROCm (AMD) or CUDA (NVIDIA).

## Diagnostic Tools

### For Older AMD GPUs (gfx1010 and similar)
If you're having issues with Conv2d operations or other GPU computations on older AMD GPUs:

```bash
# Run comprehensive diagnostic
python3 rocm_diagnostic_gfx1010.py

# If issues found, install ROCm 5.2 for better compatibility
./install_rocm_5_2_gfx1010.sh
```

**Supported older GPUs:** gfx1010, older RDNA/RDNA2 architectures with limited ROCm 6.x support.

## Quick ROCm Installation (Ubuntu)

```bash
#!/bin/bash
# Install ROCm for AMD GPUs

# Update system
sudo apt update

# Install ROCm packages
sudo apt install -y rocminfo rocm-smi rocm-cmake rocm-device-libs

# Add user to GPU groups
sudo usermod -a -G render,video $USER

echo "Log out and back in for group changes to take effect"
```

## Quick CUDA Installation (Ubuntu)

```bash
#!/bin/bash
# Install CUDA for NVIDIA GPUs

# Install NVIDIA drivers (if not already installed)
ubuntu-drivers autoinstall

# Install CUDA toolkit
sudo apt install -y nvidia-cuda-toolkit

# Verify installation
nvidia-smi
nvcc --version
```

## GPU Verification

### ROCm (AMD)
```bash
# Check GPU detection
rocminfo

# Check GPU status
rocm-smi
```

### CUDA (NVIDIA)
```bash
# Check GPU detection
nvidia-smi

# Check CUDA version
nvcc --version
```

## Python GPU Libraries

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install CuPy for CUDA
pip install cupy-cuda11x

# Install Numba for CUDA kernels
pip install numba

# Test GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Basic GPU Test

```python
#!/usr/bin/env python3
import torch

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Test basic computation
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    result = torch.matmul(x, y)
    print("GPU computation successful!")
```

## Multi-GPU Setup

For systems with multiple GPUs:

```python
import torch

# Check number of GPUs
print(f"Available GPUs: {torch.cuda.device_count()}")

# Use specific GPU
device = torch.device("cuda:0")  # First GPU
tensor = torch.randn(100, 100).to(device)

# Or use all GPUs
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

## Troubleshooting

### Common Issues:

1. **Permission denied**: Make sure user is in `video` and `render` groups
2. **GPU not detected**: Check drivers and reboot
3. **CUDA version mismatch**: Ensure PyTorch and CUDA versions match
4. **Memory errors**: Monitor GPU memory with `nvidia-smi` or `rocm-smi`

### Ubuntu LLM Installation Notes

When using LLMs or automated scripts to install GPU drivers on Ubuntu, you may encounter permission issues with `sudo` commands. Ubuntu uses PolicyKit (pkexec) for graphical authentication, which works better in these scenarios:

**Why pkexec instead of sudo:**
- `sudo` may fail in automated environments or when passwords are required
- `pkexec` provides a graphical authentication dialog that works with desktop sessions
- Better integration with Ubuntu's security policies

**Using pkexec for GPU installation:**
```bash
# Instead of: sudo apt install nvidia-driver-XXX
pkexec apt install nvidia-driver-XXX

# For ROCm installation:
pkexec apt install -y rocminfo rocm-smi rocm-cmake rocm-device-libs

# Add user to groups (may require pkexec):
pkexec usermod -a -G render,video $USER
```

**When to use pkexec vs sudo:**
- Use `pkexec` when running installation commands through LLMs or scripts
- Use `sudo` for quick commands in terminal or when you can provide password
- Both work for most administrative tasks, but `pkexec` is more reliable in automated contexts

### Reset GPU Memory:
```python
import torch
torch.cuda.empty_cache()
```

## Performance Tips

- Use appropriate batch sizes for GPU memory
- Monitor GPU utilization with `nvidia-smi` or `rocm-smi`
- Use mixed precision (FP16) for faster computation
- Profile code with `torch.profiler` or `cProfile`