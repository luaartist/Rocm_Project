#!/bin/bash
# ROCm 5.2 Installation for Older AMD GPUs (gfx1010)
# Compatible with PyTorch 2.2.2

echo "🛠️  Installing ROCm 5.2 for Older AMD GPUs..."

# Remove any existing ROCm installation
echo "Removing existing ROCm..."
sudo apt remove --purge -y rocm* hip* miopen* rocblas* rocfft* rocrand* rocsolver* hipblas* hipfft* hipsparse* rccl*

# Clean up
sudo apt autoremove -y
sudo apt autoclean

# Add ROCm 5.2 repository
echo "Adding ROCm 5.2 repository..."
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.2/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

# Update package list
sudo apt update

# Install ROCm 5.2 core (minimal for older GPUs)
echo "Installing ROCm 5.2 core components..."
sudo apt install -y \
    rocm-dev \
    rocm-libs \
    rocminfo \
    rocm-smi \
    hip-base \
    hip-runtime-amd \
    hip-dev

# Set environment variables
echo "Setting up environment variables..."
echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/hip/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib:/opt/rocm/hip/lib' >> ~/.bashrc
echo 'export ROCM_PATH=/opt/rocm' >> ~/.bashrc

# Source the environment
source ~/.bashrc

# Add user to video/render groups
sudo usermod -a -G video,render $USER

echo "✅ ROCm 5.2 installation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Log out and log back in (or reboot)"
echo "2. Install PyTorch 2.2.2: pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/rocm5.2"
echo "3. Run diagnostic: python3 rocm_diagnostic_gfx1010.py"
echo ""
echo "🔍 Your GPU (gfx1010) may have limited support - check the diagnostic output"