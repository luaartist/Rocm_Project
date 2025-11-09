#!/usr/bin/env python3
"""
ROCm GPU Diagnostic Script for Older AMD GPUs (gfx1010)
Specifically tests Conv2d operations that were failing
"""

import sys
import os

def test_rocm_versions():
    """Check ROCm installation and versions"""
    print("🔍 ROCm Version Check:")

    # Check ROCm tools
    tools = ['rocminfo', 'rocm-smi', 'hipcc']
    for tool in tools:
        result = os.system(f"which {tool} > /dev/null 2>&1")
        status = "✓" if result == 0 else "✗"
        print(f"   {tool}: {status}")

    # Check versions
    try:
        import subprocess
        result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=10)
        if 'gfx1010' in result.stdout:
            print("   ✓ Detected gfx1010 GPU")
        else:
            print("   ⚠️  GPU not detected or different architecture")
    except:
        print("   ✗ Could not run rocminfo")

def test_pytorch_setup():
    """Test PyTorch installation and basic GPU ops"""
    print("\n🔍 PyTorch Setup:")

    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")

        # Check CUDA availability (should be True on ROCm)
        cuda_available = torch.cuda.is_available()
        print(f"   CUDA available: {cuda_available}")

        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"   GPU count: {device_count}")

            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                arch = torch.cuda.get_device_properties(i).major * 10 + torch.cuda.get_device_properties(i).minor
                print(f"   GPU {i}: {name} (Compute Capability: {arch})")

            # Test basic operations
            print("\n🔍 Testing Basic Operations:")

            # Test matrix multiplication (should work)
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.matmul(x, y)
            print("   ✓ Matrix multiplication: PASS")

            # Test Conv2d (the failing operation)
            print("\n🔍 Testing Conv2d Operations:")

            # Simple Conv2d test
            conv = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1).cuda()
            input_tensor = torch.randn(1, 3, 32, 32).cuda()

            try:
                output = conv(input_tensor)
                print(f"   ✓ Conv2d (3→64, 3x3): PASS - Output shape: {output.shape}")
            except Exception as e:
                print(f"   ✗ Conv2d failed: {e}")

            # Test different Conv2d configurations
            conv_configs = [
                (1, 32, 3),   # 1→32 channels, 3x3 kernel
                (32, 64, 3),  # 32→64 channels, 3x3 kernel
                (64, 128, 1), # 64→128 channels, 1x1 kernel
            ]

            for in_ch, out_ch, ksize in conv_configs:
                try:
                    conv_test = torch.nn.Conv2d(in_ch, out_ch, kernel_size=ksize).cuda()
                    test_input = torch.randn(1, in_ch, 16, 16).cuda()
                    test_output = conv_test(test_input)
                    print(f"   ✓ Conv2d ({in_ch}→{out_ch}, {ksize}x{ksize}): PASS")
                except Exception as e:
                    print(f"   ✗ Conv2d ({in_ch}→{out_ch}, {ksize}x{ksize}): FAIL - {e}")

            return True
        else:
            print("   ✗ CUDA not available")
            return False

    except ImportError:
        print("   ✗ PyTorch not installed")
        return False

def check_rocm_pytorch_compatibility():
    """Check if ROCm and PyTorch versions are compatible"""
    print("\n🔍 ROCm/PyTorch Compatibility:")

    try:
        import torch
        pytorch_version = torch.__version__

        # Check for ROCm in version string
        if 'rocm' in pytorch_version.lower():
            print(f"   ✓ PyTorch built with ROCm: {pytorch_version}")
        else:
            print(f"   ⚠️  PyTorch not built with ROCm: {pytorch_version}")
            print("      Install with: pip install torch --index-url https://download.pytorch.org/whl/rocm5.2")

        # Check environment variables
        rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
        print(f"   ROCM_PATH: {rocm_path}")

        if os.path.exists(rocm_path):
            print("   ✓ ROCM_PATH exists")
        else:
            print("   ✗ ROCM_PATH does not exist")

    except Exception as e:
        print(f"   ✗ Error checking compatibility: {e}")

def main():
    """Run all diagnostic tests"""
    print("🚀 ROCm GPU Diagnostic for gfx1010")
    print("=" * 50)

    test_rocm_versions()
    gpu_ok = test_pytorch_setup()
    check_rocm_pytorch_compatibility()

    print("\n" + "=" * 50)
    print("📋 RECOMMENDATIONS:")

    if not gpu_ok:
        print("❌ GPU not working. Try:")
        print("   1. Install ROCm 5.2: ./install_rocm.sh")
        print("   2. Install PyTorch: pip install torch --index-url https://download.pytorch.org/whl/rocm5.2")
        print("   3. Reboot system")
    else:
        print("✅ Basic GPU operations working")
        print("   If Conv2d still fails, try ROCm 5.2 + PyTorch 2.2.2")

    print("\n🔗 Check rocm-patch project: https://github.com/hkevin01/rocm-patch")
    print("   May have patches for older GPUs like gfx1010")

if __name__ == "__main__":
    main()