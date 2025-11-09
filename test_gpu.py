#!/usr/bin/env python3
"""
Simple GPU Test Script
Tests basic GPU functionality and reports capabilities
"""

import sys

def test_pytorch_gpu():
    """Test PyTorch GPU support"""
    try:
        import torch
        print("🔍 Testing PyTorch GPU...")

        cuda_available = torch.cuda.is_available()
        print(f"   CUDA Available: {cuda_available}")

        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"   GPU Count: {device_count}")

            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"   GPU {i}: {name} ({memory:.1f} GB)")

            # Test computation
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            result = torch.matmul(x, y)
            print("   ✓ GPU computation successful")
            return True
        else:
            print("   ✗ No CUDA GPU detected")
            return False

    except ImportError:
        print("   ⚠️ PyTorch not installed")
        return False

def test_cupy_gpu():
    """Test CuPy GPU support"""
    try:
        import cupy as cp
        print("🔍 Testing CuPy GPU...")

        try:
            device = cp.cuda.Device(0)
            name = device.name.decode()
            memory = device.mem_info[1] / 1e9
            print(f"   GPU: {name} ({memory:.1f} GB)")
            print("   ✓ CuPy GPU support working")
            return True
        except:
            print("   ✗ CuPy GPU not available")
            return False

    except ImportError:
        print("   ⚠️ CuPy not installed")
        return False

def test_numba_gpu():
    """Test Numba CUDA support"""
    try:
        from numba import cuda
        print("🔍 Testing Numba CUDA...")

        available = cuda.is_available()
        print(f"   CUDA Available: {available}")

        if available:
            devices = cuda.list_devices()
            print(f"   Devices: {len(devices)}")
            print("   ✓ Numba CUDA working")
            return True
        else:
            print("   ✗ Numba CUDA not available")
            return False

    except ImportError:
        print("   ⚠️ Numba not installed")
        return False

def test_rocm_gpu():
    """Test ROCm GPU support"""
    try:
        import subprocess
        print("🔍 Testing ROCm GPU...")

        # Try rocminfo
        result = subprocess.run(['rocminfo'], capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✓ ROCm GPU detected")
            return True
        else:
            print("   ✗ ROCm GPU not detected")
            return False

    except FileNotFoundError:
        print("   ⚠️ ROCm tools not installed")
        return False

def main():
    """Run all GPU tests"""
    print("🚀 GPU Enablement Test Suite")
    print("=" * 40)

    tests = [
        ("PyTorch CUDA", test_pytorch_gpu),
        ("CuPy CUDA", test_cupy_gpu),
        ("Numba CUDA", test_numba_gpu),
        ("ROCm", test_rocm_gpu)
    ]

    results = []
    for name, test_func in tests:
        print(f"\n📋 {name}:")
        result = test_func()
        results.append(result)

    print("\n" + "=" * 40)
    print("📊 SUMMARY:")

    successful = sum(results)
    total = len(results)

    if successful > 0:
        print(f"✅ {successful}/{total} GPU frameworks working")
        print("🎉 GPU acceleration is enabled!")
    else:
        print("❌ No GPU acceleration detected")
        print("💡 Check drivers and installation")

    return successful > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)