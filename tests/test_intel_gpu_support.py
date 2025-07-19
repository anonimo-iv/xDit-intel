#!/usr/bin/env python3
"""
Test script for Intel GPU support in xDiT.

This script validates that the Intel GPU integration works correctly
and provides fallback mechanisms when Intel GPU is not available.
"""

import pytest
import torch
import sys
import os

# Add xfuser to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from xfuser.core.device_utils import (
    get_device_type,
    get_device,
    is_gpu_available,
    device_count,
    set_device,
    synchronize,
    empty_cache,
    max_memory_allocated,
    reset_peak_memory_stats,
    get_device_name,
    to_device,
    manual_seed,
    get_distributed_backend,
    supports_flash_attention,
    is_out_of_memory_error
)
from xfuser.envs import _is_cuda, _is_xpu, get_device_version


class TestDeviceUtils:
    """Test device utility functions."""
    
    def test_device_detection(self):
        """Test device type detection."""
        device_type = get_device_type()
        assert device_type in ["cuda", "xpu", "cpu"]
        
        # Test device availability
        gpu_available = is_gpu_available()
        assert isinstance(gpu_available, bool)
        
        if device_type == "xpu":
            assert _is_xpu()
            assert not _is_cuda()
        elif device_type == "cuda":
            assert _is_cuda()
            assert not _is_xpu()
        else:
            assert device_type == "cpu"
    
    def test_device_count(self):
        """Test device count function."""
        count = device_count()
        assert isinstance(count, int)
        assert count >= 1  # At least CPU
        
        if is_gpu_available():
            assert count >= 1
    
    def test_get_device(self):
        """Test device object creation."""
        device = get_device()
        assert isinstance(device, torch.device)
        
        device_type = get_device_type()
        assert str(device).startswith(device_type)
        
        # Test with device ID
        if is_gpu_available():
            device_with_id = get_device(0)
            assert isinstance(device_with_id, torch.device)
    
    def test_tensor_operations(self):
        """Test tensor operations on detected device."""
        device = get_device()
        
        # Create tensor
        tensor = torch.randn(10, 10)
        
        # Move to device
        tensor_on_device = to_device(tensor, device)
        assert tensor_on_device.device.type == device.type
        
        # Test basic operations
        result = tensor_on_device + tensor_on_device
        assert result.device.type == device.type
    
    def test_memory_functions(self):
        """Test memory management functions."""
        if is_gpu_available():
            # These should not raise errors
            reset_peak_memory_stats()
            max_mem = max_memory_allocated()
            assert isinstance(max_mem, int)
            assert max_mem >= 0
            
            # Test cache clearing
            empty_cache()  # Should not raise
    
    def test_synchronization(self):
        """Test device synchronization."""
        # Should not raise errors
        synchronize()
    
    def test_random_seed(self):
        """Test random seed setting."""
        # Should not raise errors
        manual_seed(42)
        
        # Test tensor generation with seed
        torch.manual_seed(42)
        tensor1 = torch.randn(5, 5)
        
        torch.manual_seed(42)
        tensor2 = torch.randn(5, 5)
        
        assert torch.allclose(tensor1, tensor2)
    
    def test_device_name(self):
        """Test device name retrieval."""
        if is_gpu_available():
            name = get_device_name()
            assert isinstance(name, str)
            assert len(name) > 0
    
    def test_distributed_backend(self):
        """Test distributed backend selection."""
        backend = get_distributed_backend()
        assert backend in ["nccl", "ccl", "gloo"]
        
        device_type = get_device_type()
        if device_type == "cuda":
            assert backend == "nccl"
        elif device_type == "xpu":
            assert backend == "ccl"
        else:
            assert backend == "gloo"
    
    def test_flash_attention_support(self):
        """Test flash attention support detection."""
        support = supports_flash_attention()
        assert isinstance(support, bool)
        
        device_type = get_device_type()
        if device_type == "xpu":
            # Intel GPU should not support flash attention
            assert not support
    
    def test_error_detection(self):
        """Test OOM error detection."""
        # Test with a regular exception
        regular_error = ValueError("test error")
        assert not is_out_of_memory_error(regular_error)
        
        # Test with CUDA OOM error (if CUDA available)
        if torch.cuda.is_available():
            try:
                # Try to allocate too much memory
                large_tensor = torch.zeros(10**10, device='cuda')
            except torch.cuda.OutOfMemoryError as e:
                assert is_out_of_memory_error(e)
            except Exception:
                pass  # Not enough memory to trigger OOM


class TestEnvironmentDetection:
    """Test environment detection functions."""
    
    def test_backend_detection(self):
        """Test backend detection functions."""
        cuda_available = _is_cuda()
        xpu_available = _is_xpu()
        
        # Only one should be true at most
        assert not (cuda_available and xpu_available)
        
        # Test device version retrieval
        try:
            version = get_device_version()
            assert isinstance(version, str)
        except Exception as e:
            # Should only fail if no accelerators available
            assert "No Accelerators" in str(e)


class TestIntelGPUSpecific:
    """Intel GPU specific tests."""
    
    @pytest.mark.skipif(not _is_xpu(), reason="Intel GPU not available")
    def test_intel_extension_import(self):
        """Test Intel Extension for PyTorch import."""
        try:
            import intel_extension_for_pytorch as ipex
            assert hasattr(ipex, '__version__')
        except ImportError:
            pytest.fail("Intel Extension for PyTorch not available")
    
    @pytest.mark.skipif(not _is_xpu(), reason="Intel GPU not available")
    def test_xpu_operations(self):
        """Test basic XPU operations."""
        assert torch.xpu.is_available()
        assert torch.xpu.device_count() > 0
        
        # Test device operations
        device = torch.device('xpu:0')
        tensor = torch.randn(10, 10, device=device)
        
        # Test computations
        result = torch.matmul(tensor, tensor.t())
        assert result.device.type == 'xpu'
        
        # Test synchronization
        torch.xpu.synchronize()
    
    @pytest.mark.skipif(not _is_xpu(), reason="Intel GPU not available")
    def test_intel_gpu_memory(self):
        """Test Intel GPU memory operations."""
        device = torch.device('xpu:0')
        
        # Test memory allocation
        tensor = torch.zeros(1000, 1000, device=device)
        assert tensor.device.type == 'xpu'
        
        # Test memory management
        torch.xpu.empty_cache()
        
        # Test memory statistics (if available)
        try:
            max_mem = torch.xpu.max_memory_allocated()
            assert isinstance(max_mem, int)
        except AttributeError:
            # Some versions may not have this function
            pass


class TestFallbackMechanisms:
    """Test fallback mechanisms when features are not available."""
    
    def test_flash_attention_fallback(self):
        """Test that flash attention falls back gracefully."""
        from xfuser.envs import PackagesEnvChecker
        
        checker = PackagesEnvChecker()
        has_flash_attn = checker.check_flash_attn()
        
        # Should be boolean
        assert isinstance(has_flash_attn, bool)
        
        # On Intel GPU, should be False
        if _is_xpu():
            assert not has_flash_attn
    
    def test_distributed_backend_fallback(self):
        """Test distributed backend selection."""
        backend = get_distributed_backend()
        
        # Should always return a valid backend
        assert backend in ["nccl", "ccl", "gloo"]


def run_manual_tests():
    """Run manual tests for interactive verification."""
    print("=== Manual Intel GPU Tests ===")
    
    print("\n1. Device Detection:")
    print(f"   Device type: {get_device_type()}")
    print(f"   GPU available: {is_gpu_available()}")
    print(f"   Device count: {device_count()}")
    
    if is_gpu_available():
        print(f"   Device name: {get_device_name()}")
        print(f"   Device: {get_device()}")
    
    print("\n2. Backend Selection:")
    print(f"   Distributed backend: {get_distributed_backend()}")
    print(f"   Flash attention supported: {supports_flash_attention()}")
    
    print("\n3. Environment Check:")
    print(f"   CUDA available: {_is_cuda()}")
    print(f"   Intel XPU available: {_is_xpu()}")
    
    if _is_xpu():
        print("\n4. Intel GPU Specific:")
        try:
            import intel_extension_for_pytorch as ipex
            print(f"   Intel Extension version: {ipex.__version__}")
        except ImportError:
            print("   Intel Extension: Not available")
        
        print(f"   XPU device count: {torch.xpu.device_count()}")
        for i in range(torch.xpu.device_count()):
            print(f"   XPU device {i}: {torch.xpu.get_device_name(i)}")
    
    print("\n5. Simple Computation Test:")
    try:
        device = get_device()
        tensor = torch.randn(100, 100, device=device)
        result = torch.matmul(tensor, tensor.t())
        print(f"   ✓ Computation successful on {device}")
        print(f"   Result shape: {result.shape}")
        
        # Memory test
        if is_gpu_available():
            max_mem = max_memory_allocated()
            print(f"   Peak memory: {max_mem / 1024**2:.2f} MB")
    
    except Exception as e:
        print(f"   ✗ Computation failed: {e}")


if __name__ == "__main__":
    # Run manual tests if executed directly
    run_manual_tests()
    
    # Run pytest tests
    print("\n=== Running Automated Tests ===")
    pytest.main([__file__, "-v"])