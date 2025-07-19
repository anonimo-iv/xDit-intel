"""
Intel GPU CCL (oneCCL) backend support for distributed computing.

This module provides CCL backend integration for Intel GPU distributed communication.
"""

import torch
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def is_ccl_available() -> bool:
    """Check if oneCCL backend is available."""
    try:
        # Check if CCL is available in PyTorch distributed
        import torch.distributed as dist
        return "ccl" in dist.Backend.__members__.values() or hasattr(dist.Backend, 'CCL')
    except (ImportError, AttributeError):
        return False


def setup_ccl_environment():
    """Setup environment variables for Intel GPU CCL backend."""
    # Set CCL specific environment variables
    ccl_env_vars = {
        'CCL_WORKER_COUNT': str(torch.xpu.device_count()) if hasattr(torch, 'xpu') and torch.xpu.is_available() else '1',
        'CCL_WORKER_AFFINITY': 'auto',
        'CCL_LOG_LEVEL': os.getenv('CCL_LOG_LEVEL', 'WARN'),
        'I_MPI_PIN_DOMAIN': 'auto',
        # Enable Intel GPU communication
        'CCL_PROCESS_LAUNCHER': 'none',
        'CCL_ATL_TRANSPORT': 'ofi',
    }
    
    for key, value in ccl_env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.debug(f"Set CCL environment variable: {key}={value}")


def init_ccl_process_group(
    backend: str = "ccl",
    init_method: str = "env://",
    world_size: int = -1,
    rank: int = -1,
    timeout: Optional[float] = None
):
    """Initialize CCL process group for Intel GPU distributed computing."""
    
    if not is_ccl_available():
        raise RuntimeError(
            "CCL backend is not available. Please ensure Intel Extension for PyTorch "
            "is installed with CCL support: pip install intel-extension-for-pytorch"
        )
    
    # Setup CCL environment
    setup_ccl_environment()
    
    # Import Intel Extension for PyTorch to enable CCL
    try:
        import intel_extension_for_pytorch as ipex
        logger.info(f"Using Intel Extension for PyTorch version: {ipex.__version__}")
    except ImportError:
        raise RuntimeError(
            "Intel Extension for PyTorch is required for CCL backend. "
            "Install with: pip install intel-extension-for-pytorch"
        )
    
    # Initialize process group with CCL backend
    import torch.distributed as dist
    
    # Handle timeout parameter
    timeout_delta = None
    if timeout is not None:
        import datetime
        timeout_delta = datetime.timedelta(seconds=timeout)
    
    try:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            timeout=timeout_delta
        )
        logger.info(f"CCL process group initialized with backend: {backend}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize CCL process group: {e}")
        raise


def get_ccl_backend_info():
    """Get information about CCL backend availability and configuration."""
    info = {
        'ccl_available': is_ccl_available(),
        'intel_gpu_available': hasattr(torch, 'xpu') and torch.xpu.is_available(),
        'ccl_env_vars': {},
        'ipex_available': False,
        'ipex_version': None
    }
    
    # Check Intel Extension
    try:
        import intel_extension_for_pytorch as ipex
        info['ipex_available'] = True
        info['ipex_version'] = ipex.__version__
    except ImportError:
        pass
    
    # Collect CCL environment variables
    ccl_vars = [
        'CCL_WORKER_COUNT', 'CCL_WORKER_AFFINITY', 'CCL_LOG_LEVEL',
        'CCL_PROCESS_LAUNCHER', 'CCL_ATL_TRANSPORT', 'I_MPI_PIN_DOMAIN'
    ]
    
    for var in ccl_vars:
        if var in os.environ:
            info['ccl_env_vars'][var] = os.environ[var]
    
    return info


def validate_ccl_setup():
    """Validate CCL setup for Intel GPU distributed computing."""
    info = get_ccl_backend_info()
    
    issues = []
    
    if not info['intel_gpu_available']:
        issues.append("Intel GPU (XPU) is not available")
    
    if not info['ipex_available']:
        issues.append("Intel Extension for PyTorch is not installed")
    
    if not info['ccl_available']:
        issues.append("CCL backend is not available in PyTorch distributed")
    
    if issues:
        raise RuntimeError(
            f"CCL setup validation failed:\n" + "\n".join(f"- {issue}" for issue in issues)
        )
    
    logger.info("CCL setup validation passed")
    return True


# Integration with existing xDiT distributed code
def patch_distributed_init():
    """Patch existing distributed initialization to support CCL."""
    
    # This would be called during xDiT initialization to ensure
    # CCL backend is properly configured when Intel GPU is detected
    
    from xfuser.core.device_utils import get_device_type
    
    if get_device_type() == "xpu":
        setup_ccl_environment()
        logger.info("CCL environment configured for Intel GPU")


# Example usage function
def example_ccl_distributed_setup():
    """Example of how to setup distributed training with CCL on Intel GPU."""
    
    print("Setting up CCL distributed environment for Intel GPU...")
    
    # Validate setup
    try:
        validate_ccl_setup()
        print("✓ CCL setup validation passed")
    except RuntimeError as e:
        print(f"✗ CCL setup validation failed: {e}")
        return
    
    # Initialize process group
    try:
        # These would typically come from environment variables or arguments
        world_size = int(os.getenv('WORLD_SIZE', '1'))
        rank = int(os.getenv('RANK', '0'))
        
        if world_size > 1:
            init_ccl_process_group(
                world_size=world_size,
                rank=rank
            )
            print(f"✓ CCL process group initialized (rank {rank}/{world_size})")
        else:
            print("Single GPU setup, no distributed initialization needed")
            
    except Exception as e:
        print(f"✗ Failed to initialize CCL: {e}")


if __name__ == "__main__":
    # Run example if executed directly
    example_ccl_distributed_setup()