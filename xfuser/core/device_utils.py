"""
Device abstraction utilities for supporting multiple GPU backends (CUDA, Intel XPU)
"""
import torch
from typing import Optional, Union
from xfuser.logger import init_logger

logger = init_logger(__name__)


def get_device_type() -> str:
    """Get the primary device type available."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"


def get_device(device_id: Optional[int] = None) -> torch.device:
    """Get a device object for the specified device ID."""
    device_type = get_device_type()
    if device_id is not None:
        return torch.device(f"{device_type}:{device_id}")
    else:
        return torch.device(device_type)


def is_gpu_available() -> bool:
    """Check if any GPU (CUDA or Intel XPU) is available."""
    return torch.cuda.is_available() or (hasattr(torch, 'xpu') and torch.xpu.is_available())


def device_count() -> int:
    """Get the number of available devices."""
    device_type = get_device_type()
    if device_type == "cuda":
        return torch.cuda.device_count()
    elif device_type == "xpu":
        return torch.xpu.device_count()
    else:
        return 1  # CPU


def set_device(device: Union[int, torch.device, str]):
    """Set the current device."""
    device_type = get_device_type()
    if device_type == "cuda":
        torch.cuda.set_device(device)
    elif device_type == "xpu":
        torch.xpu.set_device(device)
    # No action needed for CPU


def synchronize():
    """Synchronize the current device."""
    device_type = get_device_type()
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "xpu":
        torch.xpu.synchronize()
    # No action needed for CPU


def empty_cache():
    """Empty the device cache."""
    device_type = get_device_type()
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "xpu":
        torch.xpu.empty_cache()
    # No action needed for CPU


def max_memory_allocated(device: Optional[Union[torch.device, str, int]] = None) -> int:
    """Get the maximum memory allocated on the device."""
    device_type = get_device_type()
    if device_type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    elif device_type == "xpu":
        return torch.xpu.max_memory_allocated(device)
    else:
        return 0  # CPU doesn't track GPU memory


def reset_peak_memory_stats(device: Optional[Union[torch.device, str, int]] = None):
    """Reset peak memory statistics."""
    device_type = get_device_type()
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device_type == "xpu":
        torch.xpu.reset_peak_memory_stats(device)
    # No action needed for CPU


def get_device_name(device: Optional[Union[torch.device, str, int]] = None) -> str:
    """Get the name of the device."""
    device_type = get_device_type()
    if device_type == "cuda":
        return torch.cuda.get_device_name(device)
    elif device_type == "xpu":
        return torch.xpu.get_device_name(device)
    else:
        return "CPU"


def to_device(tensor: torch.Tensor, device: Optional[Union[torch.device, str, int]] = None) -> torch.Tensor:
    """Move tensor to the specified device."""
    if device is None:
        device = get_device()
    return tensor.to(device)


def manual_seed(seed: int, device: Optional[Union[torch.device, str, int]] = None):
    """Set random seed for the device."""
    device_type = get_device_type()
    if device_type == "cuda":
        torch.cuda.manual_seed(seed)
        if device is not None:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
    elif device_type == "xpu":
        torch.xpu.manual_seed(seed)
        if device is not None:
            with torch.xpu.device(device):
                torch.xpu.manual_seed(seed)
    # Also set the global seed
    torch.manual_seed(seed)


def get_memory_info(device: Optional[Union[torch.device, str, int]] = None):
    """Get memory information for the device."""
    device_type = get_device_type()
    if device_type == "cuda":
        return torch.cuda.mem_get_info(device)
    elif device_type == "xpu":
        # Intel XPU may not have mem_get_info, return None
        return None
    else:
        return None


def is_out_of_memory_error(error: Exception) -> bool:
    """Check if the error is an out-of-memory error."""
    device_type = get_device_type()
    if device_type == "cuda":
        return isinstance(error, torch.cuda.OutOfMemoryError)
    elif device_type == "xpu":
        # Check for Intel GPU OOM errors (may vary based on Intel Extension version)
        return "out of memory" in str(error).lower()
    else:
        return False


def supports_flash_attention() -> bool:
    """Check if the current device supports flash attention."""
    device_type = get_device_type()
    if device_type == "cuda":
        try:
            from flash_attn import flash_attn_func
            return True
        except ImportError:
            return False
    elif device_type == "xpu":
        # Intel GPUs currently don't support flash-attn
        return False
    else:
        return False


def get_distributed_backend() -> str:
    """Get the appropriate distributed backend for the device."""
    device_type = get_device_type()
    if device_type == "cuda":
        return "nccl"
    elif device_type == "xpu":
        # Intel GPU uses ccl (oneCCL) for collective communication
        return "ccl"
    else:
        return "gloo"