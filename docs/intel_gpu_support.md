# Intel GPU Support for xDiT

This document describes how to use xDiT with Intel GPUs using PyTorch's XPU backend.

## Overview

xDiT now supports Intel GPUs through PyTorch's XPU backend, allowing you to run diffusion models on:
- Intel Arc Graphics (Arc A-Series)
- Intel Core Ultra processors with integrated Arc Graphics
- Intel Data Center GPU Max Series

## Prerequisites

### Hardware Requirements
- Intel Arc Graphics family (Arc A-Series discrete GPUs)
- Intel Core Ultra processors with Intel Arc Graphics
- Intel Core Ultra Series 2 with Intel Arc Graphics  
- Intel Data Center GPU Max Series

### Software Requirements
- **PyTorch 2.5.0 or later** (includes XPU backend support)
- **Intel Extension for PyTorch 2.5.0 or later**
- **Intel GPU drivers** (latest recommended)
- **Linux or Windows 10/11**

## Installation

### 1. Install PyTorch with XPU Support

```bash
# Install PyTorch 2.5.0 or later
pip install torch>=2.5.0 torchvision torchaudio
```

### 2. Install Intel Extension for PyTorch

```bash
# Install Intel Extension for PyTorch
pip install intel-extension-for-pytorch>=2.5.0
```

### 3. Install xDiT with Intel GPU Support

```bash
# Install xDiT with Intel GPU extras (includes CCL for distributed computing)
pip install xfuser[intel-gpu]

# Or if installing from source
pip install -e .[intel-gpu]

# For distributed computing, also install oneCCL bindings
pip install oneccl_bind_pt>=2.5.0
```

## Usage

### Basic Usage

xDiT automatically detects available hardware and uses Intel GPU when available:

```python
import torch
from xfuser import xFuserPixArtAlphaPipeline

# xDiT will automatically use Intel GPU if available
pipe = xFuserPixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    torch_dtype=torch.float16  # Recommended for Intel GPU
)

image = pipe(
    prompt="A beautiful landscape",
    height=512,
    width=512,
    num_inference_steps=20
).images[0]
```

### Manual Device Selection

You can explicitly specify Intel GPU usage:

```python
import torch
from xfuser.core.device_utils import get_device, get_device_type

# Check device type
device_type = get_device_type()  # Returns "xpu" for Intel GPU
print(f"Using device: {device_type}")

# Get Intel GPU device
device = get_device(0)  # Intel GPU 0
print(f"Device: {device}")
```

### Example Script

Run the provided Intel GPU example:

```bash
python examples/intel_gpu_example.py \
    --model pixart-alpha \
    --prompt "A futuristic city skyline" \
    --height 512 \
    --width 512 \
    --num_inference_steps 20
```

## Supported Features

### ✅ Supported
- **Basic diffusion pipeline execution**
- **FP16 and BF16 precision**
- **Memory management and optimization**
- **Multi-GPU support** (multiple Intel GPUs)
- **Distributed inference** (with CCL backend)
- **PyTorch attention mechanisms**
- **Automatic mixed precision (AMP)**

### ⚠️ Limited Support
- **Flash Attention**: Not supported on Intel GPU, falls back to PyTorch attention
- **Custom CUDA kernels**: Intel GPU equivalents may not be available
- **Some optimizations**: NVIDIA-specific optimizations are disabled

### ❌ Not Supported
- **NCCL backend**: Uses CCL (oneCCL) for Intel GPU distributed computing
- **CUDA-specific libraries**: Such as custom CUDA kernels in yunchang

## Performance Optimization

### 1. Use Appropriate Precision
```python
# FP16 recommended for Intel GPU
pipe = xFuserPixArtAlphaPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)
```

### 2. Enable Intel Optimizations
```python
import intel_extension_for_pytorch as ipex

# Enable automatic mixed precision
ipex.enable_auto_mixed_precision(dtype=torch.float16)
```

### 3. Memory Management
```python
from xfuser.core.device_utils import empty_cache

# Clear GPU cache when needed
empty_cache()
```

## Distributed Computing

Intel GPU distributed computing uses CCL (oneCCL) instead of NCCL for efficient multi-GPU communication:

### Automatic Backend Selection
```python
# xDiT automatically selects appropriate backend
from xfuser.core.device_utils import get_distributed_backend

backend = get_distributed_backend()  # Returns "ccl" for Intel GPU
```

### Multi-GPU Setup

#### Prerequisites
```bash
# Install oneCCL bindings for distributed communication
pip install oneccl_bind_pt>=2.5.0
```

#### Environment Configuration
CCL environment is automatically configured by xDiT, but you can customize:

```bash
# Optional: Customize CCL settings
export CCL_WORKER_COUNT=2        # Number of Intel GPUs
export CCL_WORKER_AFFINITY=auto  # Worker affinity
export CCL_LOG_LEVEL=WARN        # Logging level
export CCL_ATL_TRANSPORT=ofi     # Transport layer
```

#### Running Distributed Inference

**Single Node, Multiple GPUs:**
```bash
# Run on 2 Intel GPUs
torchrun --nproc_per_node=2 examples/intel_gpu_distributed_example.py

# Run on 4 Intel GPUs  
torchrun --nproc_per_node=4 examples/intel_gpu_distributed_example.py
```

**Multiple Nodes:**
```bash
# Node 0 (master)
torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 \
  --master_addr=192.168.1.100 --master_port=29500 \
  examples/intel_gpu_distributed_example.py

# Node 1 (worker)
torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1 \
  --master_addr=192.168.1.100 --master_port=29500 \
  examples/intel_gpu_distributed_example.py
```

#### Distributed Example
```python
import torch.distributed as dist
from xfuser import xFuserPixArtAlphaPipeline
from xfuser.core.distributed.ccl_backend import validate_ccl_setup

# Initialize distributed environment (CCL backend auto-selected for Intel GPU)
dist.init_process_group(backend="ccl")

# Validate CCL setup
validate_ccl_setup()

# Load model with distributed configuration
pipe = xFuserPixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    torch_dtype=torch.float16
)

# Run distributed inference
# (Model automatically distributes computation across Intel GPUs)
```

### CCL Backend Features

- **Automatic Setup**: xDiT automatically configures CCL when Intel GPU is detected
- **Fallback Support**: Falls back to Gloo if CCL is unavailable
- **Environment Validation**: Built-in validation of CCL setup
- **Multi-node Support**: Supports distributed inference across multiple machines
- **Optimized Communication**: oneCCL provides optimized collective operations for Intel hardware

## Troubleshooting

### Common Issues

#### 1. Intel GPU Not Detected
```python
import torch

# Check XPU availability
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    print("Intel GPU available")
    print(f"Device count: {torch.xpu.device_count()}")
else:
    print("Intel GPU not available")
```

**Solutions:**
- Ensure Intel GPU drivers are installed
- Verify PyTorch 2.5.0+ installation
- Install Intel Extension for PyTorch

#### 2. Out of Memory Errors
**Solutions:**
- Reduce image resolution
- Use fewer inference steps
- Enable gradient checkpointing
- Use FP16 precision

#### 3. Performance Issues
**Solutions:**
- Enable Intel Extension optimizations
- Use appropriate data types (FP16/BF16)
- Ensure optimal memory layout
- Check Intel GPU utilization

### Verification Script

Check your Intel GPU setup:

```python
def check_intel_gpu_setup():
    import torch
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check XPU availability
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print("✓ Intel XPU backend available")
        device_count = torch.xpu.device_count()
        print(f"✓ Intel GPU devices: {device_count}")
        
        for i in range(device_count):
            name = torch.xpu.get_device_name(i)
            print(f"  Device {i}: {name}")
    else:
        print("✗ Intel XPU backend not available")
    
    # Check Intel Extension
    try:
        import intel_extension_for_pytorch as ipex
        print(f"✓ Intel Extension version: {ipex.__version__}")
    except ImportError:
        print("✗ Intel Extension for PyTorch not found")

check_intel_gpu_setup()
```

## Limitations

1. **Flash Attention**: Not available on Intel GPU, uses PyTorch attention
2. **CUDA Libraries**: NVIDIA-specific libraries are not supported
3. **Performance**: May differ from NVIDIA GPU performance characteristics
4. **Memory Layout**: Intel GPU may have different optimal memory patterns

## Contributing

When contributing Intel GPU support:

1. Use device-agnostic APIs from `xfuser.core.device_utils`
2. Test on both NVIDIA and Intel GPU when possible  
3. Ensure fallback mechanisms for unsupported features
4. Update documentation for Intel GPU specific behaviors

## Resources

- [Intel Extension for PyTorch Documentation](https://intel.github.io/intel-extension-for-pytorch/)
- [PyTorch XPU Documentation](https://pytorch.org/docs/stable/notes/get_start_xpu.html)
- [Intel GPU Software Installation Guide](https://dgpu-docs.intel.com/installation-guides/index.html)
- [Intel oneAPI Documentation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/documentation.html)