#!/usr/bin/env python3
"""
Example script demonstrating Intel GPU support with xDiT.

This example shows how to run diffusion models on Intel GPUs using PyTorch's XPU backend.

Prerequisites:
1. Intel Arc Graphics or Intel Data Center GPU Max Series
2. PyTorch 2.5.0 or later
3. Intel Extension for PyTorch 2.5.0 or later

Installation:
pip install torch>=2.5.0
pip install intel-extension-for-pytorch>=2.5.0
pip install xfuser[intel-gpu]

Usage:
python intel_gpu_example.py --model pixart-alpha --height 512 --width 512
"""

import argparse
import torch
import sys
import os

# Add xfuser to path if running from source
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from xfuser import xFuserPixArtAlphaPipeline
from xfuser.config import FlexibleArgumentParser
from xfuser.core.device_utils import get_device_type, is_gpu_available


def check_intel_gpu_support():
    """Check if Intel GPU support is available."""
    print("Checking Intel GPU support...")
    
    # Check PyTorch version
    torch_version = torch.__version__
    print(f"PyTorch version: {torch_version}")
    
    # Check for XPU support
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print("✓ Intel XPU backend is available")
        device_count = torch.xpu.device_count()
        print(f"✓ Found {device_count} Intel GPU(s)")
        
        for i in range(device_count):
            device_name = torch.xpu.get_device_name(i)
            print(f"  Device {i}: {device_name}")
        
        return True
    else:
        print("✗ Intel XPU backend is not available")
        print("Please ensure you have:")
        print("  1. Intel Arc Graphics or Data Center GPU Max Series")
        print("  2. PyTorch 2.5.0 or later")
        print("  3. Intel Extension for PyTorch 2.5.0 or later")
        return False


def check_intel_extension():
    """Check if Intel Extension for PyTorch is available."""
    try:
        import intel_extension_for_pytorch as ipex
        print(f"✓ Intel Extension for PyTorch version: {ipex.__version__}")
        return True
    except ImportError:
        print("✗ Intel Extension for PyTorch not found")
        print("Install with: pip install intel-extension-for-pytorch>=2.5.0")
        return False


def main():
    parser = FlexibleArgumentParser(description="xDiT Intel GPU Example")
    parser.add_argument(
        "--model",
        type=str,
        default="pixart-alpha",
        choices=["pixart-alpha", "pixart-sigma"],
        help="Model to use for generation"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Height of generated image"
    )
    parser.add_argument(
        "--width", 
        type=int,
        default=512,
        help="Width of generated image"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A beautiful landscape with mountains and a lake",
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./intel_gpu_output.png",
        help="Output path for generated image"
    )
    
    args = parser.parse_args()
    
    # Check system requirements
    print("=== Intel GPU Support Check ===")
    if not check_intel_extension():
        sys.exit(1)
    
    if not check_intel_gpu_support():
        sys.exit(1)
    
    # Initialize Intel Extension for PyTorch
    try:
        import intel_extension_for_pytorch as ipex
        # Enable XPU optimizations
        ipex.enable_auto_mixed_precision(dtype=torch.float16)
        print("✓ Intel Extension optimizations enabled")
    except Exception as e:
        print(f"Warning: Could not enable Intel optimizations: {e}")
    
    print("\n=== Device Information ===")
    device_type = get_device_type()
    print(f"Primary device type: {device_type}")
    print(f"GPU available: {is_gpu_available()}")
    
    # Model configuration based on selection
    if args.model == "pixart-alpha":
        model_id = "PixArt-alpha/PixArt-XL-2-1024-MS"
        PipelineClass = xFuserPixArtAlphaPipeline
    else:
        print(f"Model {args.model} not implemented in this example")
        sys.exit(1)
    
    print(f"\n=== Loading Model: {model_id} ===")
    
    try:
        # Initialize pipeline with Intel GPU support
        pipe = PipelineClass.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # Use FP16 for better performance on Intel GPU
        )
        
        print("✓ Model loaded successfully")
        
        # Generate image
        print(f"\n=== Generating Image ===")
        print(f"Prompt: {args.prompt}")
        print(f"Size: {args.width}x{args.height}")
        print(f"Steps: {args.num_inference_steps}")
        
        with torch.no_grad():
            image = pipe(
                prompt=args.prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                generator=torch.Generator(device=device_type).manual_seed(42),
            ).images[0]
        
        # Save image
        image.save(args.output_path)
        print(f"✓ Image saved to: {args.output_path}")
        
        # Performance statistics
        print(f"\n=== Performance Statistics ===")
        from xfuser.core.device_utils import max_memory_allocated
        max_memory = max_memory_allocated()
        if max_memory > 0:
            print(f"Peak GPU memory usage: {max_memory / 1024**3:.2f} GB")
        
    except Exception as e:
        print(f"✗ Error during generation: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure Intel GPU drivers are properly installed")
        print("2. Check that your GPU has sufficient memory")
        print("3. Try reducing image resolution or inference steps")
        print("4. Verify Intel Extension for PyTorch installation")
        sys.exit(1)
    
    print("\n=== Generation Complete ===")
    print("Intel GPU example completed successfully!")


if __name__ == "__main__":
    main()