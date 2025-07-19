#!/usr/bin/env python3
"""
Distributed inference example for Intel GPU using CCL backend.

This example demonstrates how to run distributed diffusion model inference
on multiple Intel GPUs using oneCCL for communication.

Prerequisites:
1. Multiple Intel Arc Graphics or Intel Data Center GPU Max Series
2. PyTorch 2.5.0 or later
3. Intel Extension for PyTorch 2.5.0 or later
4. oneCCL bindings for PyTorch

Installation:
pip install torch>=2.5.0
pip install intel-extension-for-pytorch>=2.5.0
pip install oneccl_bind_pt>=2.5.0
pip install xfuser[intel-gpu]

Usage:
# Single node, multiple Intel GPUs
torchrun --nproc_per_node=2 intel_gpu_distributed_example.py

# Multiple nodes
torchrun --nnodes=2 --nproc_per_node=2 --master_addr=<master_ip> intel_gpu_distributed_example.py
"""

import argparse
import torch
import torch.distributed as dist
import sys
import os

# Add xfuser to path if running from source
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from xfuser import xFuserPixArtAlphaPipeline
from xfuser.config import FlexibleArgumentParser
from xfuser.core.device_utils import get_device_type, is_gpu_available, get_device
from xfuser.core.distributed.ccl_backend import get_ccl_backend_info, validate_ccl_setup


def setup_distributed():
    """Setup distributed environment for Intel GPU."""
    
    # Initialize distributed environment
    if not dist.is_initialized():
        # Use CCL backend for Intel GPU
        backend = "ccl" if get_device_type() == "xpu" else "gloo"
        
        print(f"Initializing distributed with backend: {backend}")
        
        try:
            dist.init_process_group(backend=backend)
            print(f"✓ Distributed initialized (rank {dist.get_rank()}/{dist.get_world_size()})")
        except Exception as e:
            print(f"✗ Distributed initialization failed: {e}")
            if backend == "ccl":
                print("Falling back to gloo backend...")
                dist.init_process_group(backend="gloo")
    
    # Set device for current process
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = get_device(local_rank)
    
    print(f"Process {dist.get_rank()}: Using device {device}")
    return device, local_rank


def cleanup_distributed():
    """Cleanup distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def distributed_inference(args):
    """Run distributed inference on Intel GPU."""
    
    # Setup distributed
    device, local_rank = setup_distributed()
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"Rank {rank}/{world_size}: Starting distributed inference")
    
    # Only load model on rank 0 initially (can be optimized later)
    if rank == 0:
        print("Loading model...")
        pipe = xFuserPixArtAlphaPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
        )
        print("✓ Model loaded on rank 0")
    else:
        pipe = None
    
    # Synchronize all processes
    dist.barrier()
    
    # In a real distributed setup, you would:
    # 1. Split the model across GPUs
    # 2. Distribute the workload
    # 3. Use collective communication for synchronization
    
    if rank == 0:
        print(f"Generating images with {world_size} Intel GPUs...")
        
        # Generate multiple images distributed across GPUs
        images = []
        for i in range(args.num_images):
            if i % world_size == rank:
                print(f"Rank {rank}: Generating image {i+1}")
                
                with torch.no_grad():
                    image = pipe(
                        prompt=f"{args.prompt} (image {i+1})",
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.num_inference_steps,
                        generator=torch.Generator(device=device).manual_seed(42 + i),
                    ).images[0]
                
                # Save image
                output_path = f"{args.output_dir}/intel_gpu_distributed_{i+1}.png"
                os.makedirs(args.output_dir, exist_ok=True)
                image.save(output_path)
                print(f"✓ Rank {rank}: Saved image to {output_path}")
                
                images.append(image)
    
    # Synchronize all processes
    dist.barrier()
    
    if rank == 0:
        print(f"✓ Distributed inference completed on {world_size} Intel GPUs")


def main():
    parser = FlexibleArgumentParser(description="xDiT Intel GPU Distributed Example")
    parser.add_argument(
        "--model_id",
        type=str,
        default="PixArt-alpha/PixArt-XL-2-512-MS",
        help="Model ID to use for generation"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Height of generated images"
    )
    parser.add_argument(
        "--width", 
        type=int,
        default=512,
        help="Width of generated images"
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
        "--num_images",
        type=int,
        default=4,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./intel_gpu_distributed_output",
        help="Output directory for generated images"
    )
    
    args = parser.parse_args()
    
    # Check if we're in distributed mode
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        if rank == 0:
            print("=== Intel GPU Distributed Setup Check ===")
            
            # Check CCL backend info
            ccl_info = get_ccl_backend_info()
            print(f"CCL available: {ccl_info['ccl_available']}")
            print(f"Intel GPU available: {ccl_info['intel_gpu_available']}")
            print(f"Intel Extension available: {ccl_info['ipex_available']}")
            if ccl_info['ipex_version']:
                print(f"Intel Extension version: {ccl_info['ipex_version']}")
            
            # Validate CCL setup
            try:
                validate_ccl_setup()
                print("✓ CCL setup validation passed")
            except RuntimeError as e:
                print(f"✗ CCL setup validation failed: {e}")
                print("Continuing with available backend...")
            
            print(f"\n=== Starting Distributed Inference ===")
            print(f"World size: {world_size}")
            print(f"Model: {args.model_id}")
            print(f"Images to generate: {args.num_images}")
        
        # Run distributed inference
        try:
            distributed_inference(args)
        finally:
            cleanup_distributed()
            
    else:
        print("This script should be run with torchrun for distributed execution.")
        print("Example: torchrun --nproc_per_node=2 intel_gpu_distributed_example.py")
        print("\nFor single GPU inference, use intel_gpu_example.py instead.")
        sys.exit(1)


if __name__ == "__main__":
    main()