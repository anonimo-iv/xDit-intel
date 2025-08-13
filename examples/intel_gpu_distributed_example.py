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
# Single node, multiple Intel GPUs with torchrun
torchrun --nproc_per_node=2 intel_gpu_distributed_example.py

# Multiple nodes with torchrun
torchrun --nnodes=2 --nproc_per_node=2 --master_addr=<master_ip> intel_gpu_distributed_example.py

# Single node, multiple Intel GPUs with mpiexec
mpiexec -n 2 python intel_gpu_distributed_example.py

# Multiple nodes with mpiexec
mpiexec -hosts host1,host2 -n 4 python intel_gpu_distributed_example.py
"""

import argparse
import torch
import torch.distributed as dist
import sys
import os

# Add xfuser to path if running from source
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# CRITICAL: Setup MPI environment variables BEFORE any imports or MPI operations
# Check if we're in MPI mode and set environment variables early
if 'PMI_RANK' in os.environ:
    os.environ['RANK'] = os.environ['PMI_RANK']
    os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', '1')
    os.environ['LOCAL_RANK'] = '0'  # Will be corrected later
elif 'OMPI_COMM_WORLD_RANK' in os.environ:
    os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    os.environ['WORLD_SIZE'] = os.environ.get('OMPI_COMM_WORLD_SIZE', '1')
    os.environ['LOCAL_RANK'] = '0'  # Will be corrected later

# Set master address and port if not set
if 'MASTER_ADDR' not in os.environ:
    os.environ['MASTER_ADDR'] = 'localhost'
if 'MASTER_PORT' not in os.environ:
    os.environ['MASTER_PORT'] = '12355'

# Setup CCL environment BEFORE importing anything else
is_mpi = 'PMI_RANK' in os.environ or 'OMPI_COMM_WORLD_RANK' in os.environ
if is_mpi:
    # Force MPI-specific CCL settings
    os.environ['CCL_PROCESS_LAUNCHER'] = 'pmix'
    os.environ['CCL_ATL_TRANSPORT'] = 'mpi'
    os.environ['CCL_KVS_MODE'] = 'mpi'
    os.environ['CCL_KVS_USE_MPI_RANKS'] = '1'
    os.environ['CCL_ATL_SYNC_COLL'] = '1'
    os.environ['CCL_OP_SYNC'] = '1'
    os.environ['CCL_WORKER_COUNT'] = '1'  # Start with 1 worker to avoid issues
    os.environ['CCL_LOG_LEVEL'] = 'warn'  # Reduce verbosity
    os.environ['CCL_ZE_IPC_EXCHANGE'] = 'drmfd'

# Intel GPU specific imports - import oneccl_bindings_for_pytorch before torch.distributed
try:
    import intel_extension_for_pytorch as ipex
    import oneccl_bindings_for_pytorch as torch_ccl
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

# Initialize MPI if we're in MPI mode to prevent "MPI routine before initializing" errors
if is_mpi:
    try:
        from mpi4py import MPI
        # MPI.Init() is called automatically on import
        print(f"MPI initialized: rank {MPI.COMM_WORLD.Get_rank()}/{MPI.COMM_WORLD.Get_size()}")
    except ImportError:
        print("Warning: mpi4py not available, CCL may have issues with MPI transport")

from xfuser import xFuserPixArtAlphaPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.device_utils import get_device_type, is_gpu_available, get_device
from xfuser.core.distributed.ccl_backend import get_ccl_backend_info, validate_ccl_setup


def setup_distributed():
    """Setup distributed environment for Intel GPU."""
    
    # The xfuser's init_distributed_environment will handle all the setup
    # including MPI detection and CCL configuration
    
    # Just get the device for current process
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    global_rank = int(os.environ.get('RANK', '0'))
    
    # Debug device visibility
    if get_device_type() == "xpu":
        import torch
        print(f"Process rank={global_rank}, local_rank={local_rank}: XPU device count = {torch.xpu.device_count()}")
        print(f"Available XPU devices: {[torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count())]}")
    
    device = get_device(local_rank)
    
    print(f"Process rank={global_rank}, local_rank={local_rank}: Using device {device}")
    return device, local_rank


def cleanup_distributed():
    """Cleanup distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def distributed_inference(args, world_size):
    """Run distributed inference on Intel GPU."""
    
    # Don't set ZE_AFFINITY_MASK - let Intel GPU runtime auto-detect all XPUs
    # This allows proper device assignment across all available XPUs
    
    # For distributed inference, we need to set parallel degrees that multiply to world_size
    
    # For distributed inference, configure parallelism appropriately
    # Option 1: Use data parallelism (each GPU processes different batches)
    # Option 2: Use sequence parallelism (split sequence across GPUs)
    
    # Use ring parallelism as requested
    print(f"Using ring parallelism with ring_degree={world_size}")
    engine_args = xFuserArgs(
        model=args.model_id,
        height=args.height,
        width=args.width,
        ulysses_degree=1,
        ring_degree=world_size,  # Use ring attention across all GPUs
        data_parallel_degree=1,  # No data parallelism
    )
    print(f"Created xFuserArgs with data_parallel_degree={engine_args.data_parallel_degree}")
    
    # xfuser will handle all distributed initialization in create_config
    engine_config, input_config = engine_args.create_config()
    
    # Now we can safely get the distributed info
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device, local_rank = setup_distributed()
    
    # All ranks must load the model for distributed inference
    print(f"Rank {rank}: Loading model...")
    pipe = xFuserPixArtAlphaPipeline.from_pretrained(
        args.model_id,
        engine_config=engine_config,
        torch_dtype=torch.float16,
    )
    print(f"✓ Rank {rank}: Model loaded")
    
    # Apply IPEX optimization to prevent segfault
    if IPEX_AVAILABLE and hasattr(torch, 'xpu') and torch.xpu.is_available() and not args.no_ipex_optimize:
        print(f"Rank {rank}: Applying IPEX optimization...")
        
        # Optimize transformer
        if hasattr(pipe, 'transformer'):
            pipe.transformer = pipe.transformer.to(device)
            pipe.transformer.eval()
            pipe.transformer = ipex.optimize(pipe.transformer, dtype=torch.float16)
            print(f"✓ Rank {rank}: Transformer optimized with IPEX")
        
        # Optimize VAE
        if hasattr(pipe, 'vae') and pipe.vae is not None:
            pipe.vae = pipe.vae.to(device)
            pipe.vae.eval()
            pipe.vae = ipex.optimize(pipe.vae, dtype=torch.float16)
            print(f"✓ Rank {rank}: VAE optimized with IPEX")
        
        # Optimize text encoder if present
        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
            pipe.text_encoder = pipe.text_encoder.to(device)
            pipe.text_encoder.eval()
            pipe.text_encoder = ipex.optimize(pipe.text_encoder, dtype=torch.float16)
            print(f"✓ Rank {rank}: Text encoder optimized with IPEX")
        
        print(f"✓ Rank {rank}: IPEX optimization complete")
    else:
        print(f"⚠️ Rank {rank}: IPEX optimization not available")
    
    # Skip barriers to avoid hangs
    # dist.barrier()
    
    # Prepare for distributed inference
    print(f"Rank {rank}: Preparing for inference...")
    pipe.prepare_run(input_config)
    
    # Generate images distributed across GPUs
    print(f"Rank {rank}: Generating images with {world_size} Intel GPUs...")
    
    # Generate images based on parallelism mode
    images = []
    is_data_parallel = engine_args.data_parallel_degree > 1
    
    for i in range(args.num_images):
        if is_data_parallel:
            # Data parallelism: each rank generates different images
            if i % world_size == rank:
                print(f"Rank {rank}: Generating image {i+1}")
                with torch.no_grad():
                    output = pipe(
                        prompt=f"{args.prompt}",
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.num_inference_steps,
                        generator=torch.Generator(device=device).manual_seed(42 + i),
                        output_type="pil",
                    )
                    image = output.images[0]
                    # Save image
                    output_path = f"{args.output_dir}/intel_gpu_distributed_rank{rank}_img{i+1}.png"
                    os.makedirs(args.output_dir, exist_ok=True)
                    image.save(output_path)
                    print(f"✓ Rank {rank}: Saved image to {output_path}")
                    images.append(image)
        else:
            # Ring parallelism: all ranks participate in each image
            print(f"Rank {rank}: Contributing to image {i+1} generation with ring parallelism")
            with torch.no_grad():
                output = pipe(
                    prompt=f"{args.prompt}",
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_inference_steps,
                    generator=torch.Generator(device=device).manual_seed(42 + i),
                    output_type="pil" if rank == 0 else "latent",
                )
                if rank == 0 and hasattr(output, 'images'):
                    image = output.images[0]
                    output_path = f"{args.output_dir}/intel_gpu_distributed_img{i+1}.png"
                    os.makedirs(args.output_dir, exist_ok=True)
                    image.save(output_path)
                    print(f"✓ Rank {rank}: Saved image to {output_path}")
                    images.append(image)
    
    # Skip final barrier to avoid hangs
    # dist.barrier()
    
    if rank == 0:
        print(f"✓ Distributed inference completed on {world_size} Intel GPUs")


def main():
    parser = FlexibleArgumentParser(description="xDiT Intel GPU Distributed Example")
    parser.add_argument(
        "--model_id",
        type=str,
        default="PixArt-alpha/PixArt-XL-2-512x512",
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
    parser.add_argument(
        "--no_ipex_optimize",
        action="store_true",
        help="Disable IPEX optimization (may cause segfault with sequence parallelism)"
    )
    
    args = parser.parse_args()
    
    # Check if we're in distributed mode (handles both torchrun and MPI)
    is_distributed = any(key in os.environ for key in ['RANK', 'PMI_RANK', 'OMPI_COMM_WORLD_RANK'])
    
    if is_distributed:
        # Get rank info from environment
        if 'RANK' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ.get('WORLD_SIZE', '1'))
        elif 'PMI_RANK' in os.environ:
            rank = int(os.environ['PMI_RANK'])
            world_size = int(os.environ.get('PMI_SIZE', '1'))
        elif 'OMPI_COMM_WORLD_RANK' in os.environ:
            rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
            world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1'))
        else:
            rank = 0
            world_size = 1
        
        if rank == 0:
            print("=== Intel GPU Distributed Setup Check ===")
            
            # Check CCL backend info
            ccl_info = get_ccl_backend_info()
            print(f"CCL available: {ccl_info['ccl_available']}")
            print(f"Intel GPU available: {ccl_info['intel_gpu_available']}")
            print(f"Intel Extension available: {ccl_info['ipex_available']}")
            if ccl_info['ipex_version']:
                print(f"Intel Extension version: {ccl_info['ipex_version']}")
            
            print(f"\n=== Starting Distributed Inference ===")
            print(f"World size: {world_size}")
            print(f"Model: {args.model_id}")
            print(f"Images to generate: {args.num_images}")
        
        # Run distributed inference
        try:
            distributed_inference(args, world_size)
        finally:
            cleanup_distributed()
            
    else:
        print("This script should be run with torchrun or mpiexec for distributed execution.")
        print("Examples:")
        print("  torchrun --nproc_per_node=2 intel_gpu_distributed_example.py")
        print("  mpirun -n 2 python intel_gpu_distributed_example.py")
        print("\nFor single GPU inference, use intel_gpu_example.py instead.")
        sys.exit(1)


if __name__ == "__main__":
    main()