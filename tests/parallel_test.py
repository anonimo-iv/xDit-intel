import torch
import os
from diffusers import StableDiffusion3Pipeline, FluxPipeline

from xfuser import xFuserArgs
from xfuser.parallel import xDiTParallel
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import get_world_group

# Import sp_aurora MPI utilities if available
try:
    from sp_aurora.mpi_utils import (
        setup_mpi_distributed,
        detect_mpi_environment,
    )
    HAS_SP_AURORA_MPI = True
except ImportError:
    HAS_SP_AURORA_MPI = False


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()

    # Check if we're in MPI environment and set up accordingly
    if HAS_SP_AURORA_MPI and detect_mpi_environment():
        print("Detected MPI environment, using sp_aurora MPI setup")
        setup_mpi_distributed()

    local_rank = get_world_group().local_rank
    
    # Determine device based on available hardware
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = f"xpu:{local_rank}"
        # Set Intel GPU specific environment variables
        os.environ.setdefault('CCL_BACKEND', 'native')
        os.environ.setdefault('CCL_ATL_TRANSPORT', 'mpi')
    elif torch.cuda.is_available():
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    
    pipe = StableDiffusion3Pipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        torch_dtype=torch.float16,
    ).to(device)

    paralleler = xDiTParallel(pipe, engine_config, input_config)

    paralleler(
        height=input_config.height,
        width=input_config.height,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        generator=torch.Generator(device=device.split(':')[0]).manual_seed(input_config.seed),
    )
    if input_config.output_type == "pil":
        paralleler.save("results", "stable_diffusion_3")


if __name__ == "__main__":
    main()
