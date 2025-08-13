# This file implements USP with torch version < '2.5.0'
import torch
from torch.nn import functional as F

import torch.distributed._functional_collectives as ft_c

from sp_aurora.globals import PROCESS_GROUP
from sp_aurora.ring_pytorch_attn import ring_pytorch_attn_forward as ring_flash_attn_forward
from sp_aurora.ring_pytorch_attn import ring_pytorch_attn_func

from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_ulysses_parallel_world_size,
    get_ring_parallel_world_size,
)

from xfuser.envs import PACKAGES_CHECKER
env_info = PACKAGES_CHECKER.get_packages_info()
HAS_FLASH_ATTN = env_info["has_flash_attn"]


def ring_attn(query, key, value, dropout_p=0.0, is_causal=False):
    query = query.transpose(1,2).contiguous()
    key = key.transpose(1,2).contiguous()
    value = value.transpose(1,2).contiguous()
    if HAS_FLASH_ATTN:
        out, *_ = ring_flash_attn_forward(
            PROCESS_GROUP.RING_PG,
            query,
            key,
            value,
            softmax_scale=query.shape[-1] ** (-0.5),
            dropout_p=dropout_p,
            causal=is_causal,
        )
    else:
        out = ring_pytorch_attn_func(
            query,
            key,
            value,
            dropout_p=dropout_p,
            softmax_scale=query.shape[-1] ** (-0.5),
            causal=is_causal,
            group=PROCESS_GROUP.RING_PG,
        )
    out = out.transpose(1,2).contiguous()
    return out


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    """
    When tracing the code, the result tensor is not an AsyncCollectiveTensor,
    so we cannot call ``wait()``.
    """
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


def _sdpa_all_to_all_single(x):
    x_shape = x.shape
    
    # Intel GPU safety: ensure tensor is contiguous before flatten
    if not x.is_contiguous():
        x = x.contiguous()
    
    # Validate tensor size to prevent overflow
    total_size = x.numel()
    if total_size == 0:
        return x
    
    x = x.flatten()
    
    # Add synchronization for Intel GPU
    if hasattr(torch, 'xpu') and x.is_xpu:
        torch.xpu.synchronize()
    
    x = ft_c.all_to_all_single(x, output_split_sizes=None, input_split_sizes=None, group=PROCESS_GROUP.ULYSSES_PG)
    x = _maybe_wait(x)
    
    # Validate shape before reshape
    if x.numel() != total_size:
        raise RuntimeError(f"All-to-all changed tensor size: expected {total_size}, got {x.numel()}")
    
    x = x.reshape(x_shape)
    return x


def _ft_c_input_all_to_all(x):
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    assert h % world_size == 0, "h must be divisible by world_size, got {} and {}".format(h, world_size)

    # Intel GPU safety: validate tensor before operations
    if hasattr(torch, 'xpu') and x.is_xpu:
        # Ensure tensor is not too large for Intel GPU
        max_elements = 2**31 - 1  # Prevent int32 overflow
        if x.numel() > max_elements:
            raise RuntimeError(f"Tensor too large for Intel GPU: {x.numel()} elements")

    # Make contiguous before permute to ensure memory layout is safe
    if not x.is_contiguous():
        x = x.contiguous()
    
    x = x.permute(1, 0, 2, 3).contiguous()
    x = _sdpa_all_to_all_single(x)
    
    # Calculate expected sequence length
    expected_seq = s * world_size
    
    # Safer reshape with explicit size validation
    try:
        x = x.reshape(world_size, h // world_size, b, expected_seq // world_size, d)
        x = x.permute(2, 1, 0, 3, 4)
        x = x.reshape(b, h // world_size, expected_seq, d)
    except RuntimeError as e:
        # Provide detailed error information
        raise RuntimeError(
            f"Failed to reshape tensor in _ft_c_input_all_to_all. "
            f"Shape: {x.shape}, Target: ({world_size}, {h // world_size}, {b}, {expected_seq // world_size}, {d}). "
            f"Error: {str(e)}"
        )
    
    return x


def _ft_c_output_all_to_all(x):
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    assert s % world_size == 0, "s must be divisible by world_size, got {} and {}".format(s, world_size)

    # Intel GPU safety: validate tensor before operations
    if hasattr(torch, 'xpu') and x.is_xpu:
        # Ensure tensor is not too large for Intel GPU
        max_elements = 2**31 - 1  # Prevent int32 overflow
        if x.numel() > max_elements:
            raise RuntimeError(f"Tensor too large for Intel GPU: {x.numel()} elements")

    # Make contiguous before permute to ensure memory layout is safe
    if not x.is_contiguous():
        x = x.contiguous()
    
    x = x.permute(2, 0, 1, 3).contiguous()
    x = _sdpa_all_to_all_single(x)
    
    # Calculate expected heads
    expected_heads = h * world_size
    
    # Safer reshape with explicit size validation
    try:
        x = x.reshape(world_size, s // world_size, b, expected_heads // world_size, d)
        x = x.permute(2, 0, 3, 1, 4)
        x = x.reshape(b, expected_heads, s // world_size, d)
    except RuntimeError as e:
        # Provide detailed error information
        raise RuntimeError(
            f"Failed to reshape tensor in _ft_c_output_all_to_all. "
            f"Shape: {x.shape}, Target: ({world_size}, {s // world_size}, {b}, {expected_heads // world_size}, {d}). "
            f"Error: {str(e)}"
        )
    
    return x


@torch.compiler.disable
def USP(query, key, value, dropout_p=0.0, is_causal=False):
    # Intel GPU safety: add input validation
    if hasattr(torch, 'xpu') and query.is_xpu:
        # Validate input tensors
        for tensor, name in [(query, "query"), (key, "key"), (value, "value")]:
            if not tensor.is_contiguous():
                # Force contiguous for Intel GPU safety
                tensor = tensor.contiguous()
            if tensor.numel() == 0:
                raise ValueError(f"{name} tensor is empty")
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                raise ValueError(f"{name} tensor contains NaN or Inf values")
    
    try:
        if get_sequence_parallel_world_size() == 1:
            out = F.scaled_dot_product_attention(
                query, key, value, dropout_p=dropout_p, is_causal=is_causal
            )
        elif get_ulysses_parallel_world_size() == 1:
            out = ring_attn(query, key, value, dropout_p=dropout_p, is_causal=is_causal)
        elif get_ulysses_parallel_world_size() > 1:
            query = _ft_c_input_all_to_all(query)
            key = _ft_c_input_all_to_all(key)
            value = _ft_c_input_all_to_all(value)

            if get_ring_parallel_world_size() == 1:
                out = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=dropout_p, is_causal=is_causal
                )
            else:
                out = ring_attn(query, key, value, dropout_p=dropout_p, is_causal=is_causal)

            out = _ft_c_output_all_to_all(out)
    except Exception as e:
        # Log detailed error information for debugging
        import traceback
        error_msg = (
            f"USP attention failed:\n"
            f"Query shape: {query.shape}, Key shape: {key.shape}, Value shape: {value.shape}\n"
            f"Ulysses world size: {get_ulysses_parallel_world_size()}, "
            f"Ring world size: {get_ring_parallel_world_size()}\n"
            f"Error: {str(e)}\n"
            f"Traceback: {traceback.format_exc()}"
        )
        raise RuntimeError(error_msg) from e
        
    return out
