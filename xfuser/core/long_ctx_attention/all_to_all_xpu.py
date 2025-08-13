"""
Custom All-to-All implementation for Intel XPU that avoids driver segfaults
"""
import torch
import torch.distributed as dist


def seq_all_to_all_4d_xpu_safe(group, input_tensor, scatter_idx, gather_idx):
    """
    Safe all-to-all implementation for Intel XPU that avoids complex tensor operations.
    
    This implementation uses explicit memory copies instead of view operations to avoid
    the Intel GPU driver segfault caused by complex memory access patterns.
    
    Args:
        group: Process group for communication
        input_tensor: Input tensor to redistribute
        scatter_idx: Dimension to scatter (split across ranks)
        gather_idx: Dimension to gather (combine from ranks)
    
    Returns:
        Output tensor with dimensions redistributed
    """
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    device = input_tensor.device
    
    # Get tensor shape
    shape = list(input_tensor.shape)
    scatter_dim_size = shape[scatter_idx]
    gather_dim_size = shape[gather_idx]
    
    # Validate dimensions
    assert scatter_dim_size % world_size == 0, \
        f"Scatter dimension size {scatter_dim_size} must be divisible by world size {world_size}"
    
    # Calculate chunk sizes
    scatter_chunk_size = scatter_dim_size // world_size
    
    # Build output shape
    output_shape = shape.copy()
    output_shape[scatter_idx] = scatter_chunk_size
    output_shape[gather_idx] = gather_dim_size * world_size
    
    # Step 1: Split input tensor along scatter dimension
    # Use narrow to avoid complex view operations
    input_chunks = []
    for i in range(world_size):
        start = i * scatter_chunk_size
        chunk = input_tensor.narrow(scatter_idx, start, scatter_chunk_size)
        # Force contiguous to avoid view-related issues
        input_chunks.append(chunk.contiguous())
    
    # Step 2: Prepare send/recv buffers
    # Create explicit buffers to avoid in-place operations
    send_buffers = input_chunks
    recv_buffers = [torch.empty_like(send_buffers[0]) for _ in range(world_size)]
    
    # Step 3: All-to-all communication
    # Synchronize before communication
    if hasattr(torch, 'xpu'):
        torch.xpu.synchronize()
    
    # Use individual send/recv operations to avoid complex all-to-all
    requests = []
    for i in range(world_size):
        if i != rank:
            # Send chunk i to rank i
            req = dist.isend(send_buffers[i], dst=i, group=group)
            requests.append(req)
            # Receive chunk from rank i
            req = dist.irecv(recv_buffers[i], src=i, group=group)
            requests.append(req)
        else:
            # Copy own chunk
            recv_buffers[i] = send_buffers[i].clone()
    
    # Wait for all communications to complete
    for req in requests:
        req.wait()
    
    # Synchronize after communication
    if hasattr(torch, 'xpu'):
        torch.xpu.synchronize()
    
    # Step 4: Concatenate received chunks along gather dimension
    # Reorder chunks based on rank
    ordered_chunks = []
    for i in range(world_size):
        ordered_chunks.append(recv_buffers[i])
    
    # Concatenate along gather dimension
    output_tensor = torch.cat(ordered_chunks, dim=gather_idx)
    
    # Ensure output is contiguous
    output_tensor = output_tensor.contiguous()
    
    return output_tensor


def seq_all_to_all_4d_xpu_backward(group, grad_output, scatter_idx, gather_idx):
    """
    Backward pass for the safe XPU all-to-all operation.
    Simply reverses the scatter and gather indices.
    """
    return seq_all_to_all_4d_xpu_safe(group, grad_output, gather_idx, scatter_idx)


class SeqAllToAll4DXPU(torch.autograd.Function):
    """
    Autograd function for safe XPU all-to-all operation.
    """
    
    @staticmethod
    def forward(ctx, group, input_tensor, scatter_idx, gather_idx):
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        return seq_all_to_all_4d_xpu_safe(group, input_tensor, scatter_idx, gather_idx)
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, seq_all_to_all_4d_xpu_backward(
            ctx.group, grad_output, ctx.scatter_idx, ctx.gather_idx
        ), None, None