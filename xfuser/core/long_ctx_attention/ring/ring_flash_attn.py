from typing import List
import math
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from xfuser.core.long_ctx_attention import xFuserLongContextAttention
from xfuser.core.cache_manager.cache_manager import get_cache_manager
from sp_aurora.utils import RingComm, update_out_and_lse
from sp_aurora import RingFlashAttnFunc
from sp_aurora.kernels import select_flash_attn_impl, AttnType
# intel_ring_flash_attn_forward might not be available in sp_aurora
try:
    from sp_aurora.intel_ring_flash_attn import intel_ring_flash_attn_forward
except ImportError:
    intel_ring_flash_attn_forward = None

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
except ImportError:
    flash_attn = None
    _flash_attn_forward = None
    from sp_aurora.kernels.attention import pytorch_attn_forward

def xdit_ring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    attn_type=AttnType.FA,
    attn_processor=None,
    attn_layer=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="none",
    q_descale=None,
    k_descale=None,
    v_descale=None
):
    # Debug print at entry
    if hasattr(torch.distributed, 'get_rank'):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print(f"[DEBUG xdit_ring_flash_attn_forward] Rank {rank} entered, q.shape={q.shape}")
        import sys; sys.stdout.flush()
    
    # For Intel GPU without joint tensors, use sp_aurora's optimized implementation if available
    use_intel_ring = (hasattr(torch, 'xpu') and torch.xpu.is_available() and 
        joint_tensor_key is None and joint_tensor_value is None and
        attn_layer is None and intel_ring_flash_attn_forward is not None)
    
    if rank == 0:
        print(f"[DEBUG] Intel ring check: xpu={hasattr(torch, 'xpu') and torch.xpu.is_available()}, "
              f"joint_key={joint_tensor_key is None}, joint_val={joint_tensor_value is None}, "
              f"attn_layer={attn_layer is None}, intel_fn={intel_ring_flash_attn_forward is not None}")
        print(f"[DEBUG] use_intel_ring={use_intel_ring}")
    
    if use_intel_ring:
        # Debug shapes and scaling
        comm = RingComm(process_group)
        if comm.rank == 0:
            print(f"[DEBUG] Using sp_aurora intel_ring_flash_attn_forward")
            print(f"[DEBUG] Input shapes - q: {q.shape}, k: {k.shape}, v: {v.shape}")
            print(f"[DEBUG] softmax_scale: {softmax_scale}")
            print(f"[DEBUG] Input dtypes - q: {q.dtype}, k: {k.dtype}, v: {v.dtype}")
        
        # sp_aurora expects [batch, num_heads, seq_len, head_dim] format
        # but xDiT provides [batch, seq_len, num_heads, head_dim]
        q_transposed = q.transpose(1, 2).contiguous()
        k_transposed = k.transpose(1, 2).contiguous()
        v_transposed = v.transpose(1, 2).contiguous()
        
        # Use sp_aurora's Intel ring flash attention with proper LSE computation
        out, lse = intel_ring_flash_attn_forward(
            process_group,
            q_transposed,
            k_transposed,
            v_transposed,
            softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
        )
        
        # Transpose output back to xDiT format [batch, seq_len, num_heads, head_dim]
        out = out.transpose(1, 2).contiguous()
        
        if comm.rank == 0:
            print(f"[DEBUG] Output shapes - out: {out.shape}, lse: {lse.shape}")
            print(f"[DEBUG] Output sample values: {out[0,0,0,:5]}")
        
        return out, lse
    is_joint = False
    if (joint_tensor_key is not None and 
        joint_tensor_value is not None):
        supported_joint_strategy = ["front", "rear"]
        if joint_strategy not in supported_joint_strategy:
            raise ValueError(
                f"joint_strategy: {joint_strategy} not supprted. supported joint strategy: {supported_joint_strategy}"
            )
        else:
            is_joint = True
    elif (joint_tensor_key is None and 
        joint_tensor_value is None):
        pass
    else:
        raise ValueError(
            f"joint_tensor_key and joint_tensor_value should be None or not None simultaneously."
        )
    
    comm = RingComm(process_group)
    
    # Add synchronization barrier to ensure all ranks enter ring attention together
    if process_group is not None and comm.world_size > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            # Use the process group's barrier to sync
            dist.barrier(group=process_group)
    
    # Early return for single rank case - no ring communication needed
    if comm.world_size == 1:
        # Handle single rank case without ring communication
        if attn_layer is not None:
            k, v = get_cache_manager().update_and_get_kv_cache(
                new_kv=[k, v],
                layer=attn_layer,
                slice_dim=1,
                layer_type="attn",
            )
            k = k.contiguous()
            v = v.contiguous()
        
        # Handle joint tensors for single rank
        if joint_tensor_key is not None and joint_tensor_value is not None:
            if joint_strategy == "rear":
                k = torch.cat([k, joint_tensor_key], dim=1)
                v = torch.cat([v, joint_tensor_value], dim=1)
            elif joint_strategy == "front":
                k = torch.cat([joint_tensor_key, k], dim=1)
                v = torch.cat([joint_tensor_value, v], dim=1)
        
        # Use appropriate attention function for single rank
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            # Intel GPU path
            batch_size, seq_len_q, num_heads, head_dim = q.shape
            q_reshaped = q.transpose(1, 2)
            k_reshaped = k.transpose(1, 2)
            v_reshaped = v.transpose(1, 2)
            
            scores = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) * softmax_scale
            
            if causal:
                causal_mask = torch.triu(torch.ones(seq_len_q, k.shape[1], device=scores.device), diagonal=1)
                scores.masked_fill_(causal_mask.bool(), float('-inf'))
            
            lse = torch.logsumexp(scores, dim=-1, keepdim=False)
            attn_weights = torch.exp(scores - lse.unsqueeze(-1))
            
            if dropout_p > 0:
                attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)
            
            out = torch.matmul(attn_weights, v_reshaped)
            out = out.transpose(1, 2)
        else:
            # Non-Intel GPU path
            fn = select_flash_attn_impl(attn_type)
            out = fn(
                q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale,
                causal=causal, window_size=window_size, softcap=0.0,
                alibi_slopes=alibi_slopes, return_attn_probs=False
            )
            # Create dummy LSE for compatibility
            batch_size, seq_len, num_heads, head_dim = q.shape
            lse = torch.zeros(batch_size, num_heads, seq_len, device=q.device, dtype=torch.float32)
        
        lse = lse.squeeze(dim=-1).transpose(1, 2) if lse.dim() > 3 else lse.transpose(1, 2)
        return out, lse
    
    # Debug for multi-rank case
    if comm.rank == 0:
        print(f"[DEBUG] Ring attention context:")
        print(f"  - process_group size: {comm.world_size}")
        print(f"  - is_joint: {is_joint}")
        print(f"  - has attn_layer (KV cache): {attn_layer is not None}")
        print(f"  - has Intel GPU: {hasattr(torch, 'xpu') and torch.xpu.is_available()}")

    out = None
    lse = None

    next_k, next_v = None, None

    if attn_layer is not None:
        k, v = get_cache_manager().update_and_get_kv_cache(
            new_kv=[k, v],
            layer=attn_layer,
            slice_dim=1,
            layer_type="attn",
        )
        k = k.contiguous()
        v = v.contiguous()

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            try:
                next_k: torch.Tensor = comm.send_recv(k)
                next_v: torch.Tensor = comm.send_recv(v)
                comm.commit()
            except RuntimeError as e:
                if comm.rank == 0:
                    print(f"[ERROR] Ring communication failed at step {step}: {e}")
                    print(f"[ERROR] k shape: {k.shape}, v shape: {v.shape}")
                    print(f"[ERROR] Ring group size: {comm.world_size}, rank: {comm.rank}")
                raise

        if is_joint and joint_strategy == "rear":
            if step + 1 == comm.world_size:
                key = torch.cat([k, joint_tensor_key], dim=1)
                value = torch.cat([v, joint_tensor_value], dim=1)
            else:
                key, value = k, v
        elif is_joint and joint_strategy == "front":
            if step == 0:
                key = torch.cat([joint_tensor_key, k], dim=1)
                value = torch.cat([joint_tensor_value, v], dim=1)
            else:
                key, value = k, v
        else:
            key, value = k, v

        if not causal or step <= comm.rank:
            # For Intel GPU, compute attention with proper LSE
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                # Handle dimension mismatch for joint tensors
                query_to_use = q
                if q.shape[1] != key.shape[1]:
                    # This happens with joint tensors - truncate query to match key
                    query_to_use = q[:, :key.shape[1], :, :]
                
                # Compute attention scores and LSE manually for correct accumulation
                # Input format: (batch, seq_len, num_heads, head_dim)
                batch_size, seq_len_q, num_heads, head_dim = query_to_use.shape
                seq_len_k = key.shape[1]
                
                # Reshape for matmul: (batch, num_heads, seq_len, head_dim)
                q_reshaped = query_to_use.transpose(1, 2)
                k_reshaped = key.transpose(1, 2)
                v_reshaped = value.transpose(1, 2)
                
                # Compute attention scores
                scores = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) * softmax_scale
                
                # Apply causal mask if needed
                if causal and step == 0:
                    causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=scores.device), diagonal=1)
                    scores.masked_fill_(causal_mask.bool(), float('-inf'))
                
                # Compute LSE (log-sum-exp) for proper accumulation
                block_lse = torch.logsumexp(scores, dim=-1, keepdim=False)  # (batch, num_heads, seq_len_q)
                
                # Compute attention weights
                attn_weights = torch.exp(scores - block_lse.unsqueeze(-1))
                
                # Apply dropout if needed
                if dropout_p > 0:
                    attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)
                
                # Compute output
                block_out = torch.matmul(attn_weights, v_reshaped)  # (batch, num_heads, seq_len_q, head_dim)
                
                # Convert back to (batch, seq_len, num_heads, head_dim)
                block_out = block_out.transpose(1, 2)
                
                # Pad output if we truncated the query
                if block_out.shape[1] != q.shape[1]:
                    padding_len = q.shape[1] - block_out.shape[1]
                    padding = torch.zeros(
                        block_out.shape[0], padding_len, block_out.shape[2], block_out.shape[3],
                        device=block_out.device, dtype=block_out.dtype
                    )
                    block_out = torch.cat([block_out, padding], dim=1)
                    
                    # Also pad LSE
                    lse_padding = torch.full(
                        (batch_size, num_heads, padding_len),
                        float('-inf'),
                        device=block_lse.device,
                        dtype=block_lse.dtype
                    )
                    block_lse = torch.cat([block_lse, lse_padding], dim=2)
                
            else:
                # For non-Intel GPU, use the flash attention implementation
                fn = select_flash_attn_impl(attn_type)
                
                # Handle dimension mismatch for joint tensors
                query_to_use = q
                if q.shape[1] != key.shape[1]:
                    query_to_use = q[:, :key.shape[1], :, :]
                
                # Try to get LSE from flash attention if possible
                if attn_type == AttnType.FA and _flash_attn_forward is not None:
                    # Use flash attention forward which returns LSE
                    # Convert to flash attention format: (batch, num_heads, seq_len, head_dim)
                    q_flash = query_to_use.transpose(1, 2)
                    k_flash = key.transpose(1, 2)
                    v_flash = value.transpose(1, 2)
                    
                    block_out, block_lse = _flash_attn_forward(
                        q_flash, k_flash, v_flash,
                        dropout_p=dropout_p,
                        softmax_scale=softmax_scale,
                        causal=causal and step == 0,
                        window_size=window_size,
                        alibi_slopes=alibi_slopes,
                        return_softmax=True
                    )
                    
                    # Convert back to (batch, seq_len, num_heads, head_dim)
                    block_out = block_out.transpose(1, 2)
                else:
                    # Fallback: Call the attention function without LSE
                    block_out = fn(
                        query_to_use,
                        key,
                        value,
                        dropout_p=dropout_p,
                        softmax_scale=softmax_scale,
                        causal=causal and step == 0,
                        window_size=window_size,
                        softcap=0.0,
                        alibi_slopes=alibi_slopes,
                        return_attn_probs=False
                    )
                    
                    # Compute LSE manually as fallback
                    batch_size, seq_len_q, num_heads, head_dim = query_to_use.shape
                    seq_len_k = key.shape[1]
                    
                    # This is an approximation - ideally we'd recompute attention scores
                    # For now, create a reasonable LSE based on output magnitude
                    block_lse = torch.log(torch.sum(torch.exp(block_out), dim=-1))  # Rough approximation
                    block_lse = block_lse.transpose(1, 2)  # (batch, num_heads, seq_len)
                
                # Pad output if we truncated the query
                if block_out.shape[1] != q.shape[1]:
                    padding_len = q.shape[1] - block_out.shape[1]
                    padding = torch.zeros(
                        block_out.shape[0], padding_len, block_out.shape[2], block_out.shape[3],
                        device=block_out.device, dtype=block_out.dtype
                    )
                    block_out = torch.cat([block_out, padding], dim=1)
                    
                    # Also pad LSE
                    lse_padding = torch.full(
                        (block_lse.shape[0], block_lse.shape[1], padding_len),
                        float('-inf'),
                        device=block_lse.device,
                        dtype=block_lse.dtype
                    )
                    block_lse = torch.cat([block_lse, lse_padding], dim=2)
            
            # Always use update_out_and_lse to ensure consistent tensor shapes
            # This function handles the case when out is None (first iteration)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            try:
                comm.wait()
                k = next_k
                v = next_v
            except RuntimeError as e:
                if comm.rank == 0:
                    print(f"[ERROR] Ring communication wait failed at step {step}: {e}")
                raise

    out = out.to(q.dtype)
    
    # No transpose needed since pytorch_attn_forward returns in correct format
    # (batch, seq_len, num_heads, head_dim)
    
    # SPARSE_SAGE is not available in sp_aurora, always squeeze LSE
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    
    
    return out, lse


class xFuserRingFlashAttnFunc(RingFlashAttnFunc):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
        attn_type,
        attn_processor,
        attn_layer,
        joint_tensor_key,
        joint_tensor_value,
        joint_strategy,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        if attn_layer is None:
            k = k.contiguous()
            v = v.contiguous()
        out, softmax_lse = xdit_ring_flash_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
            attn_type=attn_type,
            attn_processor=attn_processor,
            attn_layer=attn_layer,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value,
            joint_strategy=joint_strategy,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.softcap = 0.0
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        ctx.attn_type = attn_type
        ctx.attn_processor = attn_processor
        return out if not return_softmax else (out, softmax_lse, None)


def xdit_ring_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type=AttnType.FA,
    attn_processor=None,
    attn_layer=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="none",
    q_descale=None,
    k_descale=None,
    v_descale=None,
):
    if attn_type == AttnType.FA3:
        return xFuserRingFlashAttnFunc.apply(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            return_attn_probs,
            group,
            attn_type,
            attn_processor,
            attn_layer,
            joint_tensor_key,
            joint_tensor_value,
            joint_strategy,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale
        )
    else:
        return xFuserRingFlashAttnFunc.apply(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            return_attn_probs,
            group,
            attn_type,
            attn_processor,
            attn_layer,
            joint_tensor_key,
            joint_tensor_value,
            joint_strategy,
        )

def xdit_sana_ring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_layer=None,
):

    comm = RingComm(process_group)

    out = None

    next_k, next_v = None, None

    if attn_layer is not None:
        k, v = get_cache_manager().update_and_get_kv_cache(
            new_kv=[k, v],
            layer=attn_layer,
            slice_dim=1,
            layer_type="attn",
        )
        k = k.contiguous()
        v = v.contiguous()
        
    q = F.relu(q).permute(0, 2, 3, 1).contiguous()
    k = F.relu(k).transpose(1, 2).contiguous()
    v = v.permute(0, 2, 3, 1).contiguous()
    v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1.0)

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
            comm.commit()

        key, value = k, v

        # b x n_heads x len_seq x d
        q, key, value = q.float(), key.float(), value.float()
        block_out = value @ key @ q
        out = block_out.float() if out is None else out + block_out.float()

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    out = out.to(q.dtype)
    out = out[:, :, :-1] / (out[:, :, -1:] + torch.finfo(out.dtype).eps)
    out = out.transpose(-2, -1)
    return out

class xFuserSanaRingFlashAttnFunc(RingFlashAttnFunc):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v, 
        attn_layer, 
        group, 
    ):

        if attn_layer is None:
            k = k.contiguous()
            v = v.contiguous()
        out = xdit_sana_ring_flash_attn_forward(
            group,
            q,
            k,
            v,
            attn_layer=attn_layer,
        )
        
        ctx.group = group
        return out

def xdit_sana_ring_flash_attn_func(
        q:torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        group=None,
        attn_layer=None,
    ) -> torch.Tensor:

    return xFuserSanaRingFlashAttnFunc.apply(
        q,
        k,
        v, 
        attn_layer, 
        group, 
    )
