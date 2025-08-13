import torch
import torch.nn.functional as F
from torch import Tensor

import torch.distributed
try:
    from sp_aurora import LongContextAttention
    from sp_aurora.kernels import AttnType
    from sp_aurora.comm.all_to_all import SeqAllToAll4D
    from sp_aurora.globals import HAS_SPARSE_SAGE_ATTENTION
except (ImportError, IndexError):
    raise ImportError("Please install sp_aurora or run on Intel GPUs")


from xfuser.logger import init_logger

from xfuser.core.distributed import (
    get_ring_parallel_world_size,
    )

logger = init_logger(__name__)


class xFuserLongContextAttention(LongContextAttention):
    ring_impl_type_supported_kv_cache = ["basic"]

    def __init__(
        self,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
        use_pack_qkv: bool = False,
        use_kv_cache: bool = False,
        use_sync: bool = False,
        attn_type: AttnType = AttnType.FA,
        attn_processor: torch.nn.Module = None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    ) -> None:
        """
        Arguments:
            scatter_idx: int = 2, the scatter dimension index for Ulysses All2All
            gather_idx: int = 1, the gather dimension index for Ulysses All2All
            ring_impl_type: str = "basic", the ring implementation type, currently only support "basic"
            use_pack_qkv: bool = False, whether to use pack qkv in the input
            use_kv_cache: bool = False, whether to use kv cache in the attention layer, which is applied in PipeFusion.
            attn_type: AttnType = AttnType.FA, the attention type supported inside long context attention, including "TORCH", "FA", "FA3", "SAGE_FP16", "SAGE_FP8"
            attn_processor: nn.Module = None, the attention processor can be passed in to replace the attention processor if attn_type is do not support it.
        """
        super().__init__(
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            use_pack_qkv=use_pack_qkv,
            use_sync=use_sync,
            attn_type = attn_type,
        )
        self.use_kv_cache = use_kv_cache
        self.q_descale = q_descale
        self.k_descale = k_descale
        self.v_descale = v_descale
        if (
            use_kv_cache
            and ring_impl_type not in self.ring_impl_type_supported_kv_cache
        ):
            raise RuntimeError(
                f"ring_impl_type: {ring_impl_type} do not support SP kv cache."
            )

        if HAS_SPARSE_SAGE_ATTENTION:
            from spas_sage_attn.autotune import SparseAttentionMeansim
            if isinstance(attn_processor, SparseAttentionMeansim) and torch.distributed.get_world_size(self.ring_pg) > 1:
                raise RuntimeError("Sparse Sage attention does not support ring degree > 1.")

        self.attn_processor = attn_processor
        from xfuser.core.long_ctx_attention.ring import xdit_ring_flash_attn_func
        self.ring_attn_fn = xdit_ring_flash_attn_func

    @torch.compiler.disable
    def forward(
        self,
        attn,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        joint_tensor_query=None,
        joint_tensor_key=None,
        joint_tensor_value=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        joint_strategy="none",
    ) -> Tensor:
        # Debug print at the very beginning
        if hasattr(torch, 'xpu') and query.is_xpu and self.ulysses_pg.rank() == 0:
            print(f"[DEBUG hybrid attn] Intel GPU bypass check:")
            print(f"  joint_tensor_key is None: {joint_tensor_key is None}")
            print(f"  joint_tensor_value is None: {joint_tensor_value is None}")
            print(f"  self.ulysses_pg size: {self.ulysses_pg.size()}")
            import sys; sys.stdout.flush()
            print(f"[DEBUG] xFuserLongContextAttention forward called on XPU")
            print(f"[DEBUG] Query device: {query.device}, shape: {query.shape}")
            print(f"[DEBUG] use_pack_qkv: {self.use_pack_qkv}")
            print(f"[DEBUG] Ulysses PG size: {self.ulysses_pg.size()}, Ring PG size: {self.ring_pg.size()}")
        
        # For Intel XPU, bypass Ulysses all-to-all and use ring-only
        if hasattr(torch, 'xpu') and query.is_xpu:
            if self.ulysses_pg.rank() == 0:
                print(f"[DEBUG] Bypassing Ulysses all-to-all for Intel XPU, using ring-only attention")
            
            # Skip all the Ulysses all-to-all operations
            # Directly pass tensors to ring attention without redistribution
            out = self.ring_attn_fn(
                query,
                key, 
                value,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
                group=self.ring_pg,
                attn_type=self.attn_type,
                attn_processor=self.attn_processor,
                attn_layer=attn if self.use_kv_cache else None,
                joint_tensor_key=None,
                joint_tensor_value=None,
                joint_strategy=joint_strategy,
                q_descale=self.q_descale,
                k_descale=self.k_descale,
                v_descale=self.v_descale,
            )
            
            if type(out) == tuple:
                context_layer, _, _ = out
            else:
                context_layer = out
                
            # Return directly without output all-to-all
            return context_layer
        """forward

        Arguments:
            attn (Attention): the attention module
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args,
            joint_tensor_query: Tensor = None, a replicated tensor among processes appended to the front or rear of query, depends the joint_strategy  
            joint_tensor_key: Tensor = None, a replicated tensor among processes appended to the front or rear of key, depends the joint_strategy
            joint_tensor_value: Tensor = None, a replicated tensor among processes appended to the front or rear of value, depends the joint_strategy,
            *args: the args same as flash_attn_interface
            joint_strategy: str = "none", the joint strategy for joint attention, currently only support "front" and "rear"

        Returns:
            * output (Tensor): context output
        """
        is_joint = False
        if (joint_tensor_query is not None and 
            joint_tensor_key is not None and 
            joint_tensor_value is not None):
            supported_joint_strategy = ["front", "rear"]
            if joint_strategy not in supported_joint_strategy:
                raise ValueError(
                    f"joint_strategy: {joint_strategy} not supprted. supported joint strategy: {supported_joint_strategy}"
                )
            elif joint_strategy == "rear":
                query = torch.cat([query, joint_tensor_query], dim=1)
                is_joint = True
            else:
                query = torch.cat([joint_tensor_query, query], dim=1)
                is_joint = True
        elif (joint_tensor_query is None and 
            joint_tensor_key is None and 
            joint_tensor_value is None):
            pass
        else:
            raise ValueError(
                f"joint_tensor_query, joint_tensor_key, and joint_tensor_value should be None or not None simultaneously."
            )

        if is_joint:
            ulysses_world_size = torch.distributed.get_world_size(self.ulysses_pg)
            ulysses_rank = torch.distributed.get_rank(self.ulysses_pg)
            attn_heads_per_ulysses_rank = (
                joint_tensor_key.shape[-2] // ulysses_world_size
            )
            joint_tensor_key = joint_tensor_key[
                ...,
                attn_heads_per_ulysses_rank
                * ulysses_rank : attn_heads_per_ulysses_rank
                * (ulysses_rank + 1),
                :,
            ]
            joint_tensor_value = joint_tensor_value[
                ...,
                attn_heads_per_ulysses_rank
                * ulysses_rank : attn_heads_per_ulysses_rank
                * (ulysses_rank + 1),
                :,
            ]

        # Debug logging for Intel GPU
        if hasattr(torch, 'xpu') and query.is_xpu and torch.distributed.get_rank() == 0:
            print(f"[DEBUG USP] Before SeqAllToAll4D:")
            print(f"  Query shape: {query.shape}, dtype: {query.dtype}, device: {query.device}")
            print(f"  Key shape: {key.shape}, dtype: {key.dtype}, device: {key.device}")
            print(f"  Value shape: {value.shape}, dtype: {value.dtype}, device: {value.device}")
            print(f"  Ulysses PG size: {self.ulysses_pg.size()}, Ring PG size: {self.ring_pg.size()}")
            print(f"  scatter_idx: {self.scatter_idx}, gather_idx: {self.gather_idx}")
        
        # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
        # scatter 2, gather 1
        if self.use_pack_qkv:
            # (3*bs, seq_len/N, head_cnt, head_size)
            qkv = torch.cat([query, key, value]).contiguous()
            # (3*bs, seq_len, head_cnt/N, head_size)
            if hasattr(torch, 'xpu') and qkv.is_xpu:
                # Use safer XPU-specific implementation
                from xfuser.core.long_ctx_attention.all_to_all_xpu import SeqAllToAll4DXPU
                qkv = SeqAllToAll4DXPU.apply(
                    self.ulysses_pg, qkv, self.scatter_idx, self.gather_idx
                )
            else:
                qkv = SeqAllToAll4D.apply(
                    self.ulysses_pg, qkv, self.scatter_idx, self.gather_idx
                )
            qkv = torch.chunk(qkv, 3, dim=0)
            query_layer, key_layer, value_layer = qkv

        else:
            # Check if we're on Intel XPU
            if hasattr(torch, 'xpu') and query.is_xpu:
                # Use safer XPU-specific implementation
                from xfuser.core.long_ctx_attention.all_to_all_xpu import SeqAllToAll4DXPU
                if self.ulysses_pg.rank() == 0:
                    print(f"[DEBUG] Using XPU-safe SeqAllToAll4D for Intel GPU")
                    print(f"[DEBUG] Before SeqAllToAll4D - query shape: {query.shape}")
                
                query_layer = SeqAllToAll4DXPU.apply(
                    self.ulysses_pg, query, self.scatter_idx, self.gather_idx
                )
                key_layer = SeqAllToAll4DXPU.apply(
                    self.ulysses_pg, key, self.scatter_idx, self.gather_idx
                )
                value_layer = SeqAllToAll4DXPU.apply(
                    self.ulysses_pg, value, self.scatter_idx, self.gather_idx
                )
            else:
                # Original path for CUDA devices
                if self.ulysses_pg.rank() == 0:
                    print(f"[DEBUG] Before SeqAllToAll4D - query shape: {query.shape}")
                query_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, query, self.scatter_idx, self.gather_idx
                )
                key_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, key, self.scatter_idx, self.gather_idx
                )
                value_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, value, self.scatter_idx, self.gather_idx
                )
            
            if self.ulysses_pg.rank() == 0:
                print(f"[DEBUG] After SeqAllToAll4D - query_layer shape: {query_layer.shape}")
                print(f"[DEBUG] scatter_idx: {self.scatter_idx}, gather_idx: {self.gather_idx}")


        out = self.ring_attn_fn(
            query_layer,
            key_layer,
            value_layer,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            group=self.ring_pg,
            attn_type=self.attn_type,
            attn_processor=self.attn_processor,
            attn_layer=attn if self.use_kv_cache else None,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value,
            joint_strategy=joint_strategy,
            q_descale=self.q_descale,
            k_descale=self.k_descale,
            v_descale=self.v_descale,
        )

        if type(out) == tuple:
            context_layer, _, _ = out
        else:
            context_layer = out
        
        # Debug logging for Intel GPU
        if hasattr(torch, 'xpu') and context_layer.is_xpu and torch.distributed.get_rank() == 0:
            print(f"[DEBUG USP] After ring attention:")
            print(f"  Context layer shape: {context_layer.shape}, dtype: {context_layer.dtype}")
            print(f"  About to do output SeqAllToAll4D with gather_idx: {self.gather_idx}, scatter_idx: {self.scatter_idx}")
        
        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        # Check if we're on Intel XPU
        if hasattr(torch, 'xpu') and context_layer.is_xpu:
            # Use safer XPU-specific implementation
            from xfuser.core.long_ctx_attention.all_to_all_xpu import SeqAllToAll4DXPU
            output = SeqAllToAll4DXPU.apply(
                self.ulysses_pg, context_layer, self.gather_idx, self.scatter_idx
            )
        else:
            # Original path for CUDA devices
            output = SeqAllToAll4D.apply(
                self.ulysses_pg, context_layer, self.gather_idx, self.scatter_idx
            )

        # out e.g., [s/p::h]
        return output

class xFuserSanaLinearLongContextAttention(xFuserLongContextAttention):
    def __init__(self, 
                 scatter_idx: int = 2, 
                 gather_idx: int = 1, 
                 ring_impl_type: str = "basic", 
                 use_pack_qkv: bool = False, 
                 use_kv_cache: bool = False, 
                 attn_type: AttnType = AttnType.FA,
                 attn_processor: torch.nn.Module = None):
        super().__init__(scatter_idx, gather_idx, ring_impl_type, use_pack_qkv, use_kv_cache, attn_type, attn_processor)
        # TODO need to check the attn_type
        from xfuser.core.long_ctx_attention.ring import xdit_sana_ring_flash_attn_func
        self.ring_attn_fn = xdit_sana_ring_flash_attn_func
        # self.ring_attn_fn = xdit_sana_linear_ring_flash_attn_func
        self.ring_world_size = get_ring_parallel_world_size()
    
    def forward(
        self,
        attn,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        attn_mask: Tensor = None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
    ) -> Tensor:

        """forward

        Arguments:
            attn (Attention): the attention module
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args,
            joint_tensor_query: Tensor = None, a replicated tensor among processes appended to the front or rear of query, depends the joint_strategy  
            joint_tensor_key: Tensor = None, a replicated tensor among processes appended to the front or rear of key, depends the joint_strategy
            joint_tensor_value: Tensor = None, a replicated tensor among processes appended to the front or rear of value, depends the joint_strategy,
            *args: the args same as flash_attn_interface
            joint_strategy: str = "none", the joint strategy for joint attention, currently only support "front" and "rear"

        Returns:
            * output (Tensor): context output
        """
        # (bs, seq_len/N, head_cnt, head_size)

        # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
        # scatter 2, gather 1
        if self.use_pack_qkv:
            # (3*bs, seq_len/N, head_cnt, head_size)
            qkv = torch.cat([query, key, value]).contiguous()
            # (3*bs, seq_len, head_cnt/N, head_size)
            if hasattr(torch, 'xpu') and qkv.is_xpu:
                # Use safer XPU-specific implementation
                from xfuser.core.long_ctx_attention.all_to_all_xpu import SeqAllToAll4DXPU
                qkv = SeqAllToAll4DXPU.apply(
                    self.ulysses_pg, qkv, self.scatter_idx, self.gather_idx
                )
            else:
                qkv = SeqAllToAll4D.apply(
                    self.ulysses_pg, qkv, self.scatter_idx, self.gather_idx
                )
            qkv = torch.chunk(qkv, 3, dim=0)
            query_layer, key_layer, value_layer = qkv

        else:
            if hasattr(torch, 'xpu') and query.is_xpu:
                # Use safer XPU-specific implementation
                from xfuser.core.long_ctx_attention.all_to_all_xpu import SeqAllToAll4DXPU
                query_layer = SeqAllToAll4DXPU.apply(
                    self.ulysses_pg, query, self.scatter_idx, self.gather_idx
                )
                key_layer = SeqAllToAll4DXPU.apply(
                    self.ulysses_pg, key, self.scatter_idx, self.gather_idx
                )
                value_layer = SeqAllToAll4DXPU.apply(
                    self.ulysses_pg, value, self.scatter_idx, self.gather_idx
                )
            else:
                query_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, query, self.scatter_idx, self.gather_idx
                )
                key_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, key, self.scatter_idx, self.gather_idx
                )
                value_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, value, self.scatter_idx, self.gather_idx
                )
        
        out = self.ring_attn_fn(
            query_layer,
            key_layer,
            value_layer,
            group=self.ring_pg,
            attn_layer=attn if self.use_kv_cache else None,
        )
        out = out.transpose(1, 2)
        
        if isinstance(out, tuple):
            context_layer, _, _ = out
        else:
            context_layer = out

        # scatter 1, gather 2
        if hasattr(torch, 'xpu') and context_layer.is_xpu:
            # Use safer XPU-specific implementation
            from xfuser.core.long_ctx_attention.all_to_all_xpu import SeqAllToAll4DXPU
            output: Tensor = SeqAllToAll4DXPU.apply(
                self.ulysses_pg, context_layer, self.gather_idx, self.scatter_idx
            )
        else:
            output: Tensor = SeqAllToAll4D.apply(
                self.ulysses_pg, context_layer, self.gather_idx, self.scatter_idx
            )
        
        output = output.flatten(2, 3)

        return output