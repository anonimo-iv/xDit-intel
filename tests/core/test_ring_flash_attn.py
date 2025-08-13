import unittest
import torch
import torch.distributed as dist
import os

# Import sp_aurora components directly for testing
try:
    from sp_aurora import (
        ring_flash_attn_func,
        RingFlashAttnFunc,
        pytorch_attn_func,
        update_out_and_lse,
        RingComm,
        set_seq_parallel_pg,
        PROCESS_GROUP,
    )
    from sp_aurora.kernels import AttnType, select_flash_attn_impl
    import intel_extension_for_pytorch as ipex
    HAS_SP_AURORA = True
except ImportError:
    HAS_SP_AURORA = False
    # Fallback imports from xfuser
    from xfuser.core.long_ctx_attention.ring.ring_flash_attn import xdit_ring_flash_attn_func as ring_flash_attn_func
    
from xfuser.core.long_ctx_attention import xFuserLongContextAttention
from xfuser.model_executor.layers.attention_processor import (
    xFuserAttnProcessor2_0,
)
from diffusers.models.attention_processor import (
    Attention,
)
from xfuser.core.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)


def init_dist(backend=None):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Auto-detect backend based on available hardware
    if backend is None:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            backend = 'ccl'
            device = torch.device(f'xpu:{local_rank}')
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            backend = 'nccl'
            device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(local_rank)
        else:
            backend = 'gloo'
            device = torch.device('cpu')

    print(f"Initializing distributed environment with rank {rank}, world size {world_size}, local rank {local_rank}, backend {backend}")
    
    # Set Intel GPU environment variables if using CCL
    if backend == 'ccl':
        os.environ.setdefault('CCL_PROCESS_LAUNCHER', 'pmix')
        os.environ.setdefault('CCL_ATL_TRANSPORT', 'mpi')
        os.environ.setdefault('CCL_ZE_IPC_EXCHANGE', 'drmfd')
    
    init_distributed_environment(rank=rank, world_size=world_size)

    return rank, world_size

class TestRingFlashAttn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_size = 1
        cls.num_heads = 4
        cls.head_dim = 32
        cls.seq_len = 128
        cls.dtype = torch.float16

        
        cls.rank, cls.world_size = init_dist()
        
        # Set device based on available hardware
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            cls.device = torch.device(f'xpu:{cls.rank}')
        elif torch.cuda.is_available():
            cls.device = torch.device(f'cuda:{cls.rank}')
        else:
            cls.device = torch.device('cpu')

    def setUp(self):
        torch.manual_seed(42 + self.rank)
        
    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()

    def _create_test_tensors(self):
        """Helper to create test input tensors"""
        shape = (self.batch_size, self.seq_len, self.num_heads, self.head_dim)

        # Prepare inputs
        q = torch.randn(
            shape, device=self.device, dtype=self.dtype, requires_grad=False
        )
        k = torch.randn(
            shape, device=self.device, dtype=self.dtype, requires_grad=True
        )
        v = torch.randn(
            shape, device=self.device, dtype=self.dtype, requires_grad=True
        )

        dist.broadcast(q, src=0)
        dist.broadcast(k, src=0)
        dist.broadcast(v, src=0)

        local_q = q.chunk(self.world_size, dim=1)[self.rank]
        local_k = k.chunk(self.world_size, dim=1)[self.rank]
        local_v = v.chunk(self.world_size, dim=1)[self.rank]
        return q, k, v, local_q, local_k, local_v
    
    def test_xdit_ring_flash_attn_func(self):
        """Test ring flash attention in distributed mode"""
        q, k, v, local_q, local_k, local_v = self._create_test_tensors()

        # Run reference attention (use pytorch_attn_func for Intel GPU)
        if HAS_SP_AURORA:
            # Use sp_aurora's pytorch attention as reference
            ref_output = pytorch_attn_func(
                q, k, v,
                dropout_p=0.0,
                causal=True,
            )
        else:
            # Fallback to PyTorch's native attention
            scale = 1.0 / (self.head_dim ** 0.5)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if True:  # causal
                mask = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)
                scores.masked_fill_(mask, float('-inf'))
            attn_weights = torch.softmax(scores, dim=-1)
            ref_output = torch.matmul(attn_weights, v)
        
        ref_output = ref_output.chunk(self.world_size, dim=1)[self.rank]

        # Run ring flash attention
        if HAS_SP_AURORA:
            # Test sp_aurora's ring_flash_attn_func directly
            output = ring_flash_attn_func(
                q=local_q,
                k=local_k,
                v=local_v,
                dropout_p=0.0,
                causal=True,
                group=dist.group.WORLD
            )
        else:
            # Fallback to xDiT's implementation
            from xfuser.core.long_ctx_attention.ring.ring_flash_attn import xdit_ring_flash_attn_func
            output = xdit_ring_flash_attn_func(
                q=local_q,
                k=local_k,
                v=local_v,
                dropout_p=0.0,
                causal=True,
                window_size=(-1, -1),
                group=dist.group.WORLD
            )
        
        # Compare results
        torch.testing.assert_close(ref_output, output, rtol=1e-3, atol=1e-3)
        self.assertEqual(ref_output.shape, output.shape)

    def test_xdit_ring_flash_attn_func_joint_strategy_rear(self):
        """Test ring flash attention with joint strategy"""
        q, k, v, local_q, local_k, local_v = self._create_test_tensors()
        joint_q, joint_k, joint_v, local_joint_q, local_joint_k, local_joint_v = self._create_test_tensors()

        # Compute reference output
        if HAS_SP_AURORA:
            ref_output = pytorch_attn_func(
                q, 
                torch.cat([k, joint_k], dim=1), 
                torch.cat([v, joint_v], dim=1),
                dropout_p=0.0,
                causal=False,
            )
        else:
            # Manual attention computation
            scale = 1.0 / (self.head_dim ** 0.5)
            k_cat = torch.cat([k, joint_k], dim=1)
            v_cat = torch.cat([v, joint_v], dim=1)
            scores = torch.matmul(q, k_cat.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(scores, dim=-1)
            ref_output = torch.matmul(attn_weights, v_cat)
            
        ref_output = ref_output.chunk(self.world_size, dim=1)[self.rank]

        # Test rear joint strategy
        if HAS_SP_AURORA:
            output_rear = ring_flash_attn_func(
                q=local_q,
                k=local_k,
                v=local_v,
                dropout_p=0.0,
                causal=False,
                joint_tensor_key=joint_k,
                joint_tensor_value=joint_v,
                joint_strategy="rear"
            )
        else:
            from xfuser.core.long_ctx_attention.ring.ring_flash_attn import xdit_ring_flash_attn_func
            output_rear = xdit_ring_flash_attn_func(
                q=local_q,
                k=local_k,
                v=local_v,
                dropout_p=0.0,
                causal=False,
                window_size=(-1, -1),
                joint_tensor_key=joint_k,
                joint_tensor_value=joint_v,
                joint_strategy="rear"
            )

        torch.testing.assert_close(ref_output, output_rear, rtol=1e-3, atol=1e-3)

    def test_xdit_ring_flash_attn_func_joint_strategy_front(self):
        """Test ring flash attention with joint strategy"""
        q, k, v, local_q, local_k, local_v = self._create_test_tensors()
        joint_q, joint_k, joint_v, local_joint_q, local_joint_k, local_joint_v = self._create_test_tensors()

        # Compute reference output
        if HAS_SP_AURORA:
            ref_output = pytorch_attn_func(
                q, 
                torch.cat([joint_k, k], dim=1), 
                torch.cat([joint_v, v], dim=1),
                dropout_p=0.0,
                causal=False,
            )
        else:
            # Manual attention computation
            scale = 1.0 / (self.head_dim ** 0.5)
            k_cat = torch.cat([joint_k, k], dim=1)
            v_cat = torch.cat([joint_v, v], dim=1)
            scores = torch.matmul(q, k_cat.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(scores, dim=-1)
            ref_output = torch.matmul(attn_weights, v_cat)
            
        ref_output = ref_output.chunk(self.world_size, dim=1)[self.rank]

        # Test front joint strategy
        if HAS_SP_AURORA:
            output_front = ring_flash_attn_func(
                q=local_q,
                k=local_k,
                v=local_v,
                dropout_p=0.0,
                causal=False,
                joint_tensor_key=joint_k,
                joint_tensor_value=joint_v,
                joint_strategy="front"
            )
        else:
            from xfuser.core.long_ctx_attention.ring.ring_flash_attn import xdit_ring_flash_attn_func
            output_front = xdit_ring_flash_attn_func(
                q=local_q,
                k=local_k,
                v=local_v,
                dropout_p=0.0,
                causal=False,
                window_size=(-1, -1),
                joint_tensor_key=joint_k,
                joint_tensor_value=joint_v,
                joint_strategy="front"
            )

        torch.testing.assert_close(ref_output, output_front, rtol=1e-3, atol=1e-3)

    @unittest.skipIf(not HAS_SP_AURORA, "sp_aurora not available")
    def test_sp_aurora_components(self):
        """Test sp_aurora specific components"""
        q, k, v, local_q, local_k, local_v = self._create_test_tensors()
        
        # Test RingComm
        ring_comm = RingComm(dist.group.WORLD)
        self.assertIsNotNone(ring_comm)
        
        # Test update_out_and_lse function
        out = torch.zeros_like(local_q)
        lse = torch.zeros(local_q.shape[:-1], device=self.device, dtype=torch.float32)
        block_out = local_q.clone()
        block_lse = torch.ones(local_q.shape[:-1], device=self.device, dtype=torch.float32)
        
        # This should not raise an error
        update_out_and_lse(out, lse, block_out, block_lse)
        
        # Test kernel selection
        attn_impl = select_flash_attn_impl(
            device_type="xpu" if hasattr(torch, 'xpu') and torch.xpu.is_available() else "cuda"
        )
        self.assertIn(attn_impl, [AttnType.TORCH, AttnType.FA, AttnType.FA3])
        
        # Test PROCESS_GROUP after initialization
        if PROCESS_GROUP.initialized:
            self.assertIsNotNone(PROCESS_GROUP.RING_PG)
            self.assertEqual(PROCESS_GROUP.ring_degree, self.world_size)

    @unittest.skipIf(not HAS_SP_AURORA, "sp_aurora not available")
    def test_sp_aurora_attn_types(self):
        """Test different attention types in sp_aurora"""
        q, k, v, local_q, local_k, local_v = self._create_test_tensors()
        
        # Test with TORCH attention type
        output_torch = ring_flash_attn_func(
            q=local_q,
            k=local_k,
            v=local_v,
            dropout_p=0.0,
            causal=True,
            attn_type=AttnType.TORCH,
            group=dist.group.WORLD
        )
        
        self.assertEqual(output_torch.shape, local_q.shape)

# torchrun --nproc_per_node=2 -m unittest tests/core/test_ring_flash_attn.py
# For Intel GPU: mpiexec -n 2 python -m unittest tests/core/test_ring_flash_attn.py
if __name__ == '__main__':
    unittest.main()