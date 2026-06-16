import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from dataclasses import dataclass
from torch import Tensor
from typing import List, Optional
import math
from torch.distributed._tensor import DTensor
import random

from torchao.optim.adam import _AdamBase

TORCH_TO_TRITON_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.float8_e5m2: tl.float8e5
}

@triton.jit
def stochastic_rounding(
    x_f32,
    rand_bits,
    DTYPE: tl.constexpr
):
    if (DTYPE == tl.bfloat16):
        mask = 0x0000FFFF
    elif (DTYPE == tl.float8e4nv):
        mask = 0x000FFFFF
    elif (DTYPE == tl.float8e5):
        mask = 0x000EFFFF
    else:
        mask = 0x00000000
    x_u32 = x_f32.to(tl.uint32, bitcast=True)
    rand_bits = rand_bits & mask
    x_u32 = (x_u32 + rand_bits) & ~mask
    return x_u32.to(tl.float32, bitcast=True)

@triton.jit
def fused_adamw(
    block_to_p_addr,
    block_to_g_addr,
    block_to_m_addr,
    block_to_v_addr, 
    block_to_chunk_idx,
    block_to_numel,
    lr,
    beta1,
    beta2,
    lr_div_bias_corr1, # 
    bias_corr2_rsqrt,
    eps,
    weight_decay,
    seed,
    TILE_SIZE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    TILES_PER_BLOCK: tl.constexpr,
    PARAMS_T: tl.constexpr,
    MOMENTS_T: tl.constexpr,
    STOCHASTIC_ROUNDING: tl.constexpr
):
    block_id = tl.program_id(0)  
    chunk_tile_id = tl.program_id(1) 
    
    this_block_chunk_idx = tl.load(block_to_chunk_idx + block_id, cache_modifier=".ca")
    this_block_numel = tl.load(block_to_numel + block_id, cache_modifier=".ca")

    this_block_p_addr = tl.load(block_to_p_addr + block_id,cache_modifier=".ca").to(tl.pointer_type(PARAMS_T))
    this_block_g_addr = tl.load(block_to_g_addr + block_id,cache_modifier=".ca").to(tl.pointer_type(PARAMS_T))
    this_block_m_addr = tl.load(block_to_m_addr + block_id,cache_modifier=".ca").to(tl.pointer_type(MOMENTS_T))
    this_block_v_addr = tl.load(block_to_v_addr + block_id,cache_modifier=".ca").to(tl.pointer_type(MOMENTS_T))

    tile_offset = chunk_tile_id * (CHUNK_SIZE // TILES_PER_BLOCK)

    for i in tl.range(0, (CHUNK_SIZE // TILES_PER_BLOCK) // TILE_SIZE):

        this_iter_off = (this_block_chunk_idx * CHUNK_SIZE + tile_offset + i * TILE_SIZE + tl.arange(0, TILE_SIZE)).to(tl.int64)
        this_iter_mask = this_iter_off < this_block_numel
        #
        # tl.randint(...) generates random numbers using the philox RNG, which is a usually 10-step INT-ALU heavy operation
        # the operation should be interleaved with the loads and floating point computation, since the INT-ALU is able to run
        # parallel to the fp cuda cores
        #
        if (STOCHASTIC_ROUNDING):
            rand_bits = tl.randint(seed, tile_offset + block_id * CHUNK_SIZE + i * TILE_SIZE + tl.arange(0, TILE_SIZE))


        p = tl.load(this_iter_off + this_block_p_addr, mask = this_iter_mask, other=0.0, cache_modifier=".cg")
        g = tl.load(this_block_g_addr + this_iter_off, mask = this_iter_mask, other=0.0, cache_modifier=".cg")
        m = tl.load(this_block_m_addr + this_iter_off, mask = this_iter_mask, other=0.0, cache_modifier=".cg")
        v = tl.load(this_block_v_addr + this_iter_off, mask = this_iter_mask, other=0.0, cache_modifier=".cg")

        p_math = p.to(tl.float32)
        g_math = g.to(tl.float32)
        m_math = m.to(tl.float32)
        v_math = v.to(tl.float32)

        m_math = tl.fma((1.0 - beta1), g_math - m_math, m_math)
        v_math = tl.fma(beta2, v_math, tl.fma(-beta2, g_math * g_math, g_math * g_math))
        denom = tl.fma(tl.sqrt_rn(v_math), bias_corr2_rsqrt, eps)
        m_hat =  m_math * lr_div_bias_corr1
        update = (m_hat / denom)
        p_math -= lr * weight_decay * p_math
        p_math -= update
        
        if (STOCHASTIC_ROUNDING):
            tl.store(this_block_p_addr + this_iter_off, stochastic_rounding(p_math, rand_bits, p.dtype).to(p.dtype), mask = this_iter_mask)
        else:
            tl.store(this_block_p_addr + this_iter_off, p_math.to(p.dtype), mask = this_iter_mask)
        
        tl.store(this_block_m_addr + this_iter_off, m_math.to(m.dtype), mask = this_iter_mask)
        tl.store(this_block_v_addr + this_iter_off, v_math.to(v.dtype), mask = this_iter_mask)

@dataclass
class LaunchMetadata: 
    num_chunks: int
    block_to_p_addr: Tensor
    block_to_g_addr: Tensor
    block_to_m_addr: Tensor
    block_to_v_addr: Tensor
    block_to_step: Tensor     
    block_to_chunk_idx: Tensor
    block_to_numel: Tensor

    params_t: tl.dtype
    moments_t: tl.dtype
    

def prepare_blocks(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    steps: List,
    chunk_size: int
):
    n_tensors = len(params)
    assert n_tensors == len(grads) and n_tensors == len(exp_avg) and n_tensors == len(exp_avg_sq) and n_tensors == len(steps), "lists must be of the same length!"
    if (n_tensors == 0):
        return None
    
    device = params[0].device
    params_t = params[0].dtype
    moments_t = exp_avg[0].dtype
    
    for i in range(0, n_tensors):
        this_numel_p = params[i].numel()
        assert grads[i].numel() == this_numel_p and \
            exp_avg[i].numel() == this_numel_p and \
            exp_avg_sq[i].numel() == this_numel_p, \
            f"tensors at index {i} have different numel"
        assert grads[i].dtype == params_t and params[i].dtype == params_t, \
            f"param tensors at index {i} have different dtype"
        assert exp_avg[i].dtype == moments_t and exp_avg_sq[i].dtype == moments_t, \
            f"moment tensors at index {i} have different dtype"
        assert params[i].device == device and grads[i].device == device and \
            exp_avg[i].device == device and exp_avg_sq[i].device == device

    block_to_p_addr = []
    block_to_g_addr = []
    block_to_m_addr = []
    block_to_v_addr = []
    block_to_step = []
    block_to_numel = []
    block_to_chunk_idx = []

    for i in range(0, n_tensors):
        this_p_addr = params[i].data_ptr()
        this_g_addr = grads[i].data_ptr()
        this_m_addr = exp_avg[i].data_ptr()
        this_v_addr = exp_avg_sq[i].data_ptr()
        this_tensor_size = params[i].numel()
        this_tensor_step = steps[i]

        for j in range(0, this_tensor_size, chunk_size):
            # new block in dim 0
            block_to_p_addr.append(this_p_addr)
            block_to_g_addr.append(this_g_addr)
            block_to_m_addr.append(this_m_addr)
            block_to_v_addr.append(this_v_addr)
            block_to_step.append(this_tensor_step)
            block_to_numel.append(this_tensor_size)
            block_to_chunk_idx.append(j // chunk_size)

    meta = LaunchMetadata(
        num_chunks = len(block_to_p_addr),
        block_to_p_addr = torch.tensor(block_to_p_addr, device = device),
        block_to_g_addr = torch.tensor(block_to_g_addr, device = device),
        block_to_m_addr = torch.tensor(block_to_m_addr, device = device),
        block_to_v_addr = torch.tensor(block_to_v_addr, device = device),
        block_to_step = torch.tensor(block_to_step, device = device),
        block_to_numel = torch.tensor(block_to_numel, device = device),
        block_to_chunk_idx = torch.tensor(block_to_chunk_idx, device = device),
        params_t = TORCH_TO_TRITON_DTYPE[params_t],
        moments_t = TORCH_TO_TRITON_DTYPE[moments_t]
    )

    return meta

class _FusedAdamW(_AdamBase):

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        *,
        bf16_stochastic_round=True,
    ) -> None:
        """AdamW optimizer that supports quantized training (parameter is quantized). This optimizer should
        only be used with torchao's quantized training."""
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad=False,
            block_size=float("inf"),
            bf16_stochastic_round=bf16_stochastic_round,
            is_adamw=True,
        )

        self.defaults["fused"] = True # this setting allows the zero_grad method to use torch specialized for_each kernels

        self.TILE_SIZE = 512 
        self.TILES_PER_BLOCK = 1
        self.ILP = 2
        self.CHUNK_SIZE = self.TILE_SIZE * self.TILES_PER_BLOCK * self.ILP

    def add_param_group(self, param_group: dict) -> None:
        scalar_lr = self.defaults["lr"]
        if ("lr" in param_group.keys()):
            scalar_lr = param_group["lr"].item() if isinstance(param_group["lr"], torch.Tensor) else param_group["lr"]
        super().add_param_group(param_group)

        # convert LR to a tensor
        group = self.param_groups[-1]
        group["scalar_lr"] = scalar_lr
        group["step"] = 0.0
        group["meta"] = None
        
    def zero_grad(self) -> None:
        # avoid setting set_to_none = True, since this will make torch allocator deallocate the gradient tensors
        # which in turn requires the re-allocation of the metadata tensors, since the pointers of the grad tensors will change
        super().zero_grad(set_to_none=False) 

    def step(self, closure = None):
        if (closure):
            raise NotImplementedError("Not implemented yet")
        for group in self.param_groups:            
            group["step"] += 1.0
            if (group["meta"] is None):

                params = []
                grads = []
                exp_avgs = []
                exp_avgs_sq = []
                steps = []
                for p in group["params"]:
                    if (p.grad is None):
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError("Sparse gradient is not supported")
                    state = self.state[p]
                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0.0
                        state["exp_avg"] = self._new_buffer(p, True)
                        state["exp_avg_sq"] = self._new_buffer(p, False)

                    state["step"] += 1
                    if (isinstance(p, DTensor)):
                        params.append(p.to_local())
                        grads.append(grad.to_local())
                        exp_avgs.append(state["exp_avg"].to_local())
                        exp_avgs_sq.append(state["exp_avg_sq"].to_local())
                    else:
                        params.append(p)
                        grads.append(grad)
                        exp_avgs.append(state["exp_avg"])
                        exp_avgs_sq.append(state["exp_avg_sq"])
                    steps.append(state["step"])

                meta = prepare_blocks(
                    params,
                    grads,
                    exp_avgs,
                    exp_avgs_sq,
                    steps,
                    self.CHUNK_SIZE
                )
                group["meta"] = meta
            
            # launch the kernel
            meta = group["meta"]
            grid = (meta.num_chunks, self.TILES_PER_BLOCK)
            fused_adamw[grid](
                block_to_p_addr = meta.block_to_p_addr,
                block_to_g_addr = meta.block_to_g_addr, 
                block_to_m_addr = meta.block_to_m_addr, 
                block_to_v_addr = meta.block_to_v_addr,
                block_to_chunk_idx = meta.block_to_chunk_idx,
                block_to_numel = meta.block_to_numel,
                lr = group["scalar_lr"],
                beta1 = group["betas"][0],
                beta2 = group["betas"][1],
                lr_div_bias_corr1= group["scalar_lr"] / (1.0 - math.pow(group["betas"][0], group["step"])),
                bias_corr2_rsqrt=1.0 / math.sqrt(1.0 - group["betas"][1] ** group["step"]),
                eps = group["eps"],
                weight_decay = group["weight_decay"],
                seed = random.randint(0, 2**32 - 1),
                # tunable
                TILE_SIZE = self.TILE_SIZE,
                CHUNK_SIZE = self.CHUNK_SIZE,
                TILES_PER_BLOCK = self.TILES_PER_BLOCK,

                STOCHASTIC_ROUNDING = self.bf16_stochastic_round and meta.params_t == tl.bfloat16,
                PARAMS_T = meta.params_t,
                MOMENTS_T = meta.moments_t
            )