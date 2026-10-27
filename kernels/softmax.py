"""Row-wise softmax: a fused Triton kernel and an unfused PyTorch eager baseline.

softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))

Subtracting the row max does not change the result (it cancels in the ratio) but keeps
every exponent <= 0, so exp() cannot overflow even for very large inputs.

Based on the structure of the official Triton fused-softmax tutorial
(https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html),
simplified to one program per row.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    out_ptr,
    in_ptr,
    in_row_stride,
    out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # One program (a group of warps) owns one full row. BLOCK_SIZE is n_cols rounded up to a
    # power of two, so the whole row is held in registers between the load and the store.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols  # lanes past the end of the row do no memory access

    # READ: the only DRAM read. Each element of the row is loaded exactly once; consecutive
    # lanes touch consecutive addresses, so the load is fully coalesced. Masked-off lanes get
    # -inf, which is neutral for max() and becomes exp(-inf) = 0 for the sum.
    # fp16 inputs are upcast so the max / exp / sum are computed in fp32.
    x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=-float("inf")).to(tl.float32)

    # Everything below runs on registers: no memory traffic between these steps. In the eager
    # baseline each of these lines is a separate kernel that writes its result to DRAM and the
    # next one reads it back.
    row_max = tl.max(x, axis=0)
    numerator = tl.exp(x - row_max)
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator

    # WRITE: the only DRAM write, again one coalesced pass over the row, cast back to the
    # output dtype.
    tl.store(out_ptr + row * out_row_stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def _num_warps(block_size: int) -> int:
    # More warps for wider rows, so each thread holds a bounded number of elements.
    if block_size >= 8192:
        return 16
    if block_size >= 2048:
        return 8
    return 4


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dimension of a CUDA tensor (fp32 or fp16), one kernel launch."""
    assert x.is_cuda, "input must be a CUDA tensor"
    assert x.dtype in (torch.float32, torch.float16), f"unsupported dtype {x.dtype}"
    shape = x.shape
    n_cols = shape[-1]
    x2d = x.reshape(-1, n_cols)
    if x2d.stride(-1) != 1:  # the kernel needs each row contiguous; rows may have any stride
        x2d = x2d.contiguous()
    n_rows = x2d.shape[0]
    out = torch.empty((n_rows, n_cols), device=x.device, dtype=x.dtype)

    block_size = triton.next_power_of_2(n_cols)
    softmax_kernel[(n_rows,)](
        out,
        x2d,
        x2d.stride(0),
        out.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(block_size),
    )
    return out.reshape(shape)


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """Unfused eager baseline: the same math as separate PyTorch ops.

    Each line launches its own kernel and round-trips an M x N (or M) tensor through DRAM
    (same breakdown as the tutorial):
      max      reads MN,          writes M
      subtract reads MN + M,      writes MN
      exp      reads MN,          writes MN
      sum      reads MN,          writes M
      divide   reads MN + M,      writes MN
    Total: 5MN + 2M elements read and 3MN + 2M written, vs MN read and MN written fused.
    """
    x_max = x.max(dim=-1, keepdim=True)[0]
    z = x - x_max
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=-1, keepdim=True)
    return numerator / denominator
