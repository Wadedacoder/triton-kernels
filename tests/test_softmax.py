"""Correctness tests for kernels.softmax against a float64 reference.

Run: pytest -q
"""

import pytest
import torch

from kernels import naive_softmax, softmax

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

# Output tolerances per dtype, checked against torch.softmax computed in float64.
# fp32: the kernel computes in fp32, so errors are a few fp32 ulps (~1e-7 relative).
# fp16: the result is rounded to fp16 on store (relative step 2**-11 ~ 4.9e-4), so allow
#       1e-3 relative; atol covers outputs in fp16's subnormal range.
TOL = {
    torch.float32: dict(rtol=1e-5, atol=1e-6),
    torch.float16: dict(rtol=1e-3, atol=1e-5),
}

SHAPES = [
    (1, 1),        # single element: output must be exactly 1
    (1, 7),        # one row, tiny, non-power-of-2
    (1, 1000),     # one row
    (3, 781),      # non-power-of-2 columns (mask covers 1024 - 781 lanes)
    (128, 1000),
    (4096, 1024),  # many rows, exact power of 2 (no masked lanes)
    (17, 1823),
    (5, 65536),    # wide rows: BLOCK_SIZE = 65536 held by one program
    (2, 131072),   # very wide rows
]


def reference(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x.double(), dim=-1)


def check(out: torch.Tensor, x: torch.Tensor):
    assert out.dtype == x.dtype and out.shape == x.shape
    ref = reference(x)
    assert torch.isfinite(out).all(), "output has inf/nan"
    assert torch.allclose(out.double(), ref, **TOL[x.dtype]), (
        f"max abs err {(out.double() - ref).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_matches_reference(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    check(softmax(x), x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("scale", [1e2, 1e4], ids=["x1e2", "x1e4"])
def test_large_values_are_stable(dtype, scale):
    # exp(88.7) already overflows fp32 and exp(11.1) overflows fp16, so without subtracting
    # the row max these inputs would produce inf / nan.
    torch.manual_seed(0)
    x = torch.randn(64, 1000, device="cuda") * scale
    if dtype == torch.float16:
        x = x.clamp(-60000, 60000)  # stay inside fp16's range (max 65504)
    x = x.to(dtype)
    out = softmax(x)
    check(out, x)
    row_sums = out.float().sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), rtol=0, atol=1e-2 if dtype == torch.float16 else 1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_constant_rows(dtype):
    # All-equal rows (including large ones) must give exactly uniform probabilities.
    x = torch.full((4, 300), 1e4 if dtype == torch.float32 else 6e4, device="cuda", dtype=dtype)
    check(softmax(x), x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_masked_entries(dtype):
    # Attention-style masking: -inf entries must get probability 0.
    torch.manual_seed(0)
    x = torch.randn(32, 513, device="cuda", dtype=dtype)
    x[:, 200:] = float("-inf")
    out = softmax(x)
    check(out, x)
    assert (out[:, 200:] == 0).all()


def test_strided_rows():
    # A column slice has row stride 1000 but only 500 columns per row.
    torch.manual_seed(0)
    base = torch.randn(64, 1000, device="cuda")
    x = base[:, :500]
    assert not x.is_contiguous()
    check(softmax(x), x)


def test_non_contiguous_last_dim():
    # A transposed tensor has stride != 1 along the softmax dimension.
    torch.manual_seed(0)
    x = torch.randn(300, 64, device="cuda").t()
    check(softmax(x), x)


def test_3d_input():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 129, device="cuda")
    check(softmax(x), x)


@pytest.mark.parametrize("shape", [(1, 1), (3, 781), (4096, 1024)], ids=lambda s: f"{s[0]}x{s[1]}")
def test_naive_baseline_matches_reference(shape):
    # The eager baseline used in the benchmarks must be correct too.
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda")
    check(naive_softmax(x), x)
