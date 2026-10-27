"""Benchmark the fused Triton softmax against torch.softmax and the unfused eager version.

For each width N (rows fixed at M), reports the median time from triton.testing.do_bench and
the effective memory bandwidth
    GB/s = (bytes read + bytes written) / time = 2 * M * N * element_size / time,
i.e. the minimum traffic any softmax must move (read the input once, write the output once).
The unfused version actually moves more bytes than this, which is exactly what fusion saves.

Usage: python bench/bench_softmax.py [--rows 4096] [--out results/softmax.csv] [--plot docs/softmax_bandwidth.png]
"""

import argparse
import csv
import os
import sys

import torch
import triton
import triton.testing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kernels import naive_softmax, softmax  # noqa: E402

WIDTHS = [256, 512, 1000, 1024, 2048, 3000, 4096, 6000, 8192, 16384, 32768]
IMPLS = {
    "triton_fused": softmax,
    "torch_softmax": lambda x: torch.softmax(x, dim=-1),
    "torch_eager_unfused": naive_softmax,
}
DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


def peak_bandwidth_gbs() -> float:
    p = torch.cuda.get_device_properties(0)
    # memory_clock_rate is in kHz; GDDR transfers twice per clock; bus width is in bits
    return 2 * p.memory_clock_rate * 1e3 * (p.memory_bus_width / 8) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--out", default="results/softmax.csv")
    ap.add_argument("--plot", default="docs/softmax_bandwidth.png")
    args = ap.parse_args()

    name = torch.cuda.get_device_name(0)
    peak = peak_bandwidth_gbs()
    print(f"device: {name}, theoretical peak DRAM bandwidth {peak:.1f} GB/s")
    print(f"torch {torch.__version__}, triton {triton.__version__}")

    rows = []
    for dname, dtype in DTYPES.items():
        for n in WIDTHS:
            x = torch.randn(args.rows, n, device="cuda", dtype=dtype)
            ref = torch.softmax(x.float(), dim=-1)
            for impl, fn in IMPLS.items():
                try:
                    # sanity check before timing: same answer as torch (in fp32)
                    assert torch.allclose(fn(x).float(), ref, rtol=1e-3, atol=1e-4)
                    ms, ms_lo, ms_hi = triton.testing.do_bench(lambda: fn(x), quantiles=[0.5, 0.2, 0.8])
                except torch.OutOfMemoryError:
                    ms = ms_lo = ms_hi = float("nan")  # the unfused version needs ~4 extra M x N buffers
                torch.cuda.empty_cache()
                gbs = 2 * x.numel() * x.element_size() / (ms * 1e-3) / 1e9
                rows.append(dict(device=name, dtype=dname, rows=args.rows, cols=n, impl=impl,
                                 ms_p50=ms, ms_p20=ms_lo, ms_p80=ms_hi, gbps=gbs, pct_of_peak=100 * gbs / peak))
                print(f"{dname} {args.rows}x{n:<6} {impl:<20} {ms:8.4f} ms  {gbs:7.1f} GB/s  ({100 * gbs / peak:4.1f}% of peak)")
            del x, ref
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")

    if args.plot:
        plot(rows, peak, name, args.plot)


def plot(rows, peak, name, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, muted, grid, axis = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    colors = {"triton_fused": "#2a78d6", "torch_softmax": "#eb6834", "torch_eager_unfused": "#1baf7a"}
    labels = {"triton_fused": "Triton fused", "torch_softmax": "torch.softmax", "torch_eager_unfused": "Eager, unfused"}
    plt.rcParams.update({"figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface,
                         "axes.edgecolor": axis, "axes.labelcolor": ink2, "xtick.color": muted, "ytick.color": muted,
                         "text.color": ink, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150, sharey=True)
    for ax, dname in zip(axes, DTYPES):
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for impl in IMPLS:
            pts = [(r["cols"], r["gbps"]) for r in rows if r["dtype"] == dname and r["impl"] == impl]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=colors[impl], linewidth=2, marker="o", markersize=5,
                    markeredgecolor=surface, markeredgewidth=1.2, label=labels[impl])
        ax.axhline(peak, color=muted, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(xs[0], peak, f"theoretical peak {peak:.0f} GB/s", va="bottom", color=muted, fontsize=8.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks([256, 1024, 4096, 16384])
        ax.set_xticklabels(["256", "1K", "4K", "16K"])
        ax.set_title(f"{dname}", loc="left", fontsize=10, color=ink2)
        ax.set_xlabel(f"Columns per row ({rows[0]['rows']} rows)")
    axes[0].set_ylabel("Effective bandwidth (GB/s)\n= 2 x tensor bytes / time")
    axes[0].set_ylim(0, peak * 1.12)
    axes[0].legend(frameon=False, loc="center", labelcolor=ink2, fontsize=9)
    fig.suptitle(f"Row-wise softmax on {name}", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
