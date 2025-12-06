# test_rmsnorm_compare_3way.py
import os
import math
import csv
import torch
import numpy as np
from torch.utils.cpp_extension import load

OUTPUT_CSV = "rmsnorm_perf_summary_3way.csv"

# ===================== Load Extensions =====================
def load_old_ext():
    return load(
        name="rmsnorm_ext_old",
        sources=["rms_v1.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=True,
        build_directory=os.getcwd(),
    )

def load_new_ext():
    return load(
        name="rmsnorm_ext_new",
        sources=["rms_ultimate.cu"],  # <-- replace with your new file
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=True,
        build_directory=os.getcwd(),
    )

def load_sgl_baseline():
    try:
        import sgl_kernel as sglk
        if hasattr(sglk, "sgl_fused_add_rmsnorm"):
            return sglk.sgl_fused_add_rmsnorm
        elif hasattr(sglk, "fused_add_rmsnorm"):
            return sglk.fused_add_rmsnorm
    except Exception as e:
        print(f"[INFO] Cannot import sgl_kernel: {e}")
    return None

# ===================== Benchmark Utils =====================
def benchmark_kernel(kernel_fn, x, r, w, eps, enable_pdl, warmup_runs=20, test_runs=100):
    for _ in range(warmup_runs):
        kernel_fn(x.clone(), r.clone(), w.clone(), eps, enable_pdl)
    torch.cuda.synchronize()

    times = []
    for _ in range(test_runs):
        x_t = x.clone(); r_t = r.clone(); w_t = w.clone()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        kernel_fn(x_t, r_t, w_t, eps, enable_pdl)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    arr = np.array(times, dtype=np.float64)
    return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())

# ===================== Correctness =====================
def run_correctness(x, r, w, eps, baseline, old, new):
    res = {}
    # Baseline (if available)
    if baseline is not None:
        x_b = x.clone(); r_b = r.clone(); w_b = w.clone()
        baseline(x_b, r_b, w_b, eps, False)

    # Old
    x_o = x.clone(); r_o = r.clone(); w_o = w.clone()
    old(x_o, r_o, w_o, eps, False)

    # New
    x_n = x.clone(); r_n = r.clone(); w_n = w.clone()
    new(x_n, r_n, w_n, eps, False)

    def compare(tag, a, b):
        ok = torch.allclose(a, b, rtol=1e-5, atol=1e-6)
        diff = (a - b).abs()
        return {
            "ok": ok,
            "max_abs": diff.max().item(),
            "max_rel": (diff / (b.abs() + 1e-8)).max().item(),
        }

    if baseline is not None:
        res["old_vs_base"] = compare("old_vs_base", x_o, x_b)
        res["new_vs_base"] = compare("new_vs_base", x_n, x_b)
    res["old_vs_new"] = compare("old_vs_new", x_o, x_n)
    return res

# ===================== Performance =====================
def run_perf(B, D, baseline, old, new, eps=1e-5):
    device = "cuda"
    x = torch.randn(B, D, device=device, dtype=torch.float32)
    r = torch.randn(B, D, device=device, dtype=torch.float32)
    w = torch.randn(D, device=device, dtype=torch.float32)

    base_ms = base_std = float("nan")
    if baseline is not None:
        base_ms, base_std, _, _ = benchmark_kernel(baseline, x, r, w, eps, False)
    old_ms, old_std, _, _ = benchmark_kernel(old, x, r, w, eps, False)
    new_ms, new_std, _, _ = benchmark_kernel(new, x, r, w, eps, False)

    speedup_old = (base_ms / old_ms) if not math.isnan(base_ms) and old_ms > 0 else float("nan")
    speedup_new = (base_ms / new_ms) if not math.isnan(base_ms) and new_ms > 0 else float("nan")

    return {
        "shape": (B, D),
        "base_ms": base_ms,
        "old_ms": old_ms,
        "new_ms": new_ms,
        "speedup_old": speedup_old,
        "speedup_new": speedup_new,
    }

# ===================== Main =====================
def main():
    torch.manual_seed(123); torch.cuda.manual_seed_all(123)

    ext_old = load_old_ext()
    ext_new = load_new_ext()
    baseline = load_sgl_baseline()

    old_func = ext_old.sgl_fused_add_rmsnorm
    new_func = ext_new.sgl_fused_add_rmsnorm

    shapes = [
        (128, 4096),
        (256, 4096),
        (1024, 4096),
        (2048, 8192),
        (128, 11008),
        (256, 13824),
        (512, 14336),
        (1024, 8192),
    ]

    # Run a correctness sweep on the first workload before benchmarking.
    corr_B, corr_D = shapes[0]
    x_corr = torch.randn(corr_B, corr_D, device="cuda", dtype=torch.float32)
    r_corr = torch.randn(corr_B, corr_D, device="cuda", dtype=torch.float32)
    w_corr = torch.randn(corr_D, device="cuda", dtype=torch.float32)
    print(f"\n=== Correctness check ({corr_B}x{corr_D}) ===")
    correctness = run_correctness(x_corr, r_corr, w_corr, 1e-5, baseline, old_func, new_func)
    for tag, stats in correctness.items():
        status = "OK" if stats["ok"] else "FAIL"
        print(f"{tag}: {status} | max_abs={stats['max_abs']:.3e} | max_rel={stats['max_rel']:.3e}")

    results = []
    for B, D in shapes:
        print(f"\n=== Shape {B}x{D} ===")
        perf = run_perf(B, D, baseline, old_func, new_func)
        print(f"BASE={perf['base_ms']:.3f} ms | OLD={perf['old_ms']:.3f} ms | NEW={perf['new_ms']:.3f} ms | "
              f"Speedup(OLD)={perf['speedup_old']:.2f}x Speedup(NEW)={perf['speedup_new']:.2f}x")
        results.append(perf)

    # Write CSV
    if results:
        with open(OUTPUT_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["shape", "base_ms", "old_ms", "new_ms", "speedup_old", "speedup_new"])
            for r in results:
                shape_str = f"{r['shape'][0]}x{r['shape'][1]}"
                w.writerow([shape_str, r["base_ms"], r["old_ms"], r["new_ms"], r["speedup_old"], r["speedup_new"]])
            # Average row
            arr = np.array([[r["base_ms"], r["old_ms"], r["new_ms"], r["speedup_old"], r["speedup_new"]] for r in results], dtype=np.float64)
            avg = np.nanmean(arr, axis=0)
            w.writerow(["AVG", *avg])

        print(f"\nSaved CSV results to {OUTPUT_CSV}")

if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()