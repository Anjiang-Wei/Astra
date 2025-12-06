# bench_merge_state_3way.py
import os
import csv
import math
import torch
import numpy as np
from torch.utils.cpp_extension import load

# ===================== Config =====================
WARMUP = 20
ITERS  = 100
OUTPUT_CSV = "fp16_perf_summary_3way.csv"

# Test shapes: (n, h, d)
TEST_SHAPES = [
    (512, 32, 256),
    (512, 40, 128),
    (768, 32, 256),
    (768, 40, 128),
    (512, 64, 128),
]

# ===================== Load/Build extensions =====================
def load_old_ext():
    return load(
        name="merge_ext_old",
        sources=["merge_v1.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        build_directory=os.getcwd(),
        verbose=True,
    )

def load_new_ext():
    return load(
        name="merge_ext_new",
        sources=["merge_ultimate.cu"],  # <-- rename if your new file differs
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        build_directory=os.getcwd(),
        verbose=True,
    )

def load_sglang_merge():
    try:
        import sgl_kernel as sglk
        if hasattr(sglk, "merge_state"):
            return sglk.merge_state
        raise RuntimeError("sgl_kernel.merge_state not found")
    except Exception as e:
        print(f"[INFO] Cannot import sGL kernel: {e}")
        return None

# ===================== DType helpers =====================
def value_dtype(prec: str):
    return torch.float16 if prec == "fp16" else torch.bfloat16

def cast_values_for_sgl(v: torch.Tensor, prec: str):
    return v.to(value_dtype(prec), copy=True, non_blocking=True).contiguous()

def tolerances_for_values(prec: str):
    return (1e-2, 5e-3) if prec == "fp16" else (2e-2, 1e-2)

def tolerances_for_scores():
    return (1e-6, 1e-6)

# ===================== Timing =====================
def bench_once(callable_once, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        callable_once()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        callable_once()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    arr = np.array(times, dtype=np.float64)
    return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())

# ===================== Correctness =====================
def run_one_correctness(n, h, d, prec, base_fn, old_fn, new_fn, device="cuda"):
    # Build FP32 sources and cast value tensors to fp16/bf16 for kernels
    v_a_fp32 = torch.randn(n, h, d, device=device, dtype=torch.float32)
    v_b_fp32 = torch.randn(n, h, d, device=device, dtype=torch.float32)
    s_a_fp32 = torch.randn(n, h,     device=device, dtype=torch.float32)
    s_b_fp32 = torch.randn(n, h,     device=device, dtype=torch.float32)

    v_a_val = cast_values_for_sgl(v_a_fp32, prec)
    v_b_val = cast_values_for_sgl(v_b_fp32, prec)

    # Baseline (SGLang) returns (v_out, s_out)
    if base_fn is not None:
        v_out_base, s_out_base = base_fn(
            v_a_val.clone(), s_a_fp32.clone(), v_b_val.clone(), s_b_fp32.clone(), None, None
        )
        torch.cuda.synchronize()
        # Sanity check shapes
        assert v_out_base.shape == (n, h, d), f"base v_out shape {v_out_base.shape} != {(n,h,d)}"
        assert s_out_base.shape == (n, h),    f"base s_out shape {s_out_base.shape} != {(n,h)}"
    else:
        v_out_base = s_out_base = None

    # Old impl writes to provided outputs
    v_out_old = torch.empty_like(v_a_val)
    s_out_old = torch.empty_like(s_a_fp32)
    old_fn(v_a_val.clone(), s_a_fp32.clone(), v_b_val.clone(), s_b_fp32.clone(), v_out_old, s_out_old)
    torch.cuda.synchronize()
    assert v_out_old.shape == (n, h, d), f"old v_out shape {v_out_old.shape} != {(n,h,d)}"
    assert s_out_old.shape == (n, h),    f"old s_out shape {s_out_old.shape} != {(n,h)}"

    # New impl writes to provided outputs
    v_out_new = torch.empty_like(v_a_val)
    s_out_new = torch.empty_like(s_a_fp32)
    new_fn(v_a_val.clone(), s_a_fp32.clone(), v_b_val.clone(), s_b_fp32.clone(), v_out_new, s_out_new)
    torch.cuda.synchronize()
    assert v_out_new.shape == (n, h, d), f"new v_out shape {v_out_new.shape} != {(n,h,d)}"
    assert s_out_new.shape == (n, h),    f"new s_out shape {s_out_new.shape} != {(n,h)}"

    def compare_pair(v_ref, s_ref, v_tst, s_tst):
        v_rtol, v_atol = tolerances_for_values(prec)
        s_rtol, s_atol = tolerances_for_scores()

        a_v = v_ref.to(torch.float32); b_v = v_tst.to(torch.float32)
        a_s = s_ref.to(torch.float32); b_s = s_tst.to(torch.float32)

        ok_v = torch.allclose(b_v, a_v, rtol=v_rtol, atol=v_atol)
        ok_s = torch.allclose(b_s, a_s, rtol=s_rtol, atol=s_atol)

        dv = (b_v - a_v).abs()
        den_v = torch.maximum(a_v.abs(), b_v.abs()).clamp_min(1e-8)
        ds = (b_s - a_s).abs()
        den_s = torch.maximum(a_s.abs(), b_s.abs()).clamp_min(1e-12)

        return {
            "ok": bool(ok_v and ok_s),
            "ok_v": bool(ok_v),
            "ok_s": bool(ok_s),
            "max_abs_v": dv.max().item(),
            "max_rel_v": (dv / den_v).max().item(),
            "max_abs_s": ds.max().item(),
            "max_rel_s": (ds / den_s).max().item(),
            "v_tol": (v_rtol, v_atol),
            "s_tol": (s_rtol, s_atol),
        }

    results = {}
    if v_out_base is not None:
        results["old_vs_base"] = compare_pair(v_out_base, s_out_base, v_out_old, s_out_old)
        results["new_vs_base"] = compare_pair(v_out_base, s_out_base, v_out_new, s_out_new)
    else:
        results["old_vs_new"]  = compare_pair(v_out_old,  s_out_old,  v_out_new, s_out_new)
    return results

# ===================== Performance =====================
def run_one_perf(n, h, d, prec, base_fn, old_fn, new_fn, device="cuda"):
    v_a_fp32 = torch.randn(n, h, d, device=device, dtype=torch.float32)
    v_b_fp32 = torch.randn(n, h, d, device=device, dtype=torch.float32)
    s_a_fp32 = torch.randn(n, h,     device=device, dtype=torch.float32)
    s_b_fp32 = torch.randn(n, h,     device=device, dtype=torch.float32)

    v_a_val = cast_values_for_sgl(v_a_fp32, prec)
    v_b_val = cast_values_for_sgl(v_b_fp32, prec)

    base_mean = base_std = float("nan")
    if base_fn is not None:
        def call_base():
            base_fn(v_a_val, s_a_fp32, v_b_val, s_b_fp32, None, None)
        base_mean, base_std, _, _ = bench_once(call_base)

    v_out = torch.empty_like(v_a_val); s_out = torch.empty_like(s_a_fp32)
    def call_old():
        old_fn(v_a_val, s_a_fp32, v_b_val, s_b_fp32, v_out, s_out)
    old_mean, old_std, _, _ = bench_once(call_old)

    v_out2 = torch.empty_like(v_a_val); s_out2 = torch.empty_like(s_a_fp32)
    def call_new():
        new_fn(v_a_val, s_a_fp32, v_b_val, s_b_fp32, v_out2, s_out2)
    new_mean, new_std, _, _ = bench_once(call_new)

    speedup_old = (base_mean / old_mean) if (not math.isnan(base_mean) and old_mean > 0) else float("nan")
    speedup_new = (base_mean / new_mean) if (not math.isnan(base_mean) and new_mean > 0) else float("nan")

    return {
        "base_ms": base_mean,
        "old_ms":  old_mean,
        "new_ms":  new_mean,
        "speedup_old": speedup_old,
        "speedup_new": speedup_new,
    }

# ===================== Main =====================
def main():
    assert torch.cuda.is_available()
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    ext_old = load_old_ext()
    ext_new = load_new_ext()
    base_fn = load_sglang_merge()

    old_fn = ext_old.merge_state
    new_fn = ext_new.merge_state

    # CSV rows for FP16 only: [n, h, d, base_ms, old_ms, new_ms, speedup_old, speedup_new]
    fp16_rows = []

    for prec in ["fp16", "bf16"]:
        if prec == "bf16" and not torch.cuda.is_bf16_supported():
            continue

        print("=" * 80)
        print(f"MERGE STATE (3-way) | values={prec.upper()} scores=FP32  (baseline=SGL, old=merge.cu, new=merge_new.cu)")
        print("=" * 80)

        # Correctness
        print("\nCORRECTNESS")
        for (n, h, d) in TEST_SHAPES:
            res = run_one_correctness(n, h, d, prec, base_fn, old_fn, new_fn)
            if "old_vs_base" in res:
                ro = res["old_vs_base"]; rn = res["new_vs_base"]
                so = "PASS" if ro["ok"] else "FAIL"
                sn = "PASS" if rn["ok"] else "FAIL"
                print(f"n={n:<6} h={h:<4} d={d:<5} | "
                      f"OLD vs BASE [{so}] V(max_abs={ro['max_abs_v']:.3e}, max_rel={ro['max_rel_v']:.3e}) "
                      f"S(max_abs={ro['max_abs_s']:.3e}, max_rel={ro['max_rel_s']:.3e})   "
                      f"NEW vs BASE [{sn}] V(max_abs={rn['max_abs_v']:.3e}, max_rel={rn['max_rel_v']:.3e}) "
                      f"S(max_abs={rn['max_abs_s']:.3e}, max_rel={rn['max_rel_s']:.3e})")
            else:
                rn = res["old_vs_new"]
                s = "PASS" if rn["ok"] else "FAIL"
                print(f"n={n:<6} h={h:<4} d={d:<5} | OLD vs NEW [{s}] "
                      f"V(max_abs={rn['max_abs_v']:.3e}, max_rel={rn['max_rel_v']:.3e}) "
                      f"S(max_abs={rn['max_abs_s']:.3e}, max_rel={rn['max_rel_s']:.3e})")

        # Performance
        print("\nPERFORMANCE")
        for (n, h, d) in TEST_SHAPES:
            r = run_one_perf(n, h, d, prec, base_fn, old_fn, new_fn)
            print(f"n={n:<6} h={h:<4} d={d:<5} | "
                  f"BASE={r['base_ms']:.3f} ms  OLD={r['old_ms']:.3f} ms  NEW={r['new_ms']:.3f} ms | "
                  f"Speedup(OLD)={r['speedup_old']:.2f}x  Speedup(NEW)={r['speedup_new']:.2f}x")
            if prec == "fp16":
                fp16_rows.append([n, h, d, r["base_ms"], r["old_ms"], r["new_ms"], r["speedup_old"], r["speedup_new"]])

    # CSV (FP16 only)
    if fp16_rows:
        arr = np.array(fp16_rows, dtype=np.float64)
        avg_base = float(np.nanmean(arr[:, 3]))
        avg_old  = float(np.nanmean(arr[:, 4]))
        avg_new  = float(np.nanmean(arr[:, 5]))
        avg_spo  = float(np.nanmean(arr[:, 6]))
        avg_spn  = float(np.nanmean(arr[:, 7]))

        with open(OUTPUT_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n", "h", "d", "base_ms", "old_ms", "new_ms", "speedup_old", "speedup_new"])
            for row in fp16_rows:
                w.writerow(row)
            w.writerow(["AVG", "-", "-", avg_base, avg_old, avg_new, avg_spo, avg_spn])

        print(f"\nSaved FP16 3-way summary to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()