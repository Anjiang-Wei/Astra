# bench_silu_mul.py
import os, math, csv
import torch
import numpy as np
from torch.utils.cpp_extension import load

WARMUP = 20
ITERS  = 100
OUTPUT_CSV = "silu_mul_fp16_summary.csv"

TEST_SHAPES = [
    (16, 4096),
    (32, 5120),
    (32, 8192),
    (16, 12288),
    (64, 8192)
]
DTYPES = ["fp16", "bf16"]

def load_local_ext():
    return load(
        name="silu_mul_ext",
        sources=["silu_mul_ultimate.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=True,
        build_directory=os.getcwd(),
    )

def load_sgl_silu_and_mul():
    try:
        import sgl_kernel as sglk
        if hasattr(sglk, "silu_and_mul"):
            return sglk.silu_and_mul
        raise RuntimeError("sgl_kernel.silu_and_mul not found")
    except Exception as e:
        print(f"[INFO] Cannot import sgl_kernel.silu_and_mul: {e}")
        return None

def v_dtype(prec: str):
    p = (prec or "fp16").lower()
    if p == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

def cast_val(x: torch.Tensor, prec: str):
    return x.to(v_dtype(prec), copy=True, non_blocking=True).contiguous()

def allclose_fp(prec: str):
    if prec == "bf16" and torch.cuda.is_bf16_supported():
        return 2e-2, 1e-2
    return 1e-2, 5e-3

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
        times.append(start.elapsed_time(end))
    arr = np.array(times, dtype=np.float64)
    return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())

def correctness_check(ref: torch.Tensor, tst: torch.Tensor, prec: str):
    rtol, atol = allclose_fp(prec)
    a = ref.to(torch.float32)
    b = tst.to(torch.float32)
    ok = torch.allclose(a, b, rtol=rtol, atol=atol)
    dv = (a - b).abs()
    den = torch.maximum(a.abs(), b.abs()).clamp_min(1e-8)
    return bool(ok), dv.max().item(), (dv/den).max().item(), rtol, atol

def run_one_correctness(B, D, prec, ext_func, sgl_func):
    a = torch.randn(B, D, device="cuda", dtype=torch.float32)
    b = torch.randn(D,     device="cuda", dtype=torch.float32)
    a_val = cast_val(a, prec)
    b_val = cast_val(b, prec)

    # concat for both paths: [B, 2D] = [a, b_broadcast]
    x_full = torch.cat([a_val, b_val.unsqueeze(0).expand(B, D)], dim=-1).contiguous()

    # local (your extension)
    y_local = torch.empty(B, D, device="cuda", dtype=a_val.dtype).contiguous()
    ext_func(x_full, y_local)

    if sgl_func is None:
        return None

    # sgl
    y_sgl = torch.empty(B, D, device="cuda", dtype=a_val.dtype).contiguous()
    ret = sgl_func(x_full, y_sgl)
    if ret is not None:
        y_sgl = ret

    ok, mav, mrv, rtol, atol = correctness_check(y_sgl, y_local, prec)
    return {"ok": ok, "max_abs": mav, "max_rel": mrv, "rtol": rtol, "atol": atol}

def run_one_perf(B, D, prec, ext_func, sgl_func):
    a = torch.randn(B, D, device="cuda", dtype=torch.float32)
    b = torch.randn(D,     device="cuda", dtype=torch.float32)
    a_val = cast_val(a, prec)
    b_val = cast_val(b, prec)
    x_full = torch.cat([a_val, b_val.unsqueeze(0).expand(B, D)], dim=-1).contiguous()

    # local
    y_buf = torch.empty(B, D, device="cuda", dtype=a_val.dtype).contiguous()
    def call_local():
        ext_func(x_full, y_buf)
    local_mean, local_std, local_min, local_max = bench_once(call_local)

    # sgl
    sgl_mean = sgl_std = sgl_min = sgl_max = float("nan")
    if sgl_func is not None:
        y_sgl = torch.empty(B, D, device="cuda", dtype=a_val.dtype).contiguous()
        def call_sgl():
            ret = sgl_func(x_full, y_sgl)
            if ret is not None:
                _ = ret
        sgl_mean, sgl_std, sgl_min, sgl_max = bench_once(call_sgl)

    return {
        "local_mean": local_mean, "local_std": local_std,
        "local_min": local_min, "local_max": local_max,
        "sgl_mean": sgl_mean, "sgl_std": sgl_std,
        "sgl_min": sgl_min, "sgl_max": sgl_max,
    }

def main():
    assert torch.cuda.is_available()
    torch.manual_seed(123); torch.cuda.manual_seed_all(123)

    ext = load_local_ext()
    ext_func = ext.sgl_silu_mul
    sgl_func = load_sgl_silu_and_mul()

    # Collect FP16 rows for CSV: [B, D, base_avg_ms, local_avg_ms, speedup]
    fp16_rows = []

    for prec in DTYPES:
        if prec == "bf16" and not torch.cuda.is_bf16_supported():
            continue
        print("=" * 80)
        print(f"SiLU * gate | values={prec.upper()}  (local: a[B,D], gate b[D]; sgl: concat(a,b)->[B,2D])")
        print("=" * 80)

        print("\nCORRECTNESS")
        for (B, D) in TEST_SHAPES:
            res = run_one_correctness(B, D, prec, ext_func, sgl_func)
            if res is None:
                print(f"B={B} D={D} | no sgl_kernel available")
            else:
                status = "PASS" if res["ok"] else "FAIL"
                print(f"[{status}] B={B:<6} D={D:<6} "
                      f"| max_abs={res['max_abs']:.3e} "
                      f"max_rel={res['max_rel']:.3e} "
                      f"(rtol={res['rtol']}, atol={res['atol']})")

        print("\nPERFORMANCE")
        for (B, D) in TEST_SHAPES:
            r = run_one_perf(B, D, prec, ext_func, sgl_func)
            if not math.isnan(r["sgl_mean"]):
                speedup = r["sgl_mean"] / r["local_mean"] if r["local_mean"] > 0 else float("nan")
                tag = "LOCAL faster" if speedup > 1.10 else ("LOCAL slower" if speedup < 0.90 else "similar")
                print(f"B={B:<6} D={D:<6} | "
                      f"Local={r['local_mean']:.3f}±{r['local_std']:.3f} ms  "
                      f"SGL={r['sgl_mean']:.3f}±{r['sgl_std']:.3f} ms  "
                      f"| speedup={speedup:.2f}x {tag}")

                if prec == "fp16":
                    fp16_rows.append([B, D, r["sgl_mean"], r["local_mean"], speedup])
            else:
                print(f"B={B:<6} D={D:<6} | Local={r['local_mean']:.3f}±{r['local_std']:.3f} ms | SGL=N/A")

    # Write FP16 CSV with AVG row
    if fp16_rows:
        arr = np.array(fp16_rows, dtype=np.float64)
        avg_base = float(arr[:, 2].mean())
        avg_local = float(arr[:, 3].mean())
        avg_speedup = float(arr[:, 4].mean())  # avg of per-shape speedups
        with open(OUTPUT_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["B", "D", "base_avg_ms", "local_avg_ms", "speedup"])
            for row in fp16_rows:
                w.writerow(row)
            w.writerow(["AVG", "-", avg_base, avg_local, avg_speedup])
        print(f"\nSaved FP16 summary to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()