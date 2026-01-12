"""
Unified test framework for CUDA kernel testing.
Provides base class and specialized implementations for different kernel types.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass, field


@dataclass
class TimingStats:
    """Holds timing statistics from a benchmark run."""
    mean_ms: float
    std_ms: float
    min_ms: float = 0.0
    max_ms: float = 0.0
    p95_ms: float = 0.0
    iterations: int = 100


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    status: str  # 'success', 'error'
    generated_metrics: Dict[str, Any] = field(default_factory=dict)
    baseline_metrics: Dict[str, Any] = field(default_factory=dict)
    comparison_metrics: Dict[str, Any] = field(default_factory=dict)
    profiling_data: Dict[str, Any] = field(default_factory=dict)
    total_elements: int = 0
    memory_bytes: int = 0
    input_shape_str: str = ""
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "generated_metrics": self.generated_metrics,
            "baseline_metrics": self.baseline_metrics,
            "comparison_metrics": self.comparison_metrics,
            "profiling_data": self.profiling_data,
            "total_elements": self.total_elements,
            "memory_bytes": self.memory_bytes,
            "input_shape_str": self.input_shape_str,
            "error_message": self.error_message
        }


@dataclass
class CorrectnessResult:
    """Result of a correctness verification."""
    status: str  # 'passed', 'failed', 'error'
    max_abs_diff: Optional[float] = None
    relative_diff: Optional[float] = None
    error_message: Optional[str] = None
    should_remove: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "status": self.status,
            "max_abs_diff": self.max_abs_diff,
            "relative_diff": self.relative_diff,
            "error_message": self.error_message,
        }
        if self.should_remove:
            result["should_remove"] = True
        return result


def time_cuda_kernel(
    func: Callable,
    warmup: int = 20,
    iters: int = 100
) -> TimingStats:
    """Time a CUDA kernel with warmup and statistical analysis."""
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        func()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    arr = np.asarray(times, dtype=np.float64)
    return TimingStats(
        mean_ms=float(arr.mean()),
        std_ms=float(arr.std()),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        p95_ms=float(np.percentile(arr, 95)),
        iterations=iters
    )


class KernelTest(ABC):
    """
    Base class for kernel test cases.
    
    Each test case encapsulates:
    - Test data and parameters
    - Tensor preparation
    - Benchmark execution
    - Correctness verification
    """
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._tensors_prepared = False
    
    @property
    @abstractmethod
    def kernel_type(self) -> str:
        """Return the kernel type identifier."""
        pass
    
    @abstractmethod
    def prepare_tensors(self) -> None:
        """Prepare CUDA tensors for testing. Must be called before benchmark/correctness."""
        pass
    
    @abstractmethod
    def get_total_elements(self) -> int:
        """Return total number of elements processed."""
        pass
    
    @abstractmethod
    def get_memory_bytes(self) -> int:
        """Return total memory bytes transferred."""
        pass
    
    @abstractmethod
    def get_shape_str(self) -> str:
        """Return a string describing the input shape."""
        pass
    
    @abstractmethod
    def make_baseline_caller(self, callable_func: Callable) -> Callable:
        """Create a no-arg callable that runs the baseline kernel."""
        pass
    
    @abstractmethod
    def make_generated_caller(self, callable_func: Callable) -> Callable:
        """Create a no-arg callable that runs the generated kernel."""
        pass
    
    @abstractmethod
    def verify_correctness_impl(
        self,
        baseline_callable: Callable,
        generated_callable: Callable,
        result: CorrectnessResult
    ) -> bool:
        """
        Implementation-specific correctness verification.
        Updates result with timing and diff metrics.
        Returns True if passed.
        """
        pass
    
    def benchmark(
        self,
        generated_callable: Callable,
        baseline_callable: Callable,
        warmup: int = 20,
        iters: int = 100,
        nvml_handle: Any = None
    ) -> BenchmarkResult:
        """
        Run benchmark comparing generated vs baseline kernel.
        
        Args:
            generated_callable: Generated kernel function
            baseline_callable: Baseline kernel function
            warmup: Warmup iterations
            iters: Timed iterations
            nvml_handle: Optional NVML device handle
            
        Returns:
            BenchmarkResult with timing and comparison metrics
        """
        result = BenchmarkResult(status="unknown")
        
        try:
            if not self._tensors_prepared:
                self.prepare_tensors()
            
            result.total_elements = self.get_total_elements()
            result.memory_bytes = self.get_memory_bytes()
            result.input_shape_str = self.get_shape_str()
            
            gen_caller = self.make_generated_caller(generated_callable)
            base_caller = self.make_baseline_caller(baseline_callable)
            
            # NVML baseline
            nvml_data = None
            mem_before = None
            if nvml_handle is not None:
                try:
                    import pynvml
                    mem_before = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
                except:
                    nvml_handle = None
            
            # Benchmark generated
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gen_stats = time_cuda_kernel(gen_caller, warmup=warmup, iters=iters)
            result.generated_metrics = self._compute_metrics(gen_stats)
            
            # Benchmark baseline
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            base_stats = time_cuda_kernel(base_caller, warmup=warmup, iters=iters)
            result.baseline_metrics = self._compute_metrics(base_stats)
            
            # Comparison
            result.comparison_metrics = self._compute_comparison(
                result.generated_metrics, result.baseline_metrics
            )
            
            # NVML post
            if nvml_handle is not None and mem_before is not None:
                try:
                    import pynvml
                    mem_after = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
                    nvml_data = {
                        "memory_used_mb": (mem_after.used - mem_before.used) / (1024 * 1024),
                        "gpu_utilization_percent": util.gpu,
                        "memory_utilization_percent": util.memory
                    }
                except:
                    pass
            
            result.profiling_data = self._compute_profiling(
                result.generated_metrics, nvml_data
            )
            result.status = "success"
            
        except Exception as e:
            result.status = "error"
            result.error_message = str(e)
        
        return result
    
    def verify_correctness(
        self,
        baseline_callable: Callable,
        generated_callable: Callable,
    ) -> CorrectnessResult:
        """
        Verify correctness by comparing generated vs baseline output.
        
        Args:
            baseline_callable: Baseline kernel function
            generated_callable: Generated kernel function
            
        Returns:
            CorrectnessResult with timing and accuracy metrics
        """
        result = CorrectnessResult(status="unknown")
        
        try:
            if not self._tensors_prepared:
                self.prepare_tensors()
            
            passed = self.verify_correctness_impl(
                baseline_callable, generated_callable, result
            )
            result.status = "passed" if passed else "failed"
            
        except Exception as e:
            error_msg = str(e)
            result.status = "error"
            result.error_message = error_msg
            
            # Check for configuration errors
            is_config_error = any(kw in error_msg.lower() for kw in [
                "kernel launch failed", "kernel failed"
            ])
            if is_config_error:
                result.should_remove = True
        
        return result
    
    def _compute_metrics(self, stats: TimingStats) -> Dict[str, Any]:
        """Compute performance metrics from timing stats."""
        mean_sec = stats.mean_ms / 1000.0
        total_elements = self.get_total_elements()
        memory_bytes = self.get_memory_bytes()
        
        throughput = (total_elements / mean_sec) if mean_sec > 0 else 0.0
        bandwidth = (memory_bytes / mean_sec) / (1024**3) if mean_sec > 0 else 0.0
        
        return {
            "mean_time_ms": stats.mean_ms,
            "std_time_ms": stats.std_ms,
            "min_time_ms": stats.min_ms,
            "max_time_ms": stats.max_ms,
            "p95_time_ms": stats.p95_ms,
            "throughput_elements_per_second": throughput,
            "bandwidth_gb_s": bandwidth,
            "iterations": stats.iterations
        }
    
    def _compute_comparison(
        self,
        gen_metrics: Dict[str, Any],
        base_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute comparison metrics."""
        gen_time = gen_metrics["mean_time_ms"]
        base_time = base_metrics["mean_time_ms"]
        gen_throughput = gen_metrics["throughput_elements_per_second"]
        base_throughput = base_metrics["throughput_elements_per_second"]
        
        return {
            "speedup": (base_time / gen_time) if gen_time > 0 else 0.0,
            "throughput_improvement_percent": (
                (gen_throughput - base_throughput) / base_throughput * 100
            ) if base_throughput > 0 else 0.0,
            "time_improvement_percent": (
                (base_time - gen_time) / base_time * 100
            ) if base_time > 0 else 0.0
        }
    
    def _compute_profiling(
        self,
        gen_metrics: Dict[str, Any],
        nvml_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute profiling data."""
        profiling = {}
        actual_bw = gen_metrics.get("bandwidth_gb_s", 0)
        profiling["actual_memory_bandwidth_gb_s"] = actual_bw
        profiling["theoretical_memory_bandwidth_gb_s"] = actual_bw
        profiling["memory_bandwidth_efficiency_percent"] = 100.0 if actual_bw > 0 else 0.0
        
        if nvml_data and "gpu_utilization_percent" in nvml_data:
            profiling["real_gpu_utilization_percent"] = nvml_data["gpu_utilization_percent"]
            profiling["memory_utilization_percent"] = nvml_data.get("memory_utilization_percent", 0)
            profiling["memory_used_mb"] = nvml_data.get("memory_used_mb", 0)
        else:
            profiling["real_gpu_utilization_percent"] = 0.0
        
        profiling["estimated_occupancy_percent"] = 0.0
        profiling["arithmetic_intensity_ops_per_byte"] = 0.25
        
        return profiling
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert test case to dictionary (for legacy compatibility)."""
        return {"name": self.name, "description": self.description}
    
    @classmethod
    @abstractmethod
    def from_spec(cls, name: str, spec: Dict[str, Any]) -> "KernelTest":
        """Create a test instance from LLM-generated spec."""
        pass


class MergeStateTest(KernelTest):
    """Test case for merge_state kernel."""
    
    def __init__(
        self,
        name: str,
        n: int,
        h: int,
        d: int,
        v_a: Optional[np.ndarray] = None,
        v_b: Optional[np.ndarray] = None,
        s_a: Optional[np.ndarray] = None,
        s_b: Optional[np.ndarray] = None,
        description: str = "",
        value_dtype: torch.dtype = torch.float16,
        score_dtype: torch.dtype = torch.float32
    ):
        super().__init__(name, description)
        self.n = n
        self.h = h
        self.d = d
        self.value_dtype = value_dtype
        self.score_dtype = score_dtype
        
        # Store numpy arrays, will convert to tensors in prepare_tensors
        self._v_a_np = v_a
        self._v_b_np = v_b
        self._s_a_np = s_a
        self._s_b_np = s_b
        
        # CUDA tensors (populated by prepare_tensors)
        self.v_a: Optional[torch.Tensor] = None
        self.v_b: Optional[torch.Tensor] = None
        self.s_a: Optional[torch.Tensor] = None
        self.s_b: Optional[torch.Tensor] = None
    
    @property
    def kernel_type(self) -> str:
        return "mergestate"
    
    def prepare_tensors(self) -> None:
        if self._tensors_prepared:
            return
        
        if self._v_a_np is not None:
            self.v_a = torch.tensor(self._v_a_np, device='cuda', dtype=self.value_dtype).contiguous()
            self.v_b = torch.tensor(self._v_b_np, device='cuda', dtype=self.value_dtype).contiguous()
            self.s_a = torch.tensor(self._s_a_np, device='cuda', dtype=self.score_dtype).contiguous()
            self.s_b = torch.tensor(self._s_b_np, device='cuda', dtype=self.score_dtype).contiguous()
        else:
            self.v_a = torch.randn(self.n, self.h, self.d, device='cuda', dtype=self.value_dtype).contiguous()
            self.v_b = torch.randn(self.n, self.h, self.d, device='cuda', dtype=self.value_dtype).contiguous()
            self.s_a = torch.randn(self.n, self.h, device='cuda', dtype=self.score_dtype).contiguous()
            self.s_b = torch.randn(self.n, self.h, device='cuda', dtype=self.score_dtype).contiguous()
        
        self._tensors_prepared = True
    
    def get_total_elements(self) -> int:
        return self.n * self.h * self.d
    
    def get_memory_bytes(self) -> int:
        # (v_a, v_b reads + v_out write) in fp16 + (s_a, s_b reads + s_out write) in fp32
        bytes_values = (2 + 1) * self.n * self.h * self.d * 2  # fp16
        bytes_scores = (2 + 1) * self.n * self.h * 4  # fp32
        return bytes_values + bytes_scores
    
    def get_shape_str(self) -> str:
        return f"v_a=[{self.n}, {self.h}, {self.d}]"
    
    def make_baseline_caller(self, callable_func: Callable) -> Callable:
        v_a, v_b = self.v_a, self.v_b
        s_a, s_b = self.s_a, self.s_b
        
        def caller():
            v_out = torch.zeros_like(v_a)
            s_out = torch.zeros_like(s_a)
            callable_func(v_a.clone(), s_a.clone(), v_b.clone(), s_b.clone(), v_out, s_out)
        return caller
    
    def make_generated_caller(self, callable_func: Callable) -> Callable:
        return self.make_baseline_caller(callable_func)
    
    def verify_correctness_impl(
        self,
        baseline_callable: Callable,
        generated_callable: Callable,
        result: CorrectnessResult
    ) -> bool:
        print(f"Verifying correctness for {self.name}")
        v_out_base = torch.empty_like(self.v_a)
        s_out_base = torch.empty_like(self.s_a)
        v_out_gen = torch.empty_like(self.v_a)
        s_out_gen = torch.empty_like(self.s_a)
        
        v_a, v_b = self.v_a, self.v_b
        s_a, s_b = self.s_a, self.s_b
        
        # Run baseline and generated kernels once for correctness comparison
        torch.cuda.synchronize()
        baseline_callable(v_a.clone(), s_a.clone(), v_b.clone(), s_b.clone(), v_out_base, s_out_base)
        torch.cuda.synchronize()
        generated_callable(v_a.clone(), s_a.clone(), v_b.clone(), s_b.clone(), v_out_gen, s_out_gen)
        
        # Compare
        bv = v_out_base.to(torch.float32)
        gv = v_out_gen.to(torch.float32)
        bs = s_out_base.to(torch.float32)
        gs = s_out_gen.to(torch.float32)
        
        if self.value_dtype is torch.float16:
            v_rtol, v_atol = 1e-2, 5e-3
        else:
            v_rtol, v_atol = 2e-2, 1e-2
        s_rtol, s_atol = 1e-6, 1e-6
        
        ok_v = torch.allclose(gv, bv, rtol=v_rtol, atol=v_atol)
        ok_s = torch.allclose(gs, bs, rtol=s_rtol, atol=s_atol)
        
        den_v = torch.maximum(bv.abs(), gv.abs()).clamp_min(1e-8)
        den_s = torch.maximum(bs.abs(), gs.abs()).clamp_min(1e-12)
        
        result.max_abs_diff = (gv - bv).abs().max().item()
        result.relative_diff = ((gv - bv).abs() / den_v).max().item()
        
        return bool(ok_v and ok_s)
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "n": self.n, "h": self.h, "d": self.d,
            "v_a": self._v_a_np.tolist() if self._v_a_np is not None else None,
            "v_b": self._v_b_np.tolist() if self._v_b_np is not None else None,
            "s_a": self._s_a_np.tolist() if self._s_a_np is not None else None,
            "s_b": self._s_b_np.tolist() if self._s_b_np is not None else None,
        })
        return base
    
    @classmethod
    def from_spec(cls, name: str, spec: Dict[str, Any]) -> "MergeStateTest":
        """Create MergeStateTest from LLM spec."""
        description = spec.get("description", f"Test case {name}")
        
        # Check for explicit tensor data
        if all(k in spec for k in ["v_a", "v_b", "s_a", "s_b"]):
            v_a = np.array(spec["v_a"], dtype=np.float32)
            v_b = np.array(spec["v_b"], dtype=np.float32)
            s_a = np.array(spec["s_a"], dtype=np.float32)
            s_b = np.array(spec["s_b"], dtype=np.float32)
            n, h, d = v_a.shape
            return cls(name, n, h, d, v_a, v_b, s_a, s_b, description)
        
        # Generate from dimensions
        n = int(spec.get("n", 64))
        h = int(spec.get("h", 8))
        d = int(spec.get("d", 64))
        
        np.random.seed(42)
        v_a = np.random.randn(n, h, d).astype(np.float32)
        v_b = np.random.randn(n, h, d).astype(np.float32)
        s_a = np.random.randn(n, h).astype(np.float32)
        s_b = np.random.randn(n, h).astype(np.float32)
        
        return cls(name, n, h, d, v_a, v_b, s_a, s_b, description)


class SiluTest(KernelTest):
    """Test case for SiLU (Swish) kernel."""
    
    def __init__(
        self,
        name: str,
        B: int,
        D: int,
        input_2d: Optional[np.ndarray] = None,
        a: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        description: str = "",
        value_dtype: torch.dtype = torch.float16
    ):
        super().__init__(name, description)
        self.B = B
        self.D = D
        self.value_dtype = value_dtype
        
        self._input_2d_np = input_2d
        self._a_np = a
        self._b_np = b
        
        # CUDA tensors
        self.x_full: Optional[torch.Tensor] = None
        self.out_gen: Optional[torch.Tensor] = None
    
    @property
    def kernel_type(self) -> str:
        return "silu"
    
    def prepare_tensors(self) -> None:
        if self._tensors_prepared:
            return
        
        if self._input_2d_np is not None:
            self.x_full = torch.tensor(self._input_2d_np, device='cuda', dtype=self.value_dtype).contiguous()
        elif self._a_np is not None and self._b_np is not None:
            a = torch.tensor(self._a_np, device='cuda', dtype=self.value_dtype).contiguous()
            b = torch.tensor(self._b_np, device='cuda', dtype=self.value_dtype)
            if b.ndim == 1:
                b = b.unsqueeze(0).expand(a.size(0), a.size(1)).contiguous()
            self.x_full = torch.cat([a, b], dim=-1).contiguous()
        else:
            a = torch.randn(self.B, self.D, device='cuda', dtype=self.value_dtype).contiguous()
            b = torch.randn(self.B, self.D, device='cuda', dtype=self.value_dtype).contiguous()
            self.x_full = torch.cat([a, b], dim=-1).contiguous()
        
        B, D2 = self.x_full.shape
        self.B = B
        self.D = D2 // 2
        self.out_gen = torch.empty(B, self.D, device='cuda', dtype=self.value_dtype).contiguous()
        
        self._tensors_prepared = True
    
    def get_total_elements(self) -> int:
        return self.B * self.D
    
    def get_memory_bytes(self) -> int:
        elem_size = 2 if self.value_dtype == torch.float16 else 4
        return self.x_full.numel() * elem_size + self.out_gen.numel() * elem_size
    
    def get_shape_str(self) -> str:
        return f"[B={self.B}, 2D={self.D * 2}] -> [B={self.B}, D={self.D}]"
    
    def make_baseline_caller(self, callable_func: Callable) -> Callable:
        x_full = self.x_full
        out = self.out_gen
        
        def caller():
            tmp = torch.empty_like(out)
            try:
                ret = callable_func(x_full, tmp)
                _ = tmp if ret is None else ret
            except TypeError:
                ret = callable_func(x_full)
                _ = tmp if ret is None else ret
        return caller
    
    def make_generated_caller(self, callable_func: Callable) -> Callable:
        x_full = self.x_full
        out = self.out_gen
        
        def caller():
            tmp = torch.empty_like(out)
            try:
                ret = callable_func(x_full, tmp)
                _ = tmp if ret is None else ret
            except TypeError:
                ret = callable_func(x_full)
                _ = tmp if ret is None else ret
        return caller
    
    def verify_correctness_impl(
        self,
        baseline_callable: Callable,
        generated_callable: Callable,
        result: CorrectnessResult
    ) -> bool:
        y_base = torch.empty(self.B, self.D, device='cuda', dtype=self.value_dtype).contiguous()
        y_gen = torch.empty_like(y_base)
        
        x_full = self.x_full
        
        # Run baseline and generated kernels once for correctness comparison
        torch.cuda.synchronize()
        ret_b = baseline_callable(x_full.clone(), y_base)
        torch.cuda.synchronize()
        ret_g = generated_callable(x_full.clone(), y_gen)
        if ret_b is not None:
            y_base = ret_b
        if ret_g is not None:
            y_gen = ret_g
        
        passed = torch.allclose(
            y_base.to(torch.float32),
            y_gen.to(torch.float32),
            rtol=1e-3, atol=1e-4
        )
        
        return passed
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({"B": self.B, "D": self.D})
        return base
    
    @classmethod
    def from_spec(cls, name: str, spec: Dict[str, Any]) -> "SiluTest":
        """Create SiluTest from LLM spec."""
        description = spec.get("description", f"Test case {name}")
        
        # Parse dtype
        vdt = (spec.get("dtype") or "").lower()
        if vdt in ("fp32", "float32"):
            value_dtype = torch.float32
        elif vdt in ("bf16", "bfloat16"):
            value_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            value_dtype = torch.float16
        
        if "input_2d" in spec:
            input_2d = np.array(spec["input_2d"], dtype=np.float32)
            B, D2 = input_2d.shape
            return cls(name, B, D2 // 2, input_2d=input_2d, description=description, value_dtype=value_dtype)
        
        if all(k in spec for k in ["a", "b"]):
            a = np.array(spec["a"], dtype=np.float32)
            b = np.array(spec["b"], dtype=np.float32)
            B = a.shape[0]
            D = a.shape[1] if a.ndim > 1 else len(a)
            return cls(name, B, D, a=a, b=b, description=description, value_dtype=value_dtype)
        
        B = int(spec.get("B", 64))
        D = int(spec.get("D", 256))
        return cls(name, B, D, description=description, value_dtype=value_dtype)


class RMSNormTest(KernelTest):
    """Test case for RMSNorm kernel."""
    
    def __init__(
        self,
        name: str,
        B: int,
        D: int,
        input_data: Optional[np.ndarray] = None,
        residual_data: Optional[np.ndarray] = None,
        weight_data: Optional[np.ndarray] = None,
        eps: float = 1e-5,
        enable_pdl: bool = False,
        description: str = ""
    ):
        super().__init__(name, description)
        self.B = B
        self.D = D
        self.eps = eps
        self.enable_pdl = enable_pdl
        
        self._input_np = input_data
        self._residual_np = residual_data
        self._weight_np = weight_data
        
        # CUDA tensors
        self.input_tensor: Optional[torch.Tensor] = None
        self.residual_tensor: Optional[torch.Tensor] = None
        self.weight_tensor: Optional[torch.Tensor] = None
    
    @property
    def kernel_type(self) -> str:
        return "rmsnorm"
    
    def prepare_tensors(self) -> None:
        if self._tensors_prepared:
            return
        
        if self._input_np is not None:
            self.input_tensor = torch.tensor(self._input_np, device='cuda', dtype=torch.float32).contiguous()
            self.residual_tensor = torch.tensor(self._residual_np, device='cuda', dtype=torch.float32).contiguous()
            self.weight_tensor = torch.tensor(self._weight_np, device='cuda', dtype=torch.float32).contiguous()
        else:
            self.input_tensor = torch.randn(self.B, self.D, device='cuda', dtype=torch.float32).contiguous()
            self.residual_tensor = torch.randn(self.B, self.D, device='cuda', dtype=torch.float32).contiguous()
            self.weight_tensor = torch.randn(self.D, device='cuda', dtype=torch.float32).contiguous()
        
        self._tensors_prepared = True
    
    def get_total_elements(self) -> int:
        return self.B * self.D
    
    def get_memory_bytes(self) -> int:
        # input, residual, weight reads + output write ~ 4 floats per element
        return self.get_total_elements() * 4 * 4
    
    def get_shape_str(self) -> str:
        return f"[{self.B}, {self.D}] (eps={self.eps}, pdl={self.enable_pdl})"
    
    def make_baseline_caller(self, callable_func: Callable) -> Callable:
        input_t = self.input_tensor
        residual_t = self.residual_tensor
        weight_t = self.weight_tensor
        eps = self.eps
        pdl = self.enable_pdl
        
        def caller():
            callable_func(input_t.clone(), residual_t.clone(), weight_t.clone(), eps, pdl)
        return caller
    
    def make_generated_caller(self, callable_func: Callable) -> Callable:
        return self.make_baseline_caller(callable_func)
    
    def verify_correctness_impl(
        self,
        baseline_callable: Callable,
        generated_callable: Callable,
        result: CorrectnessResult
    ) -> bool:
        x_base = self.input_tensor
        r_base = self.residual_tensor
        w = self.weight_tensor
        
        # Run baseline and generated kernels once for correctness comparison
        torch.cuda.synchronize()
        xb = x_base.clone()
        rb = r_base.clone()
        b_out = baseline_callable(xb, rb, w, self.eps, self.enable_pdl)
        if b_out is None:
            b_out = xb
        
        torch.cuda.synchronize()
        xg = x_base.clone()
        rg = r_base.clone()
        g_out = generated_callable(xg, rg, w, self.eps, self.enable_pdl)
        if g_out is None:
            g_out = xg
        
        # Compare
        a = b_out.to(torch.float32)
        b = g_out.to(torch.float32)
        abs_diff = (b - a).abs()
        
        result.max_abs_diff = abs_diff.max().item()
        
        den = torch.maximum(a.abs(), b.abs()).clamp_min(1e-8)
        rel = abs_diff / den
        result.relative_diff = rel.max().item()
        
        return result.max_abs_diff < 1e-4 and result.relative_diff < 1e-3
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "input": self._input_np.tolist() if self._input_np is not None else None,
            "residual": self._residual_np.tolist() if self._residual_np is not None else None,
            "weight": self._weight_np.tolist() if self._weight_np is not None else None,
            "eps": self.eps,
            "enable_pdl": self.enable_pdl
        })
        return base
    
    @classmethod
    def from_spec(cls, name: str, spec: Dict[str, Any]) -> "RMSNormTest":
        """Create RMSNormTest from LLM spec."""
        description = spec.get("description", f"Test case {name}")
        eps = float(spec.get("eps", 1e-5))
        enable_pdl = bool(spec.get("enable_pdl", False))
        
        if all(k in spec for k in ["input", "residual", "weight"]):
            input_data = np.array(spec["input"], dtype=np.float32)
            residual_data = np.array(spec["residual"], dtype=np.float32)
            weight_data = np.array(spec["weight"], dtype=np.float32)
            B, D = input_data.shape
            return cls(name, B, D, input_data, residual_data, weight_data, eps, enable_pdl, description)
        
        B = int(spec.get("B", 64))
        D = int(spec.get("D", 256))
        return cls(name, B, D, eps=eps, enable_pdl=enable_pdl, description=description)


# =============================================================================
# Test Collection and Factory
# =============================================================================

class TestCollection:
    """Collection of kernel tests."""
    
    def __init__(self, kernel_type: str):
        self.kernel_type = kernel_type
        self.tests: Dict[str, KernelTest] = {}
    
    def add(self, test: KernelTest) -> None:
        self.tests[test.name] = test
    
    def get(self, name: str) -> Optional[KernelTest]:
        return self.tests.get(name)
    
    def __len__(self) -> int:
        return len(self.tests)
    
    def __iter__(self):
        return iter(self.tests.values())
    
    def items(self):
        return self.tests.items()
    
    def run_all_benchmarks(
        self,
        generated_callable: Callable,
        baseline_callable: Callable,
        warmup: int = 20,
        iters: int = 100
    ) -> Dict[str, BenchmarkResult]:
        """Run benchmarks for all tests."""
        results = {}
        for name, test in self.tests.items():
            results[name] = test.benchmark(
                generated_callable, baseline_callable, warmup, iters
            )
        return results
    
    def run_all_correctness(
        self,
        baseline_callable: Callable,
        generated_callable: Callable
    ) -> Dict[str, CorrectnessResult]:
        """Run correctness verification for all tests."""
        results = {}
        for name, test in self.tests.items():
            results[name] = test.verify_correctness(
                baseline_callable, generated_callable
            )
        return results


def create_test_from_spec(
    name: str,
    spec: Dict[str, Any],
    kernel_type: str
) -> Optional[KernelTest]:
    """
    Factory function to create appropriate test instance from spec.
    
    Args:
        name: Test case name
        spec: LLM-generated test specification
        kernel_type: One of 'mergestate', 'silu', 'rmsnorm'
        
    Returns:
        KernelTest instance or None if validation fails
    """
    try:
        if kernel_type == "mergestate":
            return MergeStateTest.from_spec(name, spec)
        elif kernel_type == "silu":
            return SiluTest.from_spec(name, spec)
        else:  # rmsnorm
            return RMSNormTest.from_spec(name, spec)
    except Exception as e:
        print(f"Warning: Failed to create test '{name}': {e}")
        return None


def validate_test_dimensions(test: KernelTest) -> bool:
    """Validate test dimensions are suitable for GPU execution.
    
    Note: Matches original validation logic from origin.py:
    - MergeState: strict validation (n >= 16, h >= 4, d >= 32, d % 16 == 0)
    - RMSNorm: strict validation (B >= 32, D >= 256, D % 16 == 0)
    - SiLU: no validation (original code didn't validate SiLU dimensions)
    """
    if isinstance(test, MergeStateTest):
        return test.n >= 16 and test.h >= 4 and test.d >= 32 and test.d % 16 == 0
    elif isinstance(test, SiluTest):
        return True
    elif isinstance(test, RMSNormTest):
        return test.B >= 32 and test.D >= 256 and test.D % 16 == 0
    return False


# =============================================================================
# Summary Formatting
# =============================================================================

def format_benchmark_results(
    results: Dict[str, BenchmarkResult],
    version: str
) -> str:
    """Format benchmark results summary."""
    lines = [f"📊 BENCHMARK RESULTS for {version}"]
    
    successful = [(n, r) for n, r in results.items() if r.status == "success"]
    failed = [(n, r) for n, r in results.items() if r.status != "success"]
    
    lines.append(f"✅ Successful: {len(successful)}/{len(results)}")
    
    if successful:
        successful.sort(key=lambda x: x[1].generated_metrics.get("mean_time_ms", float('inf')))
        lines.append("\n📋 Per-test case results:")
        for name, res in successful:
            gen = res.generated_metrics
            gen_time = gen.get("mean_time_ms", 0)
            gen_throughput = gen.get("throughput_elements_per_second", 0) / 1e6
            lines.append(f"  - {name}: {gen_time:.4f}ms ({gen_throughput:.1f} M elem/s)")
    
    if failed:
        lines.append(f"\n❌ Failed benchmarks: {len(failed)}")
        for name, res in failed[:3]:
            error_msg = (res.error_message or "Unknown error")[:80]
            lines.append(f"  - {name}: {error_msg}")
    
    return "\n".join(lines)


def format_correctness_results(
    results: Dict[str, CorrectnessResult],
    version: str
) -> str:
    """Format correctness verification summary."""
    lines = [f"CORRECTNESS RESULTS for {version}"]
    
    passed = sum(1 for r in results.values() if r.status == "passed")
    failed = sum(1 for r in results.values() if r.status == "failed")
    errors = sum(1 for r in results.values() if r.status == "error")
    total = len(results)
    
    all_passed = failed == 0 and errors == 0
    status = "All tests passed" if all_passed else "Some tests failed"
    lines.append(f"{status} - {passed}/{total} passed")
    lines.append("")
    lines.append(f"DETAILED CORRECTNESS VERIFICATION for {version}")
    lines.append("=" * 80)
    
    for name, res in results.items():
        if res.status == "error":
            if res.should_remove:
                lines.append(f"\nTest: {name}\n   CONFIG ERROR (will remove): {res.error_message}")
            else:
                lines.append(f"\nTest: {name}\n   ERROR: {res.error_message}")
            continue
        
        is_pass = res.status == "passed"
        try:
            abs_str = f"{(res.max_abs_diff if res.max_abs_diff is not None else 0.0):.3e}"
        except Exception:
            abs_str = str(res.max_abs_diff)
        try:
            rel_str = f"{(res.relative_diff if res.relative_diff is not None else 0.0):.3e}"
        except Exception:
            rel_str = str(res.relative_diff)
        
        lines.append(
            f"\nTest: {name}\n"
            f"   Status: {'PASSED' if is_pass else 'FAILED'}\n"
            f"   max_abs_diff: {abs_str}\n"
            f"   rel_diff: {rel_str}"
        )
        if res.error_message and res.status == "failed":
            lines.append(f"   Notes: {res.error_message}")
    
    # Summary
    lines.append("\n" + "=" * 80)
    lines.append("VERIFICATION SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Total test cases: {total}")
    lines.append(f"Passed: {passed}")
    lines.append(f"Failed: {failed}")
    lines.append(f"Errors: {errors}")
    
    return "\n".join(lines)

