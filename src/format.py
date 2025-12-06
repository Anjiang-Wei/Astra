"""
Formatting utilities for CUDA kernel optimization output.
"""

import numpy as np
from typing import Dict, Any, List


def format_benchmark_summary(benchmark_data: Dict[str, Any]) -> str:
    """Format benchmark data for optimization analysis using comprehensive metrics."""
    if not benchmark_data or not benchmark_data.get("success"):
        return "No benchmark data available"
    
    results = benchmark_data.get("results", {})
    profiling = benchmark_data.get("profiling", {})
    
    if not results:
        # Explain WHY there are no results using detailed records
        detailed = benchmark_data.get("detailed_results", {}) or {}
        stats = benchmark_data.get("summary_stats", {}) or {}
        lines = ["No performance results available"]
        if stats:
            sb = stats.get("successful_benchmarks", 0) or 0
            fb = stats.get("failed_benchmarks", 0) or 0
            te = stats.get("total_elements_processed", 0) or 0
            lines.append(f"- successful_benchmarks={sb}, failed_benchmarks={fb}, total_elements={te}")
        if detailed:
            # List a few sample reasons
            lines.append("- Sample case statuses:")
            shown = 0
            for name, rec in detailed.items():
                status = rec.get("status", "unknown")
                err = rec.get("error_message", "") or ""
                lines.append(f"  - {name}: {status}{(' | ' + err[:120]) if err else ''}")
                shown += 1
                if shown >= 5:
                    break
        return "\n".join(lines)
    
    summary = "COMPREHENSIVE CUDA PERFORMANCE ANALYSIS:\n"
    
    def _safe_int(x):
        try:
            return int(x)
        except Exception:
            return x
    
    for size, result in sorted(results.items(), key=lambda x: _safe_int(x[0])):
        time_ms = result.get("mean_time_ms", 0.0) or 0.0
        throughput = result.get("throughput_elements_per_second", 0.0) or 0.0
        prof_data = profiling.get(size, {})
        
        summary += f"  Size {size}:\n"
        summary += f"    - Execution Time: {time_ms:.4f}ms\n"
        summary += f"    - Throughput: {throughput:,.0f} elem/s\n"
        
        # Memory performance metrics
        if prof_data.get('actual_memory_bandwidth_gb_s', 0) > 0:
            summary += f"    - Actual Memory Bandwidth: {prof_data['actual_memory_bandwidth_gb_s']:.1f} GB/s\n"
        
        if prof_data.get('memory_bandwidth_efficiency_percent', 0) > 0:
            summary += f"    - Memory Bandwidth Efficiency: {prof_data['memory_bandwidth_efficiency_percent']:.1f}%\n"
        
        if prof_data.get('theoretical_memory_bandwidth_gb_s', 0) > 0:
            summary += f"    - Theoretical Peak Bandwidth: {prof_data['theoretical_memory_bandwidth_gb_s']:.1f} GB/s\n"
        
        # Compute characteristics
        if prof_data.get('arithmetic_intensity_ops_per_byte', 0) > 0:
            summary += f"    - Arithmetic Intensity: {prof_data['arithmetic_intensity_ops_per_byte']:.2f} ops/byte\n"
        
        # GPU utilization metrics
        if prof_data.get('real_gpu_utilization_percent', 0) > 0:
            summary += f"    - Real GPU Utilization: {prof_data['real_gpu_utilization_percent']:.1f}%\n"
        
        if prof_data.get('real_memory_controller_utilization_percent', 0) > 0:
            summary += f"    - Memory Controller Utilization: {prof_data['real_memory_controller_utilization_percent']:.1f}%\n"
        
        if prof_data.get('estimated_occupancy_percent', 0) > 0:
            summary += f"    - Estimated Occupancy: {prof_data['estimated_occupancy_percent']:.1f}%\n"
        
        # PyTorch profiler metrics
        if prof_data.get('pytorch_profiler_available', False):
            summary += f"    - PyTorch CUDA Events: {prof_data.get('pytorch_num_cuda_events', 0)}\n"
            cuda_time = prof_data.get('pytorch_cuda_time_us', 0) or prof_data.get('pytorch_total_cuda_time_us', 0) or 0.0
            summary += f"    - PyTorch CUDA Time: {cuda_time:.1f}μs\n"
            
            if prof_data.get('pytorch_total_flops', 0) > 0:
                summary += f"    - Measured FLOPS: {prof_data['pytorch_total_flops']:,.0f}\n"
                summary += f"    - GFLOPS/sec: {prof_data.get('pytorch_gflops_per_sec', 0):.2f}\n"
        
        # Memory usage
        if prof_data.get('memory_utilization_percent', 0) > 0:
            summary += f"    - GPU Memory Usage: {prof_data['memory_utilization_percent']:.1f}% ({prof_data.get('memory_used_mb', 0):.0f}MB)\n"
        
        # Error information
        if prof_data.get('pytorch_profiler_error'):
            summary += f"    - PyTorch Profiler Issue: {prof_data['pytorch_profiler_error']}\n"
        
        if prof_data.get('nvml_note'):
            summary += f"    - NVML Note: {prof_data['nvml_note']}\n"
        
        summary += "\n"
    
    # Comprehensive analysis based on real measurements
    summary += "OPTIMIZATION ANALYSIS BASED ON REAL MEASUREMENTS:\n"
    
    # Collect all profiling data for analysis
    all_prof_data = [profiling.get(size, {}) for size in results.keys()]
    valid_prof_data = [data for data in all_prof_data if data.get('actual_memory_bandwidth_gb_s', 0) > 0]
    
    if valid_prof_data:
        # Memory performance analysis
        avg_memory_bw = float(np.mean([data.get('actual_memory_bandwidth_gb_s', 0.0) or 0.0 for data in valid_prof_data]))
        avg_bw_efficiency = float(np.mean([data.get('memory_bandwidth_efficiency_percent', 0.0) or 0.0 for data in valid_prof_data]))
        avg_arith_intensity = float(np.mean([data.get('arithmetic_intensity_ops_per_byte', 0.0) or 0.0 for data in valid_prof_data]))
        
        summary += f"  Memory Performance:\n"
        summary += f"    - Average Memory Bandwidth: {avg_memory_bw:.1f} GB/s\n"
        summary += f"    - Average Memory Efficiency: {avg_bw_efficiency:.1f}%\n"
        summary += f"    - Average Arithmetic Intensity: {avg_arith_intensity:.2f} ops/byte\n"
        
        # Performance classification
        if avg_arith_intensity < 1.0:
            summary += f"    - CLASSIFICATION: MEMORY-BOUND kernel\n"
            if avg_bw_efficiency < 50:
                summary += f"    - PRIMARY BOTTLENECK: Poor memory access efficiency\n"
                summary += f"    - OPTIMIZATION FOCUS: Memory coalescing, vectorized loads\n"
        elif avg_arith_intensity > 3.0:
            summary += f"    - CLASSIFICATION: COMPUTE-BOUND kernel\n"
            summary += f"    - OPTIMIZATION FOCUS: Fast math functions, operation fusion\n"
        else:
            summary += f"    - CLASSIFICATION: BALANCED kernel\n"
            summary += f"    - OPTIMIZATION FOCUS: Both memory and compute optimization\n"
        
        # GPU utilization analysis
        gpu_utils = [data.get('real_gpu_utilization_percent', 0.0) or 0.0 for data in valid_prof_data if (data.get('real_gpu_utilization_percent', 0.0) or 0.0) > 0]
        occupancies = [data.get('estimated_occupancy_percent', 0.0) or 0.0 for data in valid_prof_data if (data.get('estimated_occupancy_percent', 0.0) or 0.0) > 0]
        
        if gpu_utils:
            avg_gpu_util = np.mean(gpu_utils)
            avg_occupancy = np.mean(occupancies) if occupancies else 0
            
            summary += f"  GPU Utilization:\n"
            summary += f"    - Average GPU Utilization: {avg_gpu_util:.1f}%\n"
            summary += f"    - Average Estimated Occupancy: {avg_occupancy:.1f}%\n"
            
            if avg_gpu_util < 30:
                summary += f"    - GPU UTILIZATION ISSUE: Low utilization detected\n"
                summary += f"    - RECOMMENDATION: Increase block size or reduce kernel overhead\n"
            elif avg_gpu_util >= 70:
                summary += f"    - **GPU Utilization**: Good resource utilization achieved\n"
        
        # Arithmetic intensity analysis
        arith_intensities = [data.get('arithmetic_intensity_ops_per_byte', 0.0) or 0.0 for data in valid_prof_data if (data.get('arithmetic_intensity_ops_per_byte', 0.0) or 0.0) > 0]
        
        if arith_intensities:
            avg_arith_intensity = np.mean(arith_intensities)
            summary += f"  Compute Characteristics:\n"
            summary += f"    - Arithmetic Intensity: {avg_arith_intensity:.2f} ops/byte\n"
            
            if avg_arith_intensity < 1.0:
                summary += f"    - Note: MEMORY-BOUND kernel - Focus on memory bandwidth optimization\n"
            elif avg_arith_intensity > 3.0:
                summary += f"    - Note: COMPUTE-BOUND kernel - Focus on arithmetic optimization\n"
            else:
                summary += f"    - Note: BALANCED kernel - Optimize both memory and compute\n"
        
        # PyTorch profiler analysis
        pytorch_data = [data for data in valid_prof_data if data.get('pytorch_profiler_available', False)]
        if pytorch_data:
            total_flops = sum(data.get('pytorch_total_flops', 0) for data in pytorch_data)
            avg_gflops = np.mean([data.get('pytorch_gflops_per_sec', 0) for data in pytorch_data if data.get('pytorch_gflops_per_sec', 0) > 0])
            
            summary += f"  Compute Performance:\n"
            summary += f"    - Total FLOPS Measured: {total_flops:,.0f}\n"
            if avg_gflops > 0:
                summary += f"    - Average GFLOPS/sec: {avg_gflops:.2f}\n"
    else:
        summary += f"  - Limited profiling data available for comprehensive analysis\n"
        summary += f"  - Install nvidia-ml-py for complete GPU metrics: pip install nvidia-ml-py\n"
        
        # Check for profiling issues
        for size in results.keys():
            prof_data = profiling.get(size, {})
            if prof_data.get('pytorch_profiler_error'):
                summary += f"  - PyTorch Profiler Issue: {prof_data['pytorch_profiler_error']}\n"
                break
    
    return summary


def format_test_results_for_suggestions(correctness_results: Dict[str, Any]) -> str:
    """Format detailed test results for optimization suggestions."""
    if not correctness_results or not correctness_results.get("test_results"):
        return "No detailed test results available."
    
    test_results = correctness_results.get("test_results", {})
    summary_stats = correctness_results.get("summary_stats", {})
    
    detail = f"CORRECTNESS ANALYSIS ({summary_stats.get('passed', 0)}/{summary_stats.get('total_tests', 0)} passed):\n"
    
    # List passed tests
    passed_tests = [name for name, result in test_results.items() if result.get("status") == "passed"]
    if passed_tests:
        detail += f"\n✅ PASSED TESTS ({len(passed_tests)}):\n"
        for test_name in passed_tests:
            result = test_results[test_name]
            max_abs_diff = result.get("max_abs_diff")
            rel_diff = result.get("relative_diff")
            # Safe formatting
            try:
                abs_str = f"{(max_abs_diff if max_abs_diff is not None else 0.0):.2e}"
            except Exception:
                abs_str = str(max_abs_diff)
            try:
                rel_str = f"{(rel_diff if rel_diff is not None else 0.0):.2e}"
            except Exception:
                rel_str = str(rel_diff)
            detail += f"  - {test_name}: abs_diff={abs_str} rel_diff={rel_str}\n"
    
    # List failed tests
    failed_tests = [name for name, result in test_results.items() if result.get("status") == "failed"]
    if failed_tests:
        detail += f"\n❌ FAILED TESTS ({len(failed_tests)}):\n"
        for test_name in failed_tests:
            result = test_results[test_name]
            max_abs_diff = result.get("max_abs_diff", "N/A")
            rel_diff = result.get("relative_diff", "N/A") 
            error_msg = result.get("error_message", "Unknown error")
            detail += f"  - {test_name}: abs_diff={max_abs_diff} rel_diff={rel_diff} - {error_msg}\n"
    
    # List error tests  
    error_tests = [name for name, result in test_results.items() if result.get("status") == "error"]
    if error_tests:
        detail += f"\n⚠️ ERROR TESTS ({len(error_tests)}):\n"
        for test_name in error_tests:
            result = test_results[test_name]
            error_msg = result.get("error_message", "Unknown error")
            detail += f"  - {test_name}: {error_msg}\n"
    
    return detail


def format_comparison_summary(
    prev_version: str,
    curr_version: str,
    benchmark_results: Dict[str, Any]
) -> str:
    """
    Return a concise comparison summary between two versions.
    If previous version data is unavailable, fall back to current version summary.
    
    Args:
        prev_version: Previous version string (e.g., 'v1')
        curr_version: Current version string (e.g., 'v2')
        benchmark_results: Dictionary containing benchmark results for all versions
        
    Returns:
        Formatted comparison summary string
    """
    if prev_version not in benchmark_results or curr_version not in benchmark_results:
        return format_benchmark_summary(benchmark_results.get(curr_version, {}))

    prev_data = benchmark_results[prev_version]
    curr_data = benchmark_results[curr_version]
    prev_details = prev_data.get("detailed_results", {}) or {}
    curr_details = curr_data.get("detailed_results", {}) or {}
    
    lines = [f"PERFORMANCE COMPARISON: {prev_version} → {curr_version}"]
    compared_any = False
    
    if prev_details and curr_details:
        common_tests = sorted(set(prev_details.keys()) & set(curr_details.keys()))
        for test_name in common_tests:
            prev_res = prev_details.get(test_name, {})
            curr_res = curr_details.get(test_name, {})
            if prev_res.get("status") != "success" or curr_res.get("status") != "success":
                continue
            p_gen = prev_res.get("generated_metrics", {})
            c_gen = curr_res.get("generated_metrics", {})
            p_time = p_gen.get("mean_time_ms", 0.0) or 0.0
            c_time = c_gen.get("mean_time_ms", 0.0) or 0.0
            p_tp = p_gen.get("throughput_elements_per_second", 0.0) or 0.0
            c_tp = c_gen.get("throughput_elements_per_second", 0.0) or 0.0
            time_improv = ((p_time - c_time) / p_time * 100.0) if p_time > 0 else 0.0
            tp_improv = ((c_tp - p_tp) / p_tp * 100.0) if p_tp > 0 else 0.0
            lines.append(
                f"  - {test_name}: time {p_time:.4f}ms → {c_time:.4f}ms ({time_improv:+.1f}%), "
                f"throughput {p_tp/1e6:.1f}M/s → {c_tp/1e6:.1f}M/s ({tp_improv:+.1f}%)"
            )
            compared_any = True
    
    if compared_any:
        return "\n".join(lines)
    
    # Fallback to legacy summary if detailed data unavailable
    prev_results = prev_data.get("results", {})
    curr_results = curr_data.get("results", {})
    if not prev_results or not curr_results:
        return format_benchmark_summary(benchmark_results.get(curr_version, {}))

    def _safe_int(x):
        try:
            return int(x)
        except Exception:
            return x
    
    common_sizes = sorted(set(prev_results.keys()) & set(curr_results.keys()), key=_safe_int)
    if not common_sizes:
        return format_benchmark_summary(benchmark_results.get(curr_version, {}))

    for size in common_sizes:
        p = prev_results[size]
        c = curr_results[size]
        p_time = p.get("mean_time_ms", 0.0) or 0.0
        c_time = c.get("mean_time_ms", 0.0) or 0.0
        p_tp = p.get("throughput_elements_per_second", 0.0) or 0.0
        c_tp = c.get("throughput_elements_per_second", 0.0) or 0.0
        time_improv = ((p_time - c_time) / p_time * 100.0) if p_time > 0 else 0.0
        tp_improv = ((c_tp - p_tp) / p_tp * 100.0) if p_tp > 0 else 0.0
        lines.append(
            f"  Size {size}: time {p_time:.4f}ms → {c_time:.4f}ms ({time_improv:+.1f}%), "
            f"throughput {p_tp/1e6:.1f}M/s → {c_tp/1e6:.1f}M/s ({tp_improv:+.1f}%)"
        )

    return "\n".join(lines)


def format_results(benchmark_results: Dict[str, Any]) -> str:
    """Format benchmark results for the report."""
    if not benchmark_results:
        return "No benchmark results available."
    
    formatted = ""
    for version, results in benchmark_results.items():
        if results.get("success") and "results" in results:
            formatted += f"\n### Version {version}\n"
            for size, result in results["results"].items():
                formatted += f"- Size {size}: {result['mean_time_ms']:.4f}ms\n"
    
    return formatted

