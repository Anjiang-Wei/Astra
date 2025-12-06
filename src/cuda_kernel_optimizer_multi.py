import os
import sys
import json
import time
import argparse
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import re
import importlib
import torch
import sgl_kernel as sglk
# OpenAI Agents SDK imports
from agents import Agent, Runner, function_tool, input_guardrail, GuardrailFunctionOutput, RunContextWrapper

from test import (
    KernelTest, MergeStateTest, SiluTest, RMSNormTest,
    TestCollection, create_test_from_spec, validate_test_dimensions,
    format_benchmark_results, format_correctness_results,
    BenchmarkResult, CorrectnessResult
)
from prompts import (
    get_testgen_prompt, AGENT_INSTRUCTIONS,
    get_verify_correctness_prompt, get_benchmark_prompt,
    get_suggest_prompt, get_codegen_prompt
)
from format import (
    format_benchmark_summary, format_test_results_for_suggestions,
    format_comparison_summary, format_results
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cuda_optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("agents").setLevel(logging.WARNING)

# Check for CUDA availability
CUDA_AVAILABLE = False
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        logger.info("CUDA is available.")
    else:
        logger.warning("CUDA is not available. Running in simulation mode.")
except ImportError:
    logger.warning("CUDA-related packages not found. Running in simulation mode.")

# Global state for optimization process
optimization_state = {
    "kernel_versions": {},
    "benchmark_results": {},
    "iteration_count": 0,
    "best_version": None,
    "best_performance": None,
    "generic_test_cases": {},  # Legacy format for backward compatibility
    "test_collection": None,   # New TestCollection instance
    "baseline": {},
    "compare_kind": "rmsnorm",
    "input_attention": False,  # True for attention kernels, False for regular kernels
    "generated_wrapper": None,
    "generated_export_func": None,
    "correctness_results": {}
}

# Global callable placeholders (resolved lazily on first use)
baseline_callable = None
generated_callable = None

# Cache for loaded generated modules/functions
_generated_modules: Dict[str, Any] = {}

def extract_cuda_code(response: str) -> str:
    """Extract clean CUDA code from AI response."""
    # Try to find code blocks
    for pattern in [r'```(?:cpp|cuda|c)\s*\n(.*?)\n```', r'```\s*\n(.*?)\n```']:
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        if matches:
            code = max(matches, key=len).strip()
            if is_valid_cuda_code(code):
                return code.replace('\\n', '\n')
    
    return response.strip().replace('\\n', '\n')

def is_valid_cuda_code(code: str) -> bool:
    """Basic validation for CUDA code."""
    code_lower = code.lower()
    
    has_includes = '#include' in code
    has_cuda_elements = any(element in code_lower for element in [
        '__global__', '__device__', '__host__', 'cudamalloc', 'cudamemcpy',
        'blockidx', 'threadidx', 'blockdim', 'griddim'
    ])
    
    return has_includes and has_cuda_elements

# Guardrails for input validation
@input_guardrail
async def validate_cuda_code_input(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str
) -> GuardrailFunctionOutput:
    """Validate CUDA code input."""
    if not input or len(input.strip()) < 10:
        return GuardrailFunctionOutput(
            output_info="Input too short",
            tripwire_triggered=True
        )
    
    required_elements = ['kernel', 'cuda']
    has_required = any(element.lower() in input.lower() for element in required_elements)
    
    return GuardrailFunctionOutput(
        output_info="CUDA code validation passed" if has_required else "Missing CUDA context",
        tripwire_triggered=not has_required
    )

# Function tools for agent capabilities
@function_tool
def save_kernel_code(version: str, code: str, output_dir: str = None, base_filename: str = None) -> str:
    """Save CUDA kernel code to file."""
    try:
        # Use current run's kernel directory if no output_dir specified
        if output_dir is None:
            # Find the latest run directory
            base_runs_dir = Path("cuda_optimization_runs")
            if base_runs_dir.exists():
                latest_link = base_runs_dir / "latest"
                if latest_link.exists() and latest_link.is_symlink():
                    output_dir = str(latest_link / "kernels")
                else:
                    # Fallback: find the most recent run directory
                    run_dirs = [d for d in base_runs_dir.glob("run_*") if d.is_dir()]
                    if run_dirs:
                        latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
                        output_dir = str(latest_run / "kernels")
                    else:
                        output_dir = "cuda_optimization_runs/kernels"
            else:
                output_dir = "cuda_optimization_runs/kernels"
        
        os.makedirs(output_dir, exist_ok=True)
        clean_code = extract_cuda_code(code)
        
        # Ensure proper newlines are preserved
        clean_code = clean_code.replace('\\n', '\n')
        
        # Use base_filename if provided, otherwise fallback to default
        if base_filename:
            # Remove .cu extension if present
            base_name = base_filename.replace('.cu', '')
            kernel_path = os.path.join(output_dir, f"{base_name}_{version}.cu")
        else:
            kernel_path = os.path.join(output_dir, f"kernel_{version}.cu")
        
        with open(kernel_path, "w", encoding='utf-8') as f:
            f.write(clean_code)
        
        # Store in global state with proper formatting
        optimization_state["kernel_versions"][version] = {
            "code": clean_code,
            "path": kernel_path
        }
        
        logger.info(f"Kernel {version} saved to {kernel_path}")
        return f"Successfully saved kernel {version} to {kernel_path}"
    except Exception as e:
        logger.error(f"Error saving kernel: {e}")
        return f"Error saving kernel: {e}"


def _get_cuda_arch() -> Tuple[str, str]:
    """Get CUDA architecture flag and TORCH_CUDA_ARCH_LIST value."""
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(dev)
        return f"-arch=sm_{major}{minor}", f"{major}.{minor}"
    return "-arch=sm_70", "7.0"


@function_tool
def compile_cuda_kernel(version: str, arch_flags: str = "-arch=sm_70") -> str:
    """Compile CUDA kernel code using the PyBind pipeline for parity with runtime."""
    if version not in optimization_state["kernel_versions"]:
        return f"Error: Version {version} not found"

    if not CUDA_AVAILABLE:
        return f"Error: CUDA not available - cannot compile kernel {version}"

    export_name = optimization_state.get("generated_export_func") or \
                  optimization_state.get("baseline", {}).get("func")
    if not export_name:
        return "Error: Missing export function name for PyBind compilation"

    try:
        result = compile_generated_pybind(version, export_name)
        return result
    except Exception as e:
        logger.error(f"Error compiling kernel via PyBind: {e}")
        return f"Compilation failed for {version}:\n{e}"


def _clean_json_format(text: str) -> str:
    """Clean LLM-generated JSON text by removing markdown and fixing common issues."""
    # Remove markdown code block markers
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Extract JSON object
    json_start = text.find('{')
    json_end = text.rfind('}')
    if json_start >= 0 and json_end > json_start:
        text = text[json_start:json_end + 1]
    
    # Fix trailing commas
    text = re.sub(r',\s*([}\]])', r'\1', text)
    
    return text.strip()

@function_tool
def generate_comprehensive_test_cases(test_spec: str = "") -> str:
    """Enhanced test case generation that creates KernelTest instances.
    test_spec: JSON test specification from LLM.
    Returns summary of created test instances.
    """
    try:
        compare_kind = str(optimization_state.get('compare_kind', 'rmsnorm')).lower()
        test_collection = TestCollection(compare_kind)
        
        if not test_spec.strip():
            return "Error: LLM must provide test cases. No test spec provided."
        
        # Parse LLM-provided test data
        try:
            cleaned_spec = _clean_json_format(test_spec)
            test_data = json.loads(cleaned_spec)
            print(f"Debug: LLM provided data type: {type(test_data)}")
            print(f"Debug: LLM data keys/length: {list(test_data.keys()) if isinstance(test_data, dict) else len(test_data) if isinstance(test_data, list) else 'N/A'}")
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse LLM test spec as JSON: {e}")
            return f"Error: Failed to parse test spec as JSON: {e}"
        
        # Extract test cases from various formats
        raw_cases = {}
        if isinstance(test_data, dict) and "test_cases" in test_data:
            raw_cases = test_data["test_cases"]
        elif isinstance(test_data, dict):
            for key, value in test_data.items():
                if isinstance(value, dict):
                    # Handle dimension-only spec for merge_state
                    if all(field in value for field in ["n", "h", "d"]) and not any(f in value for f in ["v_a", "input"]):
                        n, h, d = value["n"], value["h"], value["d"]
                        np.random.seed(42)
                        value = {
                            "n": n, "h": h, "d": d,
                            "v_a": np.random.randn(n, h, d).astype(np.float32).tolist(),
                            "v_b": np.random.randn(n, h, d).astype(np.float32).tolist(),
                            "s_a": np.random.randn(n, h).astype(np.float32).tolist(),
                            "s_b": np.random.randn(n, h).astype(np.float32).tolist(),
                            "description": value.get("description", f"Test case {key}")
                        }
                    # Handle placeholder format like "GENERATE_RANDOM_64x2048"
                    elif "input" in value and isinstance(value["input"], str) and "GENERATE_RANDOM" in value["input"]:
                        match = re.search(r'GENERATE_RANDOM_(\d+)x(\d+)', value["input"])
                        if match:
                            B, D = int(match.group(1)), int(match.group(2))
                            np.random.seed(42)
                            value["input"] = np.random.randn(B, D).astype(np.float32).tolist()
                            value["residual"] = np.random.randn(B, D).astype(np.float32).tolist()
                            weight_match = re.search(r'GENERATE_RANDOM_(\d+)', value.get("weight", ""))
                            weight_dim = int(weight_match.group(1)) if weight_match else D
                            value["weight"] = np.random.randn(weight_dim).astype(np.float32).tolist()
                    raw_cases[key] = value
                elif isinstance(value, list):
                    for i, case in enumerate(value):
                        if isinstance(case, dict):
                            case_name = case.get("name", f"{key}_{i+1}")
                            case_data = {k: v for k, v in case.items() if k != "name"}
                            raw_cases[case_name] = case_data
            if not raw_cases:
                raw_cases = test_data
        elif isinstance(test_data, list):
            for i, case in enumerate(test_data):
                if isinstance(case, dict):
                    case_name = case.get("name", f"llm_test_{i+1}")
                    case_data = {k: v for k, v in case.items() if k != "name"}
                    raw_cases[case_name] = case_data
        
        # Create test instances from specs
        for name, spec in raw_cases.items():
            if not isinstance(spec, dict):
                continue
            
            test = create_test_from_spec(name, spec, compare_kind)
            if test is None:
                print(f"Warning: Failed to create test instance for: {name}")
                continue
            
            # Validate dimensions
            if not validate_test_dimensions(test):
                print(f"Warning: Skipped test case: {name} - dimensions too small or not aligned")
                continue
            
            test_collection.add(test)
            print(f"Added {compare_kind} test case: {name} ({test.get_shape_str()})")
        
        if len(test_collection) == 0:
            return "Error: LLM must provide test cases. All provided cases were invalid."
        
        # Store test collection
        optimization_state["test_collection"] = test_collection
        # Keep legacy format for backward compatibility
        optimization_state["generic_test_cases"] = {
            name: test.to_dict() for name, test in test_collection.items()
        }
        
        # Generate summary
        summary = f"Generated {len(test_collection)} practical test instances:\n"
        summary += f"  - Kernel type: {compare_kind}\n"
        
        # Categorize by common patterns
        categories = {
            "Size variations": len([t for t in test_collection if any(s in t.name for s in ['small_', 'medium_', 'large_', 'size_test'])]),
            "Parameter variations": len([t for t in test_collection if 'eps_' in t.name or 'pdl_' in t.name]),
            "Edge cases": len([t for t in test_collection if any(e in t.name for e in ['small_values', 'edge_', 'non_pow2'])]),
        }
        
        for category, count in categories.items():
            if count > 0:
                summary += f"  - {category}: {count}\n"
        
        print(f"Summary: {summary}")
        return summary
        
    except Exception as e:
        logger.error(f"Error generating comprehensive test cases: {e}")
        return f"Error generating comprehensive test cases: {e}"

def verify_kernel_correctness_detailed(version: str) -> str:
    """Enhanced correctness verification using KernelTest instances."""
    try:
        import torch

        print(f"DEBUG: verify_kernel_correctness_detailed called, version={version}")

        baseline_info = optimization_state.get("baseline", {})
        global baseline_callable, generated_callable

        # Get test collection
        test_collection: TestCollection = optimization_state.get("test_collection")
        if test_collection is None or len(test_collection) == 0:
            return "Error: No test cases available. Use generate_comprehensive_test_cases first."
        
        print(f"DEBUG: Number of test cases={len(test_collection)}")

        # Validate and import generated
        generated_wrapper = optimization_state.get("generated_wrapper")
        if not generated_wrapper:
            return "Error: No generated_wrapper configured (module:function)."

        expected_module = f"gen_{version}"
        actual_module = generated_wrapper.get("module", "")
        if actual_module != expected_module:
            return (f"CRITICAL ERROR: Testing version {version} but generated_wrapper points to module {actual_module}\n"
                    f"This means we're testing the WRONG kernel version!")

        # Import baseline / generated callables
        try:
            if baseline_callable is None:
                baseline_func_name = baseline_info.get("func", "")
                baseline_callable = _import_callable("", baseline_func_name)
            if generated_callable is None:
                gw = optimization_state["generated_wrapper"]
                generated_callable = _import_callable(gw["module"], gw["func"])
        except Exception as e:
            return f"Error importing callable: {e}"

        # Run correctness tests using test instances
        with torch.no_grad():
            correctness_results = test_collection.run_all_correctness(
                baseline_callable=baseline_callable,
                generated_callable=generated_callable
            )

        # Convert results to dict format and compute summary stats
        detailed_results = {}
        summary_stats = {
            "total_tests": len(correctness_results),
            "passed": 0,
            "failed": 0,
            "errors": 0
        }

        for test_name, result in correctness_results.items():
            result_dict = result.to_dict()
            result_dict["name"] = test_name
            test = test_collection.get(test_name)
            result_dict["description"] = test.description if test else "No description"
            
            status = result.status
            if status == "passed":
                summary_stats["passed"] += 1
            elif status == "failed":
                summary_stats["failed"] += 1
            else:
                summary_stats["errors"] += 1
                if result.error_message:
                    print(f"Error: {result.error_message}")
            
            detailed_results[test_name] = result_dict

        # Store results in optimization state
        optimization_state.setdefault("detailed_correctness_results", {})[version] = {
            "summary_stats": summary_stats,
            "test_results": detailed_results,
            "overall_status": "passed" if summary_stats["failed"] == 0 and summary_stats["errors"] == 0 else "failed"
        }

        print(f"verify_kernel_correctness_detailed completed")

        # Return formatted summary
        return format_correctness_results(correctness_results, version)

    except Exception as e:
        logger.error(f"Error in detailed correctness verification: {e}")
        return f"Error in detailed correctness verification: {e}"

@function_tool
def verify_kernel_correctness(version: str) -> str:
    """Wrapper that calls the detailed verification function."""
    try:
        result = verify_kernel_correctness_detailed(version)
        return result
    except Exception as e:
        return f"Error in correctness verification: {e}"


def benchmark_kernel_detailed(version: str) -> str:
    """Enhanced kernel benchmarking using KernelTest instances."""
    try:
        global generated_callable, baseline_callable
        baseline_info = optimization_state.get("baseline", {})
        
        # Get test collection
        test_collection: TestCollection = optimization_state.get("test_collection")
        if test_collection is None or len(test_collection) == 0:
            return "Error: No test cases available. Use generate_comprehensive_test_cases first."
        
        if not optimization_state.get("generated_wrapper"):
            return "Error: No generated_wrapper configured (module:function)."
        
        # Verify version matches expected module
        generated_wrapper = optimization_state.get("generated_wrapper")
        expected_module = f"gen_{version}"
        actual_module = generated_wrapper.get("module", "")
        if actual_module != expected_module:
            error_msg = f"CRITICAL ERROR: Benchmarking version {version} but generated_wrapper points to module {actual_module}"
            error_msg += f"\nThis means we're benchmarking the WRONG kernel version!"
            print(f"DEBUG: {error_msg}")
            return error_msg
        
        print(f"DEBUG: Version verification passed: benchmarking {version} with module {expected_module}")
        
        # Import callables
        try:
            if generated_callable is None:
                gw = optimization_state["generated_wrapper"]
                generated_callable = _import_callable(gw["module"], gw["func"])
            if baseline_callable is None:
                baseline_func_name = baseline_info.get("func", "")
                baseline_callable = _import_callable("", baseline_func_name)
        except Exception as e:
            return f"Error importing callable: {e}"
        
        # NVML setup
        nvml_handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except ImportError:
            pass
        
        # Run benchmarks using test instances
        benchmark_results = test_collection.run_all_benchmarks(
            generated_callable=generated_callable,
            baseline_callable=baseline_callable,
            warmup=20,
            iters=100
        )
        
        # Convert results to dict format and compute summary stats
        detailed_results = {}
        summary_stats = {
            "total_tests": len(benchmark_results),
            "successful_benchmarks": 0,
            "failed_benchmarks": 0,
            "total_generated_time": 0.0,
            "total_baseline_time": 0.0,
            "total_elements_processed": 0
        }
        
        for test_name, result in benchmark_results.items():
            result_dict = result.to_dict()
            result_dict["name"] = test_name
            test = test_collection.get(test_name)
            result_dict["description"] = test.description if test else "No description"
            
            if result.status == "success":
                summary_stats["successful_benchmarks"] += 1
                summary_stats["total_generated_time"] += result.generated_metrics.get("mean_time_ms", 0)
                summary_stats["total_baseline_time"] += result.baseline_metrics.get("mean_time_ms", 0)
                summary_stats["total_elements_processed"] += result.total_elements
            else:
                summary_stats["failed_benchmarks"] += 1
            
            detailed_results[test_name] = result_dict
        
        # Build legacy results for backward compatibility
        legacy_results = {}
        for name, res in detailed_results.items():
            if res.get("status") == "success" and "generated_metrics" in res:
                metrics = res["generated_metrics"]
                element_count = int(metrics["throughput_elements_per_second"] * (metrics["mean_time_ms"] / 1000.0))
                legacy_results[str(max(element_count, 1))] = {
                    "mean_time_ms": metrics["mean_time_ms"],
                    "std_time_ms": metrics["std_time_ms"],
                    "throughput_elements_per_second": metrics["throughput_elements_per_second"]
                }
        
        profiling_data = {
            name: res["profiling_data"]
            for name, res in detailed_results.items()
            if res.get("status") == "success" and "profiling_data" in res
        }
        
        # Store results in optimization state
        optimization_state.setdefault("benchmark_results", {})[version] = {
            "success": True,
            "results": legacy_results,
            "detailed_results": detailed_results,
            "summary_stats": summary_stats,
            "profiling": profiling_data
        }
        
        # Return formatted summary
        return format_benchmark_results(benchmark_results, version)
        
    except Exception as e:
        logger.error(f"Error in detailed kernel benchmarking: {e}")
        return f"Error in detailed kernel benchmarking: {e}"


# Keep original function for backward compatibility
@function_tool
def benchmark_kernel(version: str, input_sizes: str = "1024,4096,16384,65536") -> str:
    """Wrapper that calls the detailed benchmarking function."""
    try:
        result = benchmark_kernel_detailed(version)
        return result
    except Exception as e:

        return f"Error in benchmarking: {e}"

# Agent definitions using OpenAI Agents SDK
orchestrator_agent = Agent(
    name="Orchestrator",
    model="o4-mini",
    instructions=AGENT_INSTRUCTIONS["orchestrator"],
    tools=[]  # Orchestrator only receives notifications, no tools needed
)

code_generation_agent = Agent(
    name="CodeGenerator", 
    model="o4-mini",
    instructions=AGENT_INSTRUCTIONS["code_generator"],
    tools=[save_kernel_code, compile_cuda_kernel],
    input_guardrails=[validate_cuda_code_input]
)

correctness_testing_agent = Agent(
    name="CorrectnessTester",
    model="o4-mini",
    instructions=AGENT_INSTRUCTIONS["correctness_tester"],
    tools=[generate_comprehensive_test_cases, verify_kernel_correctness]
)

benchmarking_agent = Agent(
    name="Benchmarker",
    model="o4-mini",
    instructions=AGENT_INSTRUCTIONS["benchmarker"],
    tools=[benchmark_kernel]
)

optimization_strategy_agent = Agent(
    name="OptimizationStrategist",
    model="o4-mini",
    instructions=AGENT_INSTRUCTIONS["optimization_strategist"],
    tools=[]  # Strategist only provides suggestions, no tools needed
)

class CUDAKernelOptimizer:
    """Main class for CUDA kernel optimization using OpenAI Agents SDK.
    
    Args:
        max_iterations: Maximum number of optimization iterations
        initial_kernel_path: Path to the initial CUDA kernel file
        baseline_module: Baseline module name for comparison
        baseline_func: Baseline function name for comparison
        generated_wrapper: Optional module:function string for generated kernels
        compare_kind: Comparison mode tag (legacy parameter)
        generated_export_func: Export function name in generated PyBind module
        input_attention: If True, treat as attention/merge_state kernel; if False, treat as rmsnorm kernel
    """
    
    def __init__(self, max_iterations: int = 3, initial_kernel_path: Optional[str] = None,
                 baseline_module: str = "sgl_kernel", baseline_func: str = "sgl_fused_add_rmsnorm",
                 generated_wrapper: Optional[str] = None, compare_kind: str = "generic",
                 generated_export_func: Optional[str] = None, input_attention: bool = False):
        self.max_iterations = max_iterations
        self.initial_kernel_path = initial_kernel_path
        # store baseline config
        optimization_state["baseline"] = {"module": (baseline_module or ""), "func": baseline_func}
        optimization_state["compare_kind"] = compare_kind
        optimization_state["input_attention"] = input_attention
        if generated_wrapper:
            try:
                mod_name, func_name = generated_wrapper.split(":", 1)
                optimization_state["generated_wrapper"] = {"module": mod_name, "func": func_name}
            except:
                logger.warning("generated_wrapper should be 'module:function', ignored")
        # export func name for generated pybind; default to baseline func if not provided
        if generated_export_func:
            optimization_state["generated_export_func"] = generated_export_func
        else:
            optimization_state["generated_export_func"] = baseline_func
        
        # Create timestamped run directory structure
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        self.runs_base_dir = Path("cuda_optimization_runs")
        self.run_dir = self.runs_base_dir / f"run_{timestamp}"
        
        # Create directory structure for this run
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "kernels").mkdir(exist_ok=True)
        (self.run_dir / "visualizations").mkdir(exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)
        
        # Create/update latest symlink for convenience
        latest_link = self.runs_base_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(f"run_{timestamp}")
        
        # Update base_dir to point to current run
        self.base_dir = self.run_dir
        
        # Create run metadata
        self._create_run_metadata(timestamp)
        
        logger.info(f"CUDA Kernel Optimizer initialized - Run directory: {self.run_dir}")
    
    def _create_run_metadata(self, timestamp: str):
        """Create metadata file for this optimization run."""
        metadata = {
            "run_id": f"run_{timestamp}",
            "timestamp": timestamp,
            "max_iterations": self.max_iterations,
            "start_time": timestamp,
            "status": "initialized"
        }
        
        metadata_path = self.run_dir / "run_metadata.json"
        with open(metadata_path, "w", encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def _generate_kernel(self, agent, prompt, version):
        """Generate kernel using the agent. The agent handles retries internally."""
        try:
            print(f"🛠️  Generating kernel {version}...")
            
            # Generate kernel - let the agent handle saving and compilation
            result = Runner.run_sync(agent, prompt)
            
            # Check if kernel was saved
            if version not in optimization_state["kernel_versions"]:
                print(f"❌ Kernel {version} generation failed - no code saved")
                return False
            
            # Show saved kernel info
            kernel_info = optimization_state["kernel_versions"][version]
            print(f"💾 Kernel {version} saved ({len(kernel_info['code'])} chars)")
            print(f"   📁 Path: {kernel_info['path']}")
            
            # Check if compilation was mentioned in the result
            if result and hasattr(result, 'final_output'):
                output = result.final_output.lower()
                if "compilation successful" in output or "compiled successfully" in output:
                    print(f"Kernel {version} generated and compiled successfully")
                    return True
                elif "compilation failed" in output or "failed to compile" in output:
                    print(f"Compilation failed for {version}")
                    return False
                else:
                    # Assume success if compilation wasn't explicitly mentioned
                    print(f"Kernel {version} generated successfully")
                    return True
            else:
                print(f"Kernel {version} generated successfully")
                return True
                
        except Exception as e:
            print(f"❌ Error generating kernel {version}: {e}")
            return False
    
    # =========================================================================
    # Step Functions for Optimization Pipeline
    # =========================================================================
    
    def _step_check_cuda(self) -> Optional[Dict]:
        """
        Step: Check CUDA availability.
        
        Returns:
            None if CUDA is available, error dict otherwise.
        """
        if not CUDA_AVAILABLE:
            error_msg = """
❌ CUDA NOT AVAILABLE - Real Hardware Required
- CUDA Available: No
- PyTorch CUDA: Not available
            """
            print(error_msg)
            logger.error("CUDA not available - optimization cannot proceed")
            return {
                "success": False, 
                "error": "CUDA hardware required for real performance optimization",
                "run_directory": str(self.run_dir),
                "cuda_available": CUDA_AVAILABLE,
            }
        logger.info("Starting refactored optimization with CUDA support")
        return None
    
    def _step_load_initial_kernel(self) -> Tuple[bool, str]:
        """
        Step 0: Load initial kernel from file and compile.
        
        Returns:
            Tuple of (success, message)
        """
        print("\n📥 Step 0: Loading initial kernel from file ...")
        
        if not self.initial_kernel_path:
            return False, "--initial-kernel-path is required in refactored pipeline"
        
        # Load kernel code
        init_code = load_kernel_from_file(self.initial_kernel_path)
        
        # Save as v1 under current run
        out_dir = str(self.run_dir / "kernels")
        base_filename = Path(self.initial_kernel_path).name
        _internal_save_kernel_code("v1", init_code, out_dir, base_filename)
        
        # Build PyBind for v1
        export_name = optimization_state.get("generated_export_func") or \
                      optimization_state.get("baseline", {}).get("func", "sgl_fused_add_rmsnorm")
        pybind_msg = compile_generated_pybind("v1", export_name)
        print(pybind_msg)
        
        # Notify orchestrator
        Runner.run_sync(
            orchestrator_agent,
            f"Init run. run_dir={self.run_dir}. Initial version=v1, compiled msg: {pybind_msg}"
        )
        
        return True, pybind_msg
    
    def _step_generate_test_cases(self) -> Tuple[bool, Dict]:
        """
        Step 1: Generate comprehensive test cases (one-time for entire optimization).
        
        Returns:
            Tuple of (success, test_cases dict)
        """
        print("\n🧪 Step 1: Generate comprehensive test cases (ONE-TIME for entire optimization)")
        
        baseline_cfg = optimization_state.get("baseline", {})
        v1_code = self._get_kernel_code("v1")
        compare_kind = str(optimization_state.get('compare_kind', '')).lower()
        
        print(f"   Mode: {compare_kind} kernel - generating test cases")
        testgen_prompt = get_testgen_prompt(compare_kind, baseline_cfg, v1_code)
        Runner.run_sync(correctness_testing_agent, testgen_prompt)
        
        gen_cases = optimization_state.get("generic_test_cases", {})
        
        if not gen_cases:
            print("No test cases found after generation.")
            return False, {}
        
        
        # Store test case summary
        test_categories = {
            "LLM-provided": len([n for n in gen_cases.keys() if n.startswith('llm_')]),
            "Size variations": len([n for n in gen_cases.keys() if any(s in n for s in ['small_', 'medium_', 'large_'])]),
            "Parameter variations": len([n for n in gen_cases.keys() if 'eps_' in n or 'pdl_' in n]),
            "Edge cases": len([n for n in gen_cases.keys() if 'small_values' in n or 'non_pow2' in n]),
        }
        
        optimization_state["test_cases_summary"] = {
            "total_cases": len(gen_cases),
            "categories": test_categories,
            "generation_timestamp": time.time()
        }
        
        print("Practical test cases generated and will be reused for all versions")
        Runner.run_sync(
            orchestrator_agent,
            f"Generated {len(gen_cases)} practical test cases for consistent comparison across all versions."
        )
        
        return True, gen_cases
    
    def _print_test_case_summary(self, gen_cases: Dict):
        """Print summary of generated test cases."""
        print(f"\n---- Generated {len(gen_cases)} Practical Test Cases (FOR ALL VERSIONS) ----")
        
        for test_name, test_data in gen_cases.items():
            description = test_data.get('description', 'No description')
            shape = self._get_test_case_shape(test_data)
            
            if all(k in test_data for k in ["v_a", "v_b", "s_a", "s_b"]):
                # merge_state format
                print(f"  📋 {test_name}: {description} (Shape: {shape})")
            else:
                # RMSNorm format
                eps = test_data.get('eps', 1e-5)
                pdl = test_data.get('enable_pdl', False)
                pdl_str = ", PDL" if pdl else ""
                eps_str = f", eps={eps}" if eps != 1e-5 else ""
                print(f"  📋 {test_name}: {description} (Shape: {shape}{eps_str}{pdl_str})")
    
    def _get_test_case_shape(self, test_data: Dict) -> str:
        """Extract shape string from test data."""
        if all(k in test_data for k in ["v_a", "v_b", "s_a", "s_b"]):
            # merge_state format
            if "n" in test_data and "h" in test_data and "d" in test_data:
                return f"{test_data['n']}x{test_data['h']}x{test_data['d']}"
            
            v_a = test_data.get("v_a")
            if isinstance(v_a, list) and len(v_a) > 0:
                if isinstance(v_a[0], list) and len(v_a[0]) > 0:
                    if isinstance(v_a[0][0], list):
                        return f"{len(v_a)}x{len(v_a[0])}x{len(v_a[0][0])}"
                    return f"{len(v_a)}x{len(v_a[0])}"
            return "Unknown"
        
        # RMSNorm format
        input_data = test_data.get("input")
        if isinstance(input_data, list) and len(input_data) > 0:
            if isinstance(input_data[0], list):
                return f"{len(input_data)}x{len(input_data[0])}"
            return f"1x{len(input_data)}"
        return "Unknown"
    
    def _step_verify_correctness(self, version: str) -> str:
        """
        Step: Verify correctness for a given version.
        
        Args:
            version: Kernel version to verify (e.g., 'v1', 'v2')
            
        Returns:
            Correctness verification result text
        """
        print(f"\n🧪 Verifying {version} correctness using unified test cases...")
        
        # Verify we're testing the right version
        if version != "v1":
            current_wrapper = optimization_state.get("generated_wrapper", {})
            expected_module = f"gen_{version}"
            if current_wrapper.get("module") != expected_module:
                error_msg = (f"❌ CRITICAL ERROR: Testing {version} but generated_wrapper "
                           f"points to {current_wrapper.get('module')}")
                print(error_msg)
                return error_msg
            print(f"✅ Verified: Testing {version} with module {expected_module}")
        
        corr_prompt = get_verify_correctness_prompt(version)
        corr_result = Runner.run_sync(correctness_testing_agent, corr_prompt)
        corr_text = getattr(corr_result, 'final_output', '') if corr_result else ''
        
        optimization_state.setdefault("correctness_results", {})[version] = corr_text
        
        if corr_text:
            print(f"\n---- Correctness Verification Results ({version}) ----")
            print(corr_text)
        
        print(f"✅ Correctness verification completed for {version}")
        Runner.run_sync(
            orchestrator_agent,
            f"Completed correctness verification for {version} using unified test cases."
        )
        
        return corr_text
    
    def _step_benchmark(self, version: str) -> Dict:
        """
        Step: Benchmark a given version.
        
        Args:
            version: Kernel version to benchmark
            
        Returns:
            Benchmark results dict
        """
        # Get correctness stats for logging
        corr_detail = optimization_state.get("detailed_correctness_results", {}).get(version, {})
        stats = corr_detail.get("summary_stats", {})
        passed = stats.get("passed", 0) if stats else 0
        total = stats.get("total_tests", 0) if stats else 0
        
        print(f"\n📊 Benchmarking {version} using unified test cases...")
        if stats:
            print(f"   Note: Correctness {passed}/{total} for {version} - proceeding with benchmark")
        else:
            print(f"   Note: No correctness data for {version} - proceeding with benchmark")
        
        # Verify we're benchmarking the right version
        if version != "v1":
            current_wrapper = optimization_state.get("generated_wrapper", {})
            expected_module = f"gen_{version}"
            if current_wrapper.get("module") != expected_module:
                print(f"❌ CRITICAL ERROR: Benchmarking {version} but generated_wrapper "
                      f"points to {current_wrapper.get('module')}")
                return {}
            print(f"✅ Verified: Benchmarking {version} with module {expected_module}")
        
        benchmark_prompt = get_benchmark_prompt(version)
        Runner.run_sync(benchmarking_agent, benchmark_prompt)
        
        bench_data = optimization_state.get("benchmark_results", {}).get(version, {})
        
        print(f"\n---- Benchmark Summary ({version}) ----")
        print(format_benchmark_summary(bench_data))
        
        Runner.run_sync(
            orchestrator_agent,
            f"Completed {version} benchmark using unified test cases."
        )
        
        return bench_data
    
    def _step_get_suggestions(self, current_version: str, next_version: str) -> str:
        """
        Step A: Get optimization suggestions from strategist.
        
        Args:
            current_version: Current kernel version
            next_version: Target version to generate
            
        Returns:
            Optimization suggestions text
        """
        print("🧠 Asking strategist for optimization suggestions ...")
        
        base_code = self._get_kernel_code(current_version)
        bench_data = optimization_state.get("benchmark_results", {}).get(current_version, {})
        
        # Get correctness results for context
        curr_corr = optimization_state.get("detailed_correctness_results", {}).get(current_version, {})
        test_results_detail = format_test_results_for_suggestions(curr_corr)
        
        # Build performance context
        it = int(current_version[1:])  # Extract iteration number from 'vN'
        prev_version = f"v{it-1}" if it > 1 else None
        
        if prev_version and prev_version in optimization_state.get("benchmark_results", {}):
            perf_context = format_comparison_summary(
                prev_version, current_version, 
                optimization_state.get("benchmark_results", {})
            )
        else:
            perf_context = format_benchmark_summary(bench_data)
        
        suggest_prompt = get_suggest_prompt(
            next_version=next_version,
            base_version=current_version,
            base_code=base_code,
            perf_context=perf_context,
            current_version=current_version,
            test_results_detail=test_results_detail
        )
        print(suggest_prompt)
        
        strat_out = Runner.run_sync(optimization_strategy_agent, suggest_prompt)
        suggestions = getattr(strat_out, 'final_output', '') if strat_out else ''
        
        print("Suggestions ready")
        print(f"\n---- Suggestions for {next_version} ----\n{suggestions}\n")
        
        Runner.run_sync(
            orchestrator_agent,
            f"Collected suggestions for {next_version}."
        )
        
        return suggestions
    
    def _step_generate_version(self, next_version: str, base_version: str, 
                                suggestions: str) -> Tuple[bool, str]:
        """
        Step B: Generate and compile the next kernel version.
        
        Args:
            next_version: Version to generate (e.g., 'v2')
            base_version: Base version to build upon
            suggestions: Optimization suggestions
            
        Returns:
            Tuple of (success, message)
        """
        print("🛠️ Generating next version code via CodeGenerator agent ...")
        
        base_code = self._get_kernel_code(base_version)
        codegen_prompt = get_codegen_prompt(
            next_version=next_version,
            base_version=base_version,
            base_code=base_code,
            suggestions=suggestions
        )
        
        # Generate kernel code
        ok = self._generate_kernel(code_generation_agent, codegen_prompt, next_version)
        if not ok:
            print(f"❌ Code generation failed for {next_version}")
            Runner.run_sync(orchestrator_agent, f"Code generation FAILED for {next_version}.")
            return False, "Code generation failed"
        
        Runner.run_sync(orchestrator_agent, f"Code generation succeeded for {next_version}.")
        
        # Compile and build PyBind
        export_name = optimization_state.get("generated_export_func") or \
                      optimization_state.get("baseline", {}).get("func", "sgl_fused_add_rmsnorm")
        pybind_msg = compile_generated_pybind(next_version, export_name)
        print(pybind_msg)
        
        if "PyBind compiled" not in pybind_msg:
            print(f"PyBind compilation failed for {next_version}")
            print(f"Error details: {pybind_msg}")
            Runner.run_sync(orchestrator_agent, f"PyBind compilation FAILED for {next_version}.")
            return False, pybind_msg
        
        # Verify wrapper points to correct version
        current_wrapper = optimization_state.get("generated_wrapper", {})
        if current_wrapper.get("module") != f"gen_{next_version}":
            msg = f"Generated wrapper mismatch: expected gen_{next_version}, got {current_wrapper.get('module')}"
            print(msg)
            Runner.run_sync(orchestrator_agent, f"Generated_wrapper mismatch for {next_version}.")
            return False, msg
        
        print(f"✅ {next_version} compilation and PyBind setup completed successfully")
        return True, pybind_msg
    
    def _run_single_iteration(self, iteration: int) -> bool:
        """
        Run a single optimization iteration.
        
        Args:
            iteration: Iteration number (1-indexed)
            
        Returns:
            True if iteration completed successfully, False otherwise
        """
        current_version = f"v{iteration}"
        next_version = f"v{iteration + 1}"
        
        print("-" * 60)
        print(f"🔁 Iteration {iteration}: {current_version} → {next_version}")
        Runner.run_sync(
            orchestrator_agent,
            f"Start iteration {iteration}. Current={current_version}, Next={next_version}."
        )
        
        # Step A: Get optimization suggestions
        suggestions = self._step_get_suggestions(current_version, next_version)
        
        # Step B: Generate next version
        success, msg = self._step_generate_version(next_version, current_version, suggestions)
        if not success:
            print(f"Skipping to next iteration due to: {msg}")
            return False
        
        # Step C: Verify correctness
        self._step_verify_correctness(next_version)
        Runner.run_sync(
            orchestrator_agent,
            f"Iteration {iteration}: correctness verification completed for {next_version}."
        )
        
        # Step D: Benchmark
        self._step_benchmark(next_version)
        Runner.run_sync(
            orchestrator_agent,
            f"Iteration {iteration}: benchmarking completed for {next_version}."
        )
        
        return True
    
    def _step_finalize(self) -> Dict:
        """
        Final step: Find best version and report results.
        
        Returns:
            Final results dict
        """
        best_version = self._find_best_version()
        Runner.run_sync(orchestrator_agent, f"Run completed. Best={best_version}")
        
        print("\n🎉 Optimization completed!")
        print("=" * 60)
        print(f"🏆 Best performing version: {best_version}")
        print(f"📈 Total iterations completed: {optimization_state['iteration_count']}")
        
        self._finalize_run_metadata(best_version, True)
        
        return {
            "success": True,
            "best_version": best_version,
            "total_iterations": optimization_state["iteration_count"],
            "run_directory": str(self.run_dir),
        }
    
    # =========================================================================
    # Main Optimization Entry Point
    # =========================================================================
    
    def optimize_kernel(self) -> Dict:
        """
        Main optimization pipeline orchestrating all steps.
        
        Pipeline:
            1. Check CUDA availability
            2. Load initial kernel (v1)
            3. Generate test cases (one-time)
            4. Verify v1 correctness
            5. Benchmark v1
            6. Iterative optimization loop:
               a. Get optimization suggestions
               b. Generate next version
               c. Verify correctness
               d. Benchmark
            7. Finalize and report
        
        Note: Benchmarking proceeds regardless of correctness verification results.
        """
        print("🚀 Starting CUDA Kernel Optimization")
        print("=" * 60)
        
        # Step: Check CUDA availability
        cuda_error = self._step_check_cuda()
        if cuda_error:
            return cuda_error
        
        try:
            # Step 0: Load initial kernel
            success, msg = self._step_load_initial_kernel()
            if not success:
                raise ValueError(msg)
            
            # Step 1: Generate test cases (one-time)
            success, gen_cases = self._step_generate_test_cases()
            if not success:
                return {
                    "success": False,
                    "error": "Failed to generate test cases",
                    "run_directory": str(self.run_dir)
                }
            
            # Step 2: Initial correctness verification for v1
            print("\n🧪 Step 2: Initial correctness verification for v1...")
            self._step_verify_correctness("v1")
            
            # Step 3: Initial benchmark for v1
            print(f"\n📊 Step 3: Benchmarking v1...")
            self._step_benchmark("v1")
            
            # Iterative optimization loop
            print(f"\n🔄 Starting {self.max_iterations} optimization iterations...")
            for it in range(1, self.max_iterations + 1):
                self._run_single_iteration(it)
            
            # Finalize
            return self._step_finalize()
            
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            print(f"\n❌ Optimization failed: {e}")
            try:
                Runner.run_sync(orchestrator_agent, f"Run failed with error: {str(e)}")
            except Exception:
                pass
            self._finalize_run_metadata(success=False)
            return {"success": False, "error": str(e), "run_directory": str(self.run_dir)}
    
    def _compare_performance(self, old_version: str, new_version: str):
        """Compare performance between two versions."""
        if (old_version not in optimization_state["benchmark_results"] or 
            new_version not in optimization_state["benchmark_results"]):
            return
        
        old_results = optimization_state["benchmark_results"][old_version]["results"]
        new_results = optimization_state["benchmark_results"][new_version]["results"]
        
        print(f"\n📈 Performance Comparison: {old_version} vs {new_version}")
        print("   Size     | Time Δ    | Throughput Δ | Status")
        print("   ---------|-----------|--------------|--------")
        
        total_improvement = 0
        count = 0
        
        for size in old_results:
            if size in new_results:
                old_time = old_results[size]["mean_time_ms"]
                new_time = new_results[size]["mean_time_ms"]
                old_throughput = old_results[size]["throughput_elements_per_second"]
                new_throughput = new_results[size]["throughput_elements_per_second"]
                
                time_improvement = ((old_time - new_time) / old_time) * 100
                throughput_improvement = ((new_throughput - old_throughput) / old_throughput) * 100
                
                status = "Faster" if time_improvement > 1 else "Slower"
                if abs(time_improvement) < 1:
                    status = "Similar"
                
                print(f"   {size:8} | {time_improvement:+8.1f}% | {throughput_improvement:+11.1f}% | {status}")
                total_improvement += time_improvement
                count += 1
        
        if count > 0:
            avg_improvement = total_improvement / count
            overall_status = "🎉 IMPROVED" if avg_improvement > 1 else "⚠️  REGRESSED" if avg_improvement < -1 else "🟡 SIMILAR"
            print(f"\n   Overall: {avg_improvement:+.1f}% time improvement {overall_status}")
            
            # Update best version tracking
            if avg_improvement > 0:
                optimization_state["best_version"] = new_version
                optimization_state["best_performance"] = avg_improvement
            
            # Show code evolution summary
            self._show_code_evolution_summary(old_version, new_version)
    
    def _show_code_evolution_summary(self, old_version: str, new_version: str):
        """Show a summary of code changes between versions."""
        if (old_version not in optimization_state["kernel_versions"] or 
            new_version not in optimization_state["kernel_versions"]):
            return
        
        old_code = optimization_state["kernel_versions"][old_version]["code"]
        new_code = optimization_state["kernel_versions"][new_version]["code"]
        
        print(f"\n   📝 Code Evolution Summary ({old_version} → {new_version}):")
        
        # Analyze optimization patterns added
        optimizations_added = []
        optimizations_removed = []
        
        # Check for new optimization patterns
        new_patterns = {
            "Fast Math": ("__expf" in new_code or "__fdividef" in new_code) and not ("__expf" in old_code or "__fdividef" in old_code),
            "Vectorized Access": ("float2" in new_code or "float4" in new_code) and not ("float2" in old_code or "float4" in old_code),
            "Shared Memory": "__shared__" in new_code and "__shared__" not in old_code,
            "Thread Sync": "__syncthreads" in new_code and "__syncthreads" not in old_code,
            "Restrict Pointers": "__restrict__" in new_code and "__restrict__" not in old_code,
            "Grid Stride": ("gridDim" in new_code and "blockDim" in new_code) and not ("gridDim" in old_code and "blockDim" in old_code),
            "Loop Unrolling": "#pragma unroll" in new_code and "#pragma unroll" not in old_code
        }
        
        # Check for removed patterns
        removed_patterns = {
            "Fast Math": ("__expf" in old_code or "__fdividef" in old_code) and not ("__expf" in new_code or "__fdividef" in new_code),
            "Vectorized Access": ("float2" in old_code or "float4" in old_code) and not ("float2" in new_code or "float4" in new_code),
            "Shared Memory": "__shared__" in old_code and "__shared__" not in new_code,
            "Thread Sync": "__syncthreads" in old_code and "__syncthreads" not in new_code,
            "Restrict Pointers": "__restrict__" in old_code and "__restrict__" not in new_code,
            "Grid Stride": ("gridDim" in old_code and "blockDim" in old_code) and not ("gridDim" in new_code and "blockDim" in new_code),
            "Loop Unrolling": "#pragma unroll" in old_code and "#pragma unroll" not in new_code
        }
        
        added = [name for name, added in new_patterns.items() if added]
        removed = [name for name, removed in removed_patterns.items() if removed]
        
        if added:
            print(f"      ✅ Added optimizations: {', '.join(added)}")
        if removed:
            print(f"      ❌ Removed optimizations: {', '.join(removed)}")
        
        # Compare code sizes
        old_lines = len(old_code.split('\n'))
        new_lines = len(new_code.split('\n'))
        line_change = new_lines - old_lines
        
        if line_change > 0:
            print(f"      📈 Code size: +{line_change} lines ({old_lines} → {new_lines})")
        elif line_change < 0:
            print(f"      📉 Code size: {line_change} lines ({old_lines} → {new_lines})")
        else:
            print(f"      📊 Code size: Same ({new_lines} lines)")
        
        if not added and not removed and line_change == 0:
            print(f"      🔄 Minor code modifications (same optimization patterns)")
        
        return
    
    def _find_best_version(self) -> str:
        """Find the best version that passes all tests and has the shortest average benchmark time.

        Selection criteria:
        1) Only consider versions whose correctness overall_status == "passed" (i.e., all tests passed).
        2) Among those, pick the one with the smallest average mean_time_ms across all benchmarked sizes.
        3) If no version passes all tests, fall back to the original heuristic (smallest time on the largest size).
        """
        if not optimization_state["benchmark_results"]:
            return "v1"

        detailed_corr = optimization_state.get("detailed_correctness_results", {})

        best_version = None
        best_avg_time = float('inf')

        # First, try to select only from versions that passed all tests
        for version, bench in optimization_state["benchmark_results"].items():
            if not (bench.get("success") and "results" in bench):
                continue

            corr_info = detailed_corr.get(version, {})
            if corr_info.get("overall_status") != "passed":
                continue

            result_entries = bench.get("results", {})
            if not result_entries:
                continue

            times = [v.get("mean_time_ms") for v in result_entries.values() if isinstance(v, dict) and "mean_time_ms" in v]
            if not times:
                continue

            avg_time = sum(times) / len(times)
            if avg_time < best_avg_time:
                best_avg_time = avg_time
                best_version = version

        if best_version is not None:
            return best_version
    
    def _get_kernel_code(self, version: str) -> str:
        """Get the kernel code for a specific version."""
        if version in optimization_state["kernel_versions"]:
            code = optimization_state["kernel_versions"][version]["code"]
            # Ensure proper newlines
            if isinstance(code, str):
                code = code.replace('\\n', '\n')
            return code
        return "Code not available"
    
       
    def _finalize_run_metadata(self, best_version: str = None, success: bool = False):
        """Finalize run metadata with completion information."""
        metadata_path = self.run_dir / "run_metadata.json"
        
        try:
            # Load existing metadata
            if metadata_path.exists():
                with open(metadata_path, "r", encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            # Update with completion info
            metadata.update({
                "status": "completed" if success else "failed",
                "end_time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                "total_iterations": optimization_state.get("iteration_count", 0),
                "best_version": best_version,
                "kernel_versions": list(optimization_state.get("kernel_versions", {}).keys()),
                "benchmark_results_available": bool(optimization_state.get("benchmark_results")),
                "visualizations_generated": (self.base_dir / "visualizations").exists(),
                "report_generated": (self.base_dir / "optimization_report.md").exists()
            })
            
            # Save updated metadata
            with open(metadata_path, "w", encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error finalizing run metadata: {e}")

def load_kernel_from_file(file_path: str) -> str:
    """Load CUDA kernel source from a file (utf-8)."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def _set_generated_wrapper(version: str, export_func: str, mod: Any) -> None:
    """Register a compiled module as the generated wrapper."""
    global generated_callable
    name = f"gen_{version}"
    generated_callable = getattr(mod, export_func)
    _generated_modules[name] = mod
    optimization_state["generated_wrapper"] = {"module": name, "func": export_func}


def compile_generated_pybind(version: str, export_func: str, extra_cuda_cflags: Optional[List[str]] = None) -> str:
    """Compile .cu file into a Python module and register for verification/benchmarking."""
    from torch.utils.cpp_extension import load
    
    name = f"gen_{version}"
    
    if version not in optimization_state["kernel_versions"]:
        return f"Error: Version {version} not found"
    
    kernel_info = optimization_state["kernel_versions"][version]
    cu_path = kernel_info["path"]
    build_dir = str(Path(cu_path).parent / f"build_{version}")
    os.makedirs(build_dir, exist_ok=True)
    
    # Set CUDA architecture
    if torch.cuda.is_available():
        arch_flag, arch_list = _get_cuda_arch()
        os.environ['TORCH_CUDA_ARCH_LIST'] = arch_list
    else:
        return "Error: CUDA not available"
    
    if extra_cuda_cflags is None:
        extra_cuda_cflags = ["-O2", "--use_fast_math", "-std=c++17"]
    
    os.environ.setdefault('MAX_JOBS', '2')
    
    # Compile with timeout
    result = {"mod": None, "error": None}
    
    def compile_task():
        try:
            result["mod"] = load(
                name=name,
                sources=[cu_path],
                extra_cuda_cflags=extra_cuda_cflags,
                extra_cflags=["-O2", "-std=c++17"],
                build_directory=build_dir,
                verbose=False,
                with_cuda=True,
            )
        except Exception as e:
            result["error"] = e
    
    thread = threading.Thread(target=compile_task, daemon=True)
    thread.start()
    thread.join(timeout=60)
    
    if thread.is_alive():
        return f"Error: Compilation timed out after 60 seconds"
    if result["error"]:
        logger.error(f"Compilation failed: {result['error']}")
        return f"Error: Compilation failed: {result['error']}"
    if result["mod"] is None:
        return "Error: Compilation returned no result"
    
    # Register the compiled module
    try:
        _set_generated_wrapper(version, export_func, result["mod"])
        return f"✅ PyBind compiled for {version} as module '{name}', func '{export_func}'"
    except AttributeError:
        return f"❌ Compiled module missing export func '{export_func}'"


def _internal_save_kernel_code(version: str, code: str, output_dir: Optional[str] = None, base_filename: str = None) -> str:
    """Internal helper to save kernel code without @function_tool decorator."""
    try:
        output_dir = output_dir or "cuda_optimization_runs/kernels"
        os.makedirs(output_dir, exist_ok=True)
        
        clean_code = extract_cuda_code(code).replace('\\n', '\n')
        base_name = base_filename.replace('.cu', '') if base_filename else "kernel"
        kernel_path = os.path.join(output_dir, f"{base_name}_{version}.cu")
        
        with open(kernel_path, "w", encoding='utf-8') as f:
            f.write(clean_code)
        
        optimization_state["kernel_versions"][version] = {"code": clean_code, "path": kernel_path}
        logger.info(f"Kernel {version} saved to {kernel_path}")
        return f"Successfully saved kernel {version} to {kernel_path}"
    except Exception as e:
        logger.error(f"Error saving kernel: {e}")
        return f"Error saving kernel: {e}"


def _import_callable(module_name: str, func_name: str):
    """Import a callable by module and function name.
    
    For baseline (empty module_name): uses sgl_kernel directly.
    For generated modules: looks up in _generated_modules cache or imports.
    """
    if not func_name:
        raise ImportError("Function name not provided")
    
    # Baseline via sgl_kernel
    if not module_name or module_name in ("sgl_kernel", "ops:sgl_kernel"):
        fn = getattr(sglk, func_name, None)
        if fn is None:
            raise AttributeError(f"sgl_kernel has no attribute '{func_name}'")
        if not callable(fn):
            raise TypeError(f"sgl_kernel.{func_name} is not callable")
        return fn
    
    # Generated module
    mod = _generated_modules.get(module_name) or importlib.import_module(module_name)
    fn = getattr(mod, func_name, None)
    if fn is None:
        raise AttributeError(f"{module_name} has no attribute '{func_name}'")
    if not callable(fn):
        raise TypeError(f"{module_name}.{func_name} is not callable")
    return fn

def main():
    parser = argparse.ArgumentParser(description='CUDA Kernel Optimizer (Refactored, general baseline mode). Benchmarking proceeds regardless of correctness results.')
    parser.add_argument('--api-key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--max-iterations', type=int, default=5, help='Maximum optimization iterations')
    parser.add_argument('--initial-kernel-path', type=str, required=True, help='Path to initial simplified CUDA kernel file')
    parser.add_argument('--baseline-module', type=str, default='', help='Baseline module to import (optional). If empty, will use sgl_kernel or sglang.sgl_kernel automatically')
    parser.add_argument('--baseline-func', type=str, default='sgl_fused_add_rmsnorm', help='Baseline function name in module')
    parser.add_argument('--generated-wrapper', type=str, default='', help="Optional 'module:function' callable to run generated kernel for generic comparison")
    parser.add_argument('--compare-kind', type=str, default='generic', help="Comparison mode tag (e.g., 'generic','rmsnorm','qita') to guide test generation prompts")
    parser.add_argument('--input-attention', action='store_true', help='Set to true for attention/merge_state kernels, false (default) for rmsnorm kernels')
    parser.add_argument('--generated-export-func', type=str, default='', help='Export function name expected in generated PyBind module (defaults to baseline-func)')
    args = parser.parse_args()
    
    os.environ['OPENAI_API_KEY'] = args.api_key
    optimizer = CUDAKernelOptimizer(
        max_iterations=args.max_iterations,
        initial_kernel_path=args.initial_kernel_path,
        baseline_module=args.baseline_module,
        baseline_func=args.baseline_func,
        generated_wrapper=(args.generated_wrapper or None),
        compare_kind=args.compare_kind,
        generated_export_func=(args.generated_export_func or None),
        input_attention=args.input_attention,
    )
    result = optimizer.optimize_kernel()
    sys.exit(0 if result.get('success') else 1)

if __name__ == "__main__":
    main()

