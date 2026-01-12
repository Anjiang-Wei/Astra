"""
Prompt templates for CUDA kernel optimization.
"""

from typing import Dict, Any


# =============================================================================
# Agent Instructions
# =============================================================================

AGENT_INSTRUCTIONS: Dict[str, str] = {
    "orchestrator": """You are the Orchestrator Agent responsible for coordinating the CUDA kernel optimization process.
Your role is to manage the workflow between specialized agents and track optimization progress.

Keep responses concise and focused on coordination tasks.
Use the update_optimization_state tool to track progress.
""",

    "code_generator": """Generate next version CUDA source code (.cu) containing:
- At least one __global__ kernel as compute implementation
- One PyBind11 export function (name must match optimization_state.generated_export_func) for Python calls, signature must match baseline (parameter order/in-place behavior)

Constraints:
- Keep same functionality and entry signature as previous version (don't change export function name and parameter list)
- Include necessary headers; don't provide main()
- Prefer fast math, coalesced access, vectorization (float2/float4) and grid-stride loops

Suggestions:
- Fuse Kernels When Possible
- Use Vectorized Access with Alignment Checks
- Warp-Level Reductions Instead of Shared Memory Loops
- Leverage FMA (Fused Multiply-Add)

Workflow: Use save_kernel_code → compile_cuda_kernel to save and compile, repeating compilation (with updated code) until it succeeds; output code only through tools, no explanatory text.
""",

    "correctness_tester": """You handle test case generation and correctness verification with unified test data.

CRITICAL: Test cases are generated ONCE and reused for ALL versions to ensure fair comparison.

For test case generation:
- When asked to generate test cases, provide 8-10 common, practical test scenarios in JSON format
- Call generate_comprehensive_test_cases(test_spec) with your JSON test data
- Include diverse but essential scenarios: small/medium/large sizes, typical LLM dimensions, edge cases, different parameters
- Keep test cases practical and representative of real usage

For correctness verification:
- When asked to verify a kernel, you MUST use the verify_kernel_correctness tool
- Call verify_kernel_correctness with the version parameter (e.g., 'v1', 'v2', etc.)
- ALWAYS use the existing unified test cases - NEVER generate new ones during verification
- The function will return detailed pass/fail results for each test case
- Results are automatically stored for optimization suggestions
- You must actually invoke the tool, not just describe what it does

IMPORTANT: Keep test suite small (8-10 cases) but representative for efficient testing.
""",

    "benchmarker": """You benchmark kernel performance using unified test cases for consistent comparison.

CRITICAL: Always use the SAME test cases across ALL kernel versions for fair comparison.

Performance benchmarking:
- When asked to benchmark a kernel, you MUST use the benchmark_kernel tool
- Call benchmark_kernel with the version parameter (e.g., 'v1', 'v2', etc.) 
- ALWAYS use the existing unified test cases - NEVER generate new ones
- The function will return detailed performance metrics for each test case
- Detailed results are automatically stored for optimization suggestions
- You must actually invoke the tool, not just describe what it does

Analysis focus:
- Results are stored internally for comparison across versions
- You must call the benchmark_kernel function to get actual performance data

IMPORTANT: Consistent test conditions ensure meaningful performance comparison.
""",

    "optimization_strategist": """You are an expert CUDA optimization strategist. Analyze performance data and provide specific, implementable suggestions only (no code).

Focus:
- Memory access (coalescing, vectorization, shared memory)
- Thread/block sizing and occupancy
- Fast math, instruction-level fusion, loop unrolling
- Architecture-specific knobs when safe

Constraints:
- Keep the current __global__ signature and functionality intact.
- Output only suggestions; do not include code snippets.
- Be concrete and actionable.
"""
}


# =============================================================================
# Runtime Prompt Templates
# =============================================================================

VERIFY_CORRECTNESS_PROMPT = "Please call verify_kernel_correctness with version '{version}' to check the correctness of kernel version {version}."

BENCHMARK_PROMPT = "Please call benchmark_kernel with version '{version}' to benchmark the performance of kernel version {version}."

SUGGEST_PROMPT = """You are an optimization strategist. Provide concrete, code-level suggestions only (no code),
for improving the next version '{next_version}' based on the previous version '{base_version}'.

PREVIOUS VERSION KERNEL CODE:
```cuda
{base_code}
```

CURRENT PERFORMANCE (vs previous if available):
{perf_context}

DETAILED TEST RESULTS for {current_version}:
{test_results_detail}

Focus on concrete, actionable ideas (memory access, vectorization, fast-math, launch config, etc.).
Analyze which test cases pass/fail and their performance to guide optimization.
Do not output code, only suggestions.
"""

CODEGEN_PROMPT = """Generate the next kernel version '{next_version}' building upon the previous version '{base_version}'.

PREVIOUS CODE ({base_version}):
```cuda
{base_code}
```

SUGGESTIONS:
{suggestions}

REQUIREMENTS:
- Keep the same functionality.
- Include proper headers; expose a __global__ kernel entry.
- Avoid main(); produce code suitable for nvcc compilation.
- Use save_kernel_code('{next_version}') then compile_cuda_kernel('{next_version}').
- Output only code via the tool (no explanation text).
"""


def get_verify_correctness_prompt(version: str) -> str:
    """Get the prompt for correctness verification."""
    return VERIFY_CORRECTNESS_PROMPT.format(version=version)


def get_benchmark_prompt(version: str) -> str:
    """Get the prompt for benchmarking."""
    return BENCHMARK_PROMPT.format(version=version)


def get_suggest_prompt(
    next_version: str,
    base_version: str,
    base_code: str,
    perf_context: str,
    current_version: str,
    test_results_detail: str
) -> str:
    """Get the optimization suggestion prompt."""
    return SUGGEST_PROMPT.format(
        next_version=next_version,
        base_version=base_version,
        base_code=base_code,
        perf_context=perf_context,
        current_version=current_version,
        test_results_detail=test_results_detail
    )


def get_codegen_prompt(
    next_version: str,
    base_version: str,
    base_code: str,
    suggestions: str
) -> str:
    """Get the code generation prompt."""
    return CODEGEN_PROMPT.format(
        next_version=next_version,
        base_version=base_version,
        base_code=base_code,
        suggestions=suggestions
    )


# =============================================================================
# Test Generation Prompt Templates
# =============================================================================

TESTGEN_PROMPTS: Dict[str, str] = {
    "mergestate": """Generate 12-18 practical test cases for merge_state kernel testing.

BASELINE: {baseline_func}

V1 KERNEL CODE FOR REFERENCE:
```cuda
{v1_code}
```

INSTRUCTIONS:
1. Generate test cases with different (n, h, d) dimensions
2. Focus on realistic attention dimensions
3. Call generate_comprehensive_test_cases(test_spec) with your JSON

DIMENSION GUIDELINES:
- n (sequence length)
- h (num_heads)
- d (head_dim)

SIMPLE JSON FORMAT:
```json
{{
  "medium_test": {{
    "n": 512, "h": 40, "d": 128,
    "description": "Medium attention test"
  }},
  "large_test": {{
    "n": 768, "h": 40, "d": 256,
    "description": "Large attention test"
  }}
}}
```

CRITICAL RULES:
- Use simple integer values for n, h, d
- No complex data structures needed
- Focus on different scale combinations
- Keep JSON syntax clean and valid
""",

    "silu": """Generate 8-10 practical test cases for SiLU-and-Mul fused kernel.

BASELINE: module={baseline_module}, func=silu_and_mul

V1 KERNEL CODE FOR REFERENCE:
```cuda
{v1_code}
```

INSTRUCTIONS:
1. Provide realistic B,D pairs (D must be multiple of 16)
2. Prefer large LLM-like shapes
3. Call generate_comprehensive_test_cases(test_spec) with your JSON

SIMPLE JSON FORMAT:
```json
{{
  "silu_test_medium": {{
    "B": 32, "D": 4096,
    "description": "SiLU*gate medium"
  }}
}}
```

RULES:
- D must be multiple of 16
- Provide several sizes to cover throughput variations
""",

    "rmsnorm": """Generate 8-10 practical test cases that will be used consistently throughout the entire optimization process.

BASELINE: module={baseline_module}, func={baseline_func}

V1 KERNEL CODE FOR REFERENCE:
```cuda
{v1_code}
```

INSTRUCTIONS:
1. Create 8-10 common, representative test scenarios in JSON format
2. Call generate_comprehensive_test_cases(test_spec) with your JSON test data
3. These test cases will be used for ALL versions (v1, v2, v3, etc.) to ensure fair comparison
4. Based on the v1 code above, generate test cases that match the expected function interface

CRITICAL DIMENSION REQUIREMENTS (to avoid alignment errors):
	•	Batch size (B): At least 32
	•	Hidden size (D)
	•	Avoid unusual dimensions such as 3×5, 7×13, etc., as they can cause misaligned address errors
EXAMPLE CORRECT FORMAT (use larger realistic dimensions):
```json
{{
  "medium_test": {{
    "input": "GENERATE_RANDOM_64x2048",
    "residual": "GENERATE_RANDOM_64x2048",
    "weight": "GENERATE_RANDOM_2048",
    "eps": 1e-5,
    "enable_pdl": false,
    "description": "Medium 64x2048 LLM test"
  }}
}}
```

CRITICAL SYNTAX RULES:
- Numbers: 1.0, 2.5, 1e-5 (NOT "1.0")
- Booleans: true, false (NOT "true")  
- NO trailing commas before closing brackets
- Use large aligned dimensions: 64x2048, 128x4096, 256x8192, 512x11008
"""
}


def get_testgen_prompt(
    compare_kind: str,
    baseline_cfg: Dict[str, Any],
    v1_code: str
) -> str:
    """
    Get the test generation prompt for a given compare_kind.
    
    Args:
        compare_kind: One of 'mergestate', 'silu', or 'rmsnorm' (default)
        baseline_cfg: Baseline configuration dict with 'module' and 'func' keys
        v1_code: The v1 kernel source code for reference
        
    Returns:
        Formatted prompt string
    """
    compare_kind = compare_kind.lower()
    
    if compare_kind == 'mergestate':
        return TESTGEN_PROMPTS["mergestate"].format(
            baseline_func=baseline_cfg.get('func', 'merge_state'),
            v1_code=v1_code
        )
    elif compare_kind == 'silu':
        return TESTGEN_PROMPTS["silu"].format(
            baseline_module=baseline_cfg.get('module', 'sgl_kernel'),
            v1_code=v1_code
        )
    else:
        return TESTGEN_PROMPTS["rmsnorm"].format(
            baseline_module=baseline_cfg.get('module', 'sgl_kernel'),
            baseline_func=baseline_cfg.get('func', 'sgl_fused_add_rmsnorm'),
            v1_code=v1_code
        )

