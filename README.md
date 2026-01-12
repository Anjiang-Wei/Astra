# Astra: A Multi-Agent System for GPU Kernel Performance Optimization
[![arXiv](https://img.shields.io/badge/arXiv-2509.21629-b31b1b.svg)](https://arxiv.org/abs/2509.07506) [![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/license/apache-2-0) 

## Usage

```bash
python3 cuda_kernel_optimizer_multi.py \
    --api-key <OPENAI_API_KEY> \
    --initial-kernel-path <PATH_TO_INIT_CUDA_FILE> \
    --compare-kind <KERNEL_TYPE> \
    --baseline-func <BASELINE_FUNCTION_NAME> \
    --generated-export-func <EXPORT_FUNCTION_NAME>
```

## Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--api-key` | Yes | - | OpenAI API key for authentication |
| `--initial-kernel-path` | Yes | - | Path to the initial CUDA kernel file (`.cu`) to optimize |
| `--max-iterations` | No | 5 | Maximum number of optimization iterations |
| `--baseline-module` | No | `sgl_kernel` | Python module containing the baseline function. If empty, automatically uses `sgl_kernel` or `sglang.sgl_kernel` |
| `--baseline-func` | No | `sgl_fused_add_rmsnorm` | Name of the baseline function to compare against |
| `--compare-kind` | No | `generic` | Comparison mode tag that guides test generation prompts (e.g., `rmsnorm`, `silu`, `mergestate`) |
| `--generated-export-func` | No | Same as `baseline-func` | Export function name expected in the generated PyBind module |

## Examples

### 1. RMSNorm Kernel Optimization

Optimize a fused add RMSNorm kernel:

```bash
python3 cuda_kernel_optimizer_multi.py \
    --api-key $OPENAI_API_KEY \
    --initial-kernel-path /path/to/rms_v1.cu \
    --compare-kind rmsnorm \
    --baseline-func fused_add_rmsnorm \
    --generated-export-func sgl_fused_add_rmsnorm
```

### 2. SiLU (Swish) Kernel Optimization

Optimize a SiLU activation with multiplication kernel:

```bash
python3 cuda_kernel_optimizer_multi.py \
    --api-key $OPENAI_API_KEY \
    --initial-kernel-path /path/to/silu_mul_v1.cu \
    --compare-kind silu \
    --baseline-func silu_and_mul \
    --generated-export-func sgl_silu_mul
```

### 3. Merge State Kernel Optimization

Optimize a merge state kernel (commonly used in attention mechanisms):

```bash
python3 cuda_kernel_optimizer_multi.py \
    --api-key $OPENAI_API_KEY \
    --initial-kernel-path /path/to/merge_v1.cu \
    --compare-kind mergestate \
    --baseline-func merge_state \
    --generated-export-func merge_state
```

## Output

Results are saved in a timestamped directory under `cuda_optimization_runs/`.

## Citation

If our research inspires you, please cite our paper:

```bibtex
@inproceedings{wei2025astra,
  title={Astra: A Multi-Agent System for GPU Kernel Performance Optimization},
  author={Wei, Anjiang and Sun, Tianran and Seenichamy, Yogesh and Song, Hang and Ouyang, Anne and Mirhoseini, Azalia and Wang, Ke and Aiken, Alex},
  booktitle={NeurIPS 2025 Fourth Workshop on Deep Learning for Code},
  year={2025}
}
```

## License

This project is licensed under the [Apache License 2.0](https://opensource.org/license/apache-2-0). 