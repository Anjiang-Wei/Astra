#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int BLOCK_SIZE = 256;

// Kernel 1: compute per-row inverse RMS (1 / sqrt(mean(x^2) + eps)), where x = input + residual.
// Each block processes one row (batch index b). Simple shared-memory reduction is used.
__global__ void compute_inv_rms_kernel(const float* __restrict__ input,
                                       const float* __restrict__ residual,
                                       int32_t batch_size,
                                       int32_t hidden_size,
                                       int32_t stride_input,
                                       int32_t stride_residual,
                                       float eps,
                                       float* __restrict__ inv_rms_out) {
  int b = blockIdx.x;
  int tid = threadIdx.x;
  if (b >= batch_size) return;

  const float* in_row  = input    + static_cast<int64_t>(b) * stride_input;
  const float* res_row = residual + static_cast<int64_t>(b) * stride_residual;

  // Parallel accumulation of squared values
  float sum = 0.f;
  for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
    float x = in_row[i] + res_row[i];
    sum += x * x;
  }

  __shared__ float sdata[BLOCK_SIZE];
  sdata[tid] = sum;
  __syncthreads();

  // In-block reduction
  for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      sdata[tid] += sdata[tid + offset];
    }
    __syncthreads();
  }

  // Compute inverse RMS for this row
  if (tid == 0) {
    float mean = sdata[0] / static_cast<float>(hidden_size);
    inv_rms_out[b] = rsqrtf(mean + eps);
  }
}

// Kernel 2: apply normalization and weight, write back to input (in-place on FP32 buffer).
// Each block processes one row (batch index b).
__global__ void apply_rms_weight_kernel(float* __restrict__ input_out,
                                        const float* __restrict__ residual,
                                        const float* __restrict__ weight,
                                        const float* __restrict__ inv_rms,
                                        int32_t batch_size,
                                        int32_t hidden_size,
                                        int32_t stride_input,
                                        int32_t stride_residual) {
  int b = blockIdx.x;
  int tid = threadIdx.x;
  if (b >= batch_size) return;

  float scale = inv_rms[b];
  float*       in_row  = input_out + static_cast<int64_t>(b) * stride_input;
  const float* res_row = residual  + static_cast<int64_t>(b) * stride_residual;

  for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
    float x = in_row[i] + res_row[i];     // residual add
    float y = x * scale * weight[i];      // RMS normalization and weight
    in_row[i] = y;                         // write back
  }
}

} // namespace

void sgl_fused_add_rmsnorm(torch::Tensor input,
                           torch::Tensor residual,
                           torch::Tensor weight,
                           double eps,
                           bool /*enable_pdl*/) {
  TORCH_CHECK(input.is_cuda(),    "input must be CUDA");
  TORCH_CHECK(residual.is_cuda(), "residual must be CUDA");
  TORCH_CHECK(weight.is_cuda(),   "weight must be CUDA");

  TORCH_CHECK(input.dim() == 2 && residual.dim() == 2, "input and residual must be (B, D)");
  TORCH_CHECK(weight.dim() == 1,                        "weight must be (D)");
  TORCH_CHECK(input.size(0) == residual.size(0), "batch size mismatch");
  TORCH_CHECK(input.size(1) == residual.size(1), "hidden size mismatch");
  TORCH_CHECK(input.size(1) == weight.size(0),   "weight size mismatch");

  const auto device = input.device();
  TORCH_CHECK(residual.device() == device, "residual must be on the same device as input");
  TORCH_CHECK(weight.device()   == device, "weight must be on the same device as input");

  const int32_t B = static_cast<int32_t>(input.size(0));
  const int32_t D = static_cast<int32_t>(input.size(1));

  // Cast to FP32 for computation (numerically stable, simpler code).
  auto input_f32    = input.contiguous().to(torch::kFloat);
  auto residual_f32 = residual.contiguous().to(torch::kFloat);
  auto weight_f32   = weight.contiguous().to(torch::kFloat);

  const int32_t stride_input    = static_cast<int32_t>(input_f32.stride(0));
  const int32_t stride_residual = static_cast<int32_t>(residual_f32.stride(0));

  // Temporary buffer: per-row inverse RMS
  auto inv_rms = torch::empty({B}, input_f32.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(B);
  dim3 block(BLOCK_SIZE);

  // Pass 1: compute inv_rms[b] for each row
  compute_inv_rms_kernel<<<grid, block, 0, stream>>>(
      input_f32.data_ptr<float>(),
      residual_f32.data_ptr<float>(),
      B, D,
      stride_input,
      stride_residual,
      static_cast<float>(eps),
      inv_rms.data_ptr<float>());

  // Pass 2: apply normalization and weight, write back to input_f32
  apply_rms_weight_kernel<<<grid, block, 0, stream>>>(
      input_f32.data_ptr<float>(),
      residual_f32.data_ptr<float>(),
      weight_f32.data_ptr<float>(),
      inv_rms.data_ptr<float>(),
      B, D,
      stride_input,
      stride_residual);

  // Copy result back to original dtype and storage of `input`
  input.copy_(input_f32.to(input.scalar_type()));
}

// PyBind11 module export to allow importing as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sgl_fused_add_rmsnorm", &sgl_fused_add_rmsnorm,
        "Simplified fused Add + RMSNorm (in-place on input)");
}