

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

// ----- math -----
__device__ __forceinline__ float silu_f(float x) {
  return x / (1.0f + expf(-x));
}

// ----- dtype convert helpers -----
template <typename T> __device__ __forceinline__ float to_float(T x);
template <> __device__ __forceinline__ float to_float<float>(float x) { return x; }
template <> __device__ __forceinline__ float to_float<at::Half>(at::Half x) {
  return __half2float(*reinterpret_cast<const __half*>(&x));
}
template <> __device__ __forceinline__ float to_float<at::BFloat16>(at::BFloat16 x) {
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&x));
}

template <typename T> __device__ __forceinline__ T from_float(float x);
template <> __device__ __forceinline__ float from_float<float>(float x) { return x; }
template <> __device__ __forceinline__ at::Half from_float<at::Half>(float x) {
  __half h = __float2half_rn(x);
  return *reinterpret_cast<at::Half*>(&h);
}
template <> __device__ __forceinline__ at::BFloat16 from_float<at::BFloat16>(float x) {
  __nv_bfloat16 b = __float2bfloat16(x);
  return *reinterpret_cast<at::BFloat16*>(&b);
}

// ----- kernel -----
template <typename scalar_t>
__global__ void silu_and_mul_concat_kernel(
    const scalar_t* __restrict__ in,   // [B, 2D]
    scalar_t* __restrict__ out,        // [B, D]
    int32_t B, int32_t D,
    int64_t in_s0,  int64_t in_s1,
    int64_t out_s0, int64_t out_s1)
{
  for (int32_t b = blockIdx.x; b < B; b += gridDim.x) {
    const scalar_t* row = in + b * in_s0;
    scalar_t* __restrict__ o = out + b * out_s0;
    const scalar_t* x_ptr = row;
    const scalar_t* g_ptr = row + D * in_s1;

    for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
      float xv = to_float(x_ptr[d * in_s1]);
      float gv = to_float(g_ptr[d * in_s1]);
      o[d * out_s1] = from_float<scalar_t>(silu_f(xv) * gv);
    }
  }
}

static inline int choose_blocks(int32_t B) {
  int max_blocks = 256;
  int blocks = B < max_blocks ? B : max_blocks;
  return blocks > 0 ? blocks : 1;
}

// API: input [B, 2D], output [B, D]
void sgl_silu_and_mul(torch::Tensor input, torch::Tensor output) {
  TORCH_CHECK(input.is_cuda() && output.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(input.dim() == 2 && output.dim() == 2, "input/out must be 2D");
  TORCH_CHECK(input.size(0) == output.size(0), "batch size mismatch");
  TORCH_CHECK(input.size(1) == 2 * output.size(1), "input.shape[-1] must be 2*output.shape[-1]");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(), "dtype mismatch");

  auto in  = input.contiguous();
  auto out = output.contiguous();

  const auto dt = in.scalar_type();
  const int32_t B = static_cast<int32_t>(in.size(0));
  const int32_t D = static_cast<int32_t>(out.size(1));

  const int64_t in_s0  = in.stride(0);
  const int64_t in_s1  = in.stride(1);
  const int64_t out_s0 = out.stride(0);
  const int64_t out_s1 = out.stride(1);

  const int blocks = choose_blocks(B);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (dt) {
    case torch::kFloat:
      silu_and_mul_concat_kernel<float><<<blocks, BLOCK_SIZE, 0, stream>>>(
          in.data_ptr<float>(), out.data_ptr<float>(),
          B, D, in_s0, in_s1, out_s0, out_s1);
      break;
    case torch::kHalf:
      silu_and_mul_concat_kernel<at::Half><<<blocks, BLOCK_SIZE, 0, stream>>>(
          in.data_ptr<at::Half>(), out.data_ptr<at::Half>(),
          B, D, in_s0, in_s1, out_s0, out_s1);
      break;
    case torch::kBFloat16:
      silu_and_mul_concat_kernel<at::BFloat16><<<blocks, BLOCK_SIZE, 0, stream>>>(
          in.data_ptr<at::BFloat16>(), out.data_ptr<at::BFloat16>(),
          B, D, in_s0, in_s1, out_s0, out_s1);
      break;
    default:
      TORCH_CHECK(false, "silu_and_mul: unsupported dtype");
  }

  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "silu_and_mul kernel launch failed");

  if (!output.is_contiguous() || output.data_ptr() != out.data_ptr()) {
    output.copy_(out);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sgl_silu_mul", &sgl_silu_and_mul, "SiLU(x) * gate with concat input ([B,2D] -> [B,D])");
}