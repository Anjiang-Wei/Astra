#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/util/Exception.h>
#include <cmath>
#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while (0)
#define CHECK_DIM(D, x) TORCH_CHECK((x).dim() == (D), #x " must have dim " #D)
#define CHECK_SHAPE(a, b) TORCH_CHECK((a).sizes() == (b).sizes(), #a " and " #b " must have the same shape")
#define CHECK_EQ(x, y) TORCH_CHECK((x) == (y), #x " must equal " #y)

__device__ __forceinline__ float to_float(float x){ return x; }
__device__ __forceinline__ float to_float(__half x){ return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x){ return __bfloat162float(x); }

template <typename T> __device__ __forceinline__ T cast_from_float(float x);
template <> __device__ __forceinline__ float cast_from_float<float>(float x){ return x; }
template <> __device__ __forceinline__ __half cast_from_float<__half>(float x){ return __float2half(x); }
template <> __device__ __forceinline__ __nv_bfloat16 cast_from_float<__nv_bfloat16>(float x){ return __float2bfloat16(x); }

template <typename T>
__global__ void merge_state_kernel_slow(
    const T* __restrict__ v_a,
    const float* __restrict__ s_a,
    const T* __restrict__ v_b,
    const float* __restrict__ s_b,
    T* __restrict__ v_out,
    float* __restrict__ s_out,
    uint32_t seq_len,
    uint32_t num_heads,
    uint32_t head_dim)
{
  const uint64_t NH = uint64_t(seq_len) * uint64_t(num_heads);
  const uint64_t NHD = NH * uint64_t(head_dim);
  uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t gstride = uint64_t(blockDim.x) * gridDim.x;

  for (uint64_t idx = gid; idx < NHD; idx += gstride) {
    const uint32_t sh = uint32_t(idx / head_dim);
    const uint32_t d  = uint32_t(idx % head_dim);

    const float sa = s_a[sh];
    const float sb = s_b[sh];
    const float smax = fmaxf(sa, sb);
    const float wa = exp2f(sa - smax);
    const float wb = exp2f(sb - smax);
    const float denom = wa + wb + 1e-12f;
    const float a_scale = wa / denom;
    const float b_scale = wb / denom;

    if (d == 0) {
      s_out[sh] = log2f(denom) + smax;
    }

    const float va = to_float(v_a[idx]);
    const float vb = to_float(v_b[idx]);
    v_out[idx] = cast_from_float<T>(a_scale * va + b_scale * vb);
  }
}

void merge_state(at::Tensor v_a, at::Tensor s_a,
                 at::Tensor v_b, at::Tensor s_b,
                 at::Tensor v_out, at::Tensor s_out)
{
  CHECK_INPUT(v_a);
  CHECK_INPUT(s_a);
  CHECK_INPUT(v_b);
  CHECK_INPUT(s_b);
  auto device = v_a.device();
  CHECK_EQ(s_a.device(), device);
  CHECK_EQ(v_b.device(), device);
  CHECK_EQ(s_b.device(), device);
  CHECK_DIM(3, v_a);
  CHECK_DIM(2, s_a);
  CHECK_DIM(3, v_b);
  CHECK_DIM(2, s_b);
  CHECK_SHAPE(v_a, v_b);
  CHECK_SHAPE(s_a, s_b);
  CHECK_EQ(v_a.size(0), s_a.size(0));
  CHECK_EQ(v_a.size(1), s_b.size(1));
  CHECK_SHAPE(v_a, v_out);
  CHECK_SHAPE(s_a, s_out);

  const uint32_t seq_len   = (uint32_t)v_a.size(0);
  const uint32_t num_heads = (uint32_t)v_a.size(1);
  const uint32_t head_dim  = (uint32_t)v_a.size(2);

  c10::cuda::OptionalCUDAGuard guard(v_a.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const uint64_t total = (uint64_t)seq_len * num_heads * head_dim;
  const uint32_t threads = 64;
  const uint32_t blocks = (uint32_t)((total + threads - 1) / threads);

  bool ok = false;
  if (v_a.scalar_type() == at::kFloat) {
    merge_state_kernel_slow<float><<<blocks, threads, 0, stream>>>(
      v_a.data_ptr<float>(),
      s_a.data_ptr<float>(),
      v_b.data_ptr<float>(),
      s_b.data_ptr<float>(),
      v_out.data_ptr<float>(),
      s_out.data_ptr<float>(),
      seq_len, num_heads, head_dim);
    ok = true;
  } else if (v_a.scalar_type() == at::kHalf) {
    merge_state_kernel_slow<__half><<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __half*>(v_a.data_ptr()),
      s_a.data_ptr<float>(),
      reinterpret_cast<const __half*>(v_b.data_ptr()),
      s_b.data_ptr<float>(),
      reinterpret_cast<__half*>(v_out.data_ptr()),
      s_out.data_ptr<float>(),
      seq_len, num_heads, head_dim);
    ok = true;
  } else if (v_a.scalar_type() == at::kBFloat16) {
    merge_state_kernel_slow<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(v_a.data_ptr()),
      s_a.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(v_b.data_ptr()),
      s_b.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(v_out.data_ptr()),
      s_out.data_ptr<float>(),
      seq_len, num_heads, head_dim);
    ok = true;
  }

  auto err = cudaGetLastError();
  TORCH_CHECK(ok && err == cudaSuccess, "merge_state kernel launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("merge_state", &merge_state, "merge_state (slow)");
}