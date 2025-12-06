#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/util/Exception.h>
#include <type_traits>
#include <cmath>
#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while (0)
#define CHECK_DIM(D, x) TORCH_CHECK((x).dim() == (D), #x " must have dim " #D)
#define CHECK_SHAPE(a, b) TORCH_CHECK((a).sizes() == (b).sizes(), #a " and " #b " must have the same shape")
#define CHECK_EQ(x, y) TORCH_CHECK((x) == (y), #x " must equal " #y)

static const int WARP_SIZE = 32;

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T> __device__ __forceinline__ T cast_from_float(float x);
template <> __device__ __forceinline__ float cast_from_float<float>(float x) { return x; }
template <> __device__ __forceinline__ __half cast_from_float<__half>(float x) { return __float2half_rn(x); }
template <> __device__ __forceinline__ __nv_bfloat16 cast_from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// Generic kernel with vectorized loads/stores
template <typename T>
__global__ void merge_state_kernel_v6(
    const T* __restrict__ v_a,
    const float* __restrict__ s_a,
    const T* __restrict__ v_b,
    const float* __restrict__ s_b,
    T* __restrict__ v_out,
    float* __restrict__ s_out,
    uint32_t seq_len,
    uint32_t num_heads,
    uint32_t head_dim) {
  uint32_t seq = blockIdx.y;
  uint32_t head = blockIdx.z;
  uint32_t sh = seq * num_heads + head;
  float sa = s_a[sh];
  float sb = s_b[sh];
  float smax = fmaxf(sa, sb);
  float wa = exp2f(sa - smax);
  float wb = exp2f(sb - smax);
  float denom = wa + wb + 1e-12f;
  float inv_den = __fdividef(1.0f, denom);
  wa *= inv_den;
  wb *= inv_den;
  if (threadIdx.x == 0) {
    s_out[sh] = log2f(denom) + smax;
  }
  uint64_t stride_head_seq = (uint64_t)num_heads * head_dim;
  uint64_t stride_head = head_dim;
  uint64_t base = (uint64_t)seq * stride_head_seq + (uint64_t)head * stride_head;

  if constexpr (std::is_same<T, float>::value) {
    uint32_t vec4_len = head_dim / 4;
    auto va4 = reinterpret_cast<const float4*>(v_a + base);
    auto vb4 = reinterpret_cast<const float4*>(v_b + base);
    auto vout4 = reinterpret_cast<float4*>(v_out + base);
    #pragma unroll 4
    for (uint32_t i = threadIdx.x; i < vec4_len; i += blockDim.x) {
      float4 a = va4[i];
      float4 b = vb4[i];
      float4 r;
      r.x = fmaf(wa, a.x, wb * b.x);
      r.y = fmaf(wa, a.y, wb * b.y);
      r.z = fmaf(wa, a.z, wb * b.z);
      r.w = fmaf(wa, a.w, wb * b.w);
      vout4[i] = r;
    }
    for (uint32_t d = vec4_len * 4 + threadIdx.x; d < head_dim; d += blockDim.x) {
      float va_v = to_float(v_a[base + d]);
      float vb_v = to_float(v_b[base + d]);
      float v = fmaf(wa, va_v, wb * vb_v);
      v_out[base + d] = cast_from_float<T>(v);
    }
  } else if constexpr (std::is_same<T, __half>::value) {
    if ((head_dim & 1) == 0) {
      uint32_t vec2_len = head_dim / 2;
      auto va2 = reinterpret_cast<const __half2*>(v_a + base);
      auto vb2 = reinterpret_cast<const __half2*>(v_b + base);
      auto vout2 = reinterpret_cast<__half2*>(v_out + base);
      #pragma unroll 2
      for (uint32_t i = threadIdx.x; i < vec2_len; i += blockDim.x) {
        float2 a_f2 = __half22float2(va2[i]);
        float2 b_f2 = __half22float2(vb2[i]);
        float2 r_f2;
        r_f2.x = fmaf(wa, a_f2.x, wb * b_f2.x);
        r_f2.y = fmaf(wa, a_f2.y, wb * b_f2.y);
        __half h0 = __float2half_rn(r_f2.x);
        __half h1 = __float2half_rn(r_f2.y);
        vout2[i] = __halves2half2(h0, h1);
      }
      for (uint32_t d = vec2_len * 2 + threadIdx.x; d < head_dim; d += blockDim.x) {
        float va_v = to_float(v_a[base + d]);
        float vb_v = to_float(v_b[base + d]);
        float v = fmaf(wa, va_v, wb * vb_v);
        v_out[base + d] = cast_from_float<T>(v);
      }
    } else {
      for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float va_v = to_float(v_a[base + d]);
        float vb_v = to_float(v_b[base + d]);
        float v = fmaf(wa, va_v, wb * vb_v);
        v_out[base + d] = cast_from_float<T>(v);
      }
    }
  } else {
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
      float va_v = to_float(v_a[base + d]);
      float vb_v = to_float(v_b[base + d]);
      float v = fmaf(wa, va_v, wb * vb_v);
      v_out[base + d] = cast_from_float<T>(v);
    }
  }
}

// Warp-only kernel for small head_dim (<32)
template <typename T>
__global__ __launch_bounds__(32, 1)
void merge_state_warp_kernel_v6(
    const T* __restrict__ v_a,
    const float* __restrict__ s_a,
    const T* __restrict__ v_b,
    const float* __restrict__ s_b,
    T* __restrict__ v_out,
    float* __restrict__ s_out,
    uint32_t seq_len,
    uint32_t num_heads,
    uint32_t head_dim) {
  uint32_t seq = blockIdx.y;
  uint32_t head = blockIdx.z;
  uint32_t sh = seq * num_heads + head;
  uint32_t lane = threadIdx.x;
  float sa = s_a[sh];
  float sb = s_b[sh];
  float smax = fmaxf(sa, sb);
  float wa = exp2f(sa - smax);
  float wb = exp2f(sb - smax);
  float denom = wa + wb + 1e-12f;
  float inv_den = __fdividef(1.0f, denom);
  wa *= inv_den;
  wb *= inv_den;
  if (lane == 0) {
    s_out[sh] = log2f(denom) + smax;
  }
  if (lane < head_dim) {
    uint64_t idx = (uint64_t)seq * num_heads * head_dim + (uint64_t)head * head_dim + lane;
    float va_v = to_float(v_a[idx]);
    float vb_v = to_float(v_b[idx]);
    float v = fmaf(wa, va_v, wb * vb_v);
    v_out[idx] = cast_from_float<T>(v);
  }
}

void merge_state(at::Tensor v_a, at::Tensor s_a,
                 at::Tensor v_b, at::Tensor s_b,
                 at::Tensor v_out, at::Tensor s_out) {
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

  uint32_t threads = (head_dim < WARP_SIZE ? WARP_SIZE : std::min<uint32_t>(1024, ((head_dim + 31)/32)*32));
  dim3 block(threads, 1, 1);
  dim3 grid(1, seq_len, num_heads);

  bool ok = false;
  if (v_a.scalar_type() == at::kFloat) {
    if (head_dim < WARP_SIZE) {
      merge_state_warp_kernel_v6<float><<<grid, block, 0, stream>>>(
        v_a.data_ptr<float>(), s_a.data_ptr<float>(),
        v_b.data_ptr<float>(), s_b.data_ptr<float>(),
        v_out.data_ptr<float>(), s_out.data_ptr<float>(),
        seq_len, num_heads, head_dim);
    } else {
      merge_state_kernel_v6<float><<<grid, block, 0, stream>>>(
        v_a.data_ptr<float>(), s_a.data_ptr<float>(),
        v_b.data_ptr<float>(), s_b.data_ptr<float>(),
        v_out.data_ptr<float>(), s_out.data_ptr<float>(),
        seq_len, num_heads, head_dim);
    }
    ok = true;
  } else if (v_a.scalar_type() == at::kHalf) {
    if (head_dim < WARP_SIZE) {
      merge_state_warp_kernel_v6<__half><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(v_a.data_ptr()), s_a.data_ptr<float>(),
        reinterpret_cast<const __half*>(v_b.data_ptr()), s_b.data_ptr<float>(),
        reinterpret_cast<__half*>(v_out.data_ptr()), s_out.data_ptr<float>(),
        seq_len, num_heads, head_dim);
    } else {
      merge_state_kernel_v6<__half><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(v_a.data_ptr()), s_a.data_ptr<float>(),
        reinterpret_cast<const __half*>(v_b.data_ptr()), s_b.data_ptr<float>(),
        reinterpret_cast<__half*>(v_out.data_ptr()), s_out.data_ptr<float>(),
        seq_len, num_heads, head_dim);
    }
    ok = true;
  } else if (v_a.scalar_type() == at::kBFloat16) {
    if (head_dim < WARP_SIZE) {
      merge_state_warp_kernel_v6<__nv_bfloat16><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(v_a.data_ptr()), s_a.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(v_b.data_ptr()), s_b.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(v_out.data_ptr()), s_out.data_ptr<float>(),
        seq_len, num_heads, head_dim);
    } else {
      merge_state_kernel_v6<__nv_bfloat16><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(v_a.data_ptr()), s_a.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(v_b.data_ptr()), s_b.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(v_out.data_ptr()), s_out.data_ptr<float>(),
        seq_len, num_heads, head_dim);
    }
    ok = true;
  }

  auto err = cudaGetLastError();
  TORCH_CHECK(ok && err == cudaSuccess, "merge_state kernel launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("merge_state", &merge_state, "merge_state (v6)");
}