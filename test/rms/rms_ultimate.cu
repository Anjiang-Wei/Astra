#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

constexpr int BLOCK_SIZE = 512;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;

__global__ void fused_add_rmsnorm_kernel(
    float* __restrict__ input,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    int32_t batch_size,
    int32_t hidden_size,
    int32_t stride_input,
    int32_t stride_residual,
    float eps) {
  int b = blockIdx.x + blockIdx.y * gridDim.x;
  if (b >= batch_size) return;
  const float* in_row = input + int64_t(b) * stride_input;
  const float* res_row = residual + int64_t(b) * stride_residual;
  float* out_row = input + int64_t(b) * stride_input;

  bool can_vec4 = (((uintptr_t)in_row & 0xF) == 0) && (hidden_size % 4 == 0);
  bool can_vec2 = !can_vec4 && (((uintptr_t)in_row & 0x7) == 0) && (hidden_size % 2 == 0);
  int tid = threadIdx.x;
  float sum = 0.0f;

  // First pass: sum of squares
  if (can_vec4) {
    int vec_size = hidden_size >> 2;
    const float4* in4 = reinterpret_cast<const float4*>(in_row);
    const float4* res4 = reinterpret_cast<const float4*>(res_row);
    #pragma unroll 8
    for (int i = tid; i < vec_size; i += BLOCK_SIZE) {
      float4 a = __ldg(&in4[i]);
      float4 r = __ldg(&res4[i]);
      float4 x;
      x.x = a.x + r.x; x.y = a.y + r.y;
      x.z = a.z + r.z; x.w = a.w + r.w;
      sum = fmaf(x.x, x.x, sum);
      sum = fmaf(x.y, x.y, sum);
      sum = fmaf(x.z, x.z, sum);
      sum = fmaf(x.w, x.w, sum);
    }
  } else if (can_vec2) {
    int vec_size = hidden_size >> 1;
    const float2* in2 = reinterpret_cast<const float2*>(in_row);
    const float2* res2 = reinterpret_cast<const float2*>(res_row);
    #pragma unroll 8
    for (int i = tid; i < vec_size; i += BLOCK_SIZE) {
      float2 a = __ldg(&in2[i]);
      float2 r = __ldg(&res2[i]);
      float2 x;
      x.x = a.x + r.x; x.y = a.y + r.y;
      sum = fmaf(x.x, x.x, sum);
      sum = fmaf(x.y, x.y, sum);
    }
  } else {
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
      float x = __ldg(&in_row[i]) + __ldg(&res_row[i]);
      sum = fmaf(x, x, sum);
    }
  }

  // Warp-level reduction
  unsigned mask = 0xffffffffu;
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(mask, sum, offset);
  }
  __shared__ float smem[WARPS_PER_BLOCK];
  int warp_id = tid >> 5;
  int lane = tid & 31;
  if (lane == 0) smem[warp_id] = sum;
  __syncthreads();

  // Compute inverse RMS scale
  if (tid == 0) {
    float block_sum = 0.0f;
    for (int i = 0; i < WARPS_PER_BLOCK; ++i) block_sum += smem[i];
    smem[0] = rsqrtf(block_sum / hidden_size + eps);
  }
  __syncthreads();
  float scale = smem[0];

  // Second pass: normalize, weight and write
  if (can_vec4) {
    int vec_size = hidden_size >> 2;
    float4* out4 = reinterpret_cast<float4*>(out_row);
    const float4* in4 = reinterpret_cast<const float4*>(in_row);
    const float4* res4 = reinterpret_cast<const float4*>(res_row);
    #pragma unroll 4
    for (int i = tid; i < vec_size; i += BLOCK_SIZE) {
      float4 a = __ldg(&in4[i]);
      float4 r = __ldg(&res4[i]);
      float4 x; x.x = a.x + r.x; x.y = a.y + r.y;
               x.z = a.z + r.z; x.w = a.w + r.w;
      int idx4 = i << 2;
      float4 w = make_float4(__ldg(&weight[idx4]), __ldg(&weight[idx4 + 1]),
                              __ldg(&weight[idx4 + 2]), __ldg(&weight[idx4 + 3]));
      float4 y;
      y.x = fmaf(x.x, scale * w.x, 0.0f);
      y.y = fmaf(x.y, scale * w.y, 0.0f);
      y.z = fmaf(x.z, scale * w.z, 0.0f);
      y.w = fmaf(x.w, scale * w.w, 0.0f);
      out4[i] = y;
    }
  } else if (can_vec2) {
    int vec_size = hidden_size >> 1;
    float2* out2 = reinterpret_cast<float2*>(out_row);
    const float2* in2 = reinterpret_cast<const float2*>(in_row);
    const float2* res2 = reinterpret_cast<const float2*>(res_row);
    #pragma unroll 4
    for (int i = tid; i < vec_size; i += BLOCK_SIZE) {
      float2 a = __ldg(&in2[i]);
      float2 r = __ldg(&res2[i]);
      float2 x; x.x = a.x + r.x; x.y = a.y + r.y;
      int idx2 = i << 1;
      float2 w; w.x = __ldg(&weight[idx2]); w.y = __ldg(&weight[idx2 + 1]);
      float2 y;
      y.x = fmaf(x.x, scale * w.x, 0.0f);
      y.y = fmaf(x.y, scale * w.y, 0.0f);
      out2[i] = y;
    }
  } else {
    __shared__ float smem_w[BLOCK_SIZE];
    for (int tile_start = 0; tile_start < hidden_size; tile_start += BLOCK_SIZE) {
      int tile_size = min(BLOCK_SIZE, hidden_size - tile_start);
      for (int i = tid; i < tile_size; i += BLOCK_SIZE) {
        smem_w[i] = __ldg(&weight[tile_start + i]);
      }
      __syncthreads();
      for (int i = tid; i < tile_size; i += BLOCK_SIZE) {
        float x = __ldg(&in_row[tile_start + i]) + __ldg(&res_row[tile_start + i]);
        out_row[tile_start + i] = fmaf(x, scale * smem_w[i], 0.0f);
      }
      __syncthreads();
    }
  }
}

void sgl_fused_add_rmsnorm(torch::Tensor input,
                           torch::Tensor residual,
                           torch::Tensor weight,
                           double eps,
                           bool /*enable_pdl*/) {
  TORCH_CHECK(input.is_cuda() && residual.is_cuda() && weight.is_cuda(), "All inputs must be CUDA");
  TORCH_CHECK(input.dim() == 2 && residual.dim() == 2, "input and residual must be 2-D");
  TORCH_CHECK(weight.dim() == 1, "weight must be 1-D");
  TORCH_CHECK(input.size(0) == residual.size(0) && input.size(1) == residual.size(1), "Mismatch sizes");
  TORCH_CHECK(input.size(1) == weight.size(0), "weight size mismatch");
  auto input_f32 = input.contiguous().to(torch::kFloat);
  auto residual_f32 = residual.contiguous().to(torch::kFloat);
  auto weight_f32 = weight.contiguous().to(torch::kFloat);
  int32_t B = static_cast<int32_t>(input_f32.size(0));
  int32_t D = static_cast<int32_t>(input_f32.size(1));
  int32_t stride_in = static_cast<int32_t>(input_f32.stride(0));
  int32_t stride_res = static_cast<int32_t>(residual_f32.stride(0));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  uint32_t max_x = 65535;
  uint32_t gx = B > max_x ? max_x : B;
  uint32_t gy = (B + gx - 1) / gx;
  dim3 grid(gx, gy);
  dim3 block(BLOCK_SIZE);
  fused_add_rmsnorm_kernel<<<grid, block, 0, stream>>>(
    input_f32.data_ptr<float>(),
    residual_f32.data_ptr<float>(),
    weight_f32.data_ptr<float>(),
    B, D, stride_in, stride_res, static_cast<float>(eps));
  input.copy_(input_f32.to(input.scalar_type()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sgl_fused_add_rmsnorm", &sgl_fused_add_rmsnorm,
        "Fused Add + RMSNorm (in-place)");
}