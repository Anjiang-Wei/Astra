#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// Fast math SiLU for float
__device__ __forceinline__ float silu_fastf(float x) {
    float y = __expf(-x);
    float r = __frcp_rn(1.0f + y);
    return __fmul_rn(x, r);
}

// Scalar converters for half and bfloat16
__device__ __forceinline__ float to_float16(at::Half h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
}
__device__ __forceinline__ at::Half from_float16(float x) {
    __half h = __float2half_rn(x);
    return *reinterpret_cast<at::Half*>(&h);
}
__device__ __forceinline__ float to_float_bf16(at::BFloat16 b) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&b));
}
__device__ __forceinline__ at::BFloat16 from_float_bf16(float x) {
    __nv_bfloat16 b = __float2bfloat16(x);
    return *reinterpret_cast<at::BFloat16*>(&b);
}

// Float kernel with float4 vectorization
__global__ __launch_bounds__(BLOCK_SIZE,4)
void silu_mul_float_kernel(const float* __restrict__ __align__(16) input,
                           float* __restrict__ __align__(16) output,
                           int32_t B, int32_t D) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    const int W = 4;
    int total_vec = D / W;
    int vec_idx = bx * BLOCK_SIZE + tx;
    const float* row_in = input + by * (2 * D);
    float* row_out = output + by * D;
    const float4* x4 = reinterpret_cast<const float4*>(row_in);
    const float4* g4 = reinterpret_cast<const float4*>(row_in + D);
    float4* o4 = reinterpret_cast<float4*>(row_out);
    if (vec_idx < total_vec) {
        float4 xv = x4[vec_idx];
        float4 gv = g4[vec_idx];
        float4 ov;
        #pragma unroll
        for (int i = 0; i < W; i++) {
            float x = ((&xv.x)[i]);
            float g = ((&gv.x)[i]);
            ((&ov.x)[i]) = silu_fastf(x) * g;
        }
        o4[vec_idx] = ov;
    }
    // tail elements
    if (bx == gridDim.x - 1) {
        int d = total_vec * W + tx;
        if (d < D) {
            float x = row_in[d];
            float g = row_in[D + d];
            row_out[d] = silu_fastf(x) * g;
        }
    }
}

// Half2 kernel
__global__ __launch_bounds__(BLOCK_SIZE,4)
void silu_mul_half_kernel(const at::Half* __restrict__ __align__(8) input,
                          at::Half* __restrict__ __align__(8) output,
                          int32_t B, int32_t D) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    const int W = 2;
    int total_vec = D / W;
    int vec_idx = bx * BLOCK_SIZE + tx;
    const at::Half* row_in = input + by * (2 * D);
    at::Half* row_out = output + by * D;
    const __half2* x2 = reinterpret_cast<const __half2*>(row_in);
    const __half2* g2 = reinterpret_cast<const __half2*>(row_in + D);
    __half2* o2 = reinterpret_cast<__half2*>(row_out);
    if (vec_idx < total_vec) {
        __half2 xv2 = x2[vec_idx];
        __half2 gv2 = g2[vec_idx];
        float2 xv_f2 = __half22float2(xv2);
        float2 gv_f2 = __half22float2(gv2);
        float2 ov_f2;
        ov_f2.x = silu_fastf(xv_f2.x) * gv_f2.x;
        ov_f2.y = silu_fastf(xv_f2.y) * gv_f2.y;
        o2[vec_idx] = __float22half2_rn(ov_f2);
    }
    if (bx == gridDim.x - 1) {
        int d = total_vec * W + tx;
        if (d < D) {
            float x = to_float16(row_in[d]);
            float g = to_float16(row_in[D + d]);
            row_out[d] = from_float16(silu_fastf(x) * g);
        }
    }
}

// Scalar BFloat16 kernel
__global__ __launch_bounds__(BLOCK_SIZE,4)
void silu_mul_bf16_kernel(const at::BFloat16* __restrict__ input,
                           at::BFloat16* __restrict__ output,
                           int32_t B, int32_t D) {
    int b = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    const at::BFloat16* row_in = input + b * (2 * D);
    at::BFloat16* row_out = output + b * D;
    for (int d = idx; d < D; d += stride) {
        float x = to_float_bf16(row_in[d]);
        float g = to_float_bf16(row_in[D + d]);
        row_out[d] = from_float_bf16(silu_fastf(x) * g);
    }
}

// Compute grid.x based on vector width
static inline int grid_x(int D, int W) {
    int total_vec = (D + W - 1) / W;
    return (total_vec + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

// Exported function
void sgl_silu_and_mul(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.is_cuda() && output.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(input.dim()==2 && output.dim()==2, "input/out must be 2D");
    TORCH_CHECK(input.size(0)==output.size(0), "batch size mismatch");
    TORCH_CHECK(input.size(1)==2*output.size(1), "input.shape[-1] must be 2*output.shape[-1]");
    TORCH_CHECK(input.scalar_type()==output.scalar_type(), "dtype mismatch");
    auto in = input.contiguous();
    auto out = output.contiguous();
    int32_t B = in.size(0);
    int32_t D = out.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 grid;
    grid.y = B;
    switch (in.scalar_type()) {
        case torch::kFloat: {
            grid.x = grid_x(D, 4);
            silu_mul_float_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(in.data_ptr<float>(), out.data_ptr<float>(), B, D);
            break;
        }
        case torch::kHalf: {
            grid.x = grid_x(D, 2);
            silu_mul_half_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(in.data_ptr<at::Half>(), out.data_ptr<at::Half>(), B, D);
            break;
        }
        case torch::kBFloat16: {
            grid.x = (D + BLOCK_SIZE - 1) / BLOCK_SIZE;
            silu_mul_bf16_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(in.data_ptr<at::BFloat16>(), out.data_ptr<at::BFloat16>(), B, D);
            break;
        }
        default:
            TORCH_CHECK(false, "silu_and_mul: unsupported dtype");
    }
    TORCH_CHECK(cudaGetLastError()==cudaSuccess, "silu_and_mul kernel launch failed");
    if (!output.is_contiguous() || output.data_ptr()!=out.data_ptr()) {
        output.copy_(out);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgl_silu_mul", &sgl_silu_and_mul, "SiLU(x) * gate with concat input ([B,2D] -> [B,D])");
}