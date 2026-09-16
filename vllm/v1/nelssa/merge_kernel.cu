// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// NELSSA fused LSE merge kernel (Phase 2). Replaces the ~14 torch ops of
// merge_with_mask_static with a single CUDA kernel that updates gpu_output
// in place.
//
// Grid = (M, H) blocks, block = D threads: each block handles one (long-slot,
// head) pair, each thread one head-dim element. Every thread recomputes the
// 2-scalar LSE weights (cheap, L2-cached) and forms the weighted sum — no
// shared memory or sync, keeping the tiny kernel low-latency.
//
// Self-restore: pad slots (valid[m]==0) get w_gpu=1, w_cpu=0, so the merged
// value equals the original gpu_output element -> writing it back is a no-op.
// Pad slots point at non-long rows, so no other row is corrupted.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <limits>

__global__ void merge_lse_kernel(
    __nv_bfloat16* gpu_out,        // [num_reqs, H, D] bf16 (in-place)
    const float* gpu_lse,          // [num_reqs, H, 1] f32
    const __nv_bfloat16* cpu_out,  // [M, H, D] bf16 (pad)
    const float* cpu_lse,          // [M, H, 1] f32 (pad)
    const int64_t* indices,        // [M] int64
    const float* valid,           // [M] f32 (1=valid, 0=pad)
    int H, int D) {
  const int m = blockIdx.x;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t row = indices[m];

  // LSE weights over gpu_lse and cpu_lse (pad slots -> cpu_lse = -inf -> w_cpu=0).
  const float g_lse = gpu_lse[row * H + h];
  float c_lse = (valid[m] > 0.0f) ? cpu_lse[m * H + h] : -std::numeric_limits<float>::infinity();
  const float mx = fmaxf(g_lse, c_lse);
  const float s = __expf(g_lse - mx) + __expf(c_lse - mx);
  const float w_gpu = (s > 0.0f) ? __expf(g_lse - mx) / s : 1.0f;
  const float w_cpu = (s > 0.0f) ? __expf(c_lse - mx) / s : 0.0f;

  // Weighted sum, one element per thread, written back in place.
  const int64_t g_off = (row * H + h) * D + tid;
  const int64_t c_off = (m * H + h) * D + tid;
  gpu_out[g_off] = __float2bfloat16_rn(__bfloat162float(gpu_out[g_off]) * w_gpu + __bfloat162float(cpu_out[c_off]) * w_cpu);
}

// Host entry. Called from Python as vllm_merge_ext.merge_lse(...).
void merge_lse(
    torch::Tensor gpu_out,   // [num_reqs, H, D] bf16 GPU (in-place)
    torch::Tensor gpu_lse,   // [num_reqs, H, 1] f32 GPU
    torch::Tensor cpu_out,   // [M, H, D] bf16 GPU (pad)
    torch::Tensor cpu_lse,   // [M, H, 1] f32 GPU (pad)
    torch::Tensor indices,   // [M] int64 GPU
    torch::Tensor valid) {    // [M] f32 GPU (1=valid, 0=pad)
  TORCH_CHECK(gpu_out.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(gpu_out.dim() == 3 && gpu_out.dtype() == torch::kBFloat16,
              "gpu_out must be [N, H, D] bf16");
  TORCH_CHECK(gpu_lse.size(2) == 1 && gpu_lse.dtype() == torch::kFloat32,
              "gpu_lse must be [N, H, 1] f32");
  TORCH_CHECK(cpu_out.dtype() == torch::kBFloat16, "cpu_out must be bf16");
  TORCH_CHECK(cpu_lse.size(2) == 1 && cpu_lse.dtype() == torch::kFloat32,
              "cpu_lse must be [M, H, 1] f32");
  TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
  TORCH_CHECK(valid.dtype() == torch::kFloat32, "valid must be f32");

  const int M = static_cast<int>(indices.size(0));
  const int H = static_cast<int>(gpu_out.size(1));
  const int D = static_cast<int>(gpu_out.size(2));
  TORCH_CHECK(D <= 1024, "head_dim must be <= 1024 (block dim limit)");

  merge_lse_kernel<<<dim3(M, H), D, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<__nv_bfloat16*>(gpu_out.data_ptr<at::BFloat16>()),
      gpu_lse.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(cpu_out.data_ptr<at::BFloat16>()),
      cpu_lse.data_ptr<float>(),
      indices.data_ptr<int64_t>(),
      valid.data_ptr<float>(),
      H, D);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("merge_lse", &merge_lse,
        "NELSSA fused LSE merge: updates gpu_output in place using LSE weights "
        "from gpu_lse / cpu_lse. Pad slots (valid==0) self-restore.");
}
