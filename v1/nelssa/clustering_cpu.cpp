#include <torch/extension.h>
#include <omp.h>
#include <cstring>

// CPU Multi-threaded KV Cache Reorganization
// Similar to RetrievalAttention's WaveBufferCPU::construct_func()
void reorganize_by_clusters_cpu(
    torch::Tensor keys_dst,
    torch::Tensor values_dst,
    const torch::Tensor keys_src,
    const torch::Tensor values_src,
    const torch::Tensor clusters,
    const torch::Tensor cluster_size,
    int num_kv_heads,
    int n_centroids
) {
    TORCH_CHECK(keys_dst.is_contiguous(), "keys_dst must be contiguous");
    TORCH_CHECK(values_dst.is_contiguous(), "values_dst must be contiguous");
    TORCH_CHECK(keys_src.is_contiguous(), "keys_src must be contiguous");
    TORCH_CHECK(values_src.is_contiguous(), "values_src must be contiguous");
    TORCH_CHECK(clusters.is_contiguous(), "clusters must be contiguous");
    TORCH_CHECK(cluster_size.is_contiguous(), "cluster_size must be contiguous");

    int head_dim = keys_src.size(2);

    // Get data pointers
    auto keys_dst_acc = keys_dst.accessor<at::Half, 3>();
    auto values_dst_acc = values_dst.accessor<at::Half, 3>();
    auto keys_src_acc = keys_src.accessor<at::Half, 3>();
    auto values_src_acc = values_src.accessor<at::Half, 3>();
    auto clusters_acc = clusters.accessor<int, 2>();
    auto cluster_size_acc = cluster_size.accessor<int, 2>();

    // Use OpenMP for parallel processing across KV heads
    #pragma omp parallel for
    for (int kv_head = 0; kv_head < num_kv_heads; kv_head++) {
        int start_idx = 0;

        for (int centroid_idx = 0; centroid_idx < n_centroids; centroid_idx++) {
            int size = cluster_size_acc[kv_head][centroid_idx];
            if (size <= 0) continue;

            // Vectorized memcpy for each token
            for (int j = 0; j < size; j++) {
                int src_token_idx = clusters_acc[kv_head][centroid_idx][j];

                // Use memcpy for efficient memory copy (similar to RetrievalAttention)
                std::memcpy(
                    keys_dst_acc[kv_head][start_idx + j],
                    keys_src_acc[kv_head][src_token_idx],
                    head_dim * sizeof(at::Half)
                );
                std::memcpy(
                    values_dst_acc[kv_head][start_idx + j],
                    values_src_acc[kv_head][src_token_idx],
                    head_dim * sizeof(at::Half)
                );
            }
            start_idx += size;
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reorganize_by_clusters_cpu", &reorganize_by_clusters_cpu,
          "Multi-threaded CPU KV Cache reorganization by clusters");
}