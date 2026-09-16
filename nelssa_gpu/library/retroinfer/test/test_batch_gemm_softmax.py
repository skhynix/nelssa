import torch
import math
from retroinfer_kernels import batch_gemm_softmax
import time

DTYPE = torch.bfloat16


def torch_batch_gemm_softmax(A, B):
    """
    A: query vector, shape: (batch_size*group_num, group_size, dim), gpu torch tensor
    B: centroids vector, shape: (batch_size*group_num, n_clusters, dim), gpu torch tensor
    """
    # [batch_size*group_num, group_size, n_centroids]
    dist = torch.bmm(
        A, B.transpose(1, 2)
    )  # [batch_size*group_num, group_size, n_centroids]
    dist = dist / math.sqrt(A.size(-1))
    dist = torch.softmax(
        dist, dim=-1
    )  # [batch_size*group_num, group_size, n_centroids]
    return dist


def test_batch_gemm_softmax():
    batch_size = 16
    group_num = 8
    group_size = 4
    n_clusters = 8200  # must be multiple of 8
    dim = 128

    queries = torch.randn(
        (batch_size * group_num, group_size, dim), device="cuda", dtype=DTYPE
    ).contiguous()
    centroids = torch.randn(
        (batch_size * group_num, n_clusters, dim), device="cuda", dtype=DTYPE
    ).contiguous()

    # buffer
    gemm_o = torch.zeros(
        (batch_size, group_num, group_size, n_clusters), device="cuda", dtype=DTYPE
    ).contiguous()
    softmax_o = torch.zeros(
        (batch_size * group_num, group_size, n_clusters), device="cuda", dtype=DTYPE
    ).contiguous()
    n_clusters_256 = (n_clusters + 256 - 1) // 256
    _norm = torch.zeros(
        (batch_size * group_num, group_size, n_clusters_256),
        device="cuda",
        dtype=torch.float32,
    ).contiguous()
    _sum = torch.zeros(
        (batch_size * group_num, group_size, n_clusters_256),
        device="cuda",
        dtype=torch.float32,
    ).contiguous()
    torch.cuda.synchronize()

    start = time.time()
    batch_gemm_softmax(
        queries,
        centroids,
        gemm_o,
        _norm,
        _sum,
        softmax_o,
        batch_size * group_num,
        group_size,
        n_clusters,
        dim,
        1 / math.sqrt(dim),
        0,
    )
    torch.cuda.synchronize()
    print(f"cuda time {1000 * (time.time() - start):.4f} ms")

    start = time.time()
    softmax_ref = torch_batch_gemm_softmax(queries, centroids)
    torch.cuda.synchronize()
    print(f"torch time {1000 * (time.time() - start):.4f} ms")

    assert softmax_o.shape == softmax_ref.shape, (
        f"shape mismatch: {softmax_o.shape.shape} vs {softmax_ref.shape.shape}"
    )
    assert torch.allclose(softmax_o, softmax_ref, atol=1e-3), (
        f"{torch.max(torch.abs(softmax_o - softmax_ref))}"
    )


if __name__ == "__main__":
    for _ in range(10):
        test_batch_gemm_softmax()
        print("pass")
