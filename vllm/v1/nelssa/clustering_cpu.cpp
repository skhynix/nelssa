#include <torch/extension.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <nvToolsExt.h>
#include <omp.h>
#include <sched.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>

// ============================================================================
// NELSSA similarity-search NVTX sub-ranges (profiling only).
// Brackets each Prepare step with an NVTX range (gated by NELSSA_NVTX=1) to
// localize which step creates CPU gaps between kernels. Zero overhead when
// NELSSA_NVTX is unset (the bool is read once and cached).
// ============================================================================
namespace {
bool _sim_nvtx_on() {
    static const bool on = (std::getenv("NELSSA_NVTX") != nullptr
                            && std::string(std::getenv("NELSSA_NVTX")) == "1");
    return on;
}
// Push/pop an NVTX range directly (not RAII) so a range can span across code
// that needs locals declared before it (e.g. num_long is set in pass1 and read
// in every later step). Pair each _sim_nvtx_push with a _sim_nvtx_pop.
void _sim_nvtx_push(const char* msg) { if (_sim_nvtx_on()) nvtxRangePushA(msg); }
void _sim_nvtx_pop() { if (_sim_nvtx_on()) nvtxRangePop(); }

// ---- CORELOG: shared worker-core mapping probe (NELSSA_ATTN_CORELOG=1) ----
// Verifies OMP/intra-op workers get a 1:1 thread-to-core spread and measures
// thread-count overhead (1- vs 8-thread, cold vs warm). Each call logs the
// caller's + every OMP worker's core/affinity once per (tag) per process.

// ---- NELSSA fused-OMP per-worker core pinning ----
// Pins each fused CPU-attention OMP worker to cores[tid] (this parallel region
// only). GOMP_CPU_AFFINITY is avoided here because it also drags the EngineCore
// process's OMP/intra-op workers onto the same cores, causing AMX/LLC
// contention. Cores from NELSSA_FUSED_CORES, falling back to
// NELSSA_AFFINITY_CORES, falling back to none (OS schedules). Idempotent and
// safe: sched_setaffinity on a single core per worker.
const std::vector<int>& _nelssa_fused_cores() {
    static const std::vector<int> cores = []{
        std::vector<int> v;
        const char* s = std::getenv("NELSSA_FUSED_CORES");
        if (!s || !*s) s = std::getenv("NELSSA_AFFINITY_CORES");
        if (!s || !*s) return v;
        for (const char* p = s; *p; ) {
            char* end = nullptr;
            long c = std::strtol(p, &end, 10);
            if (end == p) { if (!*p) break; ++p; continue; }
            v.push_back((int)c);
            p = end;
            if (*p == ',') ++p;
        }
        return v;
    }();
    return cores;
}

// Pin the calling OMP worker to fused_cores[tid] (1:1). Called inside a
// #pragma omp parallel region. Returns true if it pinned.
bool _nelssa_pin_omp_worker(int tid) {
    const auto& cores = _nelssa_fused_cores();
    if (cores.empty() || tid < 0 || tid >= (int)cores.size()) return false;
    cpu_set_t mask; CPU_ZERO(&mask);
    CPU_SET(cores[tid], &mask);
    return sched_setaffinity(0, sizeof(mask), &mask) == 0;
}

bool _corelog_on() {
    static const bool on = (std::getenv("NELSSA_ATTN_CORELOG") != nullptr
                            && std::string(std::getenv("NELSSA_ATTN_CORELOG")) == "1");
    return on;
}

// Read one core's current frequency (kHz) from sysfs. Returns -1 on failure.
// intel_pstate exposes /sys/devices/system/cpu/<N>/cpufreq/scaling_cur_freq.
int _corelog_read_freq_khz(int core) {
    char path[96];
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", core);
    FILE* f = std::fopen(path, "re");  // 'e' = O_CLOEXEC
    if (!f) return -1;
    int khz = -1;
    if (std::fscanf(f, "%d", &khz) != 1) khz = -1;
    std::fclose(f);
    return khz;
}

// Sample the current frequency (kHz) of the calling thread's own core, then
// ALSO read the per-worker cores (72,80,...) reported by a transient OMP
// region. Returns one sampled reading per OMP worker (size = max_threads).
// Used to measure whether turbo drops under pipeline contention vs microbench.
std::vector<int> _corelog_sample_freqs() {
    const int max_threads = omp_get_max_threads();
    std::vector<int> f(max_threads, -1);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int c = sched_getcpu();
        f[tid] = _corelog_read_freq_khz(c);
    }
    return f;
}

// Log the caller thread's core + each OMP worker's core/affinity. Each distinct
// `tag` logs at most once per process (std::call_once). Safe to call from any
// OMP-eligible function; spawns a transient `#pragma omp parallel` to sample.
void _corelog_coremap(const char* tag) {
    if (!_corelog_on()) return;
    static std::once_flag flags[8];
    static const char* tags[8] = {"preQK", "preSM", "preAV",
                                  "gather", "gatherQK", "gatherSM", "gatherAV", ""};
    int idx = -1;
    for (int i = 0; i < 8; ++i) if (strcmp(tag, tags[i]) == 0) idx = i;
    if (idx < 0) return;
    std::call_once(flags[idx], [&]{
        const int main_cpu = sched_getcpu();
        cpu_set_t mmask; CPU_ZERO(&mmask);
        std::string main_aff;
        if (sched_getaffinity(0, sizeof(mmask), &mmask) == 0) {
            for (int c = 0; c < CPU_SETSIZE; ++c)
                if (CPU_ISSET(c, &mmask)) {
                    if (!main_aff.empty()) main_aff += ",";
                    main_aff += std::to_string(c);
                }
        }
        const int max_threads = omp_get_max_threads();
        std::vector<int> cores(max_threads, -1);
        std::vector<std::string> affs(max_threads);
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            cores[tid] = sched_getcpu();
            cpu_set_t mask; CPU_ZERO(&mask);
            if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
                std::string s;
                for (int c = 0; c < CPU_SETSIZE; ++c)
                    if (CPU_ISSET(c, &mask)) {
                        if (!s.empty()) s += ",";
                        s += std::to_string(c);
                    }
                affs[tid] = s;
            }
        }
        std::string line = "[NELSSA][CORELOG-CPP] ";
        line += tag;
        line += " main_cpu=" + std::to_string(main_cpu)
              + " main_aff=" + (main_aff.empty() ? "?" : main_aff)
              + " omp_workers=" + std::to_string(max_threads) + " cores=[";
        for (int i = 0; i < max_threads; ++i) { if (i) line += ","; line += std::to_string(cores[i]); }
        line += "] affinities={";
        for (int i = 0; i < max_threads; ++i) {
            if (i) line += " | ";
            line += "t" + std::to_string(i) + ":" + (affs[i].empty() ? "?" : affs[i]);
        }
        line += "}";
        fprintf(stderr, "%s\n", line.c_str());
        fflush(stderr);
    });
}
}  // namespace

// ============================================================================
// NELSSA CPU KV Cache reorganization by clusters.
//
// Mirrors the PyTorch fallback in vllm/v1/nelssa/clustering.py
// (reorganize_by_clusters_cpu): for each kv_head, the offloaded KV buffer is
// re-laid-out from token-sequential order to cluster-contiguous order using
// the per-head cluster assignment.
//
// Tensor layout (must match the PyTorch path):
//   keys_src / keys_dst : [kv_heads, total_tokens, head_dim]  (row-major)
//   clusters            : [kv_heads, n_centroids, max_cluster_size] (int32)
//                         cluster[h][c][j] = a global token id within the head
//                         (0 <= id < total_tokens) to place at cluster slot j.
//   cluster_size        : [kv_heads, n_centroids] (int32)
//                         number of valid indices in cluster[h][c].
//
// For each kv_head h, centroid c, slot j in [0, cluster_size[h][c]):
//   dst[h][cumsum(c)+j] = src[h][ cluster[h][c][j] ]
// where cumsum(c) = sum of cluster_size[h][0..c-1] is the running offset of
// this centroid's tokens within the reorganized head. Each head is independent
// and uses a per-head stride of (total_tokens * head_dim) — NOT
// max_cluster_size * n_centroids.
// ============================================================================

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
    // Ensure src/metadata are contiguous and on CPU.
    torch::Tensor keys_src_contig = keys_src.is_contiguous() ? keys_src : keys_src.contiguous();
    torch::Tensor values_src_contig = values_src.is_contiguous() ? values_src : values_src.contiguous();
    torch::Tensor clusters_contig = clusters.is_contiguous() ? clusters : clusters.contiguous();
    torch::Tensor cluster_size_contig = cluster_size.is_contiguous() ? cluster_size : cluster_size.contiguous();

    if (clusters_contig.device().type() != torch::kCPU) {
        clusters_contig = clusters_contig.cpu();
    }
    if (cluster_size_contig.device().type() != torch::kCPU) {
        cluster_size_contig = cluster_size_contig.cpu();
    }

    TORCH_CHECK(keys_dst.is_contiguous(), "keys_dst must be contiguous");
    TORCH_CHECK(values_dst.is_contiguous(), "values_dst must be contiguous");

    const int head_dim = static_cast<int>(keys_src_contig.size(2));
    // total_tokens: per-head token count = src.shape[1]. This is the correct
    // per-head stride (in tokens) — the previous C++ path wrongly used
    // max_cluster_size * n_centroids, which mismatches the actual tensor and
    // read garbage / wrong head data.
    const int total_tokens = static_cast<int>(keys_src_contig.size(1));
    const int max_cluster_size = static_cast<int>(clusters_contig.size(2));

    const int head_elems = total_tokens * head_dim;  // per-head element stride
    const size_t vec_bytes = static_cast<size_t>(head_dim) * sizeof(at::BFloat16);

    at::BFloat16* keys_dst_ptr = keys_dst.data_ptr<at::BFloat16>();
    at::BFloat16* values_dst_ptr = values_dst.data_ptr<at::BFloat16>();
    const at::BFloat16* keys_src_ptr = keys_src_contig.data_ptr<at::BFloat16>();
    const at::BFloat16* values_src_ptr = values_src_contig.data_ptr<at::BFloat16>();
    const int* clusters_ptr = clusters_contig.data_ptr<int>();
    const int* cluster_size_ptr = cluster_size_contig.data_ptr<int>();

    int num_threads = std::min(num_kv_heads, static_cast<int>(omp_get_max_threads()));
    if (num_threads < 1) num_threads = 1;

    // Parallelize across kv_heads. Each head is fully independent: its src and
    // dst bases are separated by head_elems (total_tokens * head_dim), and the
    // cluster indices are local to that head. No head writes into another
    // head's region, so no races and no per-head base offset bookkeeping bug.
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int h = 0; h < num_kv_heads; ++h) {
        const at::BFloat16* k_src = keys_src_ptr + static_cast<size_t>(h) * head_elems;
        const at::BFloat16* v_src = values_src_ptr + static_cast<size_t>(h) * head_elems;
        at::BFloat16* k_dst = keys_dst_ptr + static_cast<size_t>(h) * head_elems;
        at::BFloat16* v_dst = values_dst_ptr + static_cast<size_t>(h) * head_elems;

        const int* head_clusters = clusters_ptr + static_cast<size_t>(h) * n_centroids * max_cluster_size;
        const int* head_sizes = cluster_size_ptr + static_cast<size_t>(h) * n_centroids;

        int dst_off = 0;  // running token offset within this reorganized head
        for (int c = 0; c < n_centroids; ++c) {
            const int size = head_sizes[c];
            if (size <= 0) continue;

            const int* idx = head_clusters + static_cast<size_t>(c) * max_cluster_size;
            at::BFloat16* k_dst_c = k_dst + static_cast<size_t>(dst_off) * head_dim;
            at::BFloat16* v_dst_c = v_dst + static_cast<size_t>(dst_off) * head_dim;

            // Copy each token vector from its source (global head-local) id to
            // the contiguous cluster slot. Bound-check defensively; an out-of
            // range id would otherwise read adjacent memory silently.
            for (int j = 0; j < size; ++j) {
                const int kv_id = idx[j];
                if (kv_id < 0 || kv_id >= total_tokens) continue;
                const at::BFloat16* k_src_t = k_src + static_cast<size_t>(kv_id) * head_dim;
                const at::BFloat16* v_src_t = v_src + static_cast<size_t>(kv_id) * head_dim;
                std::memcpy(k_dst_c + static_cast<size_t>(j) * head_dim, k_src_t, vec_bytes);
                std::memcpy(v_dst_c + static_cast<size_t>(j) * head_dim, v_src_t, vec_bytes);
            }
            dst_off += size;
        }
        // Remaining slots (dst_off .. total_tokens) are left as whatever dst
        // was initialized to (torch.empty_like) — matching the PyTorch path,
        // which only writes [0, pos) and leaves the tail untouched.
    }
}

// ============================================================================
// Batch variant: reorganize all layers of a request in one call.
//
// The single-layer `reorganize_by_clusters_cpu` above is called from a Python
// `for layer_idx` loop, which serializes 32 layers and parallelizes only across
// the 8 kv_heads (num_threads capped at num_kv_heads). On a 64+ core box that
// leaves most cores idle and the loop measured ~53s for a 68K-token request.
//
// This variant takes per-layer tensors in std::vectors and parallelizes across
// LAYERS with omp_get_max_threads() (the global OMP_NUM_THREADS), so all 32
// layers run concurrently. Layers are fully independent (separate src/dst
// buffers and per-layer cluster indices), so there are no cross-layer races.
// Within each layer the head loop is serial (the layer-level parallelism is
// the dominant lever; per-head parallelism inside would over-subscribe cores).
//
// clusters/cluster_size MUST be CPU int32 tensors — the Python caller copies
// them off the GPU before invoking, so this function never touches CUDA.
// ============================================================================

void reorganize_by_clusters_cpu_batch(
    std::vector<torch::Tensor> keys_dst,
    std::vector<torch::Tensor> values_dst,
    std::vector<torch::Tensor> keys_src,
    std::vector<torch::Tensor> values_src,
    std::vector<torch::Tensor> clusters,
    std::vector<torch::Tensor> cluster_size,
    std::vector<int64_t> total_tokens,
    int64_t num_kv_heads
) {
    const size_t num_layers = keys_src.size();
    TORCH_CHECK(values_src.size() == num_layers && keys_dst.size() == num_layers
                && values_dst.size() == num_layers && clusters.size() == num_layers
                && cluster_size.size() == num_layers
                && total_tokens.size() == num_layers,
                "reorganize_by_clusters_cpu_batch: all per-layer vectors must have equal length");

    const int head_dim = static_cast<int>(keys_src[0].size(2));
    const size_t vec_bytes = static_cast<size_t>(head_dim) * sizeof(at::BFloat16);
    const int n_threads = omp_get_max_threads();
    if (n_threads < 1) {
        // Should not happen, but guard for non-OpenMP builds.
        for (size_t l = 0; l < num_layers; ++l) {
            reorganize_by_clusters_cpu(
                keys_dst[l], values_dst[l], keys_src[l], values_src[l],
                clusters[l], cluster_size[l],
                static_cast<int>(num_kv_heads),
                static_cast<int>(clusters[l].size(1)));
        }
        return;
    }

    // Parallelize across layers. dynamic scheduling tolerates per-layer size
    // variance (token counts and cluster sizes differ slightly across layers).
    #pragma omp parallel for num_threads(n_threads) schedule(dynamic)
    for (int64_t l = 0; l < static_cast<int64_t>(num_layers); ++l) {
        torch::Tensor ks = keys_src[l].is_contiguous() ? keys_src[l] : keys_src[l].contiguous();
        torch::Tensor vs = values_src[l].is_contiguous() ? values_src[l] : values_src[l].contiguous();
        torch::Tensor cl = clusters[l].is_contiguous() ? clusters[l] : clusters[l].contiguous();
        torch::Tensor cs = cluster_size[l].is_contiguous() ? cluster_size[l] : cluster_size[l].contiguous();

        TORCH_CHECK(keys_dst[l].is_contiguous() && values_dst[l].is_contiguous(),
                    "keys_dst/values_dst must be contiguous");

        const int T = static_cast<int>(total_tokens[l]);
        const int n_centroids = static_cast<int>(cl.size(1));
        const int max_cluster_size = static_cast<int>(cl.size(2));
        const int head_elems = T * head_dim;

        const at::BFloat16* k_src = ks.data_ptr<at::BFloat16>();
        const at::BFloat16* v_src = vs.data_ptr<at::BFloat16>();
        at::BFloat16* k_dst = keys_dst[l].data_ptr<at::BFloat16>();
        at::BFloat16* v_dst = values_dst[l].data_ptr<at::BFloat16>();
        const int* clusters_ptr = cl.data_ptr<int>();
        const int* sizes_ptr = cs.data_ptr<int>();

        for (int h = 0; h < static_cast<int>(num_kv_heads); ++h) {
            const at::BFloat16* k_src_h = k_src + static_cast<size_t>(h) * head_elems;
            const at::BFloat16* v_src_h = v_src + static_cast<size_t>(h) * head_elems;
            at::BFloat16* k_dst_h = k_dst + static_cast<size_t>(h) * head_elems;
            at::BFloat16* v_dst_h = v_dst + static_cast<size_t>(h) * head_elems;

            const int* head_clusters = clusters_ptr + static_cast<size_t>(h) * n_centroids * max_cluster_size;
            const int* head_sizes = sizes_ptr + static_cast<size_t>(h) * n_centroids;

            int dst_off = 0;
            for (int c = 0; c < n_centroids; ++c) {
                const int size = head_sizes[c];
                if (size <= 0) continue;
                const int* idx = head_clusters + static_cast<size_t>(c) * max_cluster_size;
                at::BFloat16* k_dst_c = k_dst_h + static_cast<size_t>(dst_off) * head_dim;
                at::BFloat16* v_dst_c = v_dst_h + static_cast<size_t>(dst_off) * head_dim;
                for (int j = 0; j < size; ++j) {
                    const int kv_id = idx[j];
                    if (kv_id < 0 || kv_id >= T) continue;
                    const at::BFloat16* k_src_t = k_src_h + static_cast<size_t>(kv_id) * head_dim;
                    const at::BFloat16* v_src_t = v_src_h + static_cast<size_t>(kv_id) * head_dim;
                    std::memcpy(k_dst_c + static_cast<size_t>(j) * head_dim, k_src_t, vec_bytes);
                    std::memcpy(v_dst_c + static_cast<size_t>(j) * head_dim, v_src_t, vec_bytes);
                }
                dst_off += size;
            }
        }
    }
}

// ============================================================================
// NELSSA CPU sparse gather: copy the selected clusters' tokens out of a
// reorganized (cluster-contiguous) CPU KV buffer.
//
// The Python path (_gather_selected_clusters in cpu_attention.py) builds the
// destination index via torch ops (arange + gather + cumsum + scatter_ + advanced
// indexing), measured ~12ms/layer. The index prep alone (grid+compact) is ~6ms.
// Because the source buffer is already cluster-contiguous (reorg put every
// cluster in a contiguous [offset, offset+size) slice), gathering a cluster is
// a SINGLE contiguous memcpy — no compaction, no per-element advanced indexing.
//
// For each kv_head h, selected cluster p with id c = cluster_ids[h, p]:
//   start = cluster_offsets[h, c]
//   size = cluster_size[h, c]
//   dst[h, dst_off : dst_off+size] = src[h, start : start+size]   (contiguous)
//   dst_off += size
// per_head_lens[h] = dst_off. Output order matches the Python path exactly
// (cluster_ids order, within-cluster = reorg order), so results are bit-exact.
//
// Tensor layout (must match the Python path):
//   keys_src / values_src : [kv_heads, num_tokens, head_dim]  bf16, CPU
//   keys_dst / values_dst : [kv_heads, max_selected, head_dim] bf16, CPU
//       (preallocated; only [0, per_head_lens[h]) per row is written; the tail
//       is left as-is and later masked out by the caller via per_head_lens)
//   cluster_offsets : [kv_heads, n_centroids] int32, CPU
//   cluster_size    : [kv_heads, n_centroids] int32, CPU
//   cluster_ids     : [kv_heads, nprobe] int64, CPU
//   per_head_lens   : [kv_heads] int32, CPU (output)
// ============================================================================

void gather_selected_clusters_cpu(
    torch::Tensor keys_dst,
    torch::Tensor values_dst,
    const torch::Tensor keys_src,
    const torch::Tensor values_src,
    const torch::Tensor cluster_offsets,
    const torch::Tensor cluster_size,
    const torch::Tensor cluster_ids,
    torch::Tensor per_head_lens,
    int64_t num_kv_heads,
    int64_t nprobe
) {
    TORCH_CHECK(keys_dst.is_contiguous() && values_dst.is_contiguous(),
                "keys_dst/values_dst must be contiguous");
    torch::Tensor keys_src_c = keys_src.is_contiguous() ? keys_src : keys_src.contiguous();
    torch::Tensor values_src_c = values_src.is_contiguous() ? values_src : values_src.contiguous();
    torch::Tensor offsets_c = cluster_offsets.is_contiguous() ? cluster_offsets : cluster_offsets.contiguous();
    torch::Tensor sizes_c = cluster_size.is_contiguous() ? cluster_size : cluster_size.contiguous();
    torch::Tensor ids_c = cluster_ids.is_contiguous() ? cluster_ids : cluster_ids.contiguous();

    const int head_dim = static_cast<int>(keys_src_c.size(2));
    const int num_tokens = static_cast<int>(keys_src_c.size(1));
    const int max_selected = static_cast<int>(keys_dst.size(1));
    const size_t vec_bytes = static_cast<size_t>(head_dim) * sizeof(at::BFloat16);

    const at::BFloat16* k_src = keys_src_c.data_ptr<at::BFloat16>();
    const at::BFloat16* v_src = values_src_c.data_ptr<at::BFloat16>();
    at::BFloat16* k_dst = keys_dst.data_ptr<at::BFloat16>();
    at::BFloat16* v_dst = values_dst.data_ptr<at::BFloat16>();
    const int* offsets_ptr = offsets_c.data_ptr<int>();
    const int* sizes_ptr = sizes_c.data_ptr<int>();
    const int64_t* ids_ptr = ids_c.data_ptr<int64_t>();
    int* lens_ptr = per_head_lens.data_ptr<int>();

    const int src_head_elems = num_tokens * head_dim;
    const int dst_head_elems = max_selected * head_dim;
    const int n_centroids = static_cast<int>(offsets_c.size(1));

    int num_threads = std::min(static_cast<int>(num_kv_heads), omp_get_max_threads());
    if (num_threads < 1) num_threads = 1;

    // Parallelize across kv_heads. Each head is independent: it reads from its
    // own src row and writes to its own dst row, so there are no cross-head races.
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int h = 0; h < static_cast<int>(num_kv_heads); ++h) {
        const at::BFloat16* k_src_h = k_src + static_cast<size_t>(h) * src_head_elems;
        const at::BFloat16* v_src_h = v_src + static_cast<size_t>(h) * src_head_elems;
        at::BFloat16* k_dst_h = k_dst + static_cast<size_t>(h) * dst_head_elems;
        at::BFloat16* v_dst_h = v_dst + static_cast<size_t>(h) * dst_head_elems;

        const int64_t* head_ids = ids_ptr + static_cast<size_t>(h) * nprobe;
        const int* head_offsets = offsets_ptr + static_cast<size_t>(h) * n_centroids;
        const int* head_sizes = sizes_ptr + static_cast<size_t>(h) * n_centroids;

        int dst_off = 0;
        for (int p = 0; p < static_cast<int>(nprobe); ++p) {
            const int64_t c = head_ids[p];
            if (c < 0 || c >= n_centroids) continue;
            const int start = head_offsets[c];
            const int size = head_sizes[c];
            if (size <= 0) continue;
            // Bounds-clamp against both the source row and the dest row so a bad
            // cluster offset/size can never read/write out of bounds silently.
            int copy = size;
            if (start + copy > num_tokens) copy = std::max(0, num_tokens - start);
            if (dst_off + copy > max_selected) copy = std::max(0, max_selected - dst_off);
            if (copy <= 0) continue;

            const at::BFloat16* k_src_c = k_src_h + static_cast<size_t>(start) * head_dim;
            const at::BFloat16* v_src_c = v_src_h + static_cast<size_t>(start) * head_dim;
            at::BFloat16* k_dst_c = k_dst_h + static_cast<size_t>(dst_off) * head_dim;
            at::BFloat16* v_dst_c = v_dst_h + static_cast<size_t>(dst_off) * head_dim;
            std::memcpy(k_dst_c, k_src_c, static_cast<size_t>(copy) * vec_bytes);
            std::memcpy(v_dst_c, v_src_c, static_cast<size_t>(copy) * vec_bytes);
            dst_off += copy;
        }
        // Zero the tail [dst_off, max_selected) so the fused AV matmul computes
        // 0 * 0 = 0 over unused slots (NaN-free). The buffer is reused across
        // layers with a FIXED M_CAP, so without this reset a shorter layer
        // would leave the previous layer's values in the tail -> 0 * stale = 0
        // only if stale is finite; zeroing makes it unconditionally safe.
        if (dst_off < max_selected) {
            const size_t tail_elems = static_cast<size_t>(max_selected - dst_off) * head_dim;
            std::memset(k_dst_h + static_cast<size_t>(dst_off) * head_dim, 0,
                        tail_elems * sizeof(at::BFloat16));
            std::memset(v_dst_h + static_cast<size_t>(dst_off) * head_dim, 0,
                        tail_elems * sizeof(at::BFloat16));
        }
        lens_ptr[h] = dst_off;
    }
}

// ============================================================================
// NELSSA batched sparse gather for P-side CPU attention.
//
// gather_selected_clusters_cpu (above) gathers ONE request, parallelizing the
// memcpy across kv_heads (OpenMP, ~8 threads). RPCAttentionEngine.handle_batch
// used to call it N times in a Python for-loop, serializing N dispatches and
// capping parallelism at kvH regardless of N. This batched variant gathers all
// N requests in a single dispatch, parallelizing across (N * kvH) so the
// N=4 case saturates the 32 P-side cores instead of running 4 x 8-thread
// gathers back-to-back. N=1 reduces to the same 8-thread gather as before
// (no regression).
//
// Per-request metadata (cluster_offsets/cluster_size/cluster_ids) and KV
// buffers have request-varying shapes (n_centroids, nprobe, num_tokens), so
// they are passed as std::vector<torch::Tensor> — the same list interface as
// reorganize_by_clusters_cpu_batch. Raw pointers + per-req dims are extracted
// into ReqMeta BEFORE the OpenMP region so the parallel body never touches a
// torch::Tensor (refcount/GIL safety).
//
// Output layout matches what cpu_attention_fused expects:
//   keys_batch   : [N, kvH, max_tokens, D] bf16 (preallocated by caller, out)
//   values_batch : [N, kvH, max_tokens, D] bf16 (preallocated by caller, out)
//   per_head_lens: [N, kvH] int32 (out) — valid tokens per (req, head).
// Tail [dst_off, max_tokens) is left uninitialized; cpu_attention_fused masks
// it via per_head_lens (same contract as the single-req path).
// ============================================================================

void gather_selected_clusters_batch_cpu(
    torch::Tensor keys_batch,        // [N, kvH, max_tokens, D] bf16 (out)
    torch::Tensor values_batch,      // [N, kvH, max_tokens, D] bf16 (out)
    torch::Tensor per_head_lens,     // [N, kvH] int32 (out)
    std::vector<torch::Tensor> keys_src,     // N x [kvH, num_tokens_req, D] bf16
    std::vector<torch::Tensor> values_src,   // N x [kvH, num_tokens_req, D] bf16
    std::vector<torch::Tensor> cluster_offsets, // N x [kvH, n_centroids_req] int32
    std::vector<torch::Tensor> cluster_size,    // N x [kvH, n_centroids_req] int32
    std::vector<torch::Tensor> cluster_ids,     // N x [kvH, nprobe_req] int64
    int64_t num_kv_heads
) {
    TORCH_CHECK(keys_batch.is_contiguous() && values_batch.is_contiguous(),
                "keys_batch/values_batch must be contiguous");
    TORCH_CHECK(per_head_lens.is_contiguous(), "per_head_lens must be contiguous");

    const int N = static_cast<int>(keys_src.size());
    TORCH_CHECK(N > 0, "empty batch");
    TORCH_CHECK(static_cast<int>(values_src.size()) == N
                && static_cast<int>(cluster_offsets.size()) == N
                && static_cast<int>(cluster_size.size()) == N
                && static_cast<int>(cluster_ids.size()) == N,
                "all per-req vectors must have length N");

    const int kvH = static_cast<int>(num_kv_heads);
    const int head_dim = static_cast<int>(keys_batch.size(3));
    const int max_tokens = static_cast<int>(keys_batch.size(2));
    const size_t vec_bytes = static_cast<size_t>(head_dim) * sizeof(at::BFloat16);
    TORCH_CHECK(head_dim > 0 && max_tokens > 0, "bad batch shape");

    // Pre-extract raw pointers + per-req dims so the OpenMP body touches no
    // torch::Tensor (refcount/GIL safety in the parallel region).
    // CRITICAL: any tensor made contiguous() here is a NEW tensor whose storage
    // must outlive the OpenMP region. We hold the (possibly newly-allocated)
    // tensors in `held` so their refcount stays > 0 until after the gather —
    // otherwise the raw pointers below would dangle.
    struct ReqMeta {
        const at::BFloat16* k_src;
        const at::BFloat16* v_src;
        const int* offsets;
        const int* sizes;
        const int64_t* ids;
        int num_tokens;
        int n_centroids;
        int nprobe;
    };
    std::vector<ReqMeta> reqs(N);
    // Keep the contiguous-ized tensors alive for the lifetime of reqs.
    std::vector<at::Tensor> held;
    held.reserve(N * 5);

    for (int n = 0; n < N; ++n) {
        at::Tensor ks = keys_src[n].is_contiguous() ? keys_src[n] : keys_src[n].contiguous();
        at::Tensor vs = values_src[n].is_contiguous() ? values_src[n] : values_src[n].contiguous();
        at::Tensor of = cluster_offsets[n].is_contiguous() ? cluster_offsets[n] : cluster_offsets[n].contiguous();
        at::Tensor sz = cluster_size[n].is_contiguous() ? cluster_size[n] : cluster_size[n].contiguous();
        at::Tensor id = cluster_ids[n].is_contiguous() ? cluster_ids[n] : cluster_ids[n].contiguous();
        held.push_back(ks); held.push_back(vs); held.push_back(of);
        held.push_back(sz); held.push_back(id);

        TORCH_CHECK(ks.size(0) == kvH && ks.size(2) == head_dim,
                    "keys_src req shape mismatch");
        TORCH_CHECK(of.dtype() == at::kInt && sz.dtype() == at::kInt,
                    "cluster_offsets/cluster_size must be int32");
        TORCH_CHECK(id.dtype() == at::kLong, "cluster_ids must be int64");

        reqs[n].k_src = ks.data_ptr<at::BFloat16>();
        reqs[n].v_src = vs.data_ptr<at::BFloat16>();
        reqs[n].offsets = of.data_ptr<int>();
        reqs[n].sizes = sz.data_ptr<int>();
        reqs[n].ids = id.data_ptr<int64_t>();
        reqs[n].num_tokens = static_cast<int>(ks.size(1));
        reqs[n].n_centroids = static_cast<int>(of.size(1));
        reqs[n].nprobe = static_cast<int>(id.size(1));
    }

    at::BFloat16* k_batch = keys_batch.data_ptr<at::BFloat16>();
    at::BFloat16* v_batch = values_batch.data_ptr<at::BFloat16>();
    int* lens_ptr = per_head_lens.data_ptr<int>();

    // Stride for a single (req, head) row in the [N, kvH, max_tokens, D] batch.
    const int row_elems = max_tokens * head_dim;
    const int req_stride = kvH * row_elems;

    const int total_work = N * kvH;
    int num_threads = std::min(total_work, omp_get_max_threads());
    if (num_threads < 1) num_threads = 1;

    // Parallelize across (N * kvH): each (req, head) reads its own src row and
    // writes its own dst row — no cross-(req, head) races.
    _corelog_coremap("gather");
    // Pin each gather OMP worker 1:1 to fused cores (same scheme as softmax).
    #pragma omp parallel num_threads(num_threads)
    {
        _nelssa_pin_omp_worker(omp_get_thread_num());
        #pragma omp for schedule(static)
        for (int idx = 0; idx < total_work; ++idx) {
            const int n = idx / kvH;
        const int h = idx - n * kvH;

        const ReqMeta& rm = reqs[n];
        const int src_head_elems = rm.num_tokens * head_dim;

        const at::BFloat16* k_src_h = rm.k_src + static_cast<size_t>(h) * src_head_elems;
        const at::BFloat16* v_src_h = rm.v_src + static_cast<size_t>(h) * src_head_elems;
        at::BFloat16* k_dst_h = k_batch + static_cast<size_t>(n) * req_stride
                                    + static_cast<size_t>(h) * row_elems;
        at::BFloat16* v_dst_h = v_batch + static_cast<size_t>(n) * req_stride
                                    + static_cast<size_t>(h) * row_elems;

        const int64_t* head_ids = rm.ids + static_cast<size_t>(h) * rm.nprobe;
        const int* head_offsets = rm.offsets + static_cast<size_t>(h) * rm.n_centroids;
        const int* head_sizes = rm.sizes + static_cast<size_t>(h) * rm.n_centroids;

        // --- gather logic identical to gather_selected_clusters_cpu (lines 343-365) ---
        int dst_off = 0;
        for (int p = 0; p < rm.nprobe; ++p) {
            const int64_t c = head_ids[p];
            if (c < 0 || c >= rm.n_centroids) continue;
            const int start = head_offsets[c];
            const int size = head_sizes[c];
            if (size <= 0) continue;
            int copy = size;
            if (start + copy > rm.num_tokens) copy = std::max(0, rm.num_tokens - start);
            if (dst_off + copy > max_tokens) copy = std::max(0, max_tokens - dst_off);
            if (copy <= 0) continue;

            const at::BFloat16* k_src_c = k_src_h + static_cast<size_t>(start) * head_dim;
            const at::BFloat16* v_src_c = v_src_h + static_cast<size_t>(start) * head_dim;
            at::BFloat16* k_dst_c = k_dst_h + static_cast<size_t>(dst_off) * head_dim;
            at::BFloat16* v_dst_c = v_dst_h + static_cast<size_t>(dst_off) * head_dim;
            std::memcpy(k_dst_c, k_src_c, static_cast<size_t>(copy) * vec_bytes);
            std::memcpy(v_dst_c, v_src_c, static_cast<size_t>(copy) * vec_bytes);
            dst_off += copy;
        }
        lens_ptr[n * kvH + h] = dst_off;
    }
    }  // end #pragma omp parallel

    // NELSSA_ATTN_CORELOG=1: ONE-TIME gather thread-overhead probe. The gather is
    // bandwidth-bound memcpy across (N*kvH=8) heads; like the fused probe this
    // measures 1-thread vs 8-thread, and cold (1st rep) vs warm (best-of-N) to
    // separate sync overhead from cache warm-up. Runs on the REAL inputs into a
    // scratch batch so the real keys_batch/values_batch/per_head_lens are
    // untouched. Restores the original thread count afterward.
    if (_corelog_on()) {
        static std::once_flag _gprobe_flag;
        std::call_once(_gprobe_flag, [&]{
            using clk = std::chrono::steady_clock;
            const int save_omp = omp_get_max_threads();
            // scratch destination so the real buffers are never overwritten
            at::Tensor kb = at::empty_like(keys_batch);
            at::Tensor vb = at::empty_like(values_batch);
            at::Tensor lb = at::empty_like(per_head_lens);
            at::BFloat16* kb_p = kb.data_ptr<at::BFloat16>();
            at::BFloat16* vb_p = vb.data_ptr<at::BFloat16>();
            int* lb_p = lb.data_ptr<int>();
            const int tw = total_work;
            auto gather_body = [&](int nt) {
                const int nt_use = std::min(tw, nt);
                #pragma omp parallel for num_threads(nt_use) schedule(static)
                for (int idx = 0; idx < tw; ++idx) {
                    const int n = idx / kvH;
                    const int h = idx - n * kvH;
                    const ReqMeta& rm = reqs[n];
                    const int src_head_elems = rm.num_tokens * head_dim;
                    const at::BFloat16* k_src_h = rm.k_src + (size_t)h * src_head_elems;
                    const at::BFloat16* v_src_h = rm.v_src + (size_t)h * src_head_elems;
                    at::BFloat16* k_dst_h = kb_p + (size_t)n * req_stride + (size_t)h * row_elems;
                    at::BFloat16* v_dst_h = vb_p + (size_t)n * req_stride + (size_t)h * row_elems;
                    const int64_t* head_ids = rm.ids + (size_t)h * rm.nprobe;
                    const int* head_offsets = rm.offsets + (size_t)h * rm.n_centroids;
                    const int* head_sizes = rm.sizes + (size_t)h * rm.n_centroids;
                    int dst_off = 0;
                    for (int p = 0; p < rm.nprobe; ++p) {
                        const int64_t c = head_ids[p];
                        if (c < 0 || c >= rm.n_centroids) continue;
                        const int start = head_offsets[c];
                        const int size = head_sizes[c];
                        if (size <= 0) continue;
                        int copy = size;
                        if (start + copy > rm.num_tokens) copy = std::max(0, rm.num_tokens - start);
                        if (dst_off + copy > max_tokens) copy = std::max(0, max_tokens - dst_off);
                        if (copy <= 0) continue;
                        std::memcpy(k_dst_h + (size_t)dst_off * head_dim,
                                    k_src_h + (size_t)start * head_dim,
                                    (size_t)copy * vec_bytes);
                        std::memcpy(v_dst_h + (size_t)dst_off * head_dim,
                                    v_src_h + (size_t)start * head_dim,
                                    (size_t)copy * vec_bytes);
                        dst_off += copy;
                    }
                    lb_p[n * kvH + h] = dst_off;
                }
            };
            auto ms = [&](auto&& body) {
                auto t0 = clk::now(); body();
                return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            };
            omp_set_num_threads(1);
            double g1_cold = ms([&]{ gather_body(1); });
            double g1_best = 1e9; for (int r = 0; r < 20; ++r) { double v = ms([&]{ gather_body(1); }); if (v < g1_best) g1_best = v; }
            omp_set_num_threads(8);
            double g8_cold = ms([&]{ gather_body(8); });
            double g8_best = 1e9; for (int r = 0; r < 20; ++r) { double v = ms([&]{ gather_body(8); }); if (v < g8_best) g8_best = v; }
            omp_set_num_threads(save_omp);
            fprintf(stderr,
                "[NELSSA][CORELOG-CPP] gather_probe N=%d kvH=%d work=%d | "
                "1t cold=%.4f warm=%.4f | 8t cold=%.4f warm=%.4f | "
                "warmup_gain(1t)=%.4fms warmup_gain(8t)=%.4fms speedup(warm)=%.2fx\n",
                N, kvH, tw, g1_cold, g1_best, g8_cold, g8_best,
                g1_cold - g1_best, g8_cold - g8_best, g1_best / g8_best);
            fflush(stderr);
        });
    }
}

// ============================================================================
// NELSSA fused CPU attention — gather-then-GEMM-softmax-GEMM in ONE dispatch.
//
// The previous Python path (_batched_attention_f32 in cpu_attention.py) issued
// 3 separate torch ops per layer (matmul QK, softmax/exp/max/log, matmul AV) ->
// 96 dispatches across 32 layers. Each dispatch wakes the shared torch intra-op
// thread pool (~0.5-1ms each), so for the tiny Long-Request GEMMs (M~1300 from the
// 1.8% retrieval budget) the dispatch/scheduling overhead dominated the actual
// compute (0.004ms theoretical, 12ms measured).
//
// This fuses the whole attention into one C++ call so torch is entered exactly
// twice per layer (QK matmul, AV matmul) instead of three op-dispatches, and the
// softmax/max/exp/log is done in-place in C++ — NO materialization of the
// intermediate scores/attn_weights tensors (the bandwidth-bound path's biggest
// hidden cost). QK/AV still go through at::matmul, which torch routes to oneDNN
// AMX bf16 internally, so we get the fast kernel without linking dnnl directly.
//
// Layout (matches the Python path's _batched_attention_f32):
//   queries       : [num_reqs, num_heads, head_dim]            bf16, CPU
//   keys_batch    : [num_reqs, num_kv_heads, max_tokens, D]    bf16, CPU (gathered)
//   values_batch  : [num_reqs, num_kv_heads, max_tokens, D]    bf16, CPU (gathered)
//   per_head_lens : [num_reqs, num_kv_heads] int32             (valid tokens per row)
//   output (out)  : [num_reqs, num_heads, head_dim]            bf16, CPU
//   lse      (out): [num_reqs, num_heads, 1]                    float32, CPU
//
// GQA: num_heads = num_kv_heads * group_size. queries are laid out
// [num_reqs, num_kv_heads, group_size, head_dim] (vLLM GQA-contiguous layout).
// Per (req, kv_head) the work is: q[g,D] x k[M,D]^T -> s[g,M]; softmax with the
// per-row mask (len < M -> tail masked -inf); w[g,M] x v[M,D] -> o[g,D];
// lse[g] = logsumexp(s). Output ordering must match the Python path exactly so
// AttentionResultMerger.merge_with_mask stays numerically identical.
//
// Parallelism: at::matmul already parallelizes across the batch dim
// (num_reqs*num_kv_heads*group_size tiles) via oneDNN/AT. We do NOT add an outer
// OpenMP loop around the matmuls — that would oversubscribe against matmul's
// internal pool. The only OpenMP region is the C++ softmax, parallelized over
// (num_reqs*num_kv_heads*group_size) rows — pure elementwise, no torch entry.
// ============================================================================

void cpu_attention_fused(
    torch::Tensor queries,        // [N, H, D] bf16  (H = num_heads)
    torch::Tensor keys_batch,      // [N, kvH, M, D] bf16
    torch::Tensor values_batch,    // [N, kvH, M, D] bf16
    torch::Tensor per_head_lens,   // [N, kvH] int32
    torch::Tensor output,          // [N, H, D] bf16 (out)
    torch::Tensor lse,             // [N, H, 1] f32  (out)
    int64_t num_kv_heads,
    int64_t head_dim
) {
    TORCH_CHECK(queries.is_contiguous(), "queries must be contiguous");
    TORCH_CHECK(keys_batch.is_contiguous(), "keys_batch must be contiguous");
    TORCH_CHECK(values_batch.is_contiguous(), "values_batch must be contiguous");
    TORCH_CHECK(per_head_lens.is_contiguous(), "per_head_lens must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(lse.is_contiguous(), "lse must be contiguous");

    const int N = static_cast<int>(queries.size(0));
    const int H = static_cast<int>(queries.size(1));

    // NELSSA_ATTN_CORELOG=1: ONE-TIME AMX-dispatch probe. Builds a modest 2D
    // GEMM in both bf16 and float and times identical-shape matmuls. AMX (when
    // oneDNN selects the AMX ISA) accelerates bf16 far beyond the fp32 unit, so
    // bf16_matmul FASTER than float_matmul (float/bf16 > 1) => AMX ON. If bf16 is
    // slower/equal, oneDNN fell back to a non-AMX (slow software bf16) path =>
    // AMX effectively OFF for this process. Run once per process; the result is
    // process-global because oneDNN's ISA choice is. Scratch tensors only.
    if (_corelog_on()) {
        static std::once_flag _amx_flag;
        std::call_once(_amx_flag, [&]{
            using clk = std::chrono::steady_clock;
            const int K = 2048;
            at::Tensor a = at::randn({K, K}, at::dtype(at::kBFloat16));
            at::Tensor b = at::randn({K, K}, at::dtype(at::kBFloat16));
            at::Tensor af = at::randn({K, K});  // float32
            at::Tensor bf = at::randn({K, K});
            at::Tensor ob = at::empty({K, K}, at::dtype(at::kBFloat16));
            at::Tensor of = at::empty({K, K});
            // warmup
            for (int i = 0; i < 10; ++i) {
                at::matmul_out(ob, a, b);
                at::matmul_out(of, af, bf);
            }
            auto timed = [&](auto&& body) {
                double best = 1e9;
                for (int r = 0; r < 10; ++r) {
                    auto t0 = clk::now();
                    for (int i = 0; i < 5; ++i) body();
                    double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / 5;
                    if (ms < best) best = ms;
                }
                return best;
            };
            double bf16_ms = timed([&]{ at::matmul_out(ob, a, b); });
            double f_ms = timed([&]{ at::matmul_out(of, af, bf); });
            double ratio = f_ms / bf16_ms;  // >1 => bf16 faster => AMX on
            const char* verdict = (ratio > 1.5) ? "AMX_ON"
                               : (ratio > 0.8) ? "AMX_UNCERTAIN" : "AMX_OFF_bf16_slower";
            fprintf(stderr,
                "[NELSSA][CORELOG-CPP] amx_probe bf16=%.2fms float=%.2fms "
                "float/bf16=%.2f -> %s (threads=%d)\n",
                bf16_ms, f_ms, ratio, verdict, omp_get_max_threads());
            fflush(stderr);
        });
    }
    const int D = static_cast<int>(head_dim);
    const int kvH = static_cast<int>(num_kv_heads);
    const int g = H / kvH;            // group_size
    const int M = static_cast<int>(keys_batch.size(2));
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Reshape queries to [N, kvH, g, D] (GQA grouping). View is contiguous-safe
    // because queries is [N, H, D] contiguous and H = kvH*g (interleaved kv_head
    // groups in vLLM's GQA layout: [kv0_g0..kv0_g{g-1}, kv1_g0, ...]).
    auto q = queries.view({N, kvH, g, D});                          // [N,kvH,g,D]
    auto k = keys_batch;                                            // [N,kvH,M,D]
    auto v = values_batch;                                          // [N,kvH,M,D]

    // --- QK: scores = q @ k^T  ->  [N, kvH, g, M]  (one batched matmul) ---------
    // at::matmul(bf16, bf16) returns bf16. Keep scores in bf16 to avoid an extra
    // f32 buffer allocation+copy (a hidden per-layer cost). The softmax below
    // reads each bf16 element into a float, computes, and writes back as bf16
    // in-place — no separate scores_f32 tensor and no attn_weights materialization.
    // (The legacy Python path also ran softmax in bf16, so this matches it.)
    // NELSSA_ATTN_CORELOG=1: log the worker core/affinity map before QK, softmax,
    // and AV (see _corelog_coremap) to verify the matmul intra-op pool gets the
    // same 8-core 1:1 spread as the softmax OMP region, and the main RPC thread's
    // own core (it's pinned to one core; if matmul workers pile on it that core
    // shows contention).
    _corelog_coremap("preQK");
    // NELSSA_ATTN_CORELOG=1: per-phase wall-clock (QK/SM/AV) on EVERY call, collected
    // and logged periodically so the steady-state per-phase time (not just the
    // one-shot probe) can be compared 1:1 against the microbench's per-phase time.
    using _ck = std::chrono::steady_clock;
    auto _t_qk0 = _corelog_on() ? _ck::now() : _ck::time_point{};
    _sim_nvtx_push("CPUattn[QK]");
    auto scores = at::matmul(q, k.transpose(-1, -2));              // [N,kvH,g,M] bf16
    scores.mul_(scale);
    _sim_nvtx_pop();  // end CPUattn[QK]
    double _qk_ms = _corelog_on()
        ? std::chrono::duration<double, std::milli>(_ck::now() - _t_qk0).count() : 0.0;

    // --- Softmax with per-row mask, in-place over scores (bf16) ----------------
    // Parallelize across (N*kvH*g) rows. Each row: read bf16->float, find max,
    // exp/sum/normalize in float, write back bf16. The bf16<->float conversion is
    // per-element (cheap, no kernel dispatch) and avoids the two full-tensor
    // casts (.to(f32) then .to(bf16)) the previous version did. lse stays f32.
    _corelog_coremap("preSM");
    auto _t_sm0 = _corelog_on() ? _ck::now() : _ck::time_point{};
    _sim_nvtx_push("CPUattn[softmax]");
    const int rows = N * kvH * g;
    at::BFloat16* scores_ptr = scores.data_ptr<at::BFloat16>();
    const int* lens_ptr = per_head_lens.data_ptr<int>();
    float* lse_ptr = lse.data_ptr<float>();

    // Preempt-detect: each OMP worker records the core it starts on, samples the
    // core mid-loop, and the core it ends on. If a worker migrated mid-softmax
    // (the main RPC thread / EngineCore stole the core, or the scheduler moved
    // it), first != last (or mid differs). Logged once per process. The counts
    // also expose how many distinct cores each worker touched.
    static int _sm_first[128]; static int _sm_last[128];
    static int _sm_mid[128];   static int _sm_iters[128];
    static std::once_flag _sm_mig_flag;
    const int _sm_cap = std::min(omp_get_max_threads(), 128);
    // Pin each OMP worker to fused_cores[tid] 1:1 (replaces GOMP_CPU_AFFINITY,
    // which dragged EngineCore workers onto fused cores -> 1.65x contention).
    // Always run (not gated on CORELOG) so production without CORELOG still
    // gets the 1:1 spread. The OS-level affinity persists across the parallel
    // for below since the same worker threads are reused.
    #pragma omp parallel
    {
        _nelssa_pin_omp_worker(omp_get_thread_num());
    }
    if (_corelog_on()) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid < _sm_cap) { _sm_first[tid] = sched_getcpu(); _sm_iters[tid] = 0; _sm_mid[tid] = -2; }
        }
    }
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        // Decompose flat row r -> (n, kh, gi)
        const int n = r / (kvH * g);
        const int rem = r - n * (kvH * g);
        const int kh = rem / g;
        const int gi = rem - kh * g;
        const int len = lens_ptr[n * kvH + kh];     // valid tokens for this (n,kh)
        const int row_len = (len > 0 && len <= M) ? len : M;

        at::BFloat16* srow = scores_ptr + static_cast<int64_t>(r) * M;
        // Mask tail to -inf first (matches masked_fill in the Python path).
        for (int m = row_len; m < M; ++m) srow[m] = at::BFloat16(-std::numeric_limits<float>::infinity());

        // max over [0, M) (tail is -inf so it can't win).
        float mx = float(srow[0]);
        for (int m = 1; m < M; ++m) {
            const float val = float(srow[m]);
            if (val > mx) mx = val;
        }
        // exp(x - mx), sum; clamp sum to avoid div-by-zero on all-masked rows.
        float sum = 0.0f;
        for (int m = 0; m < M; ++m) {
            const float e = std::exp(float(srow[m]) - mx);
            srow[m] = at::BFloat16(e);      // overwrite scores with attn weights
            sum += e;
        }
        const float denom = (sum > 1e-9f) ? sum : 1e-9f;
        const float inv = 1.0f / denom;
        for (int m = 0; m < M; ++m) {
            srow[m] = at::BFloat16(float(srow[m]) * inv);
        }
        // Tail weights are exp(-inf - mx) = 0 already (and the Python path also
        // masks them to 0.0 after the divide), so leave them as 0.

        // lse for this (n, kh, gi): stored at output index [n, kh*g + gi, 0].
        // lse is [N, H, 1] contiguous, so the linear offset is n*H + (kh*g + gi).
        lse_ptr[n * H + (kh * g + gi)] = mx + std::log(denom);

        if (_corelog_on()) {
            int tid = omp_get_thread_num();
            if (tid < _sm_cap) {
                _sm_iters[tid]++;
                if (_sm_iters[tid] == 1) _sm_mid[tid] = sched_getcpu();
                _sm_last[tid] = sched_getcpu();
            }
        }
    }
    if (_corelog_on()) {
        std::call_once(_sm_mig_flag, [&]{
            std::string line = "[NELSSA][CORELOG-CPP] softmax_migrate workers="
                             + std::to_string(_sm_cap) + " first=[";
            for (int i = 0; i < _sm_cap; ++i) { if (i) line += ","; line += std::to_string(_sm_first[i]); }
            line += "] mid=["; for (int i = 0; i < _sm_cap; ++i) { if (i) line += ","; line += std::to_string(_sm_mid[i]); }
            line += "] last=[";  for (int i = 0; i < _sm_cap; ++i) { if (i) line += ","; line += std::to_string(_sm_last[i]); }
            line += "] iters=[";  for (int i = 0; i < _sm_cap; ++i) { if (i) line += ","; line += std::to_string(_sm_iters[i]); }
            line += "]";
            int migrated = 0;
            for (int i = 0; i < _sm_cap; ++i)
                if (_sm_first[i] != _sm_last[i] || _sm_mid[i] != _sm_first[i]) migrated++;
            line += " migrated_workers=" + std::to_string(migrated);
            fprintf(stderr, "%s\n", line.c_str());
            fflush(stderr);
        });
    }
    _sim_nvtx_pop();  // end CPUattn[softmax]
    double _sm_ms = _corelog_on()
        ? std::chrono::duration<double, std::milli>(_ck::now() - _t_sm0).count() : 0.0;

    // --- AV: output = attn_weights @ v  ->  [N, kvH, g, D]  (one batched matmul) -
    // scores now holds bf16 attn weights; matmul(bf16, bf16) -> bf16, copied
    // directly into the bf16 output view. No dtype cast needed.
    _corelog_coremap("preAV");
    auto _t_av0 = _corelog_on() ? _ck::now() : _ck::time_point{};
    _sim_nvtx_push("CPUattn[AV]");
    auto out_view = output.view({N, kvH, g, D});                   // [N,kvH,g,D] bf16
    auto o_bf16 = at::matmul(scores, v);                           // [N,kvH,g,D] bf16
    out_view.copy_(o_bf16);
    _sim_nvtx_pop();  // end CPUattn[AV]
    double _av_ms = _corelog_on()
        ? std::chrono::duration<double, std::milli>(_ck::now() - _t_av0).count() : 0.0;
    // output now holds [N, kvH, g, D] == [N, H, D] bf16 (GQA-contiguous).

    // NELSSA_ATTN_CORELOG=1: collect per-phase (QK/SM/AV) time every call, log a
    // summary every 320 calls (10 steps x 32 layers). Skips the first call (warm-up)
    // and samples each OMP worker's CPU frequency to detect turbo throttling.
    if (_corelog_on()) {
        static std::vector<double> _qk_v, _sm_v, _av_v;
        static std::vector<int> _freq_min, _freq_max;  // kHz across workers/calls
        static bool _first_call = true;
        if (_first_call) {
            _first_call = false;  // drop the one-time process warm-up call
        } else {
            _qk_v.push_back(_qk_ms); _sm_v.push_back(_sm_ms); _av_v.push_back(_av_ms);
            // sample frequencies mid-operation (after QK, before SM) so the cores
            // are actually busy when read, not idle.
            for (int khz : _corelog_sample_freqs()) {
                if (khz > 0) {
                    _freq_min.push_back(khz);
                    _freq_max.push_back(khz);
                }
            }
        }
        if (_qk_v.size() >= 320) {
            auto report = [](const char* name, std::vector<double>& v) -> std::string {
                std::sort(v.begin(), v.end());
                double sum = 0; for (double x : v) sum += x;
                double mean = sum / v.size();
                double med = v[v.size() / 2];
                double p99 = v[std::min((size_t)v.size() - 1, (size_t)std::ceil(0.99 * v.size()) - 1)];
                char buf[160];
                std::snprintf(buf, sizeof(buf), "%s: n=%zu mean=%.3f med=%.3f p99=%.3f",
                              name, v.size(), mean, med, p99);
                v.clear();
                return buf;
            };
            std::sort(_freq_min.begin(), _freq_min.end());
            int fmin = _freq_min.empty() ? -1 : _freq_min.front();
            int fmed = _freq_min.empty() ? -1 : _freq_min[_freq_min.size() / 2];
            int fmax = _freq_min.empty() ? -1 : _freq_min.back();
            fprintf(stderr, "[NELSSA][CORELOG-CPP] phase_steady %s | %s | %s\n",
                    report("QK", _qk_v).c_str(),
                    report("SM", _sm_v).c_str(),
                    report("AV", _av_v).c_str());
            fprintf(stderr,
                "[NELSSA][CORELOG-CPP] cpu_freq n=%zu min=%dMHz med=%dMHz max=%dMHz "
                "(turbo_max=3900)\n",
                _freq_min.size(), fmin / 1000, fmed / 1000, fmax / 1000);
            fflush(stderr);
            _freq_min.clear(); _freq_max.clear();
        }
    }

    // NELSSA_ATTN_CORELOG=1: ONE-TIME thread-overhead probe (runs after the real
    // result is already written to output/lse, so it never corrupts it). At N=1
    // the GEMM batch dim is kvH*g=8 and the softmax has 32 rows — small enough
    // that 8-thread sync/spawn overhead may EXCEED the actual compute. Probe
    // re-runs the SAME QK/softmax/AV on the real inputs (q,k,v,per_head_lens)
    // at thread-count 1 and 8, measuring BOTH the cold (first) rep and the warm
    // best-of-20, to separate sync overhead (cold vs warm within a thread count)
    // from genuine parallel speedup (1t vs 8t). Writes to scratch buffers and
    // restores the original thread count.
    if (_corelog_on()) {
        static std::once_flag _probe_flag;
        std::call_once(_probe_flag, [&]{
            using clk = std::chrono::steady_clock;
            const int save_torch = at::get_num_threads();
            const int save_omp = omp_get_max_threads();
            const int rows_local = N * kvH * g;
            // run_phase returns (cold, warm_best): cold = 1st rep, warm = best of 20.
            auto run_phase = [&](int nt, auto&& body) -> std::pair<double,double> {
                at::set_num_threads(nt);
                omp_set_num_threads(nt);
                double cold = 1e9, best = 1e9;
                for (int rep = 0; rep < 20; ++rep) {
                    auto t0 = clk::now();
                    body();
                    double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
                    if (rep == 0) cold = ms;
                    if (ms < best) best = ms;
                }
                return {cold, best};
            };
            auto do_probe = [&](int nt, double& qk_c, double& sm_c, double& av_c,
                                double& qk_w, double& sm_w, double& av_w) {
                at::Tensor sc = at::empty({N, kvH, g, M}, scores.options());
                at::Tensor so = at::empty_like(output);
                at::Tensor sl = at::empty_like(lse);
                auto pr_qk = run_phase(nt, [&]{ sc.copy_(at::matmul(q, k.transpose(-1, -2))); sc.mul_(scale); });
                auto pr_sm = run_phase(nt, [&]{
                    at::BFloat16* sp = sc.data_ptr<at::BFloat16>();
                    const int* lp = per_head_lens.data_ptr<int>();
                    float* lsep = sl.data_ptr<float>();
                    omp_set_num_threads(nt);
                    #pragma omp parallel for schedule(static)
                    for (int r = 0; r < rows_local; ++r) {
                        const int n = r / (kvH * g);
                        const int rem = r - n * (kvH * g);
                        const int kh = rem / g;
                        const int len = lp[n * kvH + kh];
                        const int rl = (len > 0 && len <= M) ? len : M;
                        at::BFloat16* sr = sp + (int64_t)r * M;
                        for (int m = rl; m < M; ++m) sr[m] = at::BFloat16(-std::numeric_limits<float>::infinity());
                        float mx = float(sr[0]);
                        for (int m = 1; m < M; ++m) { float val = float(sr[m]); if (val > mx) mx = val; }
                        float sum = 0.0f;
                        for (int m = 0; m < M; ++m) { float e = std::exp(float(sr[m]) - mx); sr[m] = at::BFloat16(e); sum += e; }
                        float inv = 1.0f / ((sum > 1e-9f) ? sum : 1e-9f);
                        for (int m = 0; m < M; ++m) sr[m] = at::BFloat16(float(sr[m]) * inv);
                        lsep[n * H + (kh * g + (rem - kh * g))] = mx + std::log((sum > 1e-9f) ? sum : 1e-9f);
                    }
                });
                auto pr_av = run_phase(nt, [&]{ auto ov = so.view({N, kvH, g, D}); ov.copy_(at::matmul(sc, v)); });
                qk_c = pr_qk.first; qk_w = pr_qk.second;
                sm_c = pr_sm.first; sm_w = pr_sm.second;
                av_c = pr_av.first; av_w = pr_av.second;
            };
            double qk1c, sm1c, av1c, qk1w, sm1w, av1w;
            double qk8c, sm8c, av8c, qk8w, sm8w, av8w;
            do_probe(1, qk1c, sm1c, av1c, qk1w, sm1w, av1w);
            do_probe(8, qk8c, sm8c, av8c, qk8w, sm8w, av8w);
            at::set_num_threads(save_torch);
            omp_set_num_threads(save_omp);
            auto tot = [](double a, double b, double c) { return a + b + c; };
            double t1c = tot(qk1c, sm1c, av1c), t1w = tot(qk1w, sm1w, av1w);
            double t8c = tot(qk8c, sm8c, av8c), t8w = tot(qk8w, sm8w, av8w);
            fprintf(stderr,
                "[NELSSA][CORELOG-CPP] thread_probe N=%d kvH=%d g=%d M=%d rows=%d\n"
                "  1t cold: QK=%.4f SM=%.4f AV=%.4f tot=%.4f | warm: QK=%.4f SM=%.4f AV=%.4f tot=%.4f\n"
                "  8t cold: QK=%.4f SM=%.4f AV=%.4f tot=%.4f | warm: QK=%.4f SM=%.4f AV=%.4f tot=%.4f\n"
                "  warmup_gain: 1t=%.4fms 8t=%.4fms | parallel_speedup(warm)=%.2fx (1t/8t, >1 means threads help)\n",
                N, kvH, g, M, rows_local,
                qk1c, sm1c, av1c, t1c, qk1w, sm1w, av1w, t1w,
                qk8c, sm8c, av8c, t8c, qk8w, sm8w, av8w, t8w,
                t1c - t1w, t8c - t8w, t1w / t8w);
            fflush(stderr);
        });
    }
}

// ============================================================================
// NELSSA similarity search (C++ reimplementation of perform() + _batched_search
// in vllm/v1/nelssa/similarity_search.py).
//
// WHY C++: the Python path's GPU kernels are fast (~0.067ms) but perform()
// takes ~1.6ms/layer in-pipeline — the gap is host-side Python orchestration
// (per-req loop, dispatch, torch.stack, dict lookups). Moving orchestration to
// C++ removes that while keeping the GPU op sequence/dtypes byte-identical to
// batched_centroid_search (vllm/v1/attention/ops/sparse_kv_similarity_search.py).
//
// CORE INVARIANT (do NOT change): keeping softmax/sum in fp32 broke topk, so the
// dtype chain is fixed:
//   bmm(bf16,bf16)->bf16 ; mul_(rsqrt) in-place bf16 ; softmax(-1)->bf16 ;
//   sum(1)->bf16 ; masked_fill(bool_mask, finfo(bf16).min)->bf16 ;
//   topk(largest,sorted)->(values bf16, indices int64).
// No .to(kFloat)/.to(kHalf) cast, no baddbmm(alpha=) (changes accumulation
// order). libtorch dispatches these torch:: ops to CUDA automatically since
// the input tensors live on CUDA — no CUDA compilation needed.
// ============================================================================

// sim_compute_gpu: the PURE-GPU op block of similarity search
//   (bmm -> mul_(rsqrt) -> softmax -> sum -> masked_fill -> topk).
// Separated from perform_similarity_search so it is the CUDA-Graph capture
// candidate: it takes ONLY already-materialized GPU input views (qf, cf, sf)
// and writes ONLY into the provided persistent output buffers (when present)
// via the _out variants — no CPU sync, no data-dependent branch, no allocation
// on the persistent path. The CPU-side prep (long-req selection, n_centroids,
// topk clamp logging, max_topk computation) stays in the caller.
//
// PRECONDITION (persistent path): qf/cf/sf must already be views into
// FIXED-ADDRESS buffers so a captured graph replays against the same storage.
// The allocating path (buffers empty) just runs the standard torch ops.
//
// topk_k: the k to ask topk for. For the persistent path this is the buffer's
// fixed max_k (= topk_values_buf.size(1)); the caller slices [:real_k] after.
// For the allocating path it is the natural per-call max_topk.
//
// Returns (topk_values, topk_indices) — views into the persistent buffers on
// the persistent path, freshly allocated on the allocating path.
//
// CUDA-Graph capture (NELSSA_SIM_GRAPH, default on): on the persistent path the
// pure-GPU op block is captured once per (num_long) into a dedicated side
// stream and replayed every subsequent layer for that num_long. Capture is lazy
// (first eligible call) and the caller MUST have already copied the per-layer
// query/centroids/cmask into the persistent input buffers BEFORE calling — the
// graph replays only the 6 GPU ops, reading the buffers' current contents. The
// scalar args (batch, n_cent, topk_k, rsqrt_dim) are baked into the graph at
// capture time, so one graph per num_long (batch = num_long*kvH varies). See
// sim_graph_pool below.
std::tuple<torch::Tensor, torch::Tensor> sim_compute_gpu(
    const torch::Tensor qf,     // [batch, g, D] bf16 GPU
    const torch::Tensor cf,     // [batch, n_cent, D] bf16 GPU
    const torch::Tensor sf,     // [batch, n_cent] bool GPU  (mask for masked_fill)
    int64_t n_cent,
    int64_t topk_k,
    bool use_persistent,
    const torch::Tensor scores_buf,
    const torch::Tensor sm_buf,
    const torch::Tensor dist_buf,
    const torch::Tensor topk_values_buf,
    const torch::Tensor topk_indices_buf,
    double rsqrt_dim,
    int64_t batch
) {
    // The mask `sf` is computed graph-OUTSIDE by the caller (from the per-layer
    // empty-cluster mask + the layer-invariant pad mask) and passed in already
    // shaped [batch, n_cent]. The GPU op block (bmm..topk) reads it directly.
    const torch::Tensor& sf_eff = sf;
    torch::Tensor scores, sm, dist;
    if (use_persistent) {
        scores = scores_buf.narrow(0, 0, batch);                    // [batch, g, n_cent]
        at::bmm_out(scores, qf, cf.transpose(1, 2));
    } else {
        scores = torch::bmm(qf, cf.transpose(1, 2));                // [batch, g, n_cent] bf16
    }
    scores.mul_(rsqrt_dim);                                         // in-place, bf16
    if (use_persistent) {
        sm = sm_buf.narrow(0, 0, batch);                            // [batch, g, n_cent]
        at::softmax_out(sm, scores, /*dim=*/-1);
    } else {
        sm = torch::softmax(scores, /*dim=*/-1);                   // bf16
    }
    if (use_persistent) {
        dist = dist_buf.narrow(0, 0, batch);                        // [batch, n_cent]
        at::sum_out(dist, sm, /*dim=*/1);
    } else {
        dist = torch::sum(sm, /*dim=*/1);                          // [batch, n_cent] bf16
    }
    // R3: finfo(bf16).min == the most-negative bf16. Use the exact bf16 lowest
    // (== torch.finfo(bf16).min) so the cast is exact and matches Python
    // bit-for-bit; -float_max overflows bf16 and libtorch raises.
    const at::BFloat16 DTYPE_MIN = std::numeric_limits<at::BFloat16>::lowest();
    // in-place masked_fill so the GPU op block allocates NOTHING on the
    // persistent path (graph-capture requires fixed addresses — a non-in-place
    // masked_fill would materialize a fresh dist tensor). bit-identical to the
    // non-in-place form: it writes DTYPE_MIN into the same buffer in place.
    dist.masked_fill_(sf_eff, DTYPE_MIN);                          // in-place, bf16

    torch::Tensor topk_values, topk_indices;
    if (use_persistent) {
        topk_values = topk_values_buf.narrow(0, 0, batch);          // [batch, max_k]
        topk_indices = topk_indices_buf.narrow(0, 0, batch);        // [batch, max_k]
        at::topk_out(topk_values, topk_indices, dist, topk_k,
                     /*dim=*/-1, /*largest=*/true, /*sorted=*/true);
    } else {
        auto topk = torch::topk(dist, topk_k, /*dim=*/-1,
                                /*largest=*/true, /*sorted=*/true);
        topk_values = std::get<0>(topk);   // bf16 [batch, topk_k]
        topk_indices = std::get<1>(topk);  // int64 [batch, topk_k]
    }
    return std::make_tuple(topk_values, topk_indices);
}

// ============================================================================
// CUDA-Graph pool for the sim_compute_gpu persistent path.
//
// Why per-num_long: the scalar args batch (=num_long*kvH), n_cent (=max_n_centroids)
// and topk_k (=max_k_fixed) are baked into the captured graph. n_cent and topk_k
// are constant across num_long (they come from the fixed-width buffers), but
// batch varies with num_long, so we keep one graph per num_long in [1..MAX].
// num_long==0 is the empty path (returns early, never reaches compute).
//
// Capture discipline (mirrors torch.cuda.graph's safe pattern in C++):
//   - A dedicated side stream per slot; capture happens there so the default
//     stream's other ops (the per-layer copy_ into the persistent input bufs,
//     which is graph-OUTSIDE) are NOT captured.
//   - Capture is LAZY on the first eligible call: the caller has already filled
//     the persistent input buffers with real data, so the captured ops see
//     valid inputs. Subsequent calls copy fresh per-layer data into the same
//     fixed-address buffers (graph-OUTSIDE) then replay — the graph reads the
//     buffers' current contents.
//   - The persistent path allocates NOTHING inside the captured region (all
//     _out variants + in-place masked_fill_), so the private mempool stays
//     empty and there is no cross-step allocator contention with vLLM.
//
// Stream ordering: copy_ (default stream) -> wait on side stream is NOT needed
// because replay() is launched on the side stream and the caller reads the
// output views (which point into the persistent OUTPUT buffers) AFTER replay
// on the default stream. We synchronize via events: record on the side stream
// after replay, wait on the default stream before the caller reads. Capture
// itself runs on the side stream with the default stream waited beforehand so
// the input copies are visible. This keeps graph-OUTSIDE copies and graph-
// INSIDE compute correctly ordered without a full device sync.
//
// All static state is guarded by a mutex (the RPC server runs attention per
// layer serially, but the pool is process-global and capture is a critical
// section). The graph objects own CUDA resources; they live for the process
// lifetime (never reset) — the persistent buffers they were captured against
// are also process-lifetime (allocated once in gpu_model_runner init).
// ============================================================================

namespace {
constexpr int64_t SIM_GRAPH_MAX_LONG = 4;   // == NELSSA max_num_long_requests

struct SimGraphSlot {
    bool captured = false;
    at::cuda::CUDAGraph graph;
    // capture + replay stream. Created lazily at first capture (CUDAStream has
    // no default ctor). A dedicated side stream keeps the per-layer input
    // copies (graph-OUTSIDE, default stream) out of the captured region.
    std::unique_ptr<at::cuda::CUDAStream> side_stream;
    // Event used to order the graph-OUTSIDE input copies (default stream) and
    // the graph-INSIDE compute (side stream): record on the default stream after
    // the copies, then block the side stream on it before replay; record on the
    // side stream after replay, then block the default stream on it before the
    // caller reads the output views. CUDAEvent is non-copyable (move-only), so
    // store via unique_ptr.
    std::unique_ptr<at::cuda::CUDAEvent> ready;   // default -> side (inputs ready)
    std::unique_ptr<at::cuda::CUDAEvent> done;    // side -> default (replay done)
    // Bake-in check: the scalar args at capture time. If a later call for the
    // same num_long has different scalars (shouldn't happen — buffers are fixed)
    // we refuse the graph and fall back to the eager persistent path.
    int64_t batch = 0;
    int64_t n_cent = 0;
    int64_t topk_k = 0;
};

std::mutex g_sim_graph_mutex;
SimGraphSlot g_sim_graph_slots[SIM_GRAPH_MAX_LONG + 1];  // index by num_long

// NELSSA_SIM_GRAPH: "1" (default) captures+replays the persistent GPU op block;
// "0" disables and runs the eager persistent path (graph-capturable but not
// captured — for fallback / A-B comparison). Read each call (cheap) so the env
// toggle is live without a rebuild.
bool sim_graph_enabled() {
    const char* e = std::getenv("NELSSA_SIM_GRAPH");
    if (e == nullptr) return true;          // default on
    return std::string(e) != "0";
}
}  // namespace

// Capture sim_compute_gpu's persistent path into the slot's graph, then run it
// once eagerly so the output buffers hold THIS call's results (capture records
// but does not execute). Must hold g_sim_graph_mutex.
//
// Move 2b: cmask (eq_out + bitwise_or_) is captured INSIDE the graph (6 -> 8
// ops), reading fixed-address cluster_size_batch_buf + pad_mask buffer into
// cmask_buf. The per-num_long slot pool keeps each num_long's baked shapes
// isolated (an earlier single-shared-graph fold crashed on cross-num_long
// shape mismatch; per-slot capture avoids it).
static void sim_graph_capture(SimGraphSlot& slot, int64_t num_long,
                              const torch::Tensor& qf, const torch::Tensor& cf,
                              const torch::Tensor& cluster_size_batch,
                              const torch::Tensor& pad_mask,
                              const torch::Tensor& cmask_buf,
                              int64_t n_cent,
                              int64_t topk_k, const torch::Tensor& scores_buf,
                              const torch::Tensor& sm_buf,
                              const torch::Tensor& dist_buf,
                              const torch::Tensor& topk_values_buf,
                              const torch::Tensor& topk_indices_buf,
                              double rsqrt_dim, int64_t batch) {
    // Lazily create the dedicated side stream + ordering events for this slot
    // on first capture (CUDAStream/CUDAEvent have no usable default state here).
    if (!slot.side_stream) {
        slot.side_stream = std::make_unique<at::cuda::CUDAStream>(
            at::cuda::getStreamFromPool(/*isHighPriority=*/false,
                                        qf.device().index()));
        slot.ready = std::make_unique<at::cuda::CUDAEvent>();
        slot.done = std::make_unique<at::cuda::CUDAEvent>();
    }
    at::cuda::CUDAStream default_stream = at::cuda::getCurrentCUDAStream(
        qf.device().index());
    // Order the graph-OUTSIDE input copies (already submitted on the default
    // stream) before capture: record `ready` on the default stream, then make
    // the side stream block on it so the capture sees the filled input buffers
    // (query/centroids/cluster_size, all copied graph-OUTSIDE by the caller;
    // pad_mask is layer-invariant, written once per step).
    slot.ready->record(default_stream);
    slot.ready->block(*slot.side_stream);
    at::cuda::CUDAStreamGuard sg(*slot.side_stream);
    // capture_begin/end bracket the 8 GPU ops (eq_out, bitwise_or_, bmm..topk);
    // the persistent path allocates nothing, so the private mempool stays empty.
    slot.graph.capture_begin(/*pool=*/{0, 0},
                             cudaStreamCaptureModeRelaxed);
    // Move 2b: cmask = (cluster_size==0) | pad_mask, computed INSIDE the graph
    // (fixed-address in/out, graph-safe, alloc-free).
    auto cmask_view = cmask_buf.narrow(0, 0, num_long);        // [num_long,kvH,n_cent] bool
    at::eq_out(cmask_view, cluster_size_batch, /*other=*/0);
    cmask_view.bitwise_or_(pad_mask);
    auto sf = cmask_view.view({batch, n_cent});
    sim_compute_gpu(qf, cf, sf, n_cent, topk_k, /*use_persistent=*/true,
                    scores_buf, sm_buf, dist_buf, topk_values_buf,
                    topk_indices_buf, rsqrt_dim, batch);
    // capture_end() with keep_graph=false (default) ALSO instantiates the graph
    // automatically — calling instantiate() explicitly here would raise
    // "instantiate() is intended to be called by the user only when
    // keep_graph=true". So we let capture_end do both: end capture + create the
    // executable graph. replay() then runs that executable.
    slot.graph.capture_end();
    slot.captured = true;
    slot.batch = batch;
    slot.n_cent = n_cent;
    slot.topk_k = topk_k;
    // Run the captured graph once so the output buffers hold THIS call's
    // results (capture alone records but does not execute).
    slot.graph.replay();
    // Order the replay's writes before the caller's reads: record `done` on the
    // side stream, then make the default stream block on it.
    slot.done->record(*slot.side_stream);
    slot.done->block(default_stream);
}

// Replay the slot's captured graph for the current layer. The caller has
// already copied per-layer query/centroids/cmask into the persistent INPUT
// buffers (graph-OUTSIDE); replay re-runs the 6 GPU ops reading those buffers'
// current contents and writing the persistent OUTPUT buffers. Event ordering
// (ready: default->side, done: side->default) keeps the graph-OUTSIDE copies
// correctly ordered around the graph-INSIDE compute without a full device sync.
static void sim_graph_replay(SimGraphSlot& slot) {
    at::cuda::CUDAStream default_stream = at::cuda::getCurrentCUDAStream();
    slot.ready->record(default_stream);
    slot.ready->block(*slot.side_stream);
    at::cuda::CUDAStreamGuard sg(*slot.side_stream);
    slot.graph.replay();
    slot.done->record(*slot.side_stream);
    slot.done->block(default_stream);
}

std::vector<torch::Tensor> perform_similarity_search(
    const torch::Tensor query_states,        // [num_tokens, num_heads, head_dim] GPU
    const torch::Tensor query_start_loc_cpu, // [num_reqs+1] int32 or int64 CPU
    const torch::Tensor long_mask_cpu,       // [num_reqs] bool/int CPU (already AND of long+decode+has-meta)
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    double nprobe,                          // fraction, e.g. 0.018
    // centroid source — exactly one set is non-empty:
    // (A) P/D: staging buffers + per-long-req slot list. Staging is already
    //     padded to max_n_centroids, so no _pad_to_max needed.
    // (B) single-GPU: pre-stacked per-req tensors.
    const torch::Tensor centroid_staging,     // [N_SLOTS,L,kvH,max_n_cent,D] bf16 GPU, or empty
    const torch::Tensor cluster_size_staging, // [N_SLOTS,L,kvH,max_n_cent] int32 GPU, or empty
    const torch::Tensor slots_cpu,            // [num_long] int32 CPU, or empty
    const torch::Tensor centroids_prestacked, // [num_long,kvH,max_n_cent,D] bf16 GPU, or empty
    const torch::Tensor cluster_size_prestacked,// [num_long,kvH,max_n_cent] int32 GPU, or empty
    const torch::Tensor n_centroids_cpu,      // [num_long] int32 CPU (real n_centroids per req), or empty
    const torch::Tensor pad_mask_bh,          // [num_long,kvH,n_cent] bool GPU, or empty (cached across layers)
    // Persistent scratch/output buffers (CUDA-Graph path). When non-empty these
    // are used as the output tensors of the _out variants so the GPU op
    // sequence allocates NOTHING (graph capture requires fixed addresses).
    // All sized to the max num_long (=max_num_long_requests) so [:num_long]
    // views share one base address across all graph sizes. When empty the
    // function falls back to the allocating path (legacy, non-graph).
    const torch::Tensor scores_buf,           // [max_long*kvH, g, max_n_cent] bf16 GPU, or empty
    const torch::Tensor sm_buf,               // [max_long*kvH, g, max_n_cent] bf16 GPU, or empty
    const torch::Tensor dist_buf,             // [max_long*kvH, max_n_cent] bf16 GPU, or empty
    const torch::Tensor topk_values_buf,      // [max_long*kvH, max_k] bf16 GPU, or empty
    const torch::Tensor topk_indices_buf,     // [max_long*kvH, max_k] int64 GPU, or empty
    // P3: persistent INPUT buffers. When non-empty, the query/centroid/
    // cluster_size/cmask batches are COPIED into these fixed-address buffers
    // (graph-OUTSIDE) instead of torch::stack-allocated, so sim_compute_gpu
    // reads them as stable views — making the GPU op block graph-capturable.
    const torch::Tensor query_batch_buf,      // [max_long, kvH, g, D] bf16 GPU, or empty
    const torch::Tensor centroids_batch_buf,  // [max_long, kvH, max_n_cent, D] bf16 GPU, or empty
    const torch::Tensor cluster_size_batch_buf,// [max_long, kvH, max_n_cent] int32 GPU, or empty
    const torch::Tensor cmask_buf,            // [max_long, kvH, max_n_cent] bool GPU, or empty
    int64_t layer_idx,
    int64_t /*num_layers*/
) {
    _sim_nvtx_push("SIM[total]");
    // ---- read CPU index/mask tensors ----
    TORCH_CHECK(query_start_loc_cpu.is_cpu(), "query_start_loc_cpu must be CPU");
    TORCH_CHECK(long_mask_cpu.is_cpu(), "long_mask_cpu must be CPU");

    const int64_t num_reqs = long_mask_cpu.size(0);
    // query_start_loc is int64 (Python pre-filter builds it as int64). Read
    // directly — the old toType(kInt64) was a redundant per-layer conversion.
    TORCH_CHECK(query_start_loc_cpu.scalar_type() == torch::kInt64,
                "query_start_loc_cpu must be int64 (build it in the Python pre-filter)");
    const auto* qsl_ptr = query_start_loc_cpu.data_ptr<int64_t>();

    // long_mask: Python builds it as int8 already. Read directly (no toType).
    TORCH_CHECK(long_mask_cpu.scalar_type() == torch::kInt8,
                "long_mask_cpu must be int8 (build it in the Python pre-filter)");
    const auto* lmask_ptr = long_mask_cpu.data_ptr<int8_t>();

    const int64_t group_size = num_heads / num_kv_heads;
    const double rsqrt_dim = 1.0 / std::sqrt((double)head_dim);

    // ---- pass 1: collect eligible long reqs (long AND decode==1 token) ----
    // buffer_key / remote-id / dict lookups are done in the PYTHON glue; C++
    // receives long_mask already AND'd with has-metadata. Here we only apply
    // the decode==1-token check (the remaining data-dependent filter).
    _sim_nvtx_push("SIM[pass1:cpu]");
    std::vector<int64_t> long_req_indices;   // req_idx into req_ids
    std::vector<int64_t> long_start;         // start token offset
    std::vector<int64_t> n_centroids_vec;    // real n_centroids per long req
    std::vector<int64_t> nprobe_per_req;     // round(n_centroids * nprobe)
    int64_t max_n_cent = 0;

    const bool use_staging = (centroid_staging.numel() > 0);
    // slots_cpu / n_centroids_cpu are int64 (Python builds them as int64 so the
    // C++ side skips its per-layer toType(kInt64) conversion). They are layer-
    // invariant, converted once at layer 0 by the Python pre-filter.
    if (slots_cpu.numel() > 0) {
        TORCH_CHECK(slots_cpu.scalar_type() == torch::kInt64, "slots_cpu must be int64");
    }
    if (n_centroids_cpu.numel() > 0) {
        TORCH_CHECK(n_centroids_cpu.scalar_type() == torch::kInt64, "n_centroids_cpu must be int64");
    }
    const auto* slots_ptr = (slots_cpu.numel() > 0) ? slots_cpu.data_ptr<int64_t>() : nullptr;
    const auto* n_cent_ptr = (n_centroids_cpu.numel() > 0) ? n_centroids_cpu.data_ptr<int64_t>() : nullptr;

    // For single-GPU, n_centroids comes from each prestacked slice's real dim —
    // but the prestacked tensor is padded to a uniform max, so the real
    // n_centroids must be supplied via n_centroids_cpu. If that's missing we
    // fall back to the slice's full dim (no masking).
    for (int64_t i = 0; i < num_reqs; ++i) {
        if (lmask_ptr[i] == 0) continue;
        const int64_t start = qsl_ptr[i];
        const int64_t end = qsl_ptr[i + 1];
        if (end - start != 1) continue;  // decode = exactly 1 token
        int64_t nc = 0;
        if (n_cent_ptr != nullptr) {
            nc = n_cent_ptr[long_req_indices.size()];
        } else if (!use_staging && centroids_prestacked.numel() > 0) {
            nc = centroids_prestacked.size(2);  // max_n_cent (no per-req real count)
        } else if (use_staging) {
            nc = centroid_staging.size(3);
        }
        long_req_indices.push_back(i);
        long_start.push_back(start);
        n_centroids_vec.push_back(nc);
        int64_t npr = (int64_t)std::llround((double)nc * nprobe);
        if (npr < 1) npr = 1;
        nprobe_per_req.push_back(npr);
        if (nc > max_n_cent) max_n_cent = nc;
    }
    const int64_t num_long = (int64_t)long_req_indices.size();
    _sim_nvtx_pop();  // end SIM[pass1:cpu]

    std::vector<torch::Tensor> out(5);
    // empty-result path: return empty tensors; glue handles no-long steps.
    auto opts_i32_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    if (num_long == 0) {
        out[0] = torch::empty({0, 1}, torch::TensorOptions().dtype(torch::kInt64).device(query_states.device()));
        out[1] = torch::empty({0, 1}, torch::TensorOptions().dtype(query_states.dtype()).device(query_states.device()));
        out[2] = torch::empty({0}, opts_i32_cpu);
        out[3] = torch::empty({0}, opts_i32_cpu);
        out[4] = torch::empty({0}, opts_i32_cpu);
        _sim_nvtx_pop();  // end SIM[total]
        return out;
    }

    // use_persistent requires ALL persistent buffers present and large enough
    // (outputs + the P3 input buffers). If any is missing/undersized we fall
    // back to the allocating path entirely (bit-identical either way).
    const bool use_persistent = (scores_buf.numel() > 0
        && scores_buf.size(0) >= num_long * num_kv_heads
        && scores_buf.size(2) > 0
        && query_batch_buf.numel() > 0 && query_batch_buf.size(0) >= num_long
        && centroids_batch_buf.numel() > 0 && centroids_batch_buf.size(0) >= num_long
        && cluster_size_batch_buf.numel() > 0 && cluster_size_batch_buf.size(0) >= num_long
        && cmask_buf.numel() > 0 && cmask_buf.size(0) >= num_long);
    const int64_t max_n_centroids = use_persistent ? scores_buf.size(2) : 0;

    // ---- build query batch ----
    // _extract_query: view(1, kvH, g, D).permute(1,0,2,3).contiguous() per req.
    // P3: on the persistent path, copy each per-req query view into a
    // FIXED-ADDRESS slice of query_batch_buf (graph-OUTSIDE) instead of
    // torch::stack (which allocates a fresh tensor every call). sim_compute_gpu
    // then reads query_batch_buf as a stable view — graph-capturable.
    _sim_nvtx_push("SIM[query:build+copy]");
    torch::Tensor query_batch;
    if (use_persistent && query_batch_buf.numel() > 0
        && query_batch_buf.size(0) >= num_long) {
        for (int64_t j = 0; j < num_long; ++j) {
            auto qrow = query_states.index({long_start[j]});          // [num_heads, head_dim]
            // qrow is [kvH*g, D] contiguous. view directly to [kvH, g, D] — the
            // memory layout (kvH outer, g inner) already matches the permute
            // result, so the permute + contiguous materialize kernel is dropped.
            auto q = qrow.view({num_kv_heads, group_size, head_dim}); // [kvH, g, D] contiguous
            query_batch_buf.index({j}).copy_(q);                    // [kvH,g,D] -> slot j
        }
        query_batch = query_batch_buf.narrow(0, 0, num_long);        // [num_long,kvH,g,D]
    } else {
        std::vector<torch::Tensor> query_vec;
        query_vec.reserve(num_long);
        for (int64_t j = 0; j < num_long; ++j) {
            auto qrow = query_states.index({long_start[j]});          // [num_heads, head_dim]
            auto q = qrow.view({num_kv_heads, group_size, head_dim}); // [kvH, g, D] contiguous
            query_vec.push_back(q);
        }
        query_batch = torch::stack(query_vec, /*dim=*/0);           // [num_long,kvH,g,D]
    }
    _sim_nvtx_pop();  // end SIM[query:build+copy]

    // ---- build centroid + cluster_size batches ----
    // CUDA-Graph path (use_persistent) requires a FIXED n_cent across every
    // layer so the _out buffers' last dim never triggers a resize (which
    // would move the buffer address and break graph replay). We force n_cent
    // to the buffer's width (= staging max_n_centroids) and pad both the
    // centroids and the pad mask up to it. The pad mask marks every position
    // >= each req's real n_centroids (incl. the 6072..6240 gap) as masked, so
    // the extra columns are computed but never affect topk — bit-identical to
    // the unpadded computation. The non-persistent (allocating) path keeps the
    // natural per-call n_cent (no graph, no fixed-width requirement).
    _sim_nvtx_push("SIM[centroid:build+copy]");
    // Move 3: when Python _restack_per_layer already filled the batch buffers
    // (signaled by empty prestacked tensors on the persistent, non-staging
    // path), read them directly and skip the C++ copy — avoids the double copy
    // (Python stack + C++ buf copy) of the old prestacked path.
    const bool buf_pre_filled = (use_persistent
                                 && centroids_prestacked.numel() == 0
                                 && cluster_size_prestacked.numel() == 0);
    torch::Tensor centroids_batch, cluster_size_batch;
    if (buf_pre_filled) {
        centroids_batch = centroids_batch_buf.narrow(0, 0, num_long);     // [num_long,kvH,max_n_cent,D]
        cluster_size_batch = cluster_size_batch_buf.narrow(0, 0, num_long);  // [num_long,kvH,max_n_cent]
    } else {
        std::vector<torch::Tensor> cent_vec;
        std::vector<torch::Tensor> csize_vec;
        cent_vec.reserve(num_long);
        csize_vec.reserve(num_long);
        // Phase 1 (CPU): resolve each req's centroid/cluster_size view (index/view,
        // no kernel) and collect the slices so Phase 2 can launch all copies back
        // to back with no CPU work between them.
        std::vector<torch::Tensor> c_slices;   // [kvH, max_n_cent, D] bf16 per long req
        std::vector<torch::Tensor> cs_slices;   // [kvH, max_n_cent] int32 per long req
        c_slices.reserve(num_long);
        cs_slices.reserve(num_long);
        for (int64_t j = 0; j < num_long; ++j) {
            torch::Tensor c_slice;      // [kvH, max_n_cent, D] bf16
            torch::Tensor cs_slice;      // [kvH, max_n_cent] int32
            if (use_staging) {
                int64_t slot = slots_ptr[j];
                c_slice = centroid_staging.index({slot, layer_idx});       // [kvH,max_n_cent,D]
                cs_slice = cluster_size_staging.index({slot, layer_idx});   // [kvH,max_n_cent]
            } else {
                c_slice = centroids_prestacked.index({j});                 // [kvH,max_n_cent,D]
                cs_slice = cluster_size_prestacked.index({j});              // [kvH,max_n_cent]
            }
            // On the persistent path copy the slice into the fixed-width buffer
            // (narrow-copy only the real columns; the pad mask masks the tail, so
            // no zeroing needed). This drops the per-layer zeros()+copy_ padding.
            if (use_persistent) {
                c_slices.push_back(c_slice);
                cs_slices.push_back(cs_slice);
            } else {
                cent_vec.push_back(c_slice);
                csize_vec.push_back(cs_slice);
            }
        }
        // Phase 2 (GPU): launch all copy_ kernels back to back with no CPU work
        // between them to minimize the inter-kernel launch gap.
        if (use_persistent) {
            for (int64_t j = 0; j < num_long; ++j) {
                const auto& c = c_slices[j];
                if (c.size(1) >= max_n_centroids) {
                    centroids_batch_buf.index({j}).copy_(c);
                } else {
                    centroids_batch_buf.index({j}).narrow(1, 0, c.size(1)).copy_(c);
                }
            }
            for (int64_t j = 0; j < num_long; ++j) {
                const auto& cs = cs_slices[j];
                if (cs.size(1) >= max_n_centroids) {
                    cluster_size_batch_buf.index({j}).copy_(cs);
                } else {
                    cluster_size_batch_buf.index({j}).narrow(1, 0, cs.size(1)).copy_(cs);
                }
            }
        }
        if (use_persistent) {
            centroids_batch = centroids_batch_buf.narrow(0, 0, num_long);   // [num_long,kvH,max_n_cent,D]
            cluster_size_batch = cluster_size_batch_buf.narrow(0, 0, num_long);  // [num_long,kvH,max_n_cent]
        } else {
            centroids_batch = torch::stack(cent_vec, /*dim=*/0);        // [num_long,kvH,max_n_cent,D]
            cluster_size_batch = torch::stack(csize_vec, /*dim=*/0);    // [num_long,kvH,max_n_cent]
        }
    }
    _sim_nvtx_pop();  // end SIM[centroid:build+copy]

    // The batch is uniform-padded to the staging/prestacked width (which is >=
    // max_n_cent). Use that width for the GEMM; mask out everything at/after
    // each req's real n_centroids so padded positions can never win topk.
    const int64_t n_cent = use_persistent ? max_n_centroids : centroids_batch.size(2);
    _sim_nvtx_push("SIM[cmask:build+copy]");
    // cmask = (cluster_size==0) | pad_mask. The pad mask (index >= real
    // n_centroids) is layer-invariant and precomputed once per step by the
    // Python pre-filter (pad_mask_bh); the empty-cluster mask (cluster_size==0)
    // is layer-dependent and recomputed per layer. Move 2b: on the graph path
    // both ops are captured INSIDE the graph (sim_graph_capture), so this block
    // builds cmask graph-OUTSIDE only for the non-graph paths.
    const int64_t batch = num_long * num_kv_heads;
    const int64_t max_k_fixed = use_persistent ? topk_values_buf.size(1) : 0;
    const bool use_persistent_topk = (use_persistent
        && topk_values_buf.size(0) >= batch
        && max_k_fixed >= 1
        && max_k_fixed <= n_cent);
    const bool graph_eligible = (use_persistent_topk
                                 && sim_graph_enabled()
                                 && num_long >= 1
                                 && num_long <= SIM_GRAPH_MAX_LONG
                                 && pad_mask_bh.numel() > 0);

    // Pad pad_mask_bh up to max_n_centroids with True if narrower. Usually
    // dead on the persistent path (Move 2a writes it at full width into a
    // fixed buffer); kept as a fallback for the non-persistent case.
    torch::Tensor pm = pad_mask_bh;
    if (use_persistent && pm.numel() > 0 && pm.size(2) < max_n_centroids) {
        auto pm_full = torch::ones({pm.size(0), pm.size(1), max_n_centroids},
                                   pm.options().dtype(torch::kBool));
        pm_full.narrow(2, 0, pm.size(2)).copy_(pm);
        pm = pm_full;
    }

    torch::Tensor cmask;
    // Move 2b: on the graph path, cmask (eq_out + bitwise_or_) is computed
    // INSIDE the captured graph (sim_graph_capture), so we skip the graph-
    // OUTSIDE cmask build entirely here. Only the non-graph paths build cmask
    // graph-OUTSIDE into cmask_buf / a fresh tensor and feed it as `sf`.
    if (!graph_eligible) {
        if (pad_mask_bh.numel() > 0) {
            if (use_persistent) {
                // GPU-kernel-minimized path: write the empty-cluster mask (==0)
                // DIRECTLY into the fixed-address cmask_buf via eq_out (no temp
                // bool tensor, no separate copy_), then OR the pad mask in place
                // via bitwise_or_. This is 2 GPU launches instead of the old 3
                // (==0 temp | temp-OR + copy_) AND drops the two temp bool tensor
                // allocations — the inter-kernel launch gaps (the dominant cost
                // nsys showed here) shrink with fewer host->device round trips.
                auto cmask_view = cmask_buf.narrow(0, 0, num_long);        // [num_long,kvH,n_cent] bool
                at::eq_out(cmask_view, cluster_size_batch, /*other=*/0);  // empty-mask -> buf
                cmask_view.bitwise_or_(pm);                                // OR pad mask in place
                cmask = cmask_view;
            } else {
                auto empty_mask = (cluster_size_batch == 0);             // [num_long,kvH,n_cent] bool
                cmask = empty_mask | pm;                                  // [num_long,kvH,n_cent] bool
            }
        } else {
            auto mask_range = torch::arange({n_cent}, torch::TensorOptions().dtype(torch::kInt64).device(cluster_size_batch.device()));
            auto nc_per_req_cpu = torch::tensor(n_centroids_vec,
                                                torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
            auto nc_per_req = nc_per_req_cpu.to(cluster_size_batch.device()).view({num_long, 1});
            auto pad_mask = mask_range.unsqueeze(0) >= nc_per_req;      // [num_long, n_cent] bool
            auto empty_mask = (cluster_size_batch == 0);                // [num_long,kvH,max_n_cent] bool
            auto pad_mask_bh_l = pad_mask.unsqueeze(1).expand({num_long, num_kv_heads, n_cent});
            cmask = empty_mask | pad_mask_bh_l;                         // [num_long,kvH,max_n_cent] bool
            if (use_persistent) {
                cmask_buf.narrow(0, 0, num_long).copy_(cmask);
                cmask = cmask_buf.narrow(0, 0, num_long);                // [num_long,kvH,n_cent] bool
            }
        }
    }
    _sim_nvtx_pop();  // end SIM[cmask:build+copy]

    // The GPU op block (bmm..topk) is factored into sim_compute_gpu. Move 2b:
    // on the graph path cmask is captured INSIDE the graph (8 ops), so `sf` is
    // only needed for the non-graph paths. topk_k is fixed to the buffer's
    // max_k on the persistent path so the captured graph always asks the same
    // k; Python slices [:real_k] afterward.
    auto qf = query_batch.view({batch, group_size, head_dim});
    auto cf = centroids_batch.view({batch, n_cent, head_dim});
    auto sf = cmask.defined() ? cmask.view({batch, n_cent}) : torch::Tensor();

    int64_t max_topk = 0;
    for (int64_t v : nprobe_per_req) if (v > max_topk) max_topk = v;
    if (max_topk < 1) max_topk = 1;

    // SAFETY: clamp max_topk to dist's last dim. torch::topk raises
    // "selected index k out of range" if k > dim size. This should never fire
    // (nprobe<1 => round(nc*nprobe) <= nc <= n_cent), but if it does it points
    // to a shape/nc mismatch we must see — log the offending values instead of
    // crashing so the real pipeline keeps running and we get the diagnostic.
    if (max_topk > n_cent) {
        std::fprintf(stderr,
            "[NELSSA][SIM-TOPK-ERR] layer=%ld max_topk=%ld > n_cent=%ld "
            "(num_long=%ld use_staging=%d) nprobe_per_req=[",
            (long)layer_idx, (long)max_topk, (long)n_cent,
            (long)num_long, (int)use_staging);
        for (int64_t j = 0; j < num_long; ++j)
            std::fprintf(stderr, "%ld%s", (long)nprobe_per_req[j],
                         (j + 1 < num_long) ? "," : "");
        std::fprintf(stderr, "] n_centroids_vec=[");
        for (int64_t j = 0; j < num_long; ++j)
            std::fprintf(stderr, "%ld%s", (long)n_centroids_vec[j],
                         (j + 1 < num_long) ? "," : "");
        std::fprintf(stderr, "]\n");
        std::fflush(stderr);
        max_topk = n_cent;  // clamp to avoid crash; diagnostic above tells why
    }

    const int64_t topk_k = use_persistent_topk ? max_k_fixed : max_topk;

    _sim_nvtx_push("SIM[compute:graph]");
    // CUDA-Graph path (persistent topk only): capture once per num_long (lazy),
    // replay every layer. The graph (8 ops incl. cmask) reads the per-layer
    // graph-OUTSIDE-copied input buffers and writes the persistent OUTPUT
    // buffers. Falls back to eager when the graph is disabled
    // (NELSSA_SIM_GRAPH=0) or capture is unsafe (num_long out of pool range).
    torch::Tensor topk_values, topk_indices;
    if (graph_eligible) {
        std::lock_guard<std::mutex> lk(g_sim_graph_mutex);
        SimGraphSlot& slot = g_sim_graph_slots[num_long];
        if (!slot.captured) {
            // First eligible call: capture the 8 GPU ops (cmask + bmm..topk)
            // then run once to populate the output buffers.
            sim_graph_capture(slot, num_long, qf, cf, cluster_size_batch, pm,
                              cmask_buf, n_cent, topk_k,
                              scores_buf, sm_buf, dist_buf,
                              topk_values_buf, topk_indices_buf,
                              rsqrt_dim, batch);
        } else if (slot.batch == batch && slot.n_cent == n_cent
                   && slot.topk_k == topk_k) {
            // Same scalars -> replay recomputes cmask inside the graph from the
            // freshly-copied cluster_size_batch + pm and writes the output buffers.
            sim_graph_replay(slot);
        } else {
            // Scalar mismatch (shouldn't happen with fixed buffers): fall back to
            // eager. The graph path skipped the graph-OUTSIDE cmask build, so
            // compute cmask on the fly here. Do NOT re-capture (changes baked
            // scalars).
            auto cmask_view = cmask_buf.narrow(0, 0, num_long);
            at::eq_out(cmask_view, cluster_size_batch, /*other=*/0);
            cmask_view.bitwise_or_(pm);
            std::tie(topk_values, topk_indices) = sim_compute_gpu(
                qf, cf, cmask_view.view({batch, n_cent}), n_cent, topk_k,
                /*use_persistent=*/true,
                scores_buf, sm_buf, dist_buf, topk_values_buf, topk_indices_buf,
                rsqrt_dim, batch);
        }
        // Output views into the persistent OUTPUT buffers (populated either by
        // Output views into the persistent OUTPUT buffers (the scalar-mismatch
        // branch above already set these to sim_compute_gpu's returned views
        // into the same buffers, so skip the re-slice there).
        if (!topk_values.defined() || !topk_indices.defined()) {
            topk_values = topk_values_buf.narrow(0, 0, batch);     // [batch, max_k]
            topk_indices = topk_indices_buf.narrow(0, 0, batch);    // [batch, max_k]
        }
    } else {
        // Non-graph path: cmask was computed into cmask_buf above; feed `sf`.
        std::tie(topk_values, topk_indices) = sim_compute_gpu(
            qf, cf, sf, n_cent, topk_k, use_persistent_topk,
            scores_buf, sm_buf, dist_buf, topk_values_buf, topk_indices_buf,
            rsqrt_dim, batch);
    }
    _sim_nvtx_pop();  // end SIM[compute:graph]

    _sim_nvtx_push("SIM[pack:cpu]");
    // ---- pack outputs ----
    out[0] = topk_indices;                                          // [num_long*kvH, max_topk] int64 GPU
    out[1] = topk_values;                                           // [num_long*kvH, max_topk] bf16 GPU
    out[2] = torch::tensor(nprobe_per_req, opts_i32_cpu);           // [num_long] int32 CPU
    out[3] = torch::tensor(nprobe_per_req, opts_i32_cpu);          // == topk_per_req (topk==nprobe now)
    out[4] = torch::tensor(long_req_indices, opts_i32_cpu);        // [num_long] int32 CPU
    _sim_nvtx_pop();  // end SIM[pack:cpu]
    _sim_nvtx_pop();  // end SIM[total]
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reorganize_by_clusters_cpu", &reorganize_by_clusters_cpu,
          "NELSSA multi-threaded CPU KV Cache reorganization by clusters "
          "(per-head stride = total_tokens*head_dim, OpenMP static scheduling)");
    m.def("reorganize_by_clusters_cpu_batch", &reorganize_by_clusters_cpu_batch,
          "NELSSA CPU KV Cache reorganization for a batch of layers in parallel "
          "(OpenMP across layers, per-head stride = total_tokens*head_dim)");
    m.def("gather_selected_clusters_cpu", &gather_selected_clusters_cpu,
          "NELSSA CPU sparse gather of selected clusters from a reorganized KV "
          "buffer (per-head contiguous memcpy, OpenMP across kv_heads)");
    m.def("gather_selected_clusters_batch_cpu", &gather_selected_clusters_batch_cpu,
          "NELSSA CPU sparse gather for a BATCH of requests in one dispatch: "
          "N*kvH parallel memcpy into a preallocated [N,kvH,max_tokens,D] batched "
          "buffer (OpenMP across N*num_kv_heads). Per-req variable "
          "n_centroids/nprobe/num_tokens handled via list inputs.");
    m.def("cpu_attention_fused", &cpu_attention_fused,
          "NELSSA fused CPU attention: gather'd KV -> QK matmul + in-place softmax "
          "+ AV matmul in one call (2 torch dispatches, no attn_weights tensor).");
    m.def("perform_similarity_search", &perform_similarity_search,
          "NELSSA similarity search: per-req loop + stack + batched centroid "
          "search (bmm/softmax/sum/mask/topk) + extraction in one C++ call. "
          "GPU ops dispatched to CUDA via libtorch; bit-identical to the Python "
          "batched_centroid_search path.");
}
