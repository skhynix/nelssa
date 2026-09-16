# NELSSA GPU Project

## Project Overview
NELSSA GPU is a distributed long-context LLM inference system that extends the RetroInfer architecture with PNM (Processing Near Memory) support. It implements attention-aware vector indexing to accelerate LLM inference by exploiting attention sparsity and distributing KV cache across GPU-CPU-PNM infrastructure.

## Architecture

### Core Components

**1. Cache Hub** (`cache_hub/`)
- `nelssa_cache.py`: Main NELSSA cache implementation with PNM distributed support
  - Manages steady zone, retrieval zone, and estimation zone KV cache
  - Handles batched layer transfer for KV cache offloading
  - Integrates FlashAttention for efficient attention computation
  - Supports GQA (Grouped Query Attention) with LSE (Log-Sum-Exp) tracking
- `retroinfer_cache.py`: Original RetroInfer CPU-based cache
- `retroinfer_cache_gpu.py`: GPU-only RetroInfer cache variant
- `flash_attn_cache.py`: FlashAttention-optimized cache
- `cache.py`: Base KV_Cache class defining common interface
- `kmeans.py`: K-means clustering for index construction

**2. Attention Hub** (`attn_hub/`)
- `nelssa_attn.py`: NELSSA decode attention interface
- `retroinfer_attn.py`: RetroInfer attention decoder
- `xattn.py`: XAttention implementation for prefill acceleration
- `minfer.py`: MInference implementation for sparse prefill
- `full_attn.py`: Full flash attention fallback

**3. Model Hub** (`model_hub/`)
- `llama.py`: Llama model implementation (Llama-3, Llama-3.1, DeepSeek-R1-Distill-Llama)
- `qwen.py`: Qwen model implementation (Qwen2.5-7B/72B)
- `LLM.py`: Base LLM class with prefill/decode pipeline
- `minfer_patterns.py`: MInference patterns for sparse attention
- `xattn_thresholds.py`: XAttention thresholds

**4. Communication Layer** (`nelssa_comm/`)
- `NelssaClient.py`: Client for PNM (Processing Near Memory) communication
  - gRPC-based coordination with PNM server
  - RDMA-based KV cache transfer using C++ wrapper
  - Async decode execution with doorbell polling
- Protocol buffers: `nelssa_comm_pb2.py`, `nelssa_comm_pb2_grpc.py`

**5. Configuration** (`config/`)
- `config.py`: Central configuration management
- Model-specific configs: `Llama-3-8B-Instruct-1048k.json`, `Qwen2.5-7B-Instruct.json`, etc.

## Key Concepts

### Attention Zones
1. **Steady Zone**: Static pattern region (first N tokens) with full attention
2. **Retrieval Zone**: Dynamically retrieved tokens based on vector similarity
3. **Estimation Zone**: Clustered tokens with approximate attention computation

### GQA (Grouped Query Attention)
- Multiple query heads share KV heads
- `group_size = num_heads // kv_head`
- Output shape: `[batch_groups, group_size, seq_len, head_dim]`
- `batch_groups = batch_size * kv_head`

### FlashAttention Integration
- Input shape: `[batch, seq_len, num_heads, head_dim]`
- Requires `permute(0, 2, 1, 3).contiguous()` for shape transformations
- LSE output: `[batch, num_heads]` - must be reshaped to match native format

### KV Cache Management
- **Steady Zone**: Fixed tokens at beginning (static_pattern_start + static_pattern_end)
- **Dynamic Update**: Index update every UPDATE_SEGMENT (1024 tokens)
- **Cluster-based**: K-means clustering with configurable n_centroids
- **Page-based**: 8 vectors per page, pages_per_cluster defines cluster size

## Current Development Focus

### Active Branch: `feature/nelssa_node_integ`
- Integrating NELSSA node communication with PNM support
- Recent work on FlashAttention LSE correctness (permute vs view)
- Batched layer transfer for KV cache offloading optimization

### Recent Changes
- Fixed FlashAttention input tensor transformation using `permute(0, 2, 1, 3).contiguous()` instead of `view()`
- Added steady zone attention comparison tests
- Implemented async decode execution with doorbell polling

## Running the System

### Environment Setup
```bash
conda create -n retroinfer python=3.10 -y
conda activate retroinfer
conda install -y mkl
conda install -c conda-forge libstdcxx-ng -y
pip install pip==25.0
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
pip install flashinfer-python==0.2.4 -i https://flashinfer.ai/whl/cu124/torch2.5/
```

### Simple Test
```bash
python simple_test.py --batch_size 4 --attn_type NELSSA --pnm_host 10.0.0.2 --pnm_port 50058
```

### Key Arguments
- `--attn_type`: `Full_Flash_Attn`, `RetroInfer`, or `NELSSA`
- `--retrieval_budget`: Ratio of tokens to retrieve (default: 0.018)
- `--estimation_budget`: Ratio for estimation (default: 0.232)
- `--cache_ratio`: Ratio of cache size to sequence length
- `--pnm_host`, `--pnm_port`: PNM server address for NELSSA mode

## Testing
- `tests/test_steady_zone_attention.py`: Compare native vs FlashAttention outputs
- `test_nelssa_node.py`: NELSSA node operation consistency verification
- `test_layer_attention.py`: Layer-level attention testing

## Technical Notes

### Tensor Shape Transformations
- **CRITICAL**: Use `permute()` not `view()` for dimension reordering
- `permute(0, 2, 1, 3)` actually rearranges data in memory
- `view().contiguous()` only reinterprets memory without reordering

### LSE (Log-Sum-Exp) Handling
- Native format: `[batch_groups, 1, group_size, 1]`
- FlashAttn format: `[batch, num_heads]`
- Conversion requires proper reshape through `[batch, kv_head, group_size]`

### RDMA Communication
- Uses C++ wrapper (`nelssa_wrapper.so`) for low-latency KV cache transfer
- Async execution with doorbell polling for decode operations
- gRPC for coordination, RDMA for data plane

### Wave Buffer Async Construction & Sync

#### Layer Prefill Flow (`LLM.py:layer_prefill`)
```
layer_prefill(layer_idx, start_bdx, hidden_states)
├── 1. LayerNorm (input)
├── 2. WQKV projection
├── 3. RoPE (position embedding)
├── 4. reshape: [bs, seq_len, dim] → [bs, seq_len, head, head_dim]
├── 5. prefill_update_kv_cache()  ← KV cache 업데이트 (async offload 시작)
├── 6. prefill_attention()        ← Attention 연산
├── 7. kv_cache.sync()            ← ★ Wave Buffer sync
├── 8. Wo projection + residual
└── 9. LayerNorm + MLP + residual
```

#### Sync Timing Comparison

| Mode | Construction Sync Timing |
|------|-------------------------|
| **NELSSA** | `sync()` 내에서 동기적으로 대기 (layer_prefill 이 blocking) |
| **RetroInfer** | `sync()` 은 construction 시작만, 다음 레이어 시작 시점에 이전 레이어 완료 대기 (비동기 파이프라인) |

#### NELSSA Mode (`nelssa_cache.py:sync`)
```python
def sync(self, layer_idx, start_bdx):
    if self.build_index_when_prefilling:
        self.copyevents[...].synchronize()           # offload 완료 대기
        self.wave_buffer[layer_idx].async_construction(...)  # 비동기 construction 시작
        self.wave_buffer[layer_idx].construction_sync()      # ★ construction 완료 대기
        if start_bdx == self.batch_size - 1:
            self._send_layer_to_nelssa(layer_idx)
```

#### RetroInfer Mode (`retroinfer_cache.py`)

**`sync()` 메서드**:
```python
def sync(self, layer_idx, start_bdx):
    if self.build_index_when_prefilling:
        self.copyevents[...].synchronize()           # offload 완료 대기
        self.wave_buffer[layer_idx].async_construction(...)  # ★ 시작만 하고 바로 리턴
        # construction_sync() 호출 없음!
```

**`prefill_update_kv_cache()` 에서 sync**:
```python
def prefill_update_kv_cache(self, query_states, key_states, value_states, layer_idx, start_bdx):
    if self.build_index_when_prefilling:
        # sync for the previous layer and batch finish their page organization
        if layer_idx > 0:
            self.wave_buffer[layer_idx - 1].construction_sync()  # 이전 레이어 완료 대기
        elif start_bdx > 0:  # layer_idx == 0
            self.wave_buffer[self.layer_num - 1].construction_sync()  # 마지막 레이어 완료 대기
```

**마지막 배치/레이어 처리 (`prepare_cache()`)**:
```python
def prepare_cache(self):
    """Prefill → Decode 전환 시점 호출 (LLM.inference 에서 self.move() 통해)"""
    if self.build_index_when_prefilling:
        # sync the last batch of the last layer
        torch.cuda.synchronize()
        self.wave_buffer[self.layer_num - 1].construction_sync()  # ★ 마지막 레이어 완료 대기
        
        # clear temp memory
        self.clusters_cpu, self.cluster_size_cpu = None, None
        self.temp_keys, self.temp_values = None, None
        torch.cuda.empty_cache()
    
    if not self.allocated:
        # GPU 캐시 및 computation buffer 할당
        # - Block Cache (cache_keys, cache_values)
        # - Meta Index GPU 이동 (centroids, value_sum, centroids_mask, cluster_size)
        # - Computation Buffer (gemm_o, softmax_o, execution_buffer, etc.)
        self.allocate_computation_buffer()
```

#### Pipeline Timeline (RetroInfer)

```
Prefill 단계:
┌────────────────────────────────────────────────────────┐
│ Layer 0, Batch 0: [offload]→[attn]→[async_construct]  │
│ Layer 1, Batch 0: [offload]→[attn]→[async_construct]  │
│   ↑ Layer 0 construction_sync() 호출 (다음 레이어 시작 시)   │
│ ...                                                    │
│ Layer N-1, Batch B-1: [offload]→[attn]→[async_construct] │
│   ↑ Layer N-2 construction_sync() 호출                      │
└────────────────────────────────────────────────────────┘
                    ↓
Prefill 완료 후 sampling
                    ↓
self.move() 호출  ← ★ 여기서 마지막 레이어 construction_sync()
                    ↓
prepare_cache() → allocate_computation_buffer()
                    ↓
decode_forward (CUDAGraph Capture 포함)
```

#### `prepare_cache()` Detailed Actions

| 단계 | 동작 | 목적 |
|------|------|------|
| **1** | `construction_sync()` | 마지막 레이어 async construction 완료 대기 |
| **2** | 임시 메모리 정리 | Prefill 용도 CPU/GPU 메모리 해제 |
| **3** | Block Cache 할당 | Decode 용도 GPU KV 캐시 확보 |
| **4** | Meta Index GPU 이동 | 클러스터 인덱스를 GPU 로 전송 |
| **5** | Computation Buffer 할당 | Attention 연산용 버퍼 확보 |
| **6** | 포인터 설정 | 현재 레이어 버퍼 참조 |

**핵심**: `prepare_cache()` 는 **Prefill → Decode 전환 시점**에서 모든 GPU 리소스를 준비하는 **initialization barrier** 역할을 합니다.

### Future Work: NELSSA 레이어별 비동기 KV Cache 전송

#### 현재 문제점

```python
# nelssa_cache.py:sync()
def sync(self, layer_idx, start_bdx):
    if self.build_index_when_prefilling:
        self.copyevents[...].synchronize()           # offload 완료 대기
        self.wave_buffer[layer_idx].async_construction(...)  # construction 시작
        self.wave_buffer[layer_idx].construction_sync()      # ★ blocking
        if start_bdx == self.batch_size - 1:
            self._send_layer_to_nelssa(layer_idx)  # ★ synchronous 전송
```

1. `construction_sync()` 가 blocking → 다음 레이어 진행 불가
2. `_send_layer_to_nelssa()` 도 synchronous → 전송 완료까지 대기
3. **모든 배치가 완료된 후에만 전송 시작** → 파이프라이닝 비효율

#### 개선 방안: 전이중 파이프라인 (Full-Duplex)

```
Layer 0: [construction] → [send async] ─┐
Layer 1:            [construction] → [send async] ─┤
Layer 2:                       [construction] → [send async]
                                                    ↓
                                        [wait all complete]
```

#### 구현 세부 사항

**단계 1: NelssaClient 에 비동기 전송 메서드 추가**

```python
# nelssa_comm/NelssaClient.py
def send_kv_cache_async(self, layer_idx, k_tensor, v_tensor, s_tensor, n_centroids):
    """RDMA 를 통한 비동기 KV cache 전송"""
    request_token = self.rdma_client.send_async(...)
    self.pending_sends[layer_idx] = request_token
    return request_token

def wait_send_complete(self, layer_idx):
    """특정 레이어의 전송 완료 대기"""
    if layer_idx in self.pending_sends:
        self.pending_sends[layer_idx].wait()
        del self.pending_sends[layer_idx]
```

**단계 2: `sync()` 메서드 수정 (안전한 방식)**

```python
# nelssa_cache.py
def sync(self, layer_idx, start_bdx):
    if self.build_index_when_prefilling:
        # 1. offload 완료 대기
        self.copyevents[self.layer_mapping[str(layer_idx)]].synchronize()
        
        # 2. construction 시작 (비동기)
        self.wave_buffer[layer_idx].async_construction(
            self.clusters_cpu, self.cluster_size_cpu, start_bdx
        )
        
        # 3. 마지막 배치라면: transmission 시작
        if start_bdx == self.batch_size - 1:
            self._send_layer_to_nelssa_async(layer_idx)
```

**단계 3: `_send_layer_to_nelssa_async` 구현**

```python
def _send_layer_to_nelssa_async(self, layer_idx: int):
    """Start async transmission and return immediately"""
    try:
        k_tensor_full = self.list_keys[layer_idx]
        v_tensor_full = self.list_values[layer_idx]
        s_tensor_gpu = self.cluster_size[layer_idx]
        
        # CPU 로 복사
        s_valid = s_tensor_gpu.to(device="cpu", dtype=torch.int32).contiguous()
        k_valid = k_tensor_full.to(device="cpu", dtype=self.dtype).contiguous()
        v_valid = v_tensor_full.to(device="cpu", dtype=self.dtype).contiguous()
        
        # 비동기 전송 시작
        self.nelssa_client.send_kv_cache_async(
            layer_idx, k_valid, v_valid, s_valid, self.n_centroids
        )
        
        # 전송 완료 이벤트 등록
        self.send_events[layer_idx] = torch.cuda.Event()
        self.send_events[layer_idx].record()
        
    except Exception as e:
        print(f"[NELSSA] Async Send Error: {e}", flush=True)
        raise
```

**단계 4: Prefill 완료 시 모든 전송 완료 대기**

```python
def wait_all_transmissions_complete(self):
    """Wait for all pending layer transmissions to complete"""
    for layer_idx in range(self.layer_num):
        if layer_idx in self.nelssa_client.pending_sends:
            self.nelssa_client.wait_send_complete(layer_idx)
```

#### 파이프라인 타임라인 비교

**현재 (Sync 방식):**
```
Layer 0: [construct]═══[send]═══
Layer 1:             [construct]═══[send]═══
Layer 2:                          [construct]═══[send]═══
```

**개선 후 (Async 방식):**
```
Layer 0: [construct]→[send async]────────────┐
Layer 1:            [construct]→[send async]─┤
Layer 2:                       [construct]→[send async]
                                                ↓
                                     [wait all complete]
```

#### 대안: 더 공격적인 비동기화

construction 완료도 기다리지 않고 최대 병렬성 확보:

```python
def sync(self, layer_idx, start_bdx):
    if self.build_index_when_prefilling:
        # offload 완료만 대기
        self.copyevents[...].synchronize()
        
        # construction 은 백그라운드로 (대기하지 않음)
        self.wave_buffer[layer_idx].async_construction(...)
        
        # 마지막 배치라면: transmission 시작
        if start_bdx == self.batch_size - 1:
            self._send_layer_to_nelssa_async(layer_idx)
            self.pending_layers.append(layer_idx)
```

이 경우 **Thread Pool** 이나 **CUDA Graph** 를 활용해 백그라운드에서 construction 이 완료되도록 추가 구현 필요.

#### 구현 시 고려사항

| 항목 | 안전한 방식 | 공격적 방식 |
|------|------------|------------|
| Construction Sync | 유지 | 백그라운드 |
| 전송 Sync | 비동기 | 비동기 |
| 구현 난이도 | 하 | 중 |
| 병렬성 | 중 | 상 |
| 안정성 | 상 | 중 |

**추천**: 먼저 **안전한 방식**으로 구현 후, 프로파일링 결과에 따라 공격적 방식 고려

## File Structure
```
/home/sylee/git/nelssa_gpu/
├── cache_hub/          # KV cache implementations
├── attn_hub/           # Attention computation methods
├── model_hub/          # Llama/Qwen model implementations
├── nelssa_comm/        # PNM communication layer
├── config/             # Model configurations
├── tests/              # Test files
├── benchmark/          # RULER, LongBench evaluation scripts
├── simple_test.py      # Basic functionality test
└── CLAUDE.md           # This file
```
