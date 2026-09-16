# NELSSA Cache 구현 계획

## 개요
기존 RetroInfer 의 CPU offload 방식을 NELSSA-PNM 아키텍처로 확장하여 PCIe 병목 해소를 목표로 함.

---

## 1. 디렉토리 구조

```
cache_hub/
├── cache.py              # 기존 KV_Cache 기본 클래스
├── retroinfer_cache.py   # 기존 RetroInfer 구현 (유지)
├── nelssa_cache.py       # [NEW] NELSSA-PNM 전용 캐시
├── kmeans.py             # 기존 kmeans (유용)
└── __init__.py

nelssa_comm/              # [NEW] 통신 계층
├── __init__.py
├── grpc_client.py        # GPU → NELSSA gRPC 클라이언트
├── grpc_server.py        # NELSSA 노드 gRPC 서버 (PNM 용)
├── rdma_client.py        # RDMA 전송 (선택: 더 빠른 데이터 이동)
└── protos/
    └── nelssa.proto      # gRPC 프로토콜 정의

nelssa_pnm/               # [NEW] PNM 연동 계층 (NELSSA 노드 측)
├── __init__.py
├── pnm_manager.py        # PNM 메모리 관리
├── sparse_attn.py        # PNM 내 sparse attention 연산
└── offline_softmax.py    # Offline softmax 계산
```

---

## 2. 클래스 상속 구조

```
KV_Cache (cache.py)
    ├── retroinfer_cache (기존 CPU offload)
    └── nelssa_cache (NEW, NELSSA-PNM offload)
```

---

## 3. `nelssa_cache.py` 주요 구성

### 3.1 `__init__` 파라미터 (retroinfer_cache + α)

```python
class nelssa_cache(KV_Cache):
    def __init__(
        self,
        # 기존 파라미터 (retroinfer_cache 와 동일)
        valid_start, layer_num, batch_size, max_length,
        num_key_value_heads, num_heads, head_dim, dtype,
        layer_mapping, max_new_length, static_pattern_start,
        static_pattern_end, n_centroids, n_segment,
        retrieval_budget, estimation_budget, cache_ratio,
        prefill_bsz, num_gpus, model_size,

        # [NEW] NELSSA 노드 연결 정보
        nelssa_node_address: str,      # 예: "192.168.1.100:50058"
        nelssa_node_id: str,           # 노드 식별자
        use_rdma: bool = True,         # RDMA 사용 여부
        rdma_device_id: int = 0,       # RDMA 장치 ID

        # [NEW] PNM 관련
        pnm_memory_size_gb: int = 32,  # PNM 메모리 크기
        pnm_compute_units: int = 8,    # PNM 연산 유닛 수
    ) -> None:
```

---

## 4. gRPC 프로토콜 설계 (`nelssa.proto`)

```protobuf
service NELSACacheService {
  // Prefill: KV cache 를 PNM 메모리로 전송
  rpc UploadKVCache (KVCacheRequest) returns (UploadResponse);

  // Decode: Query + top-k IDs 로 sparse attention 요청
  rpc SparseAttention (SparseAttentionRequest) returns (SparseAttentionResponse);

  // 인덱스 갱신 (필요시)
  rpc UpdateIndex (IndexUpdateRequest) returns (IndexUpdateResponse);

  // 헬스 체크
  rpc HealthCheck (HealthRequest) returns (HealthResponse);
}

message KVCacheRequest {
  string layer_id = 1;
  bytes kv_data = 2;           // 직렬화된 KV tensor
  int32 batch_size = 3;
  int32 seq_len = 4;
  int32 kv_head = 5;
  int32 head_dim = 6;
  string dtype = 7;
}

message SparseAttentionRequest {
  string layer_id = 1;
  bytes query = 2;             // [batch, 1, num_heads, head_dim]
  bytes cluster_ids = 3;       // top-k cluster IDs [batch*group, k]
  int32 n_centroids = 4;
  int32 es_cluster_num = 5;
}

message SparseAttentionResponse {
  bytes attention_output = 1;  // [batch, 1, num_heads, head_dim]
  bytes softmax_lse = 2;       // offline softmax 결과 (log-sum-exp)
  double latency_ms = 3;
}
```

---

## 5. `nelssa_cache.py` 핵심 메서드

| 메서드 | 역할 |
|--------|------|
| `connect_to_nelssa()` | gRPC/RDMA 연결 설정 |
| `upload_kv_cache_prefill()` | Prefill 후 KV 를 NELSSA 로 전송 |
| `sparse_attention_decode()` | Decode 시 PNM 에 sparse attention 요청 |
| `download_attention_result()` | PNM 결과 수신 |
| `update_index_if_needed()` | 시퀀스 길이에 따른 인덱스 갱신 |
| `close_connection()` | 연결 정리 |

---

## 6. 데이터 흐름

### Prefill Phase

```
GPU                              NELSSA Node (PNM)
 │                                    │
 │  [upload_kv_cache_prefill]         │
 │───────────────────────────────────▶│
 │  - KV cache (layer 별)             │  PNM 메모리에 저장
 │  - cluster centroids               │  클러스터 인덱스 구축
 │  - value_sum, cluster_size         │
 │                                    │
 │◀───────────────────────────────────│
 │  UploadResponse (ack)              │
 │                                    │
```

### Decode Phase

```
GPU                              NELSSA Node (PNM)
 │                                    │
 │  [sparse_attention_decode]         │
 │──────────────────────────────────▶│
 │  - query tensor                    │  PNM 에서 sparse attention
 │  - top-k cluster IDs               │  offline softmax 계산
 │                                    │
 │◀───────────────────────────────────│
 │  - attention_output (reduced)      │
 │  - softmax_lse                     │
 │                                    │
```

---

## 7. 구현 순서 (단계별)

| 단계 | 작업 |
|------|------|
| **1** | `nelssa_comm/protos/nelssa.proto` 정의 |
| **2** | gRPC 클라이언트/서버 스캐폴딩 생성 |
| **3** | `nelssa_cache.py` 기본 클래스 骨架 (retroinfer_cache 복제 후 수정) |
| **4** | `connect_to_nelssa()` 구현 |
| **5** | `upload_kv_cache_prefill()` 구현 |
| **6** | `sparse_attention_decode()` 구현 |
| **7** | RDMA 연동 (선택, 성능 최적화) |
| **8** | PNM 측 `pnm_manager.py`, `sparse_attn.py` 구현 |
| **9** | 통합 테스트 |

---

## 8. retroinfer_cache.py 와의 주요 차이점

| 항목 | retroinfer_cache | nelssa_cache |
|------|------------------|--------------|
| KV 저장 위치 | CPU 메모리 | NELSSA PNM 메모리 |
| 데이터 이동 | CPU ↔ GPU (PCIe) | GPU ↔ NELSSA (RDMA/Network) |
| Sparse Attention | GPU 에서 수행 | PNM 에서 수행 |
| WaveBufferCPU | 필요 | 불필요 (PNM 이 관리) |
| offline_softmax | GPU | PNM |

---

## 9. 아키텍처 다이어그램

```
┌─────────────┐         ┌─────────────────────────────────┐
│    GPU      │         │         NELSSA Node             │
│  (Query)    │────────▶│  ┌─────────────────────────┐    │
│             │◀────────│  │      PNM Memory         │    │
│  top-k IDs  │         │  │  - KV Cache 저장        │    │
│             │         │  │  - Sparse Attention     │    │
└─────────────┘         │  │  - Offline Softmax 결과 │    │
                        │  └─────────────────────────┘    │
                        │         (PNM 장착)              │
                        └─────────────────────────────────┘
```

---

## 10. 핵심 이점

- **PCIe 병목 해소**: KV 전체 전송 → query/top-k IDs + attention 결과 만 전송
- **Data Reduction**: PNM 이 연산 수행하여 축소된 결과만 반환
- **Offline Softmax**: PNM 에서 softmax 결과까지 계산하여 GPU 부하 감소
