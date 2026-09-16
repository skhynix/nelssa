import os
import sys
import time

import numpy as np
import torch

# 현재 파일 위치(nelssa_comm)를 import 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # 7단계에서 만든 최종 API 클래스
    from NelssaClient import NelssaClient
except ImportError as e:
    print(f"Import Error: {e}\nNelssaClient.py가 있는지 확인하세요.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"NelssaClient 로딩 중 에러: {e}", file=sys.stderr)
    sys.exit(1)

# --- NELSSA 서버의 RoCE IP ---
NELSSA_SERVER_IP = "10.0.0.2"
NELSSA_SERVER_PORT = 50058  # [수정] 포트 번호를 정수로 변경 (gRPC는 정수를 선호)

# --- [중요] 님의 `nelssa_attention.hpp`와 일치하는 수동 Config ---
DUMMY_MODEL_CONFIG = {
    "num_layers": 1,
    "num_heads": 32,
    "kv_heads": 8,
    "head_dim": 128,
    "n_clusters": 1024,
    "cache_unit_size": 16,
    "n_probe": 100,
}


def create_dummy_kv_bin_files(data_path="test_kv_data"):
    """
    테스트용 더미 KV 캐시 (.bin) 파일 생성
    (NELSSAAttention이 RAM에 로드할 float32 데이터)
    """
    print(f"테스트용 더미 KV 캐시 파일 생성 중: {data_path}")
    os.makedirs(data_path, exist_ok=True)

    cfg = DUMMY_MODEL_CONFIG

    # [수정] nelssa_attention.hpp L:104의 계산식과 일치시킴
    batch_groups = 1 * cfg["kv_heads"]
    kv_size_elements = (
        batch_groups * cfg["n_clusters"] * cfg["cache_unit_size"] * cfg["head_dim"]
    )

    # float 배열 생성 (NumPy 사용이 빠름)
    k_data = np.random.rand(kv_size_elements).astype(np.float32)
    v_data = np.random.rand(kv_size_elements).astype(np.float32)

    k_file = os.path.join(data_path, "layer_0_keys.bin")
    v_file = os.path.join(data_path, "layer_0_values.bin")

    k_data.tofile(k_file)
    v_data.tofile(v_file)

    print("더미 KV 캐시 파일 생성 완료.")
    return k_file, v_file


def load_bin_to_pinned_tensor(file_path, shape):
    """.bin 파일을 읽어 float16 Pinned Memory 텐서로 변환 및 Reshape"""
    file_data = np.fromfile(file_path, dtype=np.float32)
    # 서버가 sizeof(uint16_t)로 할당하므로, float16(half)으로 변환하여 바이트 크기를 맞춤
    tensor = torch.from_numpy(file_data).half().pin_memory()
    return tensor.view(shape)


def run_test():

    # --- 1. 클라이언트 초기화 ---
    print("--- 1. NELSSA 클라이언트 초기화 ---")
    try:
        # [중요] 수동 Config를 주입하여 클라이언트 생성
        client = NelssaClient(NELSSA_SERVER_IP, NELSSA_SERVER_PORT, DUMMY_MODEL_CONFIG)
    except Exception as e:
        print(f"클라이언트 초기화 실패: {e}")
        return
    print("--- 초기화 성공 ---")

    # --- 2. Prefill (실제 .bin 파일 전송) ---
    print("\n--- 2. Prefill API (실제 .bin 파일 전송) ---")

    k_file_path, v_file_path = create_dummy_kv_bin_files()

    try:
        # .bin 파일을 Pinned Memory 텐서로 로드 및 Reshape (FIX 1 적용됨)
        cfg = DUMMY_MODEL_CONFIG
        bsz = 1
        kv_heads = cfg["kv_heads"]
        n_clusters = cfg["n_clusters"]
        cache_unit_size = cfg["cache_unit_size"]
        head_dim = cfg["head_dim"]

        # 기대 shape: [bsz, kv_heads, n_clusters * cache_unit_size, head_dim]
        k_shape = (bsz, kv_heads, n_clusters * cache_unit_size, head_dim)
        v_shape = (bsz, kv_heads, n_clusters * cache_unit_size, head_dim)

        k_tensor_pinned = load_bin_to_pinned_tensor(k_file_path, k_shape)
        v_tensor_pinned = load_bin_to_pinned_tensor(v_file_path, v_shape)

        # [추가] NelssaClient.send_kv_cache에 필요한 추가 텐서 생성
        # size_cache_tensor shape: [bsz, kv_heads, n_clusters] (더미 데이터)
        size_cache_tensor = torch.randint(
            0, 16, (bsz, kv_heads, n_clusters), dtype=torch.int32
        ).pin_memory()
        current_n_clusters = n_clusters

        start = time.time()
        # [API 호출] send_kv_cache 호출 (인자 업데이트)
        client.send_kv_cache(
            layer_idx=0,
            k_cache_tensor=k_tensor_pinned,
            v_cache_tensor=v_tensor_pinned,
            size_cache_tensor=size_cache_tensor,
            current_n_clusters=current_n_clusters,
        )
        end = time.time()

        total_mb = (k_tensor_pinned.nbytes + v_tensor_pinned.nbytes) / (1024 * 1024)
        print(
            f" > K/V Cache ({total_mb:.2f} MB) 파일 전송 성공! (시간: {end - start:.2f}s)"
        )

    except Exception as e:
        print(f"Prefill API 실패: {e}")
        import traceback

        traceback.print_exc()
        return
    print("--- Prefill API 테스트 성공 ---")

    # --- 3. Decode (실제 C++ Attention 연산 테스트) ---
    print("\n--- 3. Decode API (실제 텐서 전송 및 C++ 연산) ---")
    try:
        # Config 기반 파라미터
        cfg = DUMMY_MODEL_CONFIG
        bsz = 1
        num_heads = cfg["num_heads"]
        kv_heads = cfg["kv_heads"]
        head_dim = cfg["head_dim"]
        n_probe = cfg["n_probe"]

        # [수정] execute_decode_batched API에 맞게 텐서 생성
        # queries shape = [bsz, 1, n_heads, head_dim]
        dummy_query_tensor = torch.randn(
            bsz, 1, num_heads, head_dim, dtype=torch.float32
        ).pin_memory()

        # cluster_ids shape = [bsz * kv_heads, n_probe]
        dummy_cluster_ids = torch.randint(
            0, cfg["n_clusters"], (bsz * kv_heads, n_probe), dtype=torch.int32
        ).pin_memory()

        start = time.time()
        # [API 호출] execute_decode_batched 호출
        result_tensor = client.execute_decode_batched(
            layer_idx=0,
            bsz=bsz,
            queries=dummy_query_tensor,
            cluster_ids=dummy_cluster_ids,
        )
        end = time.time()

        latency_ms = (end - start) * 1000
        print(f" > Decode 요청/응답 성공! (E2E Latency: {latency_ms:.2f} ms)")
        print(f" > 받은 결과 텐서 Shape: {result_tensor.shape}")

        # 기대 shape: [bsz * kv_heads, 1, group_size, head_dim + 2]
        group_size = num_heads // kv_heads
        expected_shape = (bsz * kv_heads, 1, group_size, head_dim + 2)

        assert result_tensor.shape == expected_shape
        print(f" > 텐서 Shape 검증 성공: {result_tensor.shape} == {expected_shape}")

        # 값이 0이 아닌지 확인
        assert torch.any(result_tensor != 0)
        print(" > 결과 텐서 데이터 검증 성공 (결과가 0이 아님)")

    except Exception as e:
        print(f"Decode API 실패: {e}")
        import traceback

        traceback.print_exc()
        return
    print("--- Decode API 테스트 성공 ---")

    print("\n[최종 결론] 모든 API가 성공적으로 통합 및 테스트되었습니다.")
    print("`NelssaClient` 클래스를 RetroInfer에 이식할 준비가 완료되었습니다.")


if __name__ == "__main__":
    run_test()
