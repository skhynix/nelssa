import sys
import uuid
from pathlib import Path

# Add current directory to path for relative imports
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

import grpc
import numpy as np
import torch

try:
    import nelssa_comm_pb2
    import nelssa_comm_pb2_grpc
except ImportError:
    print(
        "오류: nelssa_comm_pb2.py 파일이 없습니다. .proto 파일을 다시 컴파일하세요.",
        file=sys.stderr,
    )
    sys.exit(1)


try:
    import nelssa_wrapper
except ImportError:
    print(
        "오류: nelssa_wrapper.so 파일을 찾을 수 없습니다. C++ 래퍼를 다시 컴파일하세요.",
        file=sys.stderr,
    )
    sys.exit(1)


DTYPE_MAP = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}


class NelssaClient:
    def __init__(self, host: str, port: int, model_config: dict):
        print(f"[NelssaClient] Initializing... (Host: {host}:{port})")
        self.host = host
        self.port = port
        self.channel = None
        self.stub = None
        self.cpp_client = None
        self.model_config = model_config

        try:
            # 1. C++ RDMA 엔진 초기화
            head_dim_val = self.model_config.get("head_dim", 0)
            if head_dim_val == 0:
                raise ValueError("model_config에 'head_dim'이 0이거나 없습니다.")
            self.cpp_client = nelssa_wrapper.NelssaWrapper(head_dim_val)

            my_info = self.cpp_client.get_my_info()
            print("- C++ RDMA engine(decode buffer) init - DONE")

            # 2. gRPC 채널 연결
            self.channel = grpc.insecure_channel(f"{host}:{port}")
            self.stub = nelssa_comm_pb2_grpc.NelssaCoordinatorStub(self.channel)

            # 3. gRPC InitializeSession
            print("- gRPC InitializeSession trying...")
            dtype = DTYPE_MAP[model_config.get("kv_dtype", torch.float16)]
            print(f"dtype : {dtype}, type : {type(dtype)}, string type : {str}")
            config_proto = nelssa_comm_pb2.ModelConfig(
                num_layers=model_config.get("num_layers", 0),
                num_heads=model_config.get("num_heads", 0),
                kv_heads=model_config.get("kv_heads", 0),
                head_dim=model_config.get("head_dim", 0),
                n_clusters=model_config.get("n_clusters", 0),
                cache_unit_size=model_config.get("cache_unit_size", 0),
                n_probe=model_config.get("n_probe", 0),
                dtype=dtype,
            )

            self.num_heads = model_config.get("num_heads", 0)
            self.kv_heads = model_config.get("kv_heads", 0)
            self.head_dim = model_config.get("head_dim", 0)
            for attr, name in [
                (self.num_heads, "num_heads"),
                (self.kv_heads, "kv_heads"),
                (self.head_dim, "head_dim"),
            ]:
                if attr == 0:
                    raise ValueError(f"{name} cannot be zero")

            client_rdma_info = nelssa_comm_pb2.RdmaConnInfo(
                addr=my_info["addr"],
                rkey=my_info["rkey"],
                qp_num=my_info["qp_num"],
                lid=my_info["lid"],
                gid=my_info["gid"],
            )

            req = nelssa_comm_pb2.InitializeSessionRequest(
                config=config_proto, client_decode_info=client_rdma_info
            )
            resp = self.stub.InitializeSession(req, timeout=10.0)

            if not resp.success:
                raise RuntimeError("NELSSA server reporeted InitializeSession fail!")

            server_info = resp.server_decode_info
            print(f" > Handshake 성공! NELSSA QP: {server_info.qp_num}")

            # 4. C++ RDMA 엔진에 NELSSA 주소 등록
            self.cpp_client.connect(
                peer_addr=server_info.addr,
                peer_rkey=server_info.rkey,
                qpn=server_info.qp_num,
                lid=server_info.lid,
                gid_bytes=server_info.gid,
            )
            print("[NelssaClient] Client initialize & RDMA connected")

        except Exception as e:
            print(f"[NelssaClient] init faileed: {e}")
            if self.channel:
                self.channel.close()
            raise

    def __del__(self):
        if self.channel:
            self.channel.close()
            print("[NelssaClient] gRPC channel closed..")

    # --- Prefill API ---
    def send_kv_cache(
        self,
        layer_idx: int,
        k_cache_tensor: torch.Tensor,
        v_cache_tensor: torch.Tensor,
        size_cache_tensor: torch.Tensor,
        current_n_clusters: int,
    ):  # [인자 추가]
        bsz = k_cache_tensor.shape[0]
        kv_heads = k_cache_tensor.shape[1]
        seqlen = k_cache_tensor.shape[2]

        # if(layer_idx==0):
        #     # print(k_cache_tensor.shape)
        #     # print("write csv")
        #     # test = k_cache_tensor.view(-1, 128)
        #     # np_k = test.numpy()
        #     # np.savetxt('tensor.csv', np_k, delimiter=',', fmt='%.6f')
        #     with open("tensor_all.csv", "w") as f:
        #         for b in range(bsz):
        #             for h in range(k_cache_tensor.shape[1]):
        #                 f.write(f"# batch {b}, head {h}\n")
        #                 sub_tensor = k_cache_tensor[b, h].cpu().numpy()
        #                 np.savetxt(f, sub_tensor, delimiter=',', fmt='%.5f')
        #                 f.write("\n")
        #     with open("cluster_size.csv", "w") as f:
        #         for idx in range(bsz * head_num):
        #             size_tensor = size_cache_tensor[idx].cpu().numpy()
        #             size_tensor = size_tensor.reshape(1, -1)
        #             np.savetxt(f, size_tensor, delimiter=',', fmt='%d')

        k_cache_tensor = k_cache_tensor.view(-1)
        v_cache_tensor = v_cache_tensor.view(-1)
        size_cache_tensor = size_cache_tensor.view(-1)

        # 1. Pinning
        if not k_cache_tensor.is_pinned():
            k_cache_tensor = k_cache_tensor.pin_memory()
        if not v_cache_tensor.is_pinned():
            v_cache_tensor = v_cache_tensor.pin_memory()
        if not size_cache_tensor.is_pinned():
            size_cache_tensor = size_cache_tensor.pin_memory()

        # 2. Pointers & Sizes
        k_ptr = k_cache_tensor.data_ptr()
        v_ptr = v_cache_tensor.data_ptr()
        s_ptr = size_cache_tensor.data_ptr()

        k_bytes = k_cache_tensor.nbytes
        v_bytes = v_cache_tensor.nbytes
        s_bytes = size_cache_tensor.nbytes

        req_id = f"kv_{layer_idx}_{uuid.uuid4()}"

        try:
            # 3. Buffer Allocation Request
            # [중요] Proto 변수명과 일치시킴
            start_req = nelssa_comm_pb2.SendKVStartRequest(
                request_id=req_id,
                layer_idx=layer_idx,
                k_cache_bytes=k_bytes,
                v_cache_bytes=v_bytes,
                size_cache_bytes=s_bytes,  # size_cache_bytes
                current_n_clusters=current_n_clusters,  # current_n_clusters
            )
            start_resp = self.stub.SendKVStart(start_req, timeout=10.0)

            # 4. RDMA Transfer
            # K-Cache

            self.cpp_client.send_large_data(
                k_ptr, k_bytes, start_resp.k_buffer_addr, start_resp.k_buffer_rkey
            )
            # V-Cache
            self.cpp_client.send_large_data(
                v_ptr, v_bytes, start_resp.v_buffer_addr, start_resp.v_buffer_rkey
            )
            # Size-Cache (Proto: size_buffer_addr)
            self.cpp_client.send_large_data(
                s_ptr, s_bytes, start_resp.size_buffer_addr, start_resp.size_buffer_rkey
            )

            # 5. Completion
            complete_req = nelssa_comm_pb2.SendKVCompleteRequest(
                request_id=req_id,
                layer_idx=layer_idx,
                batch_size=bsz,
                kv_heads=kv_heads,
                seqlen=seqlen,
            )
            self.stub.SendKVComplete(complete_req, timeout=100.0)

        except Exception as e:
            print(f"[NelssaClient] send_kv_cache 실패 (Layer {layer_idx}): {e}")
            raise

    # --- Decode API (Batched) ---
    def execute_decode_batched(
        self, layer_idx: int, bsz: int, queries: torch.Tensor, cluster_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        LSE 통계값(+2)을 포함한 결과를 받아서 반환합니다.
        queries shape = [bsz, 1, n_heads, head_dims]
        cluster_ids shape = [bsz * kv_heads, dynamic_nprobe?]
        """
        # 1. GPU -> CPU & Numpy
        # queries = queries.to(device='cpu', dtype=torch.float32).view(bsz, 1, self.num_heads, self.head_dim).contiguous()
        # cluster_ids = cluster_ids.to(device='cpu', dtype=torch.int32).view(bsz, self.kv_heads, -1).contiguous()
        cluster_ids = cluster_ids.view(bsz, self.kv_heads, -1)
        try:
            # 2. C++ Call
            result_bytes = self.cpp_client.execute_decode_batched(
                layer_idx, queries, cluster_ids
            )
        except Exception as e:
            print(f"ExecuteDecode(Batched) C++ 래퍼 호출 실패: {e}")
            raise

        # 2. Deserialize & Reshape
        try:
            batch_group = queries.shape[0] * self.kv_heads
            group_size = self.num_heads // self.kv_heads
            head_dim = queries.shape[3]

            result_tensor = torch.frombuffer(result_bytes, dtype=torch.float32).reshape(
                batch_group, 1, group_size, head_dim + 2
            )

            return result_tensor

        except Exception as e:
            print(f"ExecuteDecode(Batched) 응답 역직렬화 실패: {e}")
            print(f"- Received Bytes: {len(result_bytes)}")
            print(f"- Expected Shape: [{batch_group}, 1, {group_size}, {head_dim} + 2]")
            raise

    # --- Decode API (Batched) ---
    def execute_decode_batched_async(
        self,
        layer_idx: int,
        bsz: int,
        queries_tensor: torch.Tensor,
        cluster_ids_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        LSE 통계값(+2)을 포함한 결과를 받아서 반환합니다.
        """

        # 1. GPU -> CPU & dtype conversion
        # queries_tensor = queries_tensor.to(
        #     device='cpu', dtype=torch.float16
        # ).view(bsz, 1, self.num_heads, self.head_dim).contiguous()
        # cluster_ids_tensor = cluster_ids_tensor.to(
        #     device='cpu', dtype=torch.int64
        # ).view(bsz, self.kv_heads, -1).contiguous()
        # 2. C++ Call
        self.cpp_client.execute_decode_batched_async_pinned(
            layer_idx, queries_tensor, cluster_ids_tensor
        )
        # except Exception as e:
        #     print(f"ExecuteDecode(Batched) C++ 래퍼 호출 실패: {e}")
        #     raise

    def poll_doorbell(
        self,
        layer_idx: int,
        queries_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        LSE 통계값(+2)을 포함한 결과를 받아서 반환합니다.
        """
        # query shape = batch_group, 1, group_size, head_dim
        # dtype 에 따라 element size 결정
        if queries_tensor.dtype == torch.float16:
            element_size = 2  # float16
            np_dtype = np.float16
        elif queries_tensor.dtype == torch.float32:
            element_size = 4  # float32
            np_dtype = np.float32
        else:
            raise ValueError(f"Unsupported dtype: {queries_tensor.dtype}")

        result_size = (
            queries_tensor.shape[0]
            * queries_tensor.shape[2]
            * (queries_tensor.shape[3] + 2)
            * element_size
        )
        result_bytes = self.cpp_client.poll_doorbell(layer_idx, result_size)

        # Deserialize & Reshape
        try:
            batch_group = queries_tensor.shape[0]
            group_size = queries_tensor.shape[2]
            head_dim = queries_tensor.shape[3]

            arr = np.frombuffer(result_bytes, dtype=np_dtype).copy()
            # result_tensor = torch.from_numpy(arr).reshape(
            #     batch_group, 1, group_size, head_dim + 2
            # )
            result_tensor = torch.from_numpy(arr)
            return result_tensor

            # arr = np.frombuffer(result_bytes, dtype=np_dtype).copy()

            # # 새 format: [attn_results][sum_values][max_values]
            # num_heads_total = batch_group * group_size
            # attn_elements = num_heads_total * head_dim
            # sum_elements = num_heads_total
            # max_elements = num_heads_total

            # attn_arr = arr[:attn_elements].reshape(batch_group, 1, group_size, head_dim)
            # sum_arr = arr[attn_elements : attn_elements + sum_elements].reshape(
            #     batch_group, 1, group_size, 1
            # )
            # max_arr = arr[attn_elements + sum_elements :].reshape(
            #     batch_group, 1, group_size, 1
            # )

            # retrieval_out = torch.from_numpy(attn_arr)
            # r_sum = torch.from_numpy(sum_arr)
            # r_max = torch.from_numpy(max_arr)

            # return retrieval_out, r_sum, r_max

        except Exception as e:
            print(f"ExecuteDecode(Batched) 응답 역직렬화 실패: {e}")
            print(f"- Received Bytes: {len(result_bytes)}")
            print(f"- Expected Shape: [{batch_group}, 1, {group_size}, {head_dim} + 2]")
            raise
