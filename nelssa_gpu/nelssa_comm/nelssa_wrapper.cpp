#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h> // Numpy 배열 처리를 위해
#include <stdexcept>
#include <string>
#include <memory>
#include <chrono> // 디버깅용
#include <torch/extension.h>

#include "rdma_common.hpp"

namespace py = pybind11;

/**
 * @brief Python에서 RDMA 작업을 C++로 호출하기 위한 래퍼 클래스
 * [수정] execute_decode_batched API 추가
 */
class NelssaWrapper {
    std::unique_ptr<RdmaEngine> rdma; // Decode용 64KB RDMA 엔진
    int head_dim_; // 생성자에서 받은 head_dim (단일 API 응답용)

public:
    // 생성자: head_dim을 받도록 수정
    NelssaWrapper(int head_dim_val) : head_dim_(head_dim_val) {
        if (head_dim_val <= 0) {
            throw std::runtime_error("NelssaWrapper: head_dim must be positive.");
        }
        rdma.reset(new RdmaEngine(DECODE_MSG_SIZE));
    }

    struct alignas(8) DecodeHeader {
        int32_t opcode;
        int32_t layer_idx;
        int32_t q_bsz;
        int32_t num_heads;
        int32_t kv_heads;
        int32_t head_dim;
        int32_t n_probe;
    };

    // --- [기존] Handshake API ---
    py::dict get_my_info() {
        py::dict d;
        d["addr"] = (uint64_t)rdma->res.recv_buf;
        d["rkey"] = rdma->res.recv_mr->rkey;
        d["qp_num"] = rdma->res.qp->qp_num;
        d["lid"] = 0;

        uint8_t gid[16];
        rdma->get_my_gid(gid);
        d["gid"] = py::bytes((char*)gid, 16);
        return d;
    }

    void connect(uint64_t peer_addr, uint32_t peer_rkey, uint32_t qpn, uint32_t lid, std::string gid_bytes) {
        rdma->res.peer_addr = peer_addr;
        rdma->res.peer_rkey = peer_rkey;
        rdma->connect_qp(qpn, (uint16_t)lid, (uint8_t*)gid_bytes.c_str());
    }

    // --- [기존] Prefill API ---
    void send_large_data(
        uint64_t local_data_ptr, uint64_t num_bytes,
        uint64_t remote_addr, uint32_t remote_rkey
    ) {
        const size_t chunk_size = rdma->res.buf_size; // 64KB
        char* local_ptr = (char*)local_data_ptr;

        uint64_t bytes_sent = 0;
        while (bytes_sent < num_bytes) {
            size_t current_chunk_size = std::min((uint64_t)chunk_size, num_bytes - bytes_sent);

            rdma->post_write(
                local_ptr + bytes_sent,
                current_chunk_size,
                remote_addr + bytes_sent,
                remote_rkey
            );

            bytes_sent += current_chunk_size;
        }
    }

    // --- [기존] 단일 Decode API (Doorbell 0x01) ---
    py::bytes execute_step(py::bytes payload) {
        std::string payload_str = payload;
        const char* payload_data = payload_str.c_str();
        size_t payload_len = payload_str.length();

        if (payload_len > rdma->res.buf_size) {
            throw std::runtime_error("Decode(Single) payload exceeds buffer size");
        }

        rdma->post_write(payload_data, payload_len, rdma->res.peer_addr, rdma->res.peer_rkey);

        volatile char* my_doorbell = (volatile char*)rdma->res.recv_buf;
        *my_doorbell = 0;

        auto start_time = std::chrono::steady_clock::now();
        while(*my_doorbell != 1) { // 0x01 응답 대기
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() > 5) {
                throw std::runtime_error("execute_step: Timed out waiting for response (0x01)");
            }
        }

        size_t result_size = this->head_dim_ * sizeof(float);
        return py::bytes(rdma->res.recv_buf + 1, result_size);
    }

    // --- [신규] 배치 Decode API (Doorbell 0x02) ---
    py::bytes execute_decode_batched(
        int layer_idx,
        py::array_t<float> queries,     // [num_heads, head_dim]
        py::array_t<int> cluster_ids    // [kv_heads, n_probe]
    ) {
        // queries shape = [bsz, 1, n_heads, head_dims]
        // ids shape = [bsz, kv_heads, dynamic_nprobe?]
        py::buffer_info query_buf = queries.request();
        py::buffer_info ids_buf = cluster_ids.request();

        // 1. 메타데이터 추출
        int q_bsz = query_buf.shape[0];
        int num_heads = query_buf.shape[2];
        int head_dim = query_buf.shape[3];
        int ids_bsz = ids_buf.shape[0];
        int kv_heads = ids_buf.shape[1];
        int n_probe = ids_buf.shape[2];
        assert(q_bsz == ids_bsz);

        // 2. 페이로드 포장 (Packing)
        char* send_buf = rdma->res.send_buf;
        char* p = send_buf; // Moving pointer

        // [Header]
        *(p++) = 0x02; // Doorbell (Batched)
        *(reinterpret_cast<int*>(p)) = layer_idx; p += 4;
        *(reinterpret_cast<int*>(p)) = q_bsz; p += 4;
        *(reinterpret_cast<int*>(p)) = num_heads; p += 4;
        *(reinterpret_cast<int*>(p)) = kv_heads;  p += 4;
        *(reinterpret_cast<int*>(p)) = head_dim;  p += 4;
        *(reinterpret_cast<int*>(p)) = n_probe;   p += 4;

        // [Data]
        size_t query_data_bytes = query_buf.size * query_buf.itemsize;
        size_t ids_data_bytes = ids_buf.size * ids_buf.itemsize;
        size_t header_size = p - send_buf;

        // TODO: This part is still under development
        // if (header_size + query_data_bytes + ids_data_bytes > rdma->res.buf_size) {
        //     throw std::runtime_error("Decode(Batched) payload exceeds 64KB buffer size");
        // }

        memcpy(p, query_buf.ptr, query_data_bytes); p += query_data_bytes;
        memcpy(p, ids_buf.ptr, ids_data_bytes);     p += ids_data_bytes;

        size_t total_payload_len = p - send_buf;

        // 3. RDMA 전송 및 응답 대기
        volatile char* my_doorbell = (volatile char*)rdma->res.recv_buf;
        *my_doorbell = 0; // 응답 대기 전, 내 수신 버퍼 초기화

        // TODO: This part is still under development
        /////////////////////////////////////////////////////////////////////////////////////////////
        // if (q_bsz * num_heads * (head_dim + 2) * sizeof(float) > rdma->res.buf_size) {
        //     std::cerr << "[Fatal] Output size (" << q_bsz * num_heads * (head_dim + 2) * sizeof(float) << ") exceeds buffer size!" << std::endl;
        //     *my_doorbell = 0; continue;
        // }
        // 1. [Data WR] Flag(첫 1바이트)를 제외한 나머지 데이터 전송
        struct ibv_send_wr data_wr = {};
        struct ibv_sge data_sge;

        if (total_payload_len > 1) {
            data_sge.addr = (uint64_t)(send_buf + 1); // 데이터 시작점 (Offset 1)
            data_sge.length = total_payload_len - 1;        // 데이터 길이
            data_sge.lkey = rdma->res.send_mr->lkey;

            data_wr.wr_id = 100;
            data_wr.opcode = IBV_WR_RDMA_WRITE;
            data_wr.sg_list = &data_sge;
            data_wr.num_sge = 1;
            data_wr.send_flags = 0; // 시그널 안 보냄 (성능)
            data_wr.wr.rdma.remote_addr = rdma->res.peer_addr + 1; // 원격지 Offset 1
            data_wr.wr.rdma.rkey = rdma->res.peer_rkey;

            // 다음 WR 연결
            // 만약 데이터가 없으면(rv_size <= 1) 이 WR은 생략되거나 조정 필요하지만,
            // 통상적으로 Attention 결과는 1바이트보다 큼.
        }
        // 2. [Signal WR] 완료 신호 (send_buf[0]에 있는 '2') 전송
        struct ibv_send_wr signal_wr = {};
        struct ibv_sge signal_sge;

        // [수정] 스택 변수 대신 이미 등록된 send_buf[0]을 사용해야 안전함 (LKEY 보호)
        signal_sge.addr = (uint64_t)send_buf;
        signal_sge.length = 1;
        signal_sge.lkey = rdma->res.send_mr->lkey;

        signal_wr.wr_id = 101;
        signal_wr.opcode = IBV_WR_RDMA_WRITE;
        signal_wr.sg_list = &signal_sge;
        signal_wr.num_sge = 1;
        signal_wr.send_flags = IBV_SEND_SIGNALED; // 완료 확인용
        signal_wr.wr.rdma.remote_addr = rdma->res.peer_addr; // 원격지 Offset 0 (Flag 위치)
        signal_wr.wr.rdma.rkey = rdma->res.peer_rkey;
        signal_wr.next = NULL;

        // WR 연결 (Data -> Signal)
        struct ibv_send_wr *head_wr = &signal_wr; // 기본값 (데이터 없을 때)
        if (total_payload_len > 1) {
            data_wr.next = &signal_wr;
            head_wr = &data_wr;
        }

        struct ibv_send_wr *bad_wr = nullptr;
        if (ibv_post_send(rdma->res.qp, head_wr, &bad_wr)) {
            throw std::runtime_error("Failed to post linked send");
        }

        // 완료 대기 (Poll CQ)
        // signal_wr만 SIGNALED이므로 완료 이벤트는 1개만 발생함
        struct ibv_wc wc;
        int num_wc = 0;
        while (num_wc == 0) {
            num_wc = ibv_poll_cq(rdma->res.cq, 1, &wc);
        }
        if (wc.status != IBV_WC_SUCCESS) {
            std::cerr << "[RDMA Error] WC Status: " << wc.status << std::endl;
        }
        ////////////////////////////////////////////////////////////////////////////////////////////
        // if (total_payload_len <= rdma->res.buf_size){
        //     rdma->post_write(send_buf, total_payload_len, rdma->res.peer_addr, rdma->res.peer_rkey);
        // } else {
        //     send_large_data(
        //         (uint64_t)send_buf,
        //         total_payload_len,
        //         rdma->res.peer_addr,
        //         rdma->res.peer_rkey
        //     );
        // }

        auto start_time = std::chrono::steady_clock::now();
        while(*my_doorbell != 2) { // 0x02 응답 대기
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() > 15) {
                throw std::runtime_error("execute_decode_batched: Timed out waiting for response (0x02)");
            }
        }

        // 4. 응답 반환 (Doorbell 1바이트 제외)
        // size_t result_size_bytes = num_heads * head_dim * sizeof(float); //ORIGIN
        size_t result_size_bytes = q_bsz * num_heads * (head_dim + 2) * sizeof(float); // lse info add!

        return py::bytes(rdma->res.recv_buf + 1, result_size_bytes);
    }

    void execute_decode_batched_async(
        int layer_idx,
        py::array_t<float> queries_np,     // [num_heads, head_dim]
        py::array_t<int> cluster_ids_np    // [kv_heads, n_probe]
    ) {
        py::buffer_info query_buf = queries_np.request();
        py::buffer_info ids_buf = cluster_ids_np.request();

        // 1. 메타데이터 추출
        int q_bsz = query_buf.shape[0];
        int num_heads = query_buf.shape[2];
        int head_dim = query_buf.shape[3];
        int kv_heads = ids_buf.shape[0] / q_bsz;
        int ids_bsz = ids_buf.shape[0] / kv_heads;
        // int ids_bsz = ids_buf.shape[0];
        // int kv_heads = ids_buf.shape[1];
        int n_probe = ids_buf.shape[1];
        assert(q_bsz == ids_bsz);



        // 2. 페이로드 포장 (Packing)
        char* send_buf = rdma->res.send_buf;
        char* p = send_buf; // Moving pointer

        // [Header]
        *(p++) = 0x02; // Doorbell (Batched)
        *(reinterpret_cast<int*>(p)) = layer_idx; p += 4;
        *(reinterpret_cast<int*>(p)) = q_bsz; p += 4;
        *(reinterpret_cast<int*>(p)) = num_heads; p += 4;
        *(reinterpret_cast<int*>(p)) = kv_heads;  p += 4;
        *(reinterpret_cast<int*>(p)) = head_dim;  p += 4;
        *(reinterpret_cast<int*>(p)) = n_probe;   p += 4;

        // [Data]
        size_t query_data_bytes = query_buf.size * query_buf.itemsize;
        size_t ids_data_bytes = ids_buf.size * ids_buf.itemsize;
        size_t header_size = p - send_buf;

        // TODO: This part is still under development
        // if (header_size + query_data_bytes + ids_data_bytes > rdma->res.buf_size) {
        //     throw std::runtime_error("Decode(Batched) payload exceeds 64KB buffer size");
        // }

        memcpy(p, query_buf.ptr, query_data_bytes); p += query_data_bytes;
        memcpy(p, ids_buf.ptr, ids_data_bytes);     p += ids_data_bytes;

        size_t total_payload_len = p - send_buf;

        // 3. RDMA 전송 및 응답 대기
        volatile char* my_doorbell = (volatile char*)rdma->res.recv_buf;
        *my_doorbell = 0; // 응답 대기 전, 내 수신 버퍼 초기화

        // TODO: This part is still under development
        if (total_payload_len <= rdma->res.buf_size){
            rdma->post_write(send_buf, total_payload_len, rdma->res.peer_addr, rdma->res.peer_rkey);
        } else {
            send_large_data(
                (uint64_t)send_buf,
                total_payload_len,
                rdma->res.peer_addr,
                rdma->res.peer_rkey
            );
        }

        // auto start_time = std::chrono::steady_clock::now();
        // while(*my_doorbell != 2) { // 0x02 응답 대기
        //     auto now = std::chrono::steady_clock::now();
        //     if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() > 15) {
        //         throw std::runtime_error("execute_decode_batched: Timed out waiting for response (0x02)");
        //     }
        // }

        // // 4. 응답 반환 (Doorbell 1바이트 제외)
        // // size_t result_size_bytes = num_heads * head_dim * sizeof(float); //ORIGIN
        // size_t result_size_bytes = num_heads * (head_dim + 2) * sizeof(float); // lse info add!

        // return py::bytes(rdma->res.recv_buf + 1, result_size_bytes);
    }

    void execute_decode_batched_async_pinned(
        int layer_idx,
        const torch::Tensor queries,      // expected: float16 CPU tensor
        const torch::Tensor cluster_ids   // expected: int32 CPU tensor
    ) {
        // 1. 기본 검증
        // TORCH_CHECK(queries.device().is_cpu(), "queries must be a CPU tensor");
        // TORCH_CHECK(cluster_ids.device().is_cpu(), "cluster_ids must be a CPU tensor");

        // TORCH_CHECK(queries.scalar_type() == torch::kFloat32,
        //             "queries must be float32");
        // TORCH_CHECK(cluster_ids.scalar_type() == torch::kInt32,
        //             "cluster_ids must be int32");

        // TORCH_CHECK(queries.is_contiguous(), "queries must be contiguous");
        // TORCH_CHECK(cluster_ids.is_contiguous(), "cluster_ids must be contiguous");

        // TORCH_CHECK(queries.dim() == 4, "queries must be 4D");
        // TORCH_CHECK(cluster_ids.dim() == 2, "cluster_ids must be 2D");

        int q_bsz = static_cast<int>(queries.size(0));
        int num_heads = static_cast<int>(queries.size(2));
        int head_dim = static_cast<int>(queries.size(3));

        int ids_dim0 = static_cast<int>(cluster_ids.size(0));
        int n_probe = static_cast<int>(cluster_ids.size(1));

        TORCH_CHECK(q_bsz > 0, "q_bsz must be > 0");
        TORCH_CHECK(ids_dim0 % q_bsz == 0,
                    "cluster_ids.size(0) must be divisible by q_bsz");
        TORCH_CHECK(queries.scalar_type() == torch::kFloat16,
            "queries must be float16");

        int kv_heads = ids_dim0 / q_bsz;
        int ids_bsz = ids_dim0 / kv_heads;

        TORCH_CHECK(q_bsz == ids_bsz, "q_bsz != ids_bsz");

        // raw pointer 얻기 (at::Half = torch.float16)
        const at::Half* queries_ptr = queries.data_ptr<at::Half>();
        const int64_t* cluster_ids_ptr = cluster_ids.data_ptr<int64_t>();

        // 2. 페이로드 포장
        char* send_buf = rdma->res.send_buf;
        char* p = send_buf;

        *(p++) = 0x02; // Doorbell (Batched)

        *(reinterpret_cast<int32_t*>(p)) = static_cast<int32_t>(layer_idx); p += 4;
        *(reinterpret_cast<int32_t*>(p)) = static_cast<int32_t>(q_bsz);     p += 4;
        *(reinterpret_cast<int32_t*>(p)) = static_cast<int32_t>(num_heads); p += 4;
        *(reinterpret_cast<int32_t*>(p)) = static_cast<int32_t>(kv_heads);  p += 4;
        *(reinterpret_cast<int32_t*>(p)) = static_cast<int32_t>(head_dim);  p += 4;
        *(reinterpret_cast<int32_t*>(p)) = static_cast<int32_t>(n_probe);   p += 4;

        size_t query_data_bytes = queries.numel() * sizeof(uint16_t);
        size_t ids_data_bytes   = cluster_ids.numel() * sizeof(int64_t);

        std::memcpy(p, queries_ptr, query_data_bytes);
        p += query_data_bytes;

        std::memcpy(p, cluster_ids_ptr, ids_data_bytes);
        p += ids_data_bytes;

        size_t total_payload_len = static_cast<size_t>(p - send_buf);

        // 3. RDMA 전송 (Linked WR: Data 먼저, Doorbell 마지막)
        volatile char* my_doorbell = reinterpret_cast<volatile char*>(rdma->res.recv_buf);
        *my_doorbell = 0;

        // Data WR (Offset 1~Y 먼저 전송)
        struct ibv_send_wr data_wr = {};
        struct ibv_sge data_sge;

        if (total_payload_len > 1) {
            data_sge.addr = (uint64_t)(send_buf + 1);
            data_sge.length = total_payload_len - 1;
            data_sge.lkey = rdma->res.send_mr->lkey;

            data_wr.wr_id = 100;
            data_wr.opcode = IBV_WR_RDMA_WRITE;
            data_wr.sg_list = &data_sge;
            data_wr.num_sge = 1;
            data_wr.send_flags = 0;  // 시그널 안 함
            data_wr.wr.rdma.remote_addr = rdma->res.peer_addr + 1;
            data_wr.wr.rdma.rkey = rdma->res.peer_rkey;
        }

        // Signal WR (Doorbell 마지막 전송)
        struct ibv_send_wr signal_wr = {};
        struct ibv_sge signal_sge;

        signal_sge.addr = (uint64_t)send_buf;
        signal_sge.length = 1;
        signal_sge.lkey = rdma->res.send_mr->lkey;

        signal_wr.wr_id = 101;
        signal_wr.opcode = IBV_WR_RDMA_WRITE;
        signal_wr.sg_list = &signal_sge;
        signal_wr.num_sge = 1;
        signal_wr.send_flags = IBV_SEND_SIGNALED;
        signal_wr.wr.rdma.remote_addr = rdma->res.peer_addr;
        signal_wr.wr.rdma.rkey = rdma->res.peer_rkey;
        signal_wr.next = NULL;

        // 연결 (Data → Signal)
        struct ibv_send_wr *head_wr = &signal_wr;
        if (total_payload_len > 1) {
            data_wr.next = &signal_wr;
            head_wr = &data_wr;
        }

        struct ibv_send_wr *bad_wr = nullptr;
        int ret = ibv_post_send(rdma->res.qp, head_wr, &bad_wr);
        if (ret) {
            std::cerr << "[RDMA Error] ibv_post_send failed: " << strerror(ret)
                      << " (errno=" << ret << ")" << std::endl;
            std::cerr << "  total_payload_len=" << total_payload_len << std::endl;
            std::cerr << "  bad_wr wr_id=" << (bad_wr ? bad_wr->wr_id : -1) << std::endl;
            throw std::runtime_error("Failed to post linked send");
        }
    }

    py::bytes poll_doorbell(
        int layer_idx,
        int result_size
    ) {
        // py::buffer_info query_buf = queries_np.request();
        // py::buffer_info ids_buf = cluster_ids_np.request();

        // // 1. 메타데이터 추출
        // int num_heads = query_buf.shape[0];
        // int head_dim = query_buf.shape[1];
        // int kv_heads = ids_buf.shape[0];
        // int n_probe = ids_buf.shape[1];

        // // 2. 페이로드 포장 (Packing)
        // char* send_buf = rdma->res.send_buf;
        // char* p = send_buf; // Moving pointer

        // // [Header]
        // *(p++) = 0x02; // Doorbell (Batched)
        // *(reinterpret_cast<int*>(p)) = layer_idx; p += 4;
        // *(reinterpret_cast<int*>(p)) = num_heads; p += 4;
        // *(reinterpret_cast<int*>(p)) = kv_heads;  p += 4;
        // *(reinterpret_cast<int*>(p)) = head_dim;  p += 4;
        // *(reinterpret_cast<int*>(p)) = n_probe;   p += 4;

        // // [Data]
        // size_t query_data_bytes = query_buf.size * query_buf.itemsize;
        // size_t ids_data_bytes = ids_buf.size * ids_buf.itemsize;
        // size_t header_size = p - send_buf;

        // // TODO: This part is still under development
        // // if (header_size + query_data_bytes + ids_data_bytes > rdma->res.buf_size) {
        // //     throw std::runtime_error("Decode(Batched) payload exceeds 64KB buffer size");
        // // }

        // memcpy(p, query_buf.ptr, query_data_bytes); p += query_data_bytes;
        // memcpy(p, ids_buf.ptr, ids_data_bytes);     p += ids_data_bytes;

        // size_t total_payload_len = p - send_buf;
        struct ibv_wc wc;
        int num_wc = 0;
        while (num_wc == 0) {
            num_wc = ibv_poll_cq(rdma->res.cq, 1, &wc);
        }
        if (wc.status != IBV_WC_SUCCESS) {
            std::cerr << "[RDMA Error] WC Status: " << wc.status << std::endl;
        }
        // 3. RDMA 전송 및 응답 대기
        volatile char* my_doorbell = (volatile char*)rdma->res.recv_buf;
        // *my_doorbell = 0; // 응답 대기 전, 내 수신 버퍼 초기화

        // TODO: This part is still under development
        // if (total_payload_len <= rdma->res.buf_size){
        //     rdma->post_write(send_buf, total_payload_len, rdma->res.peer_addr, rdma->res.peer_rkey);
        // } else {
        //     send_large_data(
        //         (uint64_t)send_buf,
        //         total_payload_len,
        //         rdma->res.peer_addr,
        //         rdma->res.peer_rkey
        //     );
        // }

        auto start_time = std::chrono::steady_clock::now();
        while(*my_doorbell != 2) { // 0x02 응답 대기
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() > 15) {
                throw std::runtime_error("execute_decode_batched: Timed out waiting for response (0x02)");
            }
        }

        // 4. 응답 반환 (Doorbell 1바이트 제외)
        // size_t result_size_bytes = num_heads * head_dim * sizeof(float); //ORIGIN
        size_t result_size_bytes = result_size; // lse info add!

        return py::bytes(rdma->res.recv_buf + 1, result_size_bytes);
    }
};

// Python 모듈 정의 (pybind11)
PYBIND11_MODULE(nelssa_wrapper, m) {
    py::class_<NelssaWrapper>(m, "NelssaWrapper")
        .def(py::init<int>(), py::arg("head_dim")) // head_dim을 받는 생성자

        .def("get_my_info", &NelssaWrapper::get_my_info, "Get local RDMA buffer info (for Handshake)")
        .def("connect", &NelssaWrapper::connect,
             "Connect to peer QP using info from Handshake",
             py::arg("peer_addr"), py::arg("peer_rkey"), py::arg("qpn"),
             py::arg("lid"), py::arg("gid_bytes"))
        .def("send_large_data", &NelssaWrapper::send_large_data,
             "Send large data (tensor) via chunked RDMA Write (for Prefill)",
             py::arg("local_data_ptr"), py::arg("num_bytes"),
             py::arg("remote_addr"), py::arg("remote_rkey"))

        // --- [기존] 단일 API ---
        .def("execute_step", &NelssaWrapper::execute_step,
             "Execute one RDMA ping-pong (for Decode)",
             py::arg("payload"))

        // --- [신규] 배치 API ---
        .def("execute_decode_batched", &NelssaWrapper::execute_decode_batched,
             "Execute batched RDMA Decode (for RetroInfer)",
             py::arg("layer_idx"),
             py::arg("queries_np"),
             py::arg("cluster_ids_np"))
        // --- Async API ---
        .def("execute_decode_batched_async", &NelssaWrapper::execute_decode_batched_async,
             "Execute batched RDMA Decode (for RetroInfer)",
             py::arg("layer_idx"),
             py::arg("queries_np"),
             py::arg("cluster_ids_np"))
        .def("execute_decode_batched_async_pinned", &NelssaWrapper::execute_decode_batched_async_pinned,
             "Execute batched RDMA Decode (for RetroInfer)",
             py::arg("layer_idx"),
             py::arg("queries_np"),
             py::arg("cluster_ids_np"))
        .def("poll_doorbell", &NelssaWrapper::poll_doorbell,
             "Execute batched RDMA Decode (for RetroInfer)",
             py::arg("layer_idx"),
             py::arg("result_size"));
}