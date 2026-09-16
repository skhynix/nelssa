#pragma once

#include <infiniband/verbs.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <arpa/inet.h>
#include <stdlib.h>
#include <unistd.h>
#include <cerrno>
#include <iomanip>

// [안전 장치] MTU 자동 감지가 실패할 경우를 대비해 1024 강제 설정 (가장 안전)
// 1024는 모든 인피니밴드/RoCE 장비가 지원합니다.
const enum ibv_mtu FORCE_MTU = IBV_MTU_1024;
// TODO: This part is still under development
// const int DECODE_MSG_SIZE = 65536;
const int DECODE_MSG_SIZE = 4194304;

struct RdmaResource {
    struct ibv_context *ctx = nullptr;
    struct ibv_pd *pd = nullptr;
    struct ibv_cq *cq = nullptr;
    struct ibv_qp *qp = nullptr;

    char *recv_buf = nullptr;
    struct ibv_mr *recv_mr = nullptr;

    char *send_buf = nullptr;
    struct ibv_mr *send_mr = nullptr;

    size_t buf_size;

    uint32_t peer_qpn;
    uint16_t peer_lid;
    union ibv_gid peer_gid;
    uint64_t peer_addr;
    uint32_t peer_rkey;

    int active_port = -1;
    int active_gid_index = -1;
};

class RdmaEngine {
public:
    RdmaResource res;

    RdmaEngine(size_t size = DECODE_MSG_SIZE) {
        res.buf_size = size;
        setup_resources();
    }

    ~RdmaEngine() {
        if (res.qp) ibv_destroy_qp(res.qp);
        if (res.cq) ibv_destroy_cq(res.cq);
        if (res.recv_mr) ibv_dereg_mr(res.recv_mr);
        if (res.send_mr) ibv_dereg_mr(res.send_mr);
        if (res.recv_buf) free(res.recv_buf);
        if (res.send_buf) free(res.send_buf);
        if (res.pd) ibv_dealloc_pd(res.pd);
        if (res.ctx) ibv_close_device(res.ctx);
    }

    int find_active_port(struct ibv_context* ctx) {
        struct ibv_device_attr device_attr;
        ibv_query_device(ctx, &device_attr);
        for (int port = 1; port <= device_attr.phys_port_cnt; ++port) {
            struct ibv_port_attr port_attr;
            ibv_query_port(ctx, port, &port_attr);
            if (port_attr.state == IBV_PORT_ACTIVE) return port;
        }
        throw std::runtime_error("No ACTIVE port found");
    }

    int find_valid_gid_index(struct ibv_context* ctx, int port) {
        union ibv_gid gid;
        // 1순위: Index 3
        if (ibv_query_gid(ctx, port, 3, &gid) == 0) {
            bool is_zero = true;
            for (int j = 0; j < 16; ++j) if (gid.raw[j] != 0) is_zero = false;
            if (!is_zero) {
                std::cout << "[RdmaEngine] Using Priority GID Index: 3" << std::endl;
                return 3;
            }
        }
        // 2순위: 스캔
        for (int i = 0; i < 10; ++i) {
            if (i == 3) continue;
            if (ibv_query_gid(ctx, port, i, &gid) == 0) {
                bool is_zero = true;
                for (int j = 0; j < 16; ++j) if (gid.raw[j] != 0) is_zero = false;
                if (!is_zero) return i;
            }
        }
        throw std::runtime_error("No valid GID index found");
    }

    void setup_resources() {
        struct ibv_device **dev_list = ibv_get_device_list(NULL);
        if (!dev_list) throw std::runtime_error("No IB devices found");

        res.ctx = ibv_open_device(dev_list[0]);
        if (!res.ctx) throw std::runtime_error("Failed to open device");
        ibv_free_device_list(dev_list);

        res.pd = ibv_alloc_pd(res.ctx);
        res.active_port = find_active_port(res.ctx);
        res.active_gid_index = find_valid_gid_index(res.ctx, res.active_port);

        void* ptr_recv = nullptr;
        if (posix_memalign(&ptr_recv, 4096, res.buf_size) != 0) throw std::runtime_error("Failed to allocate RECV_BUF");
        res.recv_buf = (char*)ptr_recv;
        memset(res.recv_buf, 0, res.buf_size);

        void* ptr_send = nullptr;
        if (posix_memalign(&ptr_send, 4096, res.buf_size) != 0) throw std::runtime_error("Failed to allocate SEND_BUF");
        res.send_buf = (char*)ptr_send;
        memset(res.send_buf, 0, res.buf_size);

        res.recv_mr = ibv_reg_mr(res.pd, res.recv_buf, res.buf_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!res.recv_mr) throw std::runtime_error("Failed to register RECV_MR");

        res.send_mr = ibv_reg_mr(res.pd, res.send_buf, res.buf_size, IBV_ACCESS_LOCAL_WRITE);
        if (!res.send_mr) throw std::runtime_error("Failed to register SEND_MR");

        res.cq = ibv_create_cq(res.ctx, 100, NULL, NULL, 0);

        struct ibv_qp_init_attr qp_init_attr = {};
        qp_init_attr.send_cq = res.cq;
        qp_init_attr.recv_cq = res.cq;
        qp_init_attr.qp_type = IBV_QPT_RC;
        qp_init_attr.cap.max_send_wr = 100;
        qp_init_attr.cap.max_recv_wr = 100;
        qp_init_attr.cap.max_send_sge = 1;
        qp_init_attr.cap.max_recv_sge = 1;

        res.qp = ibv_create_qp(res.pd, &qp_init_attr);
        if (!res.qp) throw std::runtime_error("Failed to create QP");

        modify_qp_to_init();
    }

    void modify_qp_to_init() {
        struct ibv_qp_attr attr = {};
        attr.qp_state = IBV_QPS_INIT;
        attr.port_num = res.active_port;
        attr.pkey_index = 0;
        attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;

        if (ibv_modify_qp(res.qp, &attr,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
            throw std::runtime_error("Failed to modify QP to INIT");
        }
    }

    void connect_qp(uint32_t qpn, uint16_t lid, uint8_t* gid) {
        res.peer_qpn = qpn;
        res.peer_lid = lid;
        memcpy(res.peer_gid.raw, gid, 16);

        struct ibv_qp_attr attr = {};
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = FORCE_MTU; // [수정] 안전한 1024 강제 사용
        attr.dest_qp_num = res.peer_qpn;
        attr.rq_psn = 0;

        // [중요] Atomic 0
        attr.max_dest_rd_atomic = 0;

        attr.min_rnr_timer = 12;
        attr.ah_attr.is_global = 1;
        attr.ah_attr.dlid = 0; // RoCE 필수
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = res.active_port;

        attr.ah_attr.grh.dgid = res.peer_gid;
        attr.ah_attr.grh.sgid_index = res.active_gid_index;
        attr.ah_attr.grh.hop_limit = 1;

        if (ibv_modify_qp(res.qp, &attr,
            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {

            // [디버깅 로그] 여기서 모든 정보를 까봅니다.
            std::cerr << "\n[RDMA Error] Failed to modify QP to RTR. Errno: " << errno << std::endl;

            // 1. Peer GID가 0인지 확인 (가장 의심됨)
            std::cerr << "  > PEER GID (From Client): ";
            bool all_zero = true;
            for(int i=0; i<16; ++i) {
                std::cerr << std::hex << std::setw(2) << std::setfill('0') << (int)res.peer_gid.raw[i];
                if (i < 15) std::cerr << ":";
                if (res.peer_gid.raw[i] != 0) all_zero = false;
            }
            std::cerr << std::dec << std::endl;

            if (all_zero) {
                std::cerr << "  > [CRITICAL] Peer GID is ALL ZEROS! This causes Invalid Argument." << std::endl;
                std::cerr << "  > Solution: Recompile Server 1 (GPU) 'setup.py' to send correct GID." << std::endl;
            }

            std::cerr << "  > Peer QPN: " << res.peer_qpn << std::endl;
            std::cerr << "  > My Port: " << res.active_port << ", GID Idx: " << res.active_gid_index << std::endl;

            throw std::runtime_error("Failed to modify QP to RTR");
        }

        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = 0;
        attr.max_rd_atomic = 0; // [중요]

        if (ibv_modify_qp(res.qp, &attr,
            IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) {
            throw std::runtime_error("Failed to modify QP to RTS");
        }
    }

    void post_write(const char* data, size_t len, uint64_t remote_addr, uint32_t rkey) {
        memcpy(res.send_buf, data, len);
        struct ibv_sge sge;
        sge.addr = (uint64_t)res.send_buf;
        sge.length = len;
        sge.lkey = res.send_mr->lkey;

        struct ibv_send_wr wr, *bad_wr = NULL;
        memset(&wr, 0, sizeof(wr));
        wr.wr_id = 1;
        wr.opcode = IBV_WR_RDMA_WRITE;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = remote_addr;
        wr.wr.rdma.rkey = rkey;

        if (ibv_post_send(res.qp, &wr, &bad_wr)) {
            throw std::runtime_error("Failed to post send");
        }
        struct ibv_wc wc;
        int num_wc = 0;
        while (num_wc == 0) { num_wc = ibv_poll_cq(res.cq, 1, &wc); }
        if (wc.status != IBV_WC_SUCCESS) throw std::runtime_error("RDMA Write Failed");
    }

    void get_my_gid(uint8_t* gid_out) {
        union ibv_gid my_gid;
        ibv_query_gid(res.ctx, res.active_port, res.active_gid_index, &my_gid);
        memcpy(gid_out, my_gid.raw, 16);
    }
};
