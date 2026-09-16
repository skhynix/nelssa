#!/bin/bash
export export CUDA_VISIBLE_DEVICES=0
ATTN_TYPE=RetroInfer
params=(
    "1024000 0.01 0.232"
    "1024000 0.02 0.232"
    "1024000 0.04 0.232"
    "1024000 0.08 0.232"
)

echo "--- 전체 실험 시작: $(date) ---"
for param in "${params[@]}"; do
    # 배열에서 각 값을 추출
    set -- $param
    seqlen=$1
    budget=$2
    estimate=$3

    rm -r ruler_eval_result

    # 로그 파일 이름 설정 (예: log_248000_0.04.log)
    LOG_NAME="nia_mk3_log_${seqlen}_${budget}.log"

    echo "[현재 실행 중] Context: $seqlen, Budget: $budget (로그: $LOG_NAME)"

    # 실제 ruler_run.sh 실행 (표준 출력과 에러를 해당 로그 파일에 기록)
    bash ruler_run.sh llama-3-8b-1048k full $ATTN_TYPE $seqlen niah_multikey_3 fp16 $budget $estimate > "$LOG_NAME" 2>&1

    # 실행 결과 체크 (성공 여부 기록)
    if [ $? -eq 0 ]; then
        echo "[성공] $seqlen / $budget 완료"
    else
        echo "[실패] $seqlen / $budget 에러 발생. 로그를 확인하세요."
    fi
done

# 실험 정보를 배열로 정의 (순서: ContextLength BudgetRatio EstimateRatio)
# 질문하신 5가지 조합입니다.
# params=(
#     "256000 0.02 0.232"
#     "256000 0.04 0.232"
#     "256000 0.08 0.232"
# )

# echo "--- 전체 실험 시작: $(date) ---"
# for param in "${params[@]}"; do
#     # 배열에서 각 값을 추출
#     set -- $param
#     seqlen=$1
#     budget=$2
#     estimate=$3

#     rm -r ruler_eval_result

#     # 로그 파일 이름 설정 (예: log_248000_0.04.log)
#     LOG_NAME="qa_2_log_${seqlen}_${budget}.log"

#     echo "[현재 실행 중] Context: $seqlen, Budget: $budget (로그: $LOG_NAME)"

#     # 실제 ruler_run.sh 실행 (표준 출력과 에러를 해당 로그 파일에 기록)
#     bash ruler_run.sh llama-3-8b-1048k full $ATTN_TYPE $seqlen qa_2 fp16 $budget $estimate > "$LOG_NAME" 2>&1

#     # 실행 결과 체크 (성공 여부 기록)
#     if [ $? -eq 0 ]; then
#         echo "[성공] $seqlen / $budget 완료"
#     else
#         echo "[실패] $seqlen / $budget 에러 발생. 로그를 확인하세요."
#     fi
# done


echo "--- 모든 실험 종료: $(date) ---"
