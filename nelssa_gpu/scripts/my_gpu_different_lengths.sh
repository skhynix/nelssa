export CUDA_VISIBLE_DEVICES=0

mkdir -p gpu_different_lengths_logs

### 128K 0.01
LEN=128000
for budget in 0.01 0.02 0.04 0.08
do
    for bsz in 1 2 4 8 16 32 64
    do
        for round in 1 2
        do
            python -u my_test.py \
                --model_name /home/sylee/Llama-3-8B-Instruct-Gradient-1048k \
                --attn_type RetroInfer \
                --context_len ${LEN} \
                --task_name NIAH \
                --batch_size $bsz \
                --retrieval_budget ${budget} > gpu_different_lengths_logs/gpu_${LEN}_bsz${bsz}_budget${budget}_${round}.log 2>&1
            sleep 10
        done
    done
done

LEN=256000
for budget in 0.01 0.02 0.04 0.08
do
    for bsz in 1 2 4 8 16 32 64
    do
        for round in 1 2
        do
            python -u my_test.py \
                --model_name /home/sylee/Llama-3-8B-Instruct-Gradient-1048k \
                --attn_type RetroInfer \
                --context_len 256000 \
                --task_name NIAH \
                --batch_size $bsz \
                --retrieval_budget ${budget} > gpu_different_lengths_logs/gpu_${LEN}_bsz${bsz}_budget${budget}_${round}.log 2>&1
            sleep 10
        done
    done
done

LEN=512000
for budget in 0.01 0.02 0.04 0.08
do
    for bsz in 1 2 4 8 16
    do
        for round in 1 2
        do
            python -u my_test.py \
                --model_name /home/sylee/Llama-3-8B-Instruct-Gradient-1048k \
                --attn_type RetroInfer \
                --context_len 512000 \
                --task_name NIAH \
                --batch_size $bsz \
                --retrieval_budget ${budget} > gpu_different_lengths_logs/gpu_${LEN}_bsz${bsz}_budget${budget}_${round}.log 2>&1
            sleep 10
        done
    done
done

LEN=768000
for budget in 0.01 0.02 0.04 0.08
do
    for bsz in 1 2 4 8 16
    do
        for round in 1 2
        do
            python -u my_test.py \
                --model_name /home/sylee/Llama-3-8B-Instruct-Gradient-1048k \
                --attn_type RetroInfer \
                --context_len 768000 \
                --task_name NIAH \
                --batch_size $bsz \
                --retrieval_budget ${budget} > gpu_different_lengths_logs/gpu_${LEN}_bsz${bsz}_budget${budget}_${round}.log 2>&1
            sleep 10
        done
    done
done

LEN=1024000
for budget in 0.01 0.02 0.04 0.08
do
    for bsz in 1 2 4
    do
        for round in 1 2
        do
            python -u my_test.py \
                --model_name /home/sylee/Llama-3-8B-Instruct-Gradient-1048k \
                --attn_type RetroInfer \
                --context_len 1024000 \
                --task_name NIAH \
                --batch_size $bsz \
                --retrieval_budget ${budget} > gpu_different_lengths_logs/gpu_${LEN}_bsz${bsz}_budget${budget}_${round}.log 2>&1
            sleep 10
        done
    done
done