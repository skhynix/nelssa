# TASK
# - niah_single_1
# - niah_single_2
# - niah_multivalue
# - niah_multiquery
# - vt (variable_tracking)
# - cwe (common_words_extraction)
# - fwe (freq_words_extraction)
# - qa_1
# - qa_2
TASK=vt
LENGTH=8192
BATCH=64
python benchmark/ruler/data/prepare.py \
    --save_dir ./data/len_${LENGTH}_b_${BATCH} \
    --benchmark synthetic \
    --task ${TASK} \
    --tokenizer_path /home/sylee/dataset/Llama-3.1-8B-Instruct/ \
    --tokenizer_type hf \
    --max_seq_length ${LENGTH} \
    --model_template_type meta-chat \
    --num_samples ${BATCH}
