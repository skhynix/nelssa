# TASK
# - niah_single_1
# - niah_single_2
# - niah_single_3
# - niah_multikey_1
# - niah_multikey_2
# - niah_multikey_3
# - niah_multivalue
# - niah_multiquery
# - vt (variable_tracking)
# - cwe (common_words_extraction)
# - fwe (freq_words_extraction)
# - qa_1
# - qa_2

MODEL_NAME=/home/sylee/Llama-3-8B-Instruct-Gradient-1048k
TASK=vt
LENGTH=1024000
BATCH=1

python benchmark/ruler/data/prepare.py \
    --save_dir ./data/${LENGTH} \
    --benchmark synthetic \
    --task ${TASK} \
    --tokenizer_path ${MODEL_NAME} \
    --tokenizer_type hf \
    --max_seq_length ${LENGTH} \
    --model_template_type meta-chat \
    --num_samples ${BATCH}
