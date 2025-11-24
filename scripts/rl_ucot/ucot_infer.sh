export CUDA_VISIBLE_DEVICES=6

python inference/eval_rl_base.py \
    --base_model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --model_dir "models_outputs_rl/ucot" \
    --tokenizer_name_or_path "models_outputs_rl/ucot" \
    --eval_batch_size 8 \
    --max_new_tokens 20\
    --ucot True \

