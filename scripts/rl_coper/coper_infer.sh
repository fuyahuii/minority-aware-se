export CUDA_VISIBLE_DEVICES=0

python inference/eval_rl_coper.py \
    --base_model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --model_dir "models_outputs_rl/coper" \
    --tokenizer_name_or_path "models_outputs_rl/coper" \
    --eval_batch_size 8 \
    --max_new_tokens 200

