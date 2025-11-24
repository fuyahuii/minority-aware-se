export CUDA_VISIBLE_DEVICES=0

python inference/eval_rl_base.py \
    --base_model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --model_dir "models_outputs_rl/base" \
    --tokenizer_name_or_path "models_outputs_rl/base" \
    --eval_batch_size 8 \
    --max_new_tokens 20

