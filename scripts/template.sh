#!/bin/bash
export TOKENIZERS_PARALLELISM=True
export CUDA_VISIBLE_DEVICES=1,2,3,4
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=1800
export CUDA_LAUNCH_BLOCKING=1        
export NCCL_DEBUG=INFO               
# export WANDB_MODE=offline           

### Required variables
exp_name=${exp_name:-''}
base_model=${base_model:-''} 
model_name_or_path=${model_name_or_path:-''}
tokenizer_name_or_path=${tokenizer_name_or_path:-''}
ref_model_name_or_path_majority=${ref_model_name_or_path_majority:-''}
ref_model_name_or_path_minority=${ref_model_name_or_path_minority:-''}
cluster_map_path=${cluster_map_path:-''}
n_epochs=${n_epochs:-''}
kl_coef=${kl_coef:-''} 
llamatemplate=${llamatemplate:-'False'} # whether to use default LlamaTemplate

### Default variables
model_dir="models_outputs_rl/${exp_name}/"
config_file="./default_config_deepspeed.yaml"

batch_size="2" 
mini_batch_size="2" 
eval_batch_size="2" 
ppo_epochs="2"
num_workers="0"
learning_rate="3e-7"
weight_decay="0"
warmup_step="0"
clip_grad_norm="1"
vf_coef="0.1" #5，0.1
# kl_coef="0.01"
gamma="1.0"
lam="0.95"
adv_whitening='global'
seed="42"
max_input_length="1024"
max_new_tokens="200"
keep_num_ckpt='0'

evaluating_epoch_freq="1"
logging_epoch_freq="1"
saving_epoch_freq="1"

logging_step_freq="1"
evaluating_step_freq="100"
saving_step_freq="100"

wandb_log="True"
wandb_project="ReFT"
wandb_run_name="${exp_name}"
#########

num_processes='4'
main_process_port='6053'

mkdir -p "${model_dir}"
accelerate launch \
            --config_file "${config_file}" \
            --num_processes=${num_processes} \
            --main_process_port=${main_process_port} \
    pada-ppo.py \
            --base_model "${base_model}" \
            --model_name_or_path "${model_name_or_path}" \
            --tokenizer_name_or_path "${tokenizer_name_or_path}" \
            --train_file "${train_file}" \
            --test_file "${test_file}" \
            --model_dir "${model_dir}" \
            --batch_size "${batch_size}" \
            --mini_batch_size "${mini_batch_size}" \
            --eval_batch_size "${eval_batch_size}" \
            --ppo_epochs "${ppo_epochs}" \
            --n_epochs "${n_epochs}" \
            --num_workers "${num_workers}" \
            --learning_rate "${learning_rate}" \
            --weight_decay "${weight_decay}" \
            --warmup_step "${warmup_step}" \
            --clip_grad_norm "${clip_grad_norm}" \
            --vf_coef "${vf_coef}" \
            --kl_coef "${kl_coef}" \
            --gamma "${gamma}" \
            --lam "${lam}" \
            --evaluating_epoch_freq "${evaluating_epoch_freq}" \
            --logging_epoch_freq "${logging_epoch_freq}" \
            --saving_epoch_freq "${saving_epoch_freq}" \
            --evaluating_step_freq "${evaluating_step_freq}" \
            --logging_step_freq "${logging_step_freq}" \
            --saving_step_freq "${saving_step_freq}" \
            --seed "${seed}" \
            --max_input_length "${max_input_length}" \
            --max_new_tokens "${max_new_tokens}" \
            --wandb_log "${wandb_log}" \
            --wandb_project "${wandb_project}" \
            --wandb_run_name "${wandb_run_name}" \
            --engine "${engine}" \
            --adv_whitening "${adv_whitening}" \
            --keep_num_ckpt "${keep_num_ckpt}" \
            --llamatemplate "${llamatemplate}" \
            --ref_model_name_or_path_majority "${ref_model_name_or_path_majority}" \
            --ref_model_name_or_path_minority "${ref_model_name_or_path_minority}" \
            --cluster_map_path "${cluster_map_path}" \
            1> >(tee "${model_dir}"/"${exp_name}".log) \
            2> >(tee "${model_dir}"/"${exp_name}".err >&2)
