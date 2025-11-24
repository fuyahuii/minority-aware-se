#!/bin/bash
exp_name="rl_ucot_em_train" \
base_model="meta-llama/Meta-Llama-3-8B-Instruct" \
model_name_or_path=".../sft/..." \
tokenizer_name_or_path=".../sft/..." \
ref_model_name_or_path_majority=".../em/..." \
ref_model_name_or_path_minority=".../em/..." \
cluster_map_path=".../em/.../cluster_map.json" \
n_epochs='10' \
kl_coef='0.1' \
llamatemplate='False' \
bash template.sh

