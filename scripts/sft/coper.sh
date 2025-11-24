#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3

accelerate launch --num_processes 2 --main_process_port 29523 --multi_gpu sft_coper.py --finetune \
# --voting \
# --sampling \
# --weighted_loss \



