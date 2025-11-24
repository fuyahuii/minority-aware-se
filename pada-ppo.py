# This file is adapted from the Apache-2.0 licensed implementation by Bytedance Ltd.
# Original source: https://github.com/lqtrung1998/mwp_ReFT/blob/main/train_rl_reft.py
#
# Copyright 2023 Bytedance Ltd.
# Modifications Copyright 2025 Yahui Fu.
# 
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import pad_across_processes, broadcast
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, concatenate_datasets
from datetime import timedelta
from functools import partial
import json
import os
import random
from utils.util import set_seed, floatify, compute_ETA, discount_cumsum, do_gather, allgather, allgather_masked_whiten
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from trl import AutoModelForCausalLMWithValueHead
from trl.core import masked_mean, masked_var, masked_whiten
import numpy as np
import wandb
import shutil
from prettytable import PrettyTable
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from peft.utils import prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig,GenerationConfig
import re
import pandas as pd
from utils.CoPeR import generate_coper_prompt_inference, generate_coper_prompt_training
from utils.base_template import generate_prompt_for_training, generate_prompt_for_inference, generate_prompt_reasoning_for_training, generate_prompt_reasoning_for_inference
from utils.tokenization import tokenize_fn_truncation, collate_fn, tokenize_fn_truncation_llama3,tokenize_fn,tokenize_fn_llama3
from utils.generate import evaluate_generation
from sklearn.metrics import f1_score, classification_report
from transformers import StoppingCriteria, StoppingCriteriaList
import bisect     


tqdm = partial(tqdm, ncols=0, leave=False)

TIMEOUT = 10

def truncate_after_final_answer_precise(text: str) -> tuple[str, int]:
    """
    return truncated text and the position
    """
    m = re.search(r"(Final answer\s*:\s*(?:High|Low)\s+satisfaction)", text, re.IGNORECASE)
    if m:
        end_pos = m.end()
        return text[:end_pos], end_pos
    else:
        return text, len(text)


def extract_score_and_position(text: str):
    text_lc = text.lower()
    
    # first try to find final answer
    m = re.search(r"final answer\s*:\s*(high|low)\s+satisfaction(?!\s+or)", text_lc)
    if m:
        score = 1 if m.group(1) == "high" else 0
        reward_end_pos = m.end(1)  # final answer 结束位置
        score_source = "final_answer" 
        return score, reward_end_pos, score_source
    
    # if not, try to find final score
    m = re.search(r"final score\s*:\s*([1-5])\s*(?:$|\n)", text_lc)
    if m:
        score = int(m.group(1))
        reward_end_pos = m.end(1)  # final score 结束位置
        score_source = "final_score"
        return (0 if score <= 3 else 1), reward_end_pos, score_source

    return None, None, None


def get_precise_reward_position(tokenizer, full_text: str, char_end_pos: int) -> int:
    """
    Convert character-level end position to token-level position
    """
    if char_end_pos is None or char_end_pos >= len(full_text):
        return len(tokenizer.encode(full_text, add_special_tokens=False))
    
    text_up_to_end = full_text[:char_end_pos]
    tokens_up_to_end = tokenizer.encode(text_up_to_end, add_special_tokens=False)
    return len(tokens_up_to_end)


def compute_reward(pred, true):
    if not isinstance(pred, (int, float)):
        return 0.0
    return 1.0 if pred == true else -1.0


def logprobs_from_logits(logits, labels):
    # logits: (batch, seq_len, vocab)
    # labels: (batch, seq_len)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    # Gather the log probabilities at the label positions
    # log_probs.gather: (batch, seq_len, 1) -> squeeze(-1)
    return log_probs.gather(2, labels.unsqueeze(2)).squeeze(-1)


def prepare_deepspeed_ref_model(model):
    # Adopted from: https://github.com/huggingface/trl/blob/02f5c1d8cee73045c837d01d7f1577a57779b035/trl/trainer/ppo_trainer.py#L1399
    import deepspeed

    # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepspeed_plugin.deepspeed_config
    if model is not None:
        if hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                    }
                )

    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
    if config_kwargs["zero_optimization"]["stage"] != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model

         
def prepare_datasets_and_data_loaders(args, tokenizer):
    from collections import defaultdict
    from functools import partial
    import torch
    
    with accelerator.main_process_first():
    
        df_train = pd.read_csv("data/train_with_reasoning.csv")
        df_eval = pd.read_csv("data/valid_with_reasoning.csv")
        
        with open(args["cluster_map_path"], "r") as f:
            cluster_map = json.load(f)                     # {dialogue_id: 0/1}

        cluster_map = {str(k): int(v) for k, v in cluster_map.items()}
        
        for df in (df_train, df_eval):
            df["cluster_label"] = df["dialogue_id"].astype(str).map(cluster_map)
            print("cluster_label distribution:", df["cluster_label"].value_counts())
            if df["cluster_label"].isna().any():
                missing = df.loc[df["cluster_label"].isna(), "dialogue_id"].unique()[:10]
                raise ValueError(f"Missing cluster for dialogue_id: {missing} ...")
            
        min_df = df_train[df_train["cluster_label"] == 1]   # minority
        maj_df = df_train[df_train["cluster_label"] == 0]   # majority

        # Sample an equal number of majority samples as minority (random shuffle)
        maj_sampled = maj_df.sample(n=len(min_df), replace=False, random_state=42)
        # maj_sampled = maj_df
        # maj_sampled = maj_df.sample(n=100, replace=False, random_state=42)
        # min_df = min_df.sample(n=100, replace=False, random_state=42)
        
        boost_ratio   = args['boost_ratio']                         
        n_extra_min   = int(len(min_df) * boost_ratio)
        extra_min_df  = min_df.sample(n=n_extra_min,
                              replace=True,             
                              random_state=42)

        df_train_balanced = pd.concat(
                [maj_sampled, min_df, extra_min_df]
        ).sample(frac=1, random_state=42).reset_index(drop=True)

        accelerator.print(
            f"[Balance] after: maj={len(maj_sampled)}, "
            f"min={len(min_df)} + {len(extra_min_df)}(extra) = {len(min_df)+len(extra_min_df)}"
        )

        df_train = df_train_balanced
        
        if args["prompt"] == "coper":
            accelerator.print("Loading CoPeR prompting")
            prompts_train = df_train.apply(generate_coper_prompt_training,
                                        axis=1, result_type="expand").assign(cluster_label=df_train["cluster_label"].values)
            prompts_eval  = df_eval.apply(generate_coper_prompt_training,
                                        axis=1, result_type="expand").assign(cluster_label=df_eval["cluster_label"].values)
        elif args["prompt"]=="ucot":
            accelerator.print("Loading UCoT prompting")
            prompts_train = df_train.apply(generate_prompt_reasoning_for_training, axis=1, result_type="expand").assign(cluster_label=df_train["cluster_label"].values)
            prompts_eval = df_eval.apply(generate_prompt_reasoning_for_training, axis=1, result_type="expand").assign(cluster_label=df_eval["cluster_label"].values)
        else:
            accelerator.print("Loading base prompting")
            prompts_train=df_train.apply(generate_prompt_for_training, axis=1, result_type="expand").assign(cluster_label=df_train["cluster_label"].values)
            prompts_eval=df_eval.apply(generate_prompt_for_training, axis=1, result_type="expand").assign(cluster_label=df_eval["cluster_label"].values)
        
        raw_dataset = DatasetDict({
            'train': Dataset.from_pandas(prompts_train,preserve_index=False),
            'eval': Dataset.from_pandas(prompts_eval,preserve_index=False),
        })
        accelerator.print('Raw data:', raw_dataset)

        # if use llamatemplate
        if args['llamatemplate'] and args["prompt"] == "coper":
            accelerator.print('*****[Info] Using LlamaTemplate for tokenization****')
            tokenized_dataset = DatasetDict({
            mode: dataset.map(
                tokenize_fn_truncation_llama3, 
                fn_kwargs={'args': args, 'tokenizer': tokenizer}, 
                batched=True,
                remove_columns=[c for c in dataset.column_names if c != "cluster_label"],
                num_proc=None, 
                load_from_cache_file=True, 
                keep_in_memory=False,
            ) for mode, dataset in raw_dataset.items()
            })
        else:
            accelerator.print('****[Info] Using default tokenization function for tokenization****')
            tokenized_dataset = DatasetDict({
                mode: dataset.map(
                    tokenize_fn_truncation, 
                    fn_kwargs={'args': args, 'tokenizer': tokenizer}, 
                    batched=True,
                    remove_columns=[c for c in dataset.column_names if c != "cluster_label"],
                    num_proc=None, 
                    load_from_cache_file=True, 
                    keep_in_memory=False,
                ) for mode, dataset in raw_dataset.items()
            })
        accelerator.print('Processed data:', tokenized_dataset)

        if accelerator.is_main_process and args.get('wandb_log', False):
            wandb.config.update({
                "raw_dataset": str(raw_dataset),
                "tokenized_dataset": str(tokenized_dataset),
            })

    train_dataloader = DataLoader(
        tokenized_dataset['train'], 
        shuffle=True, 
        batch_size=args.get('batch_size', 8),
        num_workers=args.get('num_workers', 0), 
        pin_memory=True,
        collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer)
    )

    eval_dataloader = DataLoader(
        tokenized_dataset['eval'], 
        shuffle=False, 
        batch_size=args.get('eval_batch_size', 8),
        num_workers=args.get('num_workers', 0), 
        pin_memory=True,
        collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer)
    )

    return (tokenized_dataset['train'], train_dataloader), (tokenized_dataset['eval'], eval_dataloader)

def do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths=None):
    os.makedirs(save_path, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    # unwrapped_model.save_pretrained(save_path, is_main_process=accelerator.is_main_process,
    #                                 save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))
    if args.get("use_lora", False):
        unwrapped_model.save_pretrained(save_path, is_main_process=accelerator.is_main_process,save_function=accelerator.save,)
    else:
        unwrapped_model.save_pretrained(save_path, is_main_process=accelerator.is_main_process,
                                     save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))
    tokenizer.save_pretrained(save_path)
    if accelerator.is_main_process and most_recent_ckpts_paths is not None:
        most_recent_ckpts_paths.append(save_path)
        if args['keep_num_ckpt'] is not None and len(most_recent_ckpts_paths) > args['keep_num_ckpt']:
            ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
            # os.remove(ckpt_to_be_removed)
            shutil.rmtree(ckpt_to_be_removed)

                            
def rollout(args, model, ref_model_maj,ref_model_min, tokenizer, query_tensors, query_tensors_attention_mask, answer_values, cluster_labels):
    model.eval()
    accelerator.print("⇢ rollout: start generate", flush=True)
            
    with torch.no_grad():
        gen_output = accelerator.unwrap_model(model).generate(
            input_ids=query_tensors,
            attention_mask=query_tensors_attention_mask,
            top_k=50, 
            top_p=0.85, 
            temperature=0.7, 
            do_sample=True,
            # output_scores=True,
            # return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            # bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # max_length=args['max_gen_length'],
            # pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=150 if args["prompt"] == "coper" else 20,
            min_new_tokens=10,
            # repetition_penalty=2.0,
            bad_words_ids=[[tokenizer.pad_token_id]],
            # stopping_criteria=stopping_criteria,
            # no_repeat_ngram_size=3,
            # length_penalty=1.2, 
            # use_cache=False,           
        )
        accelerator.print("⇠ rollout: finish generate", flush=True)
        completed_tensors = pad_across_processes(gen_output, dim=1, pad_index=tokenizer.pad_token_id, pad_first=False)
        accelerator.print("completed_tensors shap", completed_tensors.shape, flush=True)
    
    completed_texts = []
    reward_positions = []  
    extracted_scores = []
    score_sources = []     
    
    for i, gen_seq in enumerate(completed_tensors):
        # Decoding from the generated sequence
        query_len = query_tensors[i].shape[0]
        generated_part = gen_seq[query_len:]    
        # accelerator.print(f"Sample {i} query len: {query_len}", flush=True)
        
        # Moving out left padding (if any)
        non_pad_mask = generated_part != tokenizer.pad_token_id
        if non_pad_mask.any():
            first_non_pad = non_pad_mask.nonzero()[0].item() if non_pad_mask.nonzero().numel() > 0 else 0
            generated_part = generated_part[first_non_pad:]
        
        if not args['llamatemplate']:
            # For LlamaTemplate, eos token appears multiple times in instruction and user input, so we skip this step
            # Find the first EOS token and truncate if it exists
            eos_positions = (generated_part == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                accelerator.print(f"Sample {i} EOS positions: {eos_positions}", flush=True)
                generated_part = generated_part[:eos_positions[0]]

        generated_text = tokenizer.decode(generated_part.cpu().numpy().tolist(), 
                                        skip_special_tokens=True, 
                                        clean_up_tokenization_spaces=True).strip()
        accelerator.print(f"Sample {i} generated text: {generated_text}", flush=True)
        completed_texts.append(generated_text)
        
        score, char_end_pos, source = extract_score_and_position(generated_text)
        extracted_scores.append(score)
        score_sources.append(source)
        
        # Calculating precise token-level reward position
        if char_end_pos is not None:
            token_reward_pos = get_precise_reward_position(tokenizer, generated_text, char_end_pos)
           
            generated_tokens = tokenizer.encode(generated_text, add_special_tokens=False)
            if token_reward_pos <= len(generated_tokens):
                # Display tokens near the reward position
                start_tok = max(0, token_reward_pos-5)
                end_tok = min(len(generated_tokens), token_reward_pos)
                context_tokens = generated_tokens[start_tok+1:end_tok]
                context_text = tokenizer.decode(context_tokens, skip_special_tokens=True)
                
                accelerator.print(f"Sample {i} - Token position {token_reward_pos}:")
                # accelerator.print(f"  Context tokens: {context_tokens}")
                accelerator.print(f"  Context text: '{context_text}'")
                
                # Display the exact reward token
                if token_reward_pos > 0 and token_reward_pos <= len(generated_tokens):
                    reward_token_id = generated_tokens[token_reward_pos-1]
                    reward_token_text = tokenizer.decode([reward_token_id], skip_special_tokens=True)
                    accelerator.print(f"  Reward token: ID={reward_token_id}, Text='{reward_token_text}'")
                
            absolute_reward_pos = query_len + token_reward_pos
        else:
            # If no score found, set position to the end of the generated content
            absolute_reward_pos = query_len + len(generated_part)
            accelerator.print(f"Sample {i} - No score found, using end position: {absolute_reward_pos}")

        reward_positions.append(absolute_reward_pos)
        accelerator.print(f"  Score: {score}, Source: {source}")
        
    correctness = []
    class_weights=[]
    
    CLASS_WEIGHTS = {
    0: 1.0,   # majority
    1: 2.0    # minority
    }
    for pred, tgt, source in zip(extracted_scores, answer_values, score_sources):
        
        # class_weights.append(CLASS_WEIGHTS[int(tgt)])s
        clus_id = int(cluster_labels[i].item())
        class_weights.append(CLASS_WEIGHTS[clus_id])
        
        if pred is None:
            correctness.append(-1.0)
        elif int(pred) == int(tgt):
            correctness.append(1.0)
        else:
            correctness.append(-1.0)
        accelerator.print(f"Predicted: {pred}, True: {tgt}, Source: {source}, Correctness: {correctness[-1]}")
       
        
    model_input_ids = completed_tensors
    model_attention_mask = (completed_tensors != tokenizer.pad_token_id)
    with torch.no_grad():
        # Get old logprob and val
        lm_logits, _dummy2, val = model(input_ids=model_input_ids, attention_mask=model_attention_mask)
        old_logprob = logprobs_from_logits(lm_logits[:, :-1, :], labels=model_input_ids[:, 1:])  # (bs, seqlen-1)

        # Get the ref model logprob
        # ref_logprob: first all zeros, then write in clusters
        ref_logprob = torch.zeros_like(old_logprob)

        maj_idx = (cluster_labels == 0).nonzero(as_tuple=True)[0]
        if maj_idx.numel():
            # accelerator.print(f"Majority indices: {maj_idx}, num: {maj_idx.numel()}")
            maj_logits, _, _ = ref_model_maj(
                input_ids=model_input_ids[maj_idx],
                attention_mask=model_attention_mask[maj_idx])
            ref_logprob[maj_idx] = logprobs_from_logits(
                maj_logits[:, :-1, :], model_input_ids[maj_idx][:, 1:])

        min_idx = (cluster_labels == 1).nonzero(as_tuple=True)[0]
        if min_idx.numel():
            # accelerator.print(f"Minority indices: {min_idx}, num: {min_idx.numel()}")
            min_logits, _, _ = ref_model_min(
                input_ids=model_input_ids[min_idx],
                attention_mask=model_attention_mask[min_idx])
            ref_logprob[min_idx] = logprobs_from_logits(
                min_logits[:, :-1, :], model_input_ids[min_idx][:, 1:])

    
    #mask positions and rewards
    prompt_len = query_tensors.size(1)
    mask = torch.zeros_like(model_input_ids, dtype=torch.bool)
    score_rew = np.zeros(mask.shape)
    
    for i in range(len(model_input_ids)):
        reward_pos = reward_positions[i]
        mask[i, prompt_len:reward_pos] = 1
    
        # score_rew[i, reward_pos-1] = correctness[i]*class_weights[i] 
        # accelerator.print(f"Sample {i}: Reward {correctness[i]*class_weights[i]} at position {reward_pos-1} (source: {score_sources[i]})")
        score_rew[i, reward_pos-1] = correctness[i]
        accelerator.print(f"Sample {i}: Reward {correctness[i]} at position {reward_pos-1} (source: {score_sources[i]})")
        
        if not args['llamatemplate']:
            eos_positions = (model_input_ids[i] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            for eos_pos in eos_positions:
                mask[i, eos_pos:] = 0
                score_rew[i, eos_pos:] = 0
    
    # accelerator.print("mask2", mask, mask.shape, flush=True)
    # Make the kl reward and the full reward
    kl_rew = None
    rew = score_rew
    if ref_logprob is not None:
        kl = old_logprob - ref_logprob  # (bs, seqlen-1)
        kl = (kl.float() * mask[:, :-1]).cpu().numpy()
        kl_rew = np.zeros(mask.shape)  # (bs, seqlen)
        kl_rew[:, :-1] = -kl # NOTE the minus sign
 
        kl_coef = args["kl_coef"]
        rew = score_rew + kl_coef * kl_rew
        

    # Process val ret adv logprob
    val = (val.float() * mask).cpu().numpy()
    gamma = args["gamma"]
    lam = args["lam"]
    # ret = np.zeros_like(rew)
    adv = np.zeros_like(rew)
    for i in range(len(rew)):
        cur_rew, cur_val = rew[i], val[i]
        cur_delta = -cur_val[:-1] + cur_rew[:-1] + gamma * cur_val[1:]
        cur_adv = discount_cumsum(cur_delta, discount=gamma * lam)
        cur_adv[:prompt_len] = 0
        adv[i][:-1] = cur_adv

    # lambda_return = GAE + values
    ret = adv + val  # (bs, seqlen)

    rew = torch.tensor(rew, device=mask.device, dtype=old_logprob.dtype) * mask
    score_rew = torch.tensor(score_rew, device=mask.device, dtype=old_logprob.dtype) * mask
    if kl_rew is not None:
        kl_rew = torch.tensor(kl_rew, device=mask.device, dtype=old_logprob.dtype) * mask
    ret = torch.tensor(ret, device=mask.device, dtype=old_logprob.dtype) * mask
    val = torch.tensor(val, device=mask.device, dtype=old_logprob.dtype) * mask
    adv = torch.tensor(adv, device=mask.device, dtype=old_logprob.dtype) * mask
    old_logprob = old_logprob * mask[:, :-1]
    

    model.train()
    return model_input_ids, model_attention_mask, mask, rew, score_rew, kl_rew, ret, correctness, val, old_logprob, ref_logprob, adv
          

def debug_mask_info(cur_mask, global_iter_num, mini_idx, b_inds):
    """Detailed debug information for mask and indices."""
    accelerator.print(f"[Debug] global_iter_num={global_iter_num}, mini_idx={mini_idx}")
    accelerator.print(f"[Debug] b_inds shape: {b_inds.shape}, values: {b_inds}")
    accelerator.print(f"[Debug] cur_mask shape: {cur_mask.shape}")
    accelerator.print(f"[Debug] cur_mask sum: {cur_mask.sum()}")
    accelerator.print(f"[Debug] cur_mask per sample: {cur_mask.sum(dim=1)}")
    accelerator.print(f"[Debug] cur_mask non-zero samples: {(cur_mask.sum(dim=1) > 0).sum()}")
    

def train_one_epoch(args, model, ref_model_maj,ref_model_min, train_dataset, train_dataloader, optimizer, scheduler, tokenizer,
                    global_step, global_iter_num, test_dataset, test_dataloader,
                    prefix, epoch, best_eval_log_dict, summary_log_dict, most_recent_ckpts_paths,topk_ckpts):
    model_dir = args['model_dir']
    clip_grad_norm = args.get('clip_grad_norm', None)
    vf_coef = args['vf_coef']
    evaluating_step_freq = args.get('evaluating_step_freq', 100)
    logging_step_freq = args.get('logging_step_freq', 1)
    saving_step_freq = args.get('saving_step_freq', 100)
    model.train()
    epoch_result_dict = defaultdict(list)
    with tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=not accelerator.is_main_process, desc='Train Loop') as t:
        for idx, batch in t:
            result_dict = defaultdict(list)
            # accelerator.print(f"[{global_iter_num}] batch['ppo_forward_kwargs']:", idx,batch['ppo_forward_kwargs'], flush=True)
            accelerator.print(f"[rank={accelerator.process_index}] got batch {idx}", flush=True)
            # Do rollout first
            model.eval()
            model_input_ids, model_attention_mask, mask, rew, score_rew, kl_rew, ret, correctness, val, old_logprob, ref_logprob, adv = rollout(
                args, model, ref_model_maj,ref_model_min, tokenizer,
                query_tensors=batch['ppo_forward_kwargs']['query_tensors'],
                query_tensors_attention_mask=batch['ppo_forward_kwargs']['query_tensors_attention_mask'],
                answer_values=batch['ppo_forward_kwargs']['answer_values'],
                cluster_labels=batch['ppo_forward_kwargs']['cluster_labels'],
                # src_name=train_dataset[0]['item_id'].split('_')[0],
            )  

            model.train()
            # preprocess
            raw_adv = adv
            if args['adv_whitening'] == 'global':
                adv = allgather_masked_whiten(adv, mask) # (mini_bs, seqlen)
            elif args['adv_whitening'] == 'local':
                adv = masked_whiten(adv, mask)

            batch_size_per_gpu = len(batch['ppo_forward_kwargs']['query'])
            mini_batch_size_per_gpu = args["mini_batch_size"]
            ppo_epochs = args["ppo_epochs"]
            train_stats = {}
            for _ in range(ppo_epochs):
                perms = torch.randperm(batch_size_per_gpu)
                for mini_idx in range(0, len(perms), mini_batch_size_per_gpu):
                    b_inds = perms[mini_idx: mini_idx + mini_batch_size_per_gpu]
                    # Subset to batch
                    cur_val = val[b_inds].contiguous()  # mini_bs x seqlen
                    cur_old_logprob = old_logprob[b_inds].contiguous()  # mini_bs x seqlen
                    cur_mask = mask[b_inds].contiguous()  # mini_bs x seqlen
                    cur_rew = rew[b_inds].contiguous()  # mini_bs x seqlen
                    cur_score_rew = score_rew[b_inds].contiguous() # mini_bs x seqlen
                    cur_kl_rew = None if kl_rew is None else kl_rew[b_inds].contiguous()  # mini_bs x seqlen
                    cur_ret = ret[b_inds].contiguous()  # mini_bs x seqlen
                    cur_adv = adv[b_inds].contiguous()  # mini_bs x seqlen
                    cur_raw_adv = raw_adv[b_inds].contiguous()  # mini_bs x seqlen
                    cur_model_input_ids = model_input_ids[b_inds].contiguous()  # mini_bs x seqlen
                    cur_model_attention_mask = model_attention_mask[b_inds].contiguous()  # mini_bs x seqlen
                    
                    
                    if cur_mask.sum() == 0:
                        debug_mask_info(cur_mask, global_iter_num, mini_idx, b_inds)
                        accelerator.print(f"[Critical Warning] Entire mini-batch has empty mask, skipping")
                        continue
    
                    resp_len_per_sample = torch.clamp(torch.sum(cur_mask, dim=1), min=1.0)  # (mini_bs,)
                    cur_query_mask = torch.logical_xor(cur_mask, cur_model_attention_mask)  # (mini_bs, seqlen)
                    query_len_per_sample = torch.clamp(torch.sum(cur_query_mask, dim=1), min=1.0)  # (mini_bs,)

                    # Preprocess advantage and get metrics  
                    cur_mask = cur_mask.type(cur_adv.dtype).contiguous()
                    
                    # Confirm mask is not empty
                    assert cur_mask.sum() > 0, f"Empty mask after filtering at global_iter_num={global_iter_num}"
                    
                    mean_adv, var_adv = masked_mean(cur_adv, cur_mask), masked_var(cur_adv, cur_mask)

                    # Forward current model
                    model.eval()
                    lm_logits, _, vpreds = model(input_ids=cur_model_input_ids, attention_mask=cur_model_attention_mask)
                    logprob = logprobs_from_logits(lm_logits[:, :-1, :], cur_model_input_ids[:, 1:])  # (mini_bs, seqlen-1)

                    # Compute losses
                    loss = 0

                    # policy gradient loss
                    ratio = torch.exp(logprob - cur_old_logprob)
                    pg_losses = -cur_adv[:, :-1] * ratio
                    pg_losses2 = -cur_adv[:, :-1] * torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
                    pg_loss = ((torch.max(pg_losses, pg_losses2) * cur_mask[:, :-1]).sum(dim=-1) / resp_len_per_sample).mean()

                    # value loss
                    vpredclipped = torch.max(torch.min(vpreds, cur_val + 0.2), cur_val - 0.2)
                    vf_losses1 = (vpreds - cur_ret) ** 2
                    vf_losses2 = (vpredclipped - cur_ret) ** 2
                    vf_loss = 0.5 * ((torch.max(vf_losses1, vf_losses2) * cur_mask).sum(dim=-1) / resp_len_per_sample).mean()
                    # vf_loss = 0.5 * ((torch.max(vf_losses1, vf_losses2) * cur_mask).sum() / cur_mask.sum())

                    # total loss
                    loss += pg_loss + vf_coef * vf_loss

                    # token related metrics
                    mean_query_len = torch.mean(allgather(torch.mean(query_len_per_sample)))
                    std_query_len = torch.mean(allgather(torch.std(query_len_per_sample)))
                    mean_resp_len = torch.mean(allgather(torch.mean(resp_len_per_sample)))
                    std_resp_len = torch.mean(allgather(torch.std(resp_len_per_sample)))

                    # value related metrics
                    vf_expl_var_num = masked_var(cur_ret - vpreds, cur_mask)
                    vf_expl_var_dem = masked_var(cur_ret, cur_mask)
                    vf_expl_var = 1.0 - vf_expl_var_num / (vf_expl_var_dem + 1e-8)
                    vf_expl_var = max(-1.0, vf_expl_var.item())  # the truncated value suffices
                    mean_vpred = masked_mean(vpreds, cur_mask)
                    mean_return = masked_mean(cur_ret, cur_mask)
                    mean_reward = masked_mean(cur_rew, cur_mask)
                    mean_score_reward = masked_mean(cur_score_rew, cur_mask)
                    mean_kl_reward = 0.0 if cur_kl_rew is None else masked_mean(cur_kl_rew, cur_mask)
                    mean_kcxkl_reward = args["kl_coef"] * mean_kl_reward

                    # policy related metrics
                    mean_ratio = masked_mean(ratio, cur_mask[:, :-1])
                    #mean_adv = masked_mean(cur_adv[:, :-1], cur_mask[:, :-1])
                    mean_logprob = masked_mean(logprob, cur_mask[:, :-1])
                    # sequence-level kl
                    mean_seq_kl = -1.0
                    if cur_kl_rew is not None:
                        cur_kl = -cur_kl_rew
                        seq_kl = torch.sum(cur_kl * cur_mask, dim=1)  # (mini_bs,)
                        mean_seq_kl = torch.mean(seq_kl)

                    # Update
                    epoch_result_dict['loss'].append(loss.item())

                    # accelerator.backward(loss)
                    # accelerator.deepspeed_engine_wrapped.backward(loss)
                    # runs backpropagation and handles mixed precision
                    if accelerator.distributed_type == "DEEPSPEED":
                        accelerator.deepspeed_engine_wrapped.engine.backward(loss)
                        total_grad_norm = 0.0
                        for n, p in model.named_parameters():
                            grad = deepspeed.utils.safe_get_full_grad(p)
                            if grad is None:
                                continue
                            cur_grad = grad.view(-1)
                            cur_grad_norm_sqrt = torch.norm(cur_grad, 2)
                            if cur_grad_norm_sqrt < 1e-8:
                                accelerator.print(f'{n} grad_norm_sqrt: {cur_grad_norm_sqrt}')
                            total_grad_norm += cur_grad_norm_sqrt ** 2

                        total_grad_norm = total_grad_norm ** 0.5
                        accelerator.deepspeed_engine_wrapped.engine.step()
                    else:
                        accelerator.backward(loss)
                        total_grad_norm = -1.0
                        if clip_grad_norm is not None:
                            total_grad_norm = accelerator.clip_grad_norm_(model.parameters(), clip_grad_norm)
                        optimizer.step()
                        model.zero_grad()
                        optimizer.zero_grad()

                    # Update running stats
                    n_correct, total = do_gather([sum(correctness), len(correctness)])
                    train_stats["acc"] = n_correct / total
                    train_stats["ncor"] = n_correct
                    train_stats["total"] = total
                    train_stats['pg_loss'] = pg_loss.item()
                    train_stats['vf_loss'] = vf_loss.item()
                    train_stats['vf_expl_var'] = vf_expl_var

                    for k, v in train_stats.items():
                        result_dict[k].append(v)

                    total_param_norm = 0.0
                    if accelerator.distributed_type == "DEEPSPEED":
                        for n, p in model.named_parameters():
                            param = deepspeed.utils.safe_get_full_fp32_param(p)
                            if param is None:
                                continue
                            cur_param = param.view(-1)
                            total_param_norm += torch.norm(cur_param, 2) ** 2
                        total_param_norm = total_param_norm ** 0.5
                    else:
                        total_param_norm = torch.norm(
                            torch.cat([p.view(-1) for p in model.parameters()]),
                            p=2  # L2 norm
                        )
                    
                    if accelerator.is_main_process and args['wandb_log']:
                        ### 1. Calculate majority / minority proportion in the batch
                        batch_clusters = batch['ppo_forward_kwargs']['cluster_labels'].detach().cpu()
                        maj_prop = (batch_clusters == 0).float().mean().item()
                        min_prop = 1.0 - maj_prop

                        ### 2. Roughly calculate minority precision / recall
                        targets = torch.tensor(batch['ppo_forward_kwargs']['answer_values']).cpu()
                        corr_tensor = torch.tensor(correctness)          # 1 / -1 / 0
                        preds = torch.where(corr_tensor == 1,
                                            targets,                   
                                            1 - targets)                
                        true_min = targets == 0
                        pred_min = preds == 0
                        tp = (true_min & pred_min).sum().item()
                        fp = ((~true_min) & pred_min).sum().item()
                        fn = (true_min & (~pred_min)).sum().item()
                        recall_min = tp / (tp + fn + 1e-8)
                        precision_min = tp / (tp + fp + 1e-8)    
                        
                        train_stats['min_recall'] = recall_min
                        train_stats['min_precision'] = precision_min
                        train_stats['mean_seq_kl'] = mean_seq_kl
                    
                    
                    # logging
                    if accelerator.is_main_process and args['wandb_log']:
                        wandb.log({
                            "data/maj_prop": maj_prop,
                            "data/min_prop": min_prop,
                            "metric/min_recall": recall_min,
                            "metric/min_precision": precision_min,
                        }, step=global_iter_num)
                        wandb.log({
                            "nn/total_grad_norm": total_grad_norm,
                            "nn/total_param_norm": total_param_norm,
                            "nn/lr": scheduler.get_last_lr()[0],
                        }, step=global_iter_num)
                        wandb.log({
                            "acc/acc": train_stats["acc"],
                            "acc/ncor": train_stats["ncor"],
                            "acc/total": train_stats["total"],
                        }, step=global_iter_num)
                        wandb.log({
                            "loss/loss:": loss,
                            "loss/pg_loss": pg_loss,
                            "loss/vf_loss": vf_loss,
                        }, step=global_iter_num)
                        wandb.log({
                            "tokens/mean_query_len": mean_query_len,
                            "tokens/std_query_len": std_query_len,
                            "tokens/mean_resp_len": mean_resp_len,
                            "tokens/std_resp_len": std_resp_len,
                        }, step=global_iter_num)
                        wandb.log({
                            "policy/mean_ratio": mean_ratio,
                            "policy/mean_adv": mean_adv,
                            "policy/var_adv": var_adv,
                            "policy/mean_logprob": mean_logprob,
                            "policy/mean_seq_kl": mean_seq_kl,
                        }, step=global_iter_num)
                        wandb.log({
                            "value/vf_expl_var": vf_expl_var,
                            "value/mean_vpred": mean_vpred,
                            "value/mean_return": mean_return,
                            "value/mean_reward": mean_reward,
                            "value/mean_score_reward": mean_score_reward,
                            "value/mean_kl_reward": mean_kl_reward,
                            "value/mean_kcxkl_reward": mean_kcxkl_reward,
                        }, step=global_iter_num)
                    # Update iter num
                    # torch.distributed.barrier()
                    global_iter_num += 1

            scheduler.step()
            global_step += 1
            # accelerator.empty_cache()
            # Step update metric
            epoch_result_dict['loss'].append(loss.item())
            for k, v in train_stats.items():
                epoch_result_dict[k].append(v)

            # Step evaluating
            eval_log_dict = {}
            is_best = False
            if evaluating_step_freq is not None and global_step % evaluating_step_freq == 0:
                evaluate_result_dict = {f'Eval.Gen.{k}': v for k, v in
                                        evaluate_generation(args, accelerator,model, test_dataset, test_dataloader, tokenizer).items()}
                eval_log_dict.update(evaluate_result_dict)
                if eval_log_dict['Eval.Gen.value_accuracy'] > best_eval_log_dict.get('Eval.Gen.value_accuracy_best', -1):
                    is_best = True
                    best_eval_log_dict['Eval.Gen.value_accuracy_best'] = eval_log_dict['Eval.Gen.value_accuracy']
                    if 'Eval.Gen.value_accuracy' not in summary_log_dict:
                        summary_log_dict['Eval.Gen.value_accuracy'] = []
                    summary_log_dict['Eval.Gen.value_accuracy'].append(eval_log_dict['Eval.Gen.value_accuracy'])

            # Step logging
            train_log_dict = {}
            if logging_step_freq is not None and global_step % logging_step_freq == 0:
                train_log_dict = {f'T.{k}': sum(v) / len(v) if isinstance(v, list) else v for k, v in epoch_result_dict.items()}

            if eval_log_dict or train_log_dict:
                log_dict = {'lr': scheduler.get_last_lr()[0], **train_log_dict, **eval_log_dict, **best_eval_log_dict}
                if accelerator.is_main_process and args['wandb_log']:
                    wandb.log(log_dict, step=global_step)
                    log_dict = {'wandb': args['wandb_project'] + '|' + args['wandb_run_name'], **log_dict}
                log_dict = {k: f'{v:.5g}' if isinstance(v, float) else v for k,v in log_dict.items()}
                accelerator.print(f"{prefix}[E={epoch}/{args['n_epochs']}, S={global_step}] {log_dict}")

            # Step saving
            if saving_step_freq is not None and global_step % saving_step_freq == 0:
                if evaluating_step_freq is not None and global_step % evaluating_step_freq == 0:
                    cur_score = eval_log_dict.get('Eval.Gen.value_accuracy')
                    save_topk_ckpt(cur_score, global_step, 'step',
                                args, model, tokenizer,
                                model_dir, topk_ckpts, most_recent_ckpts_paths)

                if args['keep_num_ckpt'] > 0:
                    save_path = os.path.join(model_dir, f'global_step_{global_step}')
                    do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths)

            # Keep only max_record items
            for k, v in epoch_result_dict.items():
                if len(v) > 1:
                    epoch_result_dict[k] = v[-1:]

    # Metric summary:
    epoch_result_dict = {k: (sum(v) / len(v) if isinstance(v, list) else v) for k, v in epoch_result_dict.items()}
    return epoch_result_dict, global_step, global_iter_num

def save_topk_ckpt(cur_score: float,
                   global_id: int,
                   tag: str,                         # 'step' or 'epoch'
                   args, model, tokenizer,
                   model_dir: str,
                   topk_ckpts: list[tuple[float,str]],
                   most_recent_ckpts_paths: list[str]):
    """
    Put current model into Top-K according to cur_score,
    Ensure len(topk_ckpts) <= args['n_best_ckpt'].
    """
    if cur_score is None:              
        return
    need_save = (len(topk_ckpts) < args['n_best_ckpt'] or
                 cur_score > min(m for m, _ in topk_ckpts))
    if not need_save:
        return

    ckpt_name = f"best_{cur_score:.4f}_{tag}{global_id}"
    save_path = os.path.join(model_dir, ckpt_name)
    do_checkpoint(args, model, tokenizer, save_path)

    bisect.insort(topk_ckpts, (cur_score, save_path))         # Insert and keep sorted
    topk_ckpts.sort(key=lambda x: x[0], reverse=True)         # Sort descending
    while len(topk_ckpts) > args['n_best_ckpt']:              # Remove lowest score if exceed
        _, worst_path = topk_ckpts.pop()
        if accelerator.is_main_process:
            shutil.rmtree(worst_path, ignore_errors=True)

def main(args):
    set_seed(args['seed'] + accelerator.process_index)

    if accelerator.is_main_process and args['wandb_log']:
        wandb.init(project=args['wandb_project'], name=args['wandb_run_name'])
        wandb.config.update(args)
        
    tokenizer = AutoTokenizer.from_pretrained(args['tokenizer_name_or_path'], use_fast=True,local_files_only=True)
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.padding_side = "left"  
    tokenizer.pad_token_id = 2 # token "#"
    print("[INFO] tokenizer pad_token_id:", tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id))
    
    (train_dataset, train_dataloader), (test_dataset, test_dataloader) = prepare_datasets_and_data_loaders(args, tokenizer)
    
    # bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_enable_fp32_cpu_offload=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        args["base_model"],
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
    )
    base_model = prepare_model_for_kbit_training(base_model)
    base_model.gradient_checkpointing_enable()    
    
    model = PeftModel.from_pretrained(
        base_model,
        args["model_name_or_path"],
        local_files_only=True,
        is_trainable=True
    )
    accelerator.print("[INFO] Using LoRA, only trainable weights:", model.print_trainable_parameters())
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    
    quant_config_ref_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,)
    
    quant_config_ref_8bit = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_enable_fp32_cpu_offload=False,
    )
    
    if not hasattr(model, "generation_config"):
        model.generation_config = GenerationConfig.from_model_config(model.config)
        print("[INFO] model generation config:", model.generation_config)

    # Loading ref_model (do not share the base_model instance with the main model)
    def load_ref(path):
        ref_base = AutoModelForCausalLM.from_pretrained(
            args['base_model'], torch_dtype=torch.float16, quantization_config=quant_config_ref_4bit if args['infer_8bit'] else None,low_cpu_mem_usage=True
        )
        # ref_base = prepare_model_for_kbit_training(ref_base) 
        peft = PeftModel.from_pretrained(ref_base, path, is_trainable=False,local_files_only=True)
        ref = AutoModelForCausalLMWithValueHead.from_pretrained(peft)
        ref.eval()
        if not hasattr(ref, 'generation_config'):
            ref.generation_config = GenerationConfig.from_model_config(ref.config)
        return ref
    ref_model_maj = load_ref(args['ref_model_name_or_path_majority'])
    ref_model_min = load_ref(args['ref_model_name_or_path_minority'])
    
    # optimizer
    n_epochs = args['n_epochs']
    num_training_steps = (len(train_dataloader) // accelerator.num_processes * n_epochs)
    warmup_step = args['warmup_step'] if args['warmup_step'] is not None and args['warmup_step'] >= 0 else int(0.1 * num_training_steps)
    
    if args["use_lora"]:
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if p.requires_grad],
                "weight_decay": args['weight_decay'],
            },
        ]
    else:
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "LayerNorm.weight"])],
                "weight_decay": args['weight_decay'],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm.weight"])],
                "weight_decay": 0.0,
            },
        ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=1e-8)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=num_training_steps)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step)
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader,
                                                                              test_dataloader)
    if ref_model_maj is not None:
        if accelerator.distributed_type == "DEEPSPEED":
            ref_model_maj = prepare_deepspeed_ref_model(ref_model_maj)
            ref_model_min = prepare_deepspeed_ref_model(ref_model_min)
        else:
            ref_model_maj = accelerator.prepare(ref_model_maj)
            ref_model_min = accelerator.prepare(ref_model_min)

    global_step = 0
    global_iter_num = 0
    evaluating_epoch_freq = args['evaluating_epoch_freq']
    logging_epoch_freq = args['logging_epoch_freq']
    saving_epoch_freq = args['saving_epoch_freq']
    model_dir = args['model_dir']
    best_eval_log_dict = {}
    summary_log_dict = {}
    os.makedirs(model_dir, exist_ok=True)
    most_recent_ckpts_paths = []
    topk_ckpts = []
    with tqdm(range(1, n_epochs+1), total=n_epochs, disable=False) as t:
        for epoch in t:
            kwargs = {
                'args': args,
                'model': model,
                'ref_model_maj': ref_model_maj,
                'ref_model_min': ref_model_min,
                'train_dataset': train_dataset,
                'train_dataloader': train_dataloader,
                'test_dataset': test_dataset,
                'test_dataloader': test_dataloader,
                'optimizer': optimizer,
                'scheduler': scheduler,
                'global_step': global_step,
                'global_iter_num': global_iter_num,
                'tokenizer': tokenizer,
                'prefix': '',
                'epoch': epoch,
                'best_eval_log_dict': best_eval_log_dict,
                'summary_log_dict': summary_log_dict,
                'most_recent_ckpts_paths': most_recent_ckpts_paths,
                'topk_ckpts':topk_ckpts
            }
            train_epoch_result_dict, global_step, global_iter_num = train_one_epoch(**kwargs)

            eval_log_dict = {}
            is_best = False
            if evaluating_epoch_freq is not None and epoch % evaluating_epoch_freq == 0:
                evaluate_result_dict = {f'Eval.Gen.{k}': v for k, v in
                                        evaluate_generation(args,accelerator, model, test_dataset, test_dataloader, tokenizer).items()}
                eval_log_dict.update(evaluate_result_dict)
                if eval_log_dict['Eval.Gen.value_accuracy'] > best_eval_log_dict.get('Eval.Gen.value_accuracy_best', -1):
                    is_best = True
                    best_eval_log_dict['Eval.Gen.value_accuracy_best'] = eval_log_dict['Eval.Gen.value_accuracy']
                    if 'Eval.Gen.value_accuracy' not in summary_log_dict:
                        summary_log_dict['Eval.Gen.value_accuracy'] = []
                    summary_log_dict['Eval.Gen.value_accuracy'].append(eval_log_dict['Eval.Gen.value_accuracy'])

            train_log_dict = {}
            if logging_epoch_freq is not None and epoch % logging_epoch_freq == 0:
                train_log_dict = {f'T.{k}': sum(v) / len(v) if isinstance(v, list) else v for k, v in
                                train_epoch_result_dict.items()}

            if eval_log_dict or train_log_dict:
                log_dict = {'lr': scheduler.get_last_lr()[0], **train_log_dict, **eval_log_dict, **best_eval_log_dict}
                if accelerator.is_main_process and args['wandb_log']:
                    wandb.log(log_dict, step=global_iter_num)
                    log_dict = {'wandb': args['wandb_project'] + '|' + args['wandb_run_name'] + '|' + wandb.run.id, **log_dict}

                log_dict = {k: f'{v:.5g}' if isinstance(v, float) else v for k, v in log_dict.items()}
                accelerator.print(
                    f"[Epoch={epoch}/{args['n_epochs']}, Step={global_step}] {log_dict}")

            # if saving_epoch_freq is not None and epoch % saving_epoch_freq == 0:
            #     if is_best:
            #         save_path = os.path.join(model_dir, f'best')
            #         do_checkpoint(args, model, tokenizer, save_path)
            
            if saving_epoch_freq is not None and epoch % saving_epoch_freq == 0:
                cur_score = eval_log_dict.get('Eval.Gen.value_accuracy')
                save_topk_ckpt(cur_score, epoch, 'epoch',
                            args, model, tokenizer,
                            model_dir, topk_ckpts, most_recent_ckpts_paths)
            #     if args['keep_num_ckpt'] > 0:
            #         # save the checkpoint only if keep num ckpt > 0
            #         save_path = os.path.join(args['model_dir'], f'global_step_{str(global_step)}_epoch_{epoch}')
            #         do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths)

    return 

if __name__ == '__main__':
    from transformers import HfArgumentParser

    NONE_INT = -100
    NONE_STR = 'None'


    @dataclass
    class Arguments:
        base_model: str
        model_name_or_path: str
        tokenizer_name_or_path: str
        model_dir: str
        train_file: str
        test_file: str
        batch_size: int = field(default=8)
        mini_batch_size: int = field(default=8)
        eval_batch_size: int = field(default=8)
        ppo_epochs: int = field(default=1)
        n_epochs: int = field(default=40)
        num_workers: int = field(default=0)
        learning_rate: float = field(default=2e-5)
        weight_decay: float = field(default=1e-6)
        warmup_step: int = field(default=0)
        clip_grad_norm: float = field(default=1)
        vf_coef: float = field(default=1.0)
        kl_coef: float = field(default=0.1)
        gamma: float = field(default=0.98)
        lam: float = field(default=0.95)
        evaluating_epoch_freq: int = field(default=1)
        logging_epoch_freq: int = field(default=1)
        saving_epoch_freq: int = field(default=1000)
        evaluating_step_freq: int = field(default=100)
        logging_step_freq: int = field(default=1)
        # logging_seq_str_step_freq: int = field(default=NONE_INT)
        # logging_values_step_freq: int = field(default=NONE_INT)
        saving_step_freq: int = field(default=100)
        seed: int = field(default=42)
        max_input_length: int = field(default=1024)
        max_new_tokens: int = field(default=150)
        keep_num_ckpt: int = field(default=5)
        n_best_ckpt: int  = field(default=3) 
        llamatemplate: bool= field(default=False)
        # wandb stuff
        wandb_log: bool = field(default=False)
        wandb_project: str = field(default='coper')
        wandb_run_name: str = field(default='default_run_name')
        ###
        engine: str = field(default='python')
        adv_whitening: str = field(default='global')
        use_lora: bool = field(default=True)
        ref_model_name_or_path_majority: str = field(default=None)
        ref_model_name_or_path_minority: str = field(default=None)
        cluster_map_path: str = field(default=None)
        prompt: str="ucot" #coper, ucot, base
        infer_8bit: bool = True
        boost_ratio: float = 2

    parser = HfArgumentParser(Arguments)
    (args,) = parser.parse_args_into_dataclasses()
    args = asdict(args)
    for k, v in args.items():
        if v in [NONE_INT, NONE_STR]:
            args[k] = None
    accelerator = Accelerator(gradient_accumulation_steps = 4, kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))])
    accelerator.print(args)
    accelerator.print(json.dumps(args, indent=2, ensure_ascii=False))
    #print gradient_accumulation_steps number
    accelerator.print(f"[INFO] gradient_accumulation_steps is: {accelerator.gradient_accumulation_steps}", flush=True)
    main(args)
