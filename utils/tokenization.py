import pandas as pd
import torch
import json
from collections import defaultdict


def tokenize_fn_truncation(batch, tokenizer, args):
    """tokinization function for the dataset"""
    assert tokenizer.eos_token_id is not None, (tokenizer.eos_token_id, tokenizer.eos_token)
    
    new_batch = defaultdict(list)
    all_keys = list(batch.keys())
    
    # tokenizer.truncation_side = "left"
    max_input_length = args.get('max_input_length', 1024)
    # label_max = args.get('max_new_tokens', 200)
    label_max = 200 if args['prompt']=='coper' else 20
    print("max_input_length:", max_input_length, "label_max:", label_max)
                
    for item_values in zip(*(batch[k] for k in all_keys)):
        item = {k: item_values[i] for i, k in enumerate(all_keys)}
        
        instruction = item['instruction'].strip()
        input_text = item['input'].strip()
        output = item['output'].strip()
        score = item['score']

        # prepare frefix for PPO query and full sequence
        prefix_text = f"{instruction}\n\n{input_text}"  # PPO query
        
        # true score
        # true_label = extract_score_from_reasoning(truncate_after_final_answer(output))
        true_label = 0 if score <= 3 else 1
        
        intruct_encode = tokenizer(instruction, add_special_tokens=True)
        input_text_encode = tokenizer(input_text, add_special_tokens=False)
        output_encode = tokenizer(output, add_special_tokens=False)
        
        input_text_ids=input_text_encode['input_ids'] 
        max_input_text_length = max_input_length - 100  # reserve some space for instruction
        if len(input_text_ids) > max_input_text_length:
            input_text_ids = input_text_ids[-max_input_text_length:]  # take last max_input_length tokens
        
        # prefix（for PPO）
        prefix_ids = intruct_encode['input_ids'] + input_text_ids
        prefix_attention_mask = intruct_encode['attention_mask']+ [1] * len(input_text_ids)
        
        # full sequence（input + output + eos）
        output_ids = output_encode['input_ids'][:label_max-1]  
        full_ids = prefix_ids + output_ids + [tokenizer.eos_token_id]
        labels=[-100]*len(prefix_ids) + output_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(full_ids)
            
        new_batch['input_ids'].append(full_ids)
        new_batch['labels'].append(labels)
        new_batch['attention_mask'].append(attention_mask)
        new_batch['prefix'].append(prefix_ids)
        new_batch['prefix_attention_mask'].append(prefix_attention_mask)
        
        new_batch['instruction'].append(instruction)
        new_batch['input'].append(input_text)
        new_batch['output'].append(output)
        new_batch['prefix_text'].append(prefix_text)
        new_batch['true_labels'].append(true_label)
        
        #if there is item_id, keep it; otherwise create a new one
        if 'item_id' in item:
            new_batch['item_id'].append(item['item_id'])
        else:
            new_batch['item_id'].append(f"item_{len(new_batch['input_ids'])-1}")
            
        if "cluster_label" not in item:
            raise KeyError("cluster_label missing in tokenization batch; check Dataset pipeline.")
        new_batch["cluster_label"].append(item["cluster_label"])   
    
    return new_batch


def tokenize_fn_truncation_llama3(batch, tokenizer, args):
    """tokenization function for the dataset with llama3 chat template"""
    assert tokenizer.eos_token_id is not None, (tokenizer.eos_token_id, tokenizer.eos_token)
    
    new_batch = defaultdict(list)
    all_keys = list(batch.keys())
    
    max_input_length = args.get('max_input_length', 1024)
    # label_max = args.get('max_new_tokens', 200)
    label_max= 200 if args['prompt']=='coper' else 20
                
    for item_values in zip(*(batch[k] for k in all_keys)):
        item = {k: item_values[i] for i, k in enumerate(all_keys)}
        
        instruction = item['instruction'].strip()
        input_text = item['input'].strip()
        output = item['output'].strip()
        score = item['score']

        input_encode = tokenizer(input_text, add_special_tokens=False)
        input_text_ids = input_encode['input_ids']
        
        max_input_text_length = max_input_length - 100  # reserve some space for instruction and other tokens
        if len(input_text_ids) > max_input_text_length:
            input_text_ids = input_text_ids[-max_input_text_length:]
            truncated_input_text = tokenizer.decode(input_text_ids, skip_special_tokens=True)
        else:
            truncated_input_text = input_text
        
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": truncated_input_text}
        ]
        
        prefix_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        prefix_encode = tokenizer(prefix_text, add_special_tokens=False)
        prefix_ids = prefix_encode['input_ids']
        prefix_attention_mask = [1] * len(prefix_ids)
        
        output_encode = tokenizer(output, add_special_tokens=False)
        output_ids = output_encode['input_ids']
        
        if len(output_ids) > label_max-1:
            output_ids = output_ids[:label_max-1]
        
        # SFT
        full_ids = prefix_ids + output_ids + [tokenizer.eos_token_id]
        labels = [-100] * len(prefix_ids) + output_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(full_ids)
            
        # true score
        true_label = 0 if score <= 3 else 1
        
        new_batch['input_ids'].append(full_ids)
        new_batch['labels'].append(labels)
        new_batch['attention_mask'].append(attention_mask)
        new_batch['prefix'].append(prefix_ids)
        new_batch['prefix_attention_mask'].append(prefix_attention_mask)
        
        new_batch['instruction'].append(instruction)
        new_batch['input'].append(input_text)
        new_batch['output'].append(output)
        new_batch['prefix_text'].append(prefix_text)
        new_batch['true_labels'].append(true_label)
        
        # if there is item_id, keep it; otherwise create a new one
        if 'item_id' in item:
            new_batch['item_id'].append(item['item_id'])
        else:
            new_batch['item_id'].append(f"item_{len(new_batch['input_ids'])-1}")
        
        if "cluster_label" not in item:
            raise KeyError("cluster_label missing in tokenization batch; check Dataset pipeline.")
        new_batch["cluster_label"].append(item["cluster_label"])   
    
    return new_batch

def tokenize_fn(batch, tokenizer, args):
    """tokinization function for the dataset"""
    assert tokenizer.eos_token_id is not None, (tokenizer.eos_token_id, tokenizer.eos_token)
    
    new_batch = defaultdict(list)
    all_keys = list(batch.keys())
    print("All keys in batch:", all_keys)
    
    max_input_length = args.get('max_input_length', 1024)
    label_max = args.get('max_new_tokens', 200)
    max_full_length = max_input_length + label_max
    
    for item_values in zip(*(batch[k] for k in all_keys)):
        item = {k: item_values[i] for i, k in enumerate(all_keys)}
        
        instruction = item['instruction'].strip()
        input_text = item['input'].strip()
        output = item['output'].strip()
        score =item['score']
        
        # prepare frefix for PPO query and full sequence
        prefix_text = f"{instruction}\n\n{input_text}"  # PPO query
        full_input = f"{instruction}\n\n{input_text}\n{output}"  # full sequence
        
        true_label = 0 if score <= 3 else 1
        # Tokenization
        prefix_encode = tokenizer(prefix_text, add_special_tokens=False)
        output_encode = tokenizer(output, add_special_tokens=False)
        
        # full sequence（input + output + eos）
        input_ids = prefix_encode['input_ids']
        input_ids = input_ids[:max_input_length] 
        full_ids = input_ids + output_encode['input_ids'] + [tokenizer.eos_token_id]
        labels=[-100]*len(input_ids) + output_encode['input_ids'] + [tokenizer.eos_token_id]
        attention_mask = [1] * len(full_ids)
        
        # prefix（for PPO）
        prefix_ids = prefix_encode['input_ids']
        prefix_attention_mask = prefix_encode['attention_mask']

        # Truncation
        input_ids = full_ids[:max_full_length]
        labels = labels[:max_full_length]
        attention_mask = attention_mask[:max_full_length]
        prefix_ids = prefix_ids[:max_input_length]
        prefix_attention_mask = prefix_attention_mask[:max_input_length]
        
        new_batch['input_ids'].append(input_ids)
        new_batch['labels'].append(labels)
        new_batch['attention_mask'].append(attention_mask)
        new_batch['prefix'].append(prefix_ids)
        new_batch['prefix_attention_mask'].append(prefix_attention_mask)
        
        new_batch['instruction'].append(instruction)
        new_batch['input'].append(input_text)
        new_batch['output'].append(output)
        new_batch['prefix_text'].append(prefix_text)
        new_batch['true_labels'].append(true_label)
        
        #if there is item_id, keep it; otherwise create a new one
        if 'item_id' in item:
            new_batch['item_id'].append(item['item_id'])
        else:
            new_batch['item_id'].append(f"item_{len(new_batch['input_ids'])-1}")
    
    return new_batch

def tokenize_fn_llama3(batch, tokenizer, args):
    """tokenization function for the dataset with llama3 chat template"""
    assert tokenizer.eos_token_id is not None, (tokenizer.eos_token_id, tokenizer.eos_token)
    
    new_batch = defaultdict(list)
    all_keys = list(batch.keys())
    
    max_input_length = args.get('max_input_length', 1024)
    label_max = args.get('max_new_tokens', 200)
                
    for item_values in zip(*(batch[k] for k in all_keys)):
        item = {k: item_values[i] for i, k in enumerate(all_keys)}
        
        instruction = item['instruction'].strip()
        input_text = item['input'].strip()
        output = item['output'].strip()
        score = item['score']
        
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text}
        ]
        
        # 使用chat template生成prefix（for PPO query）
        prefix_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # true score
        true_label = 0 if score <= 3 else 1
        
        # 对prefix进行tokenization
        prefix_tokens = tokenizer(
            prefix_text,
            padding=False,
            truncation=True,
            max_length=max_input_length,
            add_special_tokens=False  # chat template已经包含了special tokens
        )
        
        prefix_ids = prefix_tokens['input_ids']
        prefix_attention_mask = prefix_tokens['attention_mask']
        
        # 对output进行tokenization
        output_tokens = tokenizer(
            output,
            padding=False,
            truncation=True,
            max_length=label_max,
            add_special_tokens=False
        )
        
        output_ids = output_tokens['input_ids']
        
        # 构建完整序列（input + output + eos）
        full_ids = prefix_ids + output_ids + [tokenizer.eos_token_id]
        labels = [-100] * len(prefix_ids) + output_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(full_ids)
            
        new_batch['input_ids'].append(full_ids)
        new_batch['labels'].append(labels)
        new_batch['attention_mask'].append(attention_mask)
        new_batch['prefix'].append(prefix_ids)
        new_batch['prefix_attention_mask'].append(prefix_attention_mask)
        
        new_batch['instruction'].append(instruction)
        new_batch['input'].append(input_text)
        new_batch['output'].append(output)
        new_batch['prefix_text'].append(prefix_text)
        new_batch['true_labels'].append(true_label)
        
        # if there is item_id, keep it; otherwise create a new one
        if 'item_id' in item:
            new_batch['item_id'].append(item['item_id'])
        else:
            new_batch['item_id'].append(f"item_{len(new_batch['input_ids'])-1}")
    
    return new_batch

def collate_fn(batch, tokenizer,args):

    max_input_length = max([len(item['input_ids']) for item in batch])
    max_target_length = max([len(item['labels']) for item in batch])
    max_prefix_length = max([len(item['prefix']) for item in batch])

    # for standard sft training (right padding)
    input_ids, attention_mask, labels = [], [], []
    
    # for PPO training (left padding)
    prefix_left_padded = []
    prefix_attention_mask_left_padded = []
    labels_left_padded = []
    cluster_labels = []

    for item in batch:
        # standard SFT training: right padding
        input_ids.append(
            item['input_ids'] + [tokenizer.pad_token_id] * (max_input_length - len(item['input_ids']))
        )
        attention_mask.append(
            item['attention_mask'] + [0] * (max_input_length - len(item['attention_mask']))
        )
        labels.append(
            item['labels'] + [-100] * (max_target_length - len(item['labels']))
        )
        
        # PPO training: left padding
        labels_left_padded.append(
            [-100] * (max_target_length - len(item['labels'])) + item['labels']
        )
        prefix_left_padded.append(
            [tokenizer.pad_token_id] * (max_prefix_length - len(item['prefix'])) + item['prefix']
        )
        prefix_attention_mask_left_padded.append(
            [0] * (max_prefix_length - len(item['prefix_attention_mask'])) + item['prefix_attention_mask']
        )
        cluster_labels.append(item["cluster_label"])

    # PPO forward arguments
    ppo_forward_kwargs = {
        'query': [item['prefix_text'] for item in batch],
        'query_tensors': torch.LongTensor(prefix_left_padded),
        'query_tensors_attention_mask': torch.BoolTensor(prefix_attention_mask_left_padded),
        'answer_values': [item['true_labels'] for item in batch],  
        'item_ids': torch.LongTensor([int(item['item_id'].split('_')[-1]) if '_' in str(item['item_id']) else i for i, item in enumerate(batch)]),
        'cluster_labels': torch.LongTensor(cluster_labels)
    }
    
    # generation prefix arguments
    generate_prefix_kwargs = {
        'input_ids': torch.LongTensor(prefix_left_padded),
        'attention_mask': torch.BoolTensor(prefix_attention_mask_left_padded),
        'labels': torch.LongTensor(labels_left_padded)
    }

    # standard sft training parameters
    standard_kwargs = {
        'input_ids': torch.LongTensor(input_ids),
        'attention_mask': torch.BoolTensor(attention_mask),
        'labels': torch.LongTensor(labels),
        'true_labels': [item['true_labels'] for item in batch],
    }
    
    return {
        'ppo_forward_kwargs': ppo_forward_kwargs,
        'generate_prefix_kwargs': generate_prefix_kwargs,
        'standard_kwargs': standard_kwargs,  
    }
    

def preprocess_function(examples, tokenizer, max_length=1280, label_max_len=256):
    input_texts = [instr + "\n\n" + inp for instr, inp in zip(examples["instruction"], examples["input"])]
    output_texts = examples["output"]
    
    input_ids = []
    attention_masks = []
    labels = []
    
    for input_text, output_text in zip(input_texts, output_texts):
        input_tokens = tokenizer(
            input_text,
            padding=False,
            truncation=True,
            max_length=max_length - label_max_len,  # 1024
            add_special_tokens=True
        )["input_ids"]
        
        output_tokens = tokenizer(
            output_text,
            padding=False,
            truncation=True,
            max_length=label_max_len - 1,  # 255，留1个给EOS
            add_special_tokens=False
        )["input_ids"]
        
        full_input_ids = input_tokens + output_tokens + [tokenizer.eos_token_id]
        
        input_len = len(input_tokens)
        label = [-100] * input_len + output_tokens + [tokenizer.eos_token_id]
        
        input_ids.append(full_input_ids)
        labels.append(label)
        attention_masks.append([1] * len(full_input_ids))
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels
    }   
    
    
def tokenize_testdata(dataset, tokenizer, max_length=1024):
    """
    Output: dict: keys: input_ids, attention_mask, prompt_text
    """
    if isinstance(dataset, pd.DataFrame):
        dataset = dataset.to_dict("records")

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    
    prompts = [ex["instruction"] + "\n\n" + ex["input"] for ex in dataset]
    tokens = tokenizer(prompts, 
                       return_tensors="pt", 
                       padding=True, 
                       truncation=True, 
                       max_length=max_length, 
                       add_special_tokens=True,
                       pad_to_multiple_of=8)
    
    tokens["prompt_text"] = prompts
    
    tokenizer.padding_side = original_padding_side
    return tokens