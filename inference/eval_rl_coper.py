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
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from trl import AutoModelForCausalLMWithValueHead
import numpy as np
import pandas as pd
from utils.CoPeR import generate_coper_prompt_inference,generate_coper_prompt_training
from utils.generate import truncate_after_final_answer,extract_score_from_reasoning
from sklearn.metrics import f1_score, classification_report
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from peft.utils import prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, GenerationConfig
import re

tqdm = partial(tqdm, ncols=0, leave=False)

TIMEOUT = 10
    

def tokenize_fn_truncation(batch, args, tokenizer):
    """tokenization function for the dataset"""
    assert tokenizer.eos_token_id is not None, (tokenizer.eos_token_id, tokenizer.eos_token)
    
    new_batch = defaultdict(list)
    all_keys = list(batch.keys())
    
    max_input_length = args.get('max_input_length', 1024)
    label_max = args.get('max_new_tokens', 200)
                
    for item_values in zip(*(batch[k] for k in all_keys)):
        item = {k: item_values[i] for i, k in enumerate(all_keys)}
        
        instruction = item['instruction'].strip()
        input_text = item['input'].strip()
        # output = item['output'].strip()
        score = item['score']

        # prepare prefix for PPO query and full sequence
        prefix_text = f"{instruction}\n\n{input_text}"  # PPO query
        
        # true score
        true_label = 0 if score <= 3 else 1
        
        # Tokenization
        intruct_encode = tokenizer(instruction, add_special_tokens=False)
        input_text_encode = tokenizer(input_text, add_special_tokens=False)
        # output_encode = tokenizer(output, add_special_tokens=False)
        
        input_text_ids = input_text_encode['input_ids'] 
        if len(input_text_ids) > max_input_length:
            input_text_ids = input_text_ids[-max_input_length:]  # take last max_input_length tokens
        
        # prefix（for PPO）
        prefix_ids = intruct_encode['input_ids'] + input_text_ids
        prefix_attention_mask = intruct_encode['attention_mask'] + [1] * len(input_text_ids)
        
        new_batch['prefix'].append(prefix_ids)
        new_batch['prefix_attention_mask'].append(prefix_attention_mask)
        
        new_batch['instruction'].append(instruction)
        new_batch['input'].append(input_text)
        # new_batch['output'].append(output)
        new_batch['prefix_text'].append(prefix_text)
        new_batch['true_labels'].append(true_label)
        
        # if there is item_id, keep it; otherwise create a new one
        if 'item_id' in item:
            new_batch['item_id'].append(item['item_id'])
        else:
            new_batch['item_id'].append(f"item_{len(new_batch['prefix'])-1}")
           
    
    return new_batch

def collate_fn(batch, args, tokenizer):
    max_prefix_length = max([len(item['prefix']) for item in batch])
    
    # for PPO training (left padding)
    prefix_left_padded = []
    prefix_attention_mask_left_padded = []
    # labels_left_padded = []

    for item in batch:
        prefix_left_padded.append(
            [tokenizer.pad_token_id] * (max_prefix_length - len(item['prefix'])) + item['prefix']
        )
        prefix_attention_mask_left_padded.append(
            [0] * (max_prefix_length - len(item['prefix_attention_mask'])) + item['prefix_attention_mask']
        )

    # generation prefix arguments
    generate_prefix_kwargs = {
        'input_ids': torch.LongTensor(prefix_left_padded),
        'attention_mask': torch.BoolTensor(prefix_attention_mask_left_padded),
        # 'labels': torch.LongTensor(labels_left_padded)
    }

    return {
        'generate_prefix_kwargs': generate_prefix_kwargs,
    }

def prepare_test_dataset_and_dataloader(args, tokenizer):
    from collections import defaultdict
    from functools import partial
    import torch
    
    with accelerator.main_process_first():
        
        df_test = pd.read_csv("data/test.csv")
        
        # Generate prompts for test data
        if 'generate_coper_prompt_inference' in globals():
            prompts_test = df_test.apply(generate_coper_prompt_inference, axis=1, result_type="expand")
        else:
            raise ImportError("Function 'generate_coper_prompt_inference' not found. Please ensure it is defined.")
        
        raw_dataset = Dataset.from_pandas(prompts_test)
        accelerator.print('Raw test data:', raw_dataset)

        tokenized_dataset = raw_dataset.map(
            tokenize_fn_truncation, 
            fn_kwargs={'args': args, 'tokenizer': tokenizer}, 
            batched=True,
            remove_columns=raw_dataset.column_names,
            num_proc=None, 
            load_from_cache_file=False, 
            keep_in_memory=False,
        )
        accelerator.print('Processed test data:', tokenized_dataset)

    test_dataloader = DataLoader(
        tokenized_dataset, 
        shuffle=False, 
        batch_size=args.get('eval_batch_size', 8),
        num_workers=args.get('num_workers', 0), 
        pin_memory=True,
        collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer)
    )

    return tokenized_dataset, test_dataloader

def evaluate_generation(args, model, dataset, dataloader, tokenizer):
    model.eval()
    predictions = []
    targets = []
    
    accelerator.print("Starting evaluation...")
    
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=not accelerator.is_main_process,
                           desc='Evaluation Loop'):
        with torch.no_grad():
            output_ = accelerator.unwrap_model(model).generate(
                input_ids=batch['generate_prefix_kwargs']['input_ids'],
                attention_mask=batch['generate_prefix_kwargs']['attention_mask'],
                max_new_tokens=args.get('max_new_tokens', 200),
                min_new_tokens=5,
                output_scores=True,
                return_dict_in_generate=True,
                use_cache=True,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bad_words_ids=[[tokenizer.pad_token_id]],
            )
            
        generated_ids = output_.sequences
        generated_ids = pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True)
        generated_ids= accelerator.gather(generated_ids)

        # Decode predictions
        preds = [tokenizer.decode(g.cpu().numpy().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in generated_ids]
        predictions.extend(preds)
          
    print("Size of dataset:", len(dataset))
    print("Size of predictions:", len(predictions))
    
    
    # if accelerator.is_main_process and accelerator.is_local_main_process:
    results = []
    corr_value = 0
    
    # Initialize counters for F1 calculation
    true_labels = []
    pred_labels = []
    
    dataset= dataset.select(range(len(predictions))) if len(predictions) < len(dataset) else dataset

    for pred_txt, item in zip(predictions, dataset):
        target_score = int(item["true_labels"])          # already 0 / 1
        pred_trunc = truncate_after_final_answer(pred_txt.strip().split("Final answer: Low satisfaction or High satisfaction")[1].strip()) if "Final answer: Low satisfaction or High satisfaction" in pred_txt else None
        print(f"Predicted reasoning: {pred_trunc}")
        pred_score = extract_score_from_reasoning(pred_trunc) if pred_trunc is not None else None

        is_correct = (int(pred_score) == int(target_score)) if pred_score is not None else False
        corr_value += int(is_correct)

        accelerator.print(f"Item ID: {item['item_id']}, Target: {target_score}, Predict Score: {pred_score}")
        
        if pred_score is not None:
            true_labels.append(target_score)
            pred_labels.append(int(pred_score))

        results.append(
            {
                "item_id": item["item_id"],
                "target_score": target_score,
                "prediction": pred_txt,
                "prediction_trunc": pred_trunc,
                "prediction_score": pred_score,
                "is_correct": is_correct,
            }
        )

    # Save results
    res_path = os.path.join(args["model_dir"].rstrip("/"), "test_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    accelerator.print(f"Results saved to: {res_path}")
    
    # Calculate metrics
    if len(true_labels) > 0:
        # Calculate macro F1 score
        macro_f1 = f1_score(true_labels, pred_labels, average='macro') * 100  # Convert to percentage
        
        # Print detailed classification report
        accelerator.print("[Test Results] Classification Report:")
        accelerator.print(classification_report(true_labels, pred_labels, target_names=['Class 0', 'Class 1']))
        accelerator.print(f"[Test Results] Macro F1 Score: {macro_f1:.5g}%")
    else:
        macro_f1 = 0.0
        accelerator.print(f"[Test Results] No valid predictions found, Macro F1 Score: {macro_f1:.5g}%")
        
    accuracy = corr_value / len(true_labels) * 100
    accelerator.print(f"[Test Results] Accuracy: {accuracy:.5g}%")
    
    # Save summary metrics
    summary = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": classification_report(true_labels, pred_labels, target_names=['Class 0', 'Class 1'], output_dict=True),
        "total_samples": len(results),
        "valid_predictions": len(true_labels),
        "invalid_predictions": len(results) - len(true_labels)
    }
    
    summary_path = os.path.join(args["model_dir"].rstrip("/"), "test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    accelerator.print(f"Summary saved to: {summary_path}")
    
    value_accuracy = torch.FloatTensor([macro_f1]).to(accelerator.device)
    value_accuracy = broadcast(value_accuracy).cpu().numpy().tolist()[0]

    return {'value_accuracy': value_accuracy, 'accuracy': accuracy if accelerator.is_main_process else -1}

def load_model_and_tokenizer(args):
    """Load the trained model and tokenizer from checkpoint"""
    
    # Load fine-tuned adapter from best checkpoint
    best_checkpoint_path = os.path.join(args["model_dir"], "best_56.1511_epoch2")
    if not os.path.exists(best_checkpoint_path):
        raise FileNotFoundError(f"Best checkpoint not found at {best_checkpoint_path}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(best_checkpoint_path, use_fast=True, local_files_only=True)
    tokenizer.pad_token_id = 2
    
    # Load quantization config
    quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_enable_fp32_cpu_offload=True)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args["base_model"],
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        quantization_config=quant_config
    )
    # base_model = prepare_model_for_kbit_training(base_model)
    
    accelerator.print(f"Loading model from: {best_checkpoint_path}")
    
    model = PeftModel.from_pretrained(
        base_model,
        best_checkpoint_path,
        local_files_only=True,
        is_trainable=False  # Set to False for inference
    )
    
    # Wrap with value head
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    
    if not hasattr(model, "generation_config"):
        model.generation_config = GenerationConfig.from_model_config(model.config)
        accelerator.print("[INFO] Model generation config:", model.generation_config)

    return model, tokenizer

def main(args):
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Prepare test dataset
    test_dataset, test_dataloader = prepare_test_dataset_and_dataloader(args, tokenizer)
    
    # Prepare model for accelerate
    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    
    # Run evaluation
    accelerator.print("Starting test evaluation...")
    results = evaluate_generation(args, model, test_dataset, test_dataloader, tokenizer)
    
    if accelerator.is_main_process:
        accelerator.print("=" * 50)
        accelerator.print("FINAL TEST RESULTS:")
        accelerator.print(f"Accuracy: {results['accuracy']:.2f}%")
        accelerator.print(f"Macro F1: {results['value_accuracy']:.2f}%")
        accelerator.print("=" * 50)

if __name__ == '__main__':
    from transformers import HfArgumentParser

    NONE_INT = -100
    NONE_STR = 'None'

    @dataclass
    class TestArguments:
        base_model: str
        model_dir: str  # Directory containing the best checkpoint
        tokenizer_name_or_path: str
        eval_batch_size: int = field(default=8)
        num_workers: int = field(default=0)
        max_input_length: int = field(default=1024)
        max_new_tokens: int = field(default=150)
        use_lora: bool = field(default=True)

    parser = HfArgumentParser(TestArguments)
    (args,) = parser.parse_args_into_dataclasses()
    args = asdict(args)
    
    for k, v in args.items():
        if v in [NONE_INT, NONE_STR]:
            args[k] = None
    
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))])
    accelerator.print("Test Arguments:")
    accelerator.print(json.dumps(args, indent=2, ensure_ascii=False))
    
    main(args)