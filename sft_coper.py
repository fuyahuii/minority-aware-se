import os
os.environ["WANDB_CONSOLE"] = "off"
import numpy as np
import pandas as pd
import re
import argparse
from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import (LoraConfig, PeftConfig,get_peft_model,prepare_model_for_kbit_training)
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments, 
                          pipeline, 
                          logging,
                          EarlyStoppingCallback,
                          DataCollatorForSeq2Seq)
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)
import bitsandbytes as bnb
from utils.logger_utils import setup_logging
from utils.CoPeR import generate_coper_prompt_training, generate_coper_prompt_inference
from utils.tokenization import preprocess_function_new, tokenize_testdata,preprocess_function,preprocess_function_llama3,tokenize_testdata_llama3,preprocess_function_truncations,preprocess_function_truncations_llama3
from sklearn.utils import resample
from collections import Counter
from utils.util import WeightedLossTrainer
import random
from utils.voting_batch import predict_voting_expanded, predict_voting_batch
from transformers import StoppingCriteria, StoppingCriteriaList


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_data(args):
    X_train = pd.read_csv("data/train_with_reasoning.csv")
    X_eval = pd.read_csv("data/valid_with_reasoning.csv")
    X_test = pd.read_csv("data/test.csv")

    print(f"Original Train size: {len(X_train)}, Eval size: {len(X_eval)}, Test size: {len(X_test)}")
    
    X_train["binary_label"] = X_train["label"].apply(lambda x: 0 if x == "A" else 1)

    original_counts = Counter(X_train["binary_label"])
    print(f"Original label distribution in training set: {dict(original_counts)}")

    if args.sampling:
        # oversampling：
        df_majority = X_train[X_train["binary_label"] == 1]
        df_minority = X_train[X_train["binary_label"] == 0]

        df_minority_upsampled = resample(
            df_minority,
            replace=True,
            n_samples=int(len(df_majority) * 0.5),
            random_state=42
        )

        X_train_balanced = pd.concat([df_majority, df_minority_upsampled])
        X_train_balanced = X_train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        X_train=X_train_balanced
        balanced_counts = Counter(X_train["binary_label"])
        print(f"Balanced label distribution in training set: {dict(balanced_counts)}")

    X_train_prompt = X_train.apply(generate_coper_prompt_training, axis=1, result_type="expand")
    X_eval_prompt = X_eval.apply(generate_coper_prompt_training, axis=1, result_type="expand")
    X_test_prompt = X_test.apply(generate_coper_prompt_inference, axis=1, result_type="expand")
        
    y_true = X_test["label"].apply(lambda x: 0 if x=="A" else 1).tolist()

    train_data = Dataset.from_pandas(X_train_prompt)
    eval_data = Dataset.from_pandas(X_eval_prompt)
    test_data = Dataset.from_pandas(X_test_prompt)

    return train_data, eval_data, test_data, y_true, X_test_prompt

def truncate_after_final_answer(text):
    """
    Truncate text after the first occurrence of 'Final answer: ... satisfaction'.
    Helps avoid repeated template outputs.
    """
    match = re.search(r"(Final answer\s*:\s*(?:High|Low)\s+satisfaction)", text, re.IGNORECASE)
    if match:
        return text[:match.end()]
    return text
    
def extract_score_from_reasoning(text: str):
    text_lc = text.lower()
    
    m = re.search(r"final score\s*:\s*([1-5])\s*(?:$|\n)", text_lc)
    if m:
        score = int(m.group(1))
        return 0 if score <= 3 else 1
    
    m = re.search(r"final answer\s*:\s*(high|low)\s+satisfaction(?!\s+or)", text_lc)
    if m:
        return 1 if m.group(1) == "high" else 0
    
    return None  # If neither pattern matches, return None

def predict_pipeline(test_data, model, tokenizer, args,X_test_df,log_path):
    y_pred = []
    reasoning_outputs=[]
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, temperature=0.4, do_sample=True, top_p=0.9)
    for sample in tqdm(test_data):
        prompt = sample["instruction"] + "\n\n" + sample["input"]
        result = pipe(prompt)
        output_text = result[0]["generated_text"]
        generated = output_text[len(prompt):].strip()
        score = extract_score_from_reasoning(generated)
        
        y_pred.append(score)
        reasoning_outputs.append(generated)
        
    X_test_df = X_test_df.copy()
    X_test_df["generated_reasoning"] = reasoning_outputs
    X_test_df.to_csv(os.path.join(log_path,"inference_with_reasoning.csv"), index=False,escapechar="\\")
      
    return y_pred

class StopTokenCriteria(StoppingCriteria):
    def __init__(self, stop_strings, tokenizer, prompt_length=None):
        super().__init__()
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        for seq in input_ids:
            text = self.tokenizer.decode(
                seq[self.prompt_length:] if self.prompt_length else seq,
                skip_special_tokens=False
            )
            if any(s in text for s in self.stop_strings):
                return True
        return False

def predict(test_data, model, tokenizer, args, X_test_df,y_true,log_path):
    y_pred = []
    reasoning_outputs = []
    
    model.eval()
    model.to(model.device)

    for sample in tqdm(test_data):
        prompt = sample["instruction"] + "\n\n" + sample["input"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=512,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        input_len = len(inputs["input_ids"][0])
        generated=tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
        generated = truncate_after_final_answer(generated)
        print(generated)
        score = extract_score_from_reasoning(generated)
    
        y_pred.append(score)
        reasoning_outputs.append(generated)

    X_test_df = X_test_df.copy()
    X_test_df["generated_reasoning"] = reasoning_outputs
    X_test_df["predicted_label"] = y_pred
    X_test_df["true_label"] = y_true
    X_test_df.to_csv(os.path.join(log_path,"inference_with_reasoning.csv"), index=False, escapechar="\\")

    return y_pred
 
def predict_batch(tokenized_test, model, tokenizer, args, X_test_df, y_true, log_path):
    batch_size = getattr(args, "test_batch_size", 4)
    input_ids = tokenized_test["input_ids"]
    attention_mask = tokenized_test["attention_mask"]

    y_pred = []
    reasoning_outputs = []
    
    stopper = StopTokenCriteria(["@"], tokenizer, prompt_length=input_ids.shape[-1])
    
    tokenizer.padding_side = 'left'  # Ensure padding is on the left for Llama-3 models
    model.eval()
    model.to(model.device)
    num_samples = input_ids.shape[0]

    for batch_start in tqdm(range(0, num_samples, batch_size), desc="Predict (batch)"):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_input_ids = input_ids[batch_start:batch_end].to(model.device)
        batch_attention_mask = attention_mask[batch_start:batch_end].to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bad_words_ids=[[tokenizer.pad_token_id]],  # Avoid generating pad token
                # stopping_criteria=StoppingCriteriaList([stopper])
            )

        for i, output in enumerate(output_ids):
            input_len=batch_input_ids[i].shape[0] #if padding is on the left
            generated = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
            print("generated output:", generated)
            generated = truncate_after_final_answer(generated)
            score = extract_score_from_reasoning(generated)
            y_pred.append(score)
            reasoning_outputs.append(generated)
            
            print("score is:", score)

    X_test_df = X_test_df.copy()
    X_test_df["generated_reasoning"] = reasoning_outputs
    X_test_df["predicted_label"] = y_pred
    X_test_df["true_label"] = y_true
    X_test_df.to_csv(os.path.join(log_path, "inference_with_reasoning.csv"), index=False, escapechar="\\")
    return y_pred

def evaluate(y_true, y_pred):
    y_true_arr = []
    y_pred_arr = []
    skipped = []
    skipped_num=0

    for yt, yp in zip(y_true, y_pred):
        try:
            yp_num = float(yp)
            y_true_arr.append(yt)
            y_pred_arr.append(yp_num)
        except Exception:
            skipped.append(yp)
            skipped_num += 1
    
    if skipped:
        print(f"Skipped non-numeric y_pred values: {skipped}")
        print(f"Number of skipped predictions: {skipped_num}")

    if len(y_true_arr) == 0 or len(y_pred_arr) == 0:
        print("No valid numeric predictions to evaluate.")
        return

    y_true_arr = np.array(y_true_arr)
    y_pred_arr = np.array(y_pred_arr)
    mask = ~np.isnan(y_pred_arr)
    y_true_clean, y_pred_clean = y_true_arr[mask], y_pred_arr[mask]
    print(f"Accuracy: {accuracy_score(y_true_clean, y_pred_clean):.4f}")
    print(classification_report(y_true_clean, y_pred_clean))


def main():
    set_seed(42)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_name', type=str, help="base model name, choose from meta-llama/Llama-3.2-1B-Instruct or meta-llama/Meta-Llama-3-8B-Instruct", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument('--model_path', type=str, help="path to save trained model", default="output_coper_8bit_gpt4.1mini/model")
    parser.add_argument('--log_path', type=str, help="path to save log", default="output_coper_8bit_gpt4.1mini/log")
    parser.add_argument('--eval_strategy', type=str, help="evaluate strategy", default="steps")
    parser.add_argument('--finetune', action='store_true', help="whether finetune llama")
    parser.add_argument('--voting', action='store_true', help="whether to use voting for prediction")
    parser.add_argument('--sampling', action='store_true', help="whether to use sampling") 
    parser.add_argument('--weighted_loss', action='store_true', help="whether to use loss weight")    
    parser.add_argument('--test_batch_size', type=int, default=8, help="batch size for test/inference")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0)
    parser.add_argument('--warmup', type=int, default=0)
    args = parser.parse_args()
    print(args)
   
    save_variable = f"{args.base_model_name.split('/')[-1]}_finetune_{args.finetune}_{args.eval_strategy}_lr_{args.lr}_r_{args.lora_r}_alpha_{args.lora_alpha}_dropout_{args.lora_dropout}"
    model_path = os.path.join(args.model_path, save_variable)
    log_path = os.path.join(args.log_path, save_variable)

    # Create directory
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
    if not os.path.exists(log_path):
        os.makedirs(log_path,exist_ok=True)

    # logging file
    setup_logging(log_path, log_filename="log_generate.txt")
    print(f"Logging to {log_path}")
    
    train_data, eval_data, test_data, y_true,X_test_prompt = prepare_data(args)
    
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, use_fast=False)
    tokenizer.pad_token_id= 2 # token "@"
    print("Tokenizer pad token id:", tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id))
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        
    quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_enable_fp32_cpu_offload=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        device_map=device_map,
        torch_dtype=torch.float16,
        quantization_config=quant_config,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.config.pretraining_tp = 1
    
    if not args.finetune:
        print("No finetuning, only inference.")
        y_pred = predict(test_data, model, tokenizer,args,X_test_prompt,log_path)
        evaluate(y_true, y_pred)
    else:  
        def find_all_linear_names(model):  
            cls =  bnb.nn.Linear8bitLt
            lora_module_names = set()
            for name, module in model.named_modules():
                if isinstance(module, cls):
                    names = name.split('.')
                    lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            if 'lm_head' in lora_module_names:  # needed for 16 bit
                lora_module_names.remove('lm_head')
            return list(lora_module_names)
        modules = find_all_linear_names(model)
        print("modules to be fine-tuned:", modules)

        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=modules,
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)

        if not ddp and torch.cuda.device_count() > 1:
            # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            model.is_parallelizable = True
            model.model_parallel = True
            
        tokenized_train = train_data.map(
            lambda examples: preprocess_function_new(examples, tokenizer=tokenizer,max_length=1280, label_max_len=256),
            batched=True,
            remove_columns=train_data.column_names,
        )
        tokenized_eval = eval_data.map(
            lambda examples: preprocess_function_new(examples, tokenizer=tokenizer,max_length=1280, label_max_len=256),
            batched=True,
            remove_columns=eval_data.column_names
        )
        tokenized_test = tokenize_testdata(test_data, tokenizer, max_length=1024)

        training_arguments = TrainingArguments(
            output_dir=model_path,                    # directory to save and repository id
            num_train_epochs=15,                       # number of training epochs
            per_device_train_batch_size=8,            # batch size per device during training
            # gradient_accumulation_steps=32 // world_size,            # number of steps before performing a backward/update pass
            gradient_checkpointing=True,              # use gradient checkpointing to save memory
            optim="paged_adamw_32bit",
            logging_steps=10,   
            logging_first_step=True,   
            log_level="info",                     
            learning_rate=args.lr,                       # learning rate, based on QLoRA paper
            weight_decay=0.001,
            fp16=True,
            bf16=False,
            max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
            max_steps=-1,
            warmup_ratio=0.03,                        # warmup ratio based on QLoRA paper
            group_by_length=False,
            lr_scheduler_type="cosine",               # use cosine learning rate scheduler
            # report_to="wandb",                  # report metrics to w&b
            eval_strategy=args.eval_strategy,              # eval checkpoint every epoch
            eval_steps = 100,
            save_steps = 100,
            save_strategy=args.eval_strategy,              # save checkpoint every epoch
            save_total_limit=1,
            metric_for_best_model="loss",
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            local_rank=local_rank,
            greater_is_better=False,
            disable_tqdm=True,
            label_names=["labels"]
        )
        print(f"Training arguments: {training_arguments}")
        trainer = transformers.Trainer(
            model=model,
            args=training_arguments,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
                    )
        
        if args.weighted_loss:
            trainer = WeightedLossTrainer(
                model=model,
                args=training_arguments,
                tokenizer=tokenizer,  
                train_dataset=tokenized_train,
                eval_dataset=tokenized_eval,
                data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
        
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
        else:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            percent = 100 * trainable_params / total_params
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Total parameters:     {total_params:,}")
            print(f"Trainable %:          {percent:.4f}%")
        
        trainer.train()
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        
        if args.voting:
            print("Using voting for prediction.")
            y_pred = predict_voting_expanded(tokenized_test, model, tokenizer, args,X_test_prompt,y_true,log_path, num_samples=10)
        else:
            y_pred = predict_batch(tokenized_test, model, tokenizer,args,X_test_prompt,y_true,log_path)
        evaluate(y_true, y_pred)   

if __name__ == "__main__":
    main()
    