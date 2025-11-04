import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from trl import SFTTrainer
from trl import setup_chat_format
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
from sklearn.model_selection import train_test_split
import bitsandbytes as bnb
from utils.logger_utils import setup_logging
from utils.CoPeR import generate_coper_prompt_training, generate_coper_prompt_inference
from sklearn.utils import resample
from collections import Counter
from utils.util import WeightedLossTrainer
import random
from peft import PeftModel
from utils.voting import predict_voting
from utils.voting_batch import predict_voting_batch, predict_voting_expanded


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def truncate_after_final_answer(text):
    """
    Truncate text after the first occurrence of 'Final answer: ... satisfaction'.
    Helps avoid repeated template outputs.
    """
    match = re.search(r"(Final answer\s*:\s*(?:High|Low)\s+satisfaction)", text, re.IGNORECASE)
    if match:
        return text[:match.end()]
    return text


def extract_score_from_reasoning(output_text):
    pattern = r"final answer\s*:\s*(high|low)\s+satisfaction"
    match = re.search(pattern, output_text.lower())
    if match:
        return 1 if match.group(1) == "high" else 0
    else:
        print(f"Could not extract final answer from output:\n{output_text}")
        return None

def tokenize_testdata(dataset, tokenizer, max_length=1024):
    """
    Output: dict: keys: input_ids, attention_mask, prompt_text
    """
    if isinstance(dataset, pd.DataFrame):
        dataset = dataset.to_dict("records")

    prompts = [ex["instruction"] + "\n\n" + ex["input"] for ex in dataset]
    tokens = tokenizer(prompts, 
                       return_tensors="pt", 
                       padding=True, 
                       truncation=True, 
                       max_length=max_length, 
                       pad_to_multiple_of=8)
    
    tokens["prompt_text"] = prompts
    return tokens

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
    X_test_df.to_csv(os.path.join(log_path,"inference_with_reasoning_test.csv"), index=False,escapechar="\\")
      
    return y_pred

def predict(test_data, model, tokenizer, args, X_test_df,y_true,log_path):
    y_pred = []
    reasoning_outputs = []

    model.eval()
    model.to(model.device)

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'  # Ensure padding is on the left for generation
    for sample in tqdm(test_data):
        prompt = sample["instruction"] + "\n\n" + sample["input"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.85,
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

    tokenizer.padding_side = original_padding_side  
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_name', type=str,help="base model name, choose from meta-llama/Llama-3.2-1B-Instruct or meta-llama/Meta-Llama-3-8B-Instruct", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument('--model_path', type=str, help="path to save trained model", default="output_coper_gpt4.1mini/model")
    parser.add_argument('--log_path', type=str, help="path to save log", default="output_coper_gpt4.1mini/log")
    parser.add_argument('--test_file', type=str, default="../../../data/test.csv")
    parser.add_argument('--test_batch_size', type=int, default=4, help="batch size for test/inference")
    parser.add_argument('--voting', action='store_true', help="whether to use voting for prediction")
    parser.add_argument('--base8bit', action='store_true', help="whether to use 8-bit or 4-bit quantization for the base model")

    args = parser.parse_args()
    
    variable="Llama-3.2-1B-Instruct_finetune_True_steps_lr_0.0001_r_16_alpha_16_dropout_0"
    model_path = os.path.join(args.model_path,variable.split("/")[0])
    log_path = os.path.join(args.log_path, variable.split("/")[0])
    # Create directories if they don't exist
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # logging file
    setup_logging(log_path, log_filename="log_test857.txt")
    print(f"Logging to {log_path}")

    # Load test data
    df = pd.read_csv(args.test_file)
    X_test_prompt = df.apply(generate_coper_prompt_inference, axis=1, result_type="expand")
    y_true = df["label"].apply(lambda x: 0 if x=="A" else 1).tolist()
    test_data = Dataset.from_pandas(X_test_prompt)

    # Tokenizer
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, use_fast=False)
    special_tokens_dict = {
    "additional_special_tokens": ["[seeker]", "[supporter]"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    # Load LoRA model
    if args.base8bit:
        print("Using 8-bit quantization for the base model")
        quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_enable_fp32_cpu_offload=True,
        )
    else:
        print("Using 4-bit quantization for the base model")
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
        )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quant_config
    )

    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, model_path)

    tokenized_test = tokenize_testdata(test_data, tokenizer, max_length=1024)
    
    # Prediction
    if args.voting:
        print("Using voting for prediction")
        y_pred = predict_voting_expanded(tokenized_test, model, tokenizer,args,X_test_prompt,y_true,log_path,num_samples=10)
    else:
        print("Using pipeline for prediction")
        y_pred = predict(test_data, model, tokenizer,args,X_test_prompt,y_true,log_path)
    
    tokenizer.padding_side = original_padding_side  # Restore original padding side
    
    # Evaluation
    evaluate(y_true, y_pred)

if __name__ == "__main__":
    main()
