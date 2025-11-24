from tqdm import tqdm
import torch
import re
import os
import random
import numpy as np
import pandas as pd
import json
from sklearn.metrics import f1_score, classification_report
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import pad_across_processes, broadcast
from collections import defaultdict

def truncate_after_final_answer(text: str) -> str:
    m = re.search(r"(Final answer\s*:\s*(?:High|Low)\s+satisfaction)", text, re.IGNORECASE)
    return text[: m.end()] if m else text

def extract_score_from_reasoning(text: str):
    text_lc = text.lower()
    
    # First try to match final score anywhere in the text (remove the $ anchor)
    m = re.search(r"final score\s*:\s*([1-5])\s*(?:$|\n)", text_lc)
    if m:
        score = int(m.group(1))
        return 0 if score <= 3 else 1
    
    # Only if final score is not found, try final answer
    m = re.search(r"final answer\s*:\s*(high|low)\s+satisfaction(?!\s+or)", text_lc)
    if m:
        return 1 if m.group(1) == "high" else 0

    return None

def evaluate_generation(args,accelerator, model, dataset, dataloader, tokenizer):
    model.eval()
    predictions = []
    targets = []
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=not accelerator.is_main_process,
                           desc='Evaluation Gen Loop'):
        output_ = accelerator.unwrap_model(model).generate(
            # **batch['generate_prefix_kwargs'],
            # max_length=args['max_gen_length'],
            input_ids=batch['generate_prefix_kwargs']['input_ids'],
            attention_mask=batch['generate_prefix_kwargs']['attention_mask'],
            max_new_tokens=args.get('max_new_tokens', 200),
            output_scores=True,
            return_dict_in_generate=True,
            # num_beams=1,
            use_cache=True,
            temperature=0.7,
            # top_k=0.0, 
            top_p=0.85,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bad_words_ids=[[tokenizer.pad_token_id]], 
             
            # pad_token_id=tokenizer.eos_token_id,
        )
        generated_ids = output_.sequences
        generated_ids = pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True)

        labels = batch['generate_prefix_kwargs']['labels']
        labels = pad_across_processes(labels, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True)
        labels[labels == -100] = tokenizer.pad_token_id

        generated_ids, labels = accelerator.gather(generated_ids), accelerator.gather(labels)

        preds = [tokenizer.decode(g.cpu().numpy().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in generated_ids]
        
        predictions.extend(preds)
        target = [tokenizer.decode(t.cpu().numpy().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for t in
                  labels]
        targets.extend(target)
        
    print("size of dataset", len(dataset))
    print("size of predictions", len(predictions))
    
    predictions = predictions[:len(dataset)]
    targets = targets[:len(dataset)]
    
    if accelerator.is_main_process and accelerator.is_local_main_process:
        results = []
        corr_value = 0
        
        # Initialize counters for F1 calculation
        true_labels = []
        pred_labels = []

        for pred_txt, tar_txt, item in zip(predictions, targets, dataset):
            target_score = int(item["true_labels"])          # 已经是 0 / 1
            target_cot   = tar_txt.strip().split("Step 1")[-1].strip()
            
            # pred_trunc   = truncate_after_final_answer(pred_txt)
            # pred_trunc= truncate_after_final_answer(pred_txt.strip().split("Final answer: High satisfaction or Low satisfaction\nStep")[-1].strip())
            # pred_score   = extract_score_from_reasoning(pred_trunc)  # 0 / 1 / None
            
            # truncated_pred_txt = pred_txt.strip().split("Final answer: Low satisfaction or High satisfaction")[1].strip() 
            # print(f"pred_txt: {truncated_pred_txt}")
            pred_trunc = truncate_after_final_answer(pred_txt.strip().split("Final answer: Low satisfaction or High satisfaction")[1].strip()) if "Final answer: Low satisfaction or High satisfaction" in pred_txt else None
            accelerator.print(f"pred_trunc: {pred_trunc}")
            if pred_trunc is None:
                accelerator.print(f"pred_txt: {pred_txt}")
            pred_score = extract_score_from_reasoning(pred_trunc) if pred_trunc is not None else None

            is_correct   = (int(pred_score) == int(target_score)) if pred_score is not None else False
            corr_value += int(is_correct)

            accelerator.print(f"Item ID: {item['item_id']}, Target: {target_score}, Predict Score: {pred_score}")
            
            # Collect labels for F1 calculation (only if prediction is valid)
            if pred_score is not None:
                true_labels.append(target_score)
                pred_labels.append(int(pred_score))

            results.append(
                {
                    "item_id":          item["item_id"],
                    "target":           tar_txt,
                    "target_cot":       target_cot,
                    "target_score":     target_score,
                    "prediction":       pred_txt,
                    "prediction_trunc": pred_trunc,
                    "prediction_score": pred_score,
                    "is_correct":       is_correct,
                }
            )

        res_path = os.path.join(args["model_dir"].rstrip("/"), "_res.json")
        with open(res_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Calculate macro average F1 score using sklearn
        if len(true_labels) > 0:
            # Calculate macro F1 score
            macro_f1 = f1_score(true_labels, pred_labels, average='macro') * 100  # Convert to percentage
            
            # Optional: Print detailed classification report
            accelerator.print("[Eval Info] Classification Report:")
            accelerator.print(classification_report(true_labels, pred_labels, target_names=['Class 0', 'Class 1']))
            accelerator.print(f"[Eval Info] Macro F1 Score: {macro_f1:.5g}%")
        else:
            macro_f1 = 0.0
            accelerator.print(f"[Eval Info] No valid predictions found, Macro F1 Score: {macro_f1:.5g}%")
            
        accuracy = corr_value / len(true_labels) * 100
        accelerator.print(f"[Eval Info] Accuracy: {accuracy:.5g}%")
        
        value_accuracy = torch.FloatTensor([macro_f1]).to(accelerator.device)
    else:
        value_accuracy = torch.FloatTensor([-1.0]).to(accelerator.device)
    value_accuracy = broadcast(value_accuracy).cpu().numpy().tolist()[0]

    # Metric summary:
    model.train()
    return {'value_accuracy': value_accuracy}