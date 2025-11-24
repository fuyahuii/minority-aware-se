from tqdm import tqdm
import torch
import re
import os
import random
import numpy as np
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def predict_voting_batch(tokenized_test, model, tokenizer, args, X_test_df, y_true, log_path, num_samples=10):
    batch_size = getattr(args, "test_batch_size", 4)
    input_ids = tokenized_test["input_ids"]      # shape: (N, seq_len)
    attention_mask = tokenized_test["attention_mask"]
    prompt_texts = tokenized_test.get("prompt_text", None)
    N = input_ids.shape[0]
    
    y_pred = []
    reasoning_outputs = []
    voting_stats = []

    set_seed(42)  
    model.eval()
    model.to(model.device)

    for batch_start in tqdm(range(0, N, batch_size), desc="Predicting batches"):
        batch_end = min(batch_start + batch_size, N)
        batch_input_ids = input_ids[batch_start:batch_end].to(model.device)       # shape: (B, seq_len)
        batch_attention_mask = attention_mask[batch_start:batch_end].to(model.device)
        current_batch_size = batch_input_ids.shape[0]
        print("tokenizer padding:",tokenizer.padding_side)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                num_return_sequences=num_samples,
                pad_token_id=tokenizer.eos_token_id,
            ) 
            # outputs shape:(B * num_samples, output_seq_len)


        for i in range(current_batch_size):
            sample_results = []
            sample_reasonings = []

            for j in range(num_samples):
                seq_idx = i * num_samples + j
                # if prompt_texts is not None:
                #     input_len = len(tokenizer(prompt_texts[batch_start + i], return_tensors="pt")["input_ids"][0]) # if padding is right
                # else:
                #     input_len = (batch_input_ids[i] != tokenizer.pad_token_id).sum().item()
                input_len=batch_input_ids[i].shape[0] # if padding is left
                generated_reasoning = tokenizer.decode(
                    outputs[seq_idx][input_len:], 
                    skip_special_tokens=True
                ).strip()
                
                generated = truncate_after_final_answer(generated_reasoning)
                score = extract_score_from_reasoning(generated)
            
                if score is not None:
                    sample_results.append(score)
                    sample_reasonings.append(generated_reasoning)
                    
            if sample_results:
                best_result, best_reasoning, vote_info = majority_vote(sample_results, sample_reasonings)
                y_pred.append(best_result)
                reasoning_outputs.append(best_reasoning)
                voting_stats.append(vote_info)
            else:
                y_pred.append(None)
                reasoning_outputs.append("No valid prediction generated")
                voting_stats.append({'votes': [], 'selected': None, 'confidence': 0.0})
        
        if batch_start==0:
            print("Generated reasoning",generated_reasoning)
        
    X_test_df = X_test_df.copy()
    X_test_df["generated_reasoning"] = reasoning_outputs
    X_test_df["predicted_label"] = y_pred
    X_test_df["true_label"] = y_true
    X_test_df.to_csv(os.path.join(log_path, "inference_with_reasoning.csv"), index=False, escapechar="\\")

    save_voting_analysis(voting_stats, log_path)
    return y_pred


def predict_voting_expanded(tokenized_test, model, tokenizer, args, X_test_df, y_true, log_path, num_samples=10):
    input_ids = tokenized_test["input_ids"]
    attention_mask = tokenized_test["attention_mask"]
    
    batch_size = getattr(args, "test_batch_size", 4)
    prompt_texts = tokenized_test.get("prompt_text", None)
    N = input_ids.shape[0]
    y_pred = []
    reasoning_outputs = []
    voting_stats = []

    set_seed(42)
    model.eval()
    model.to(model.device)
    
    for batch_start in tqdm(range(0, N, batch_size), desc="Predicting batches"):
        batch_end = min(batch_start + batch_size, N)
        batch_input_ids = input_ids[batch_start:batch_end].to(model.device)
        batch_attention_mask = attention_mask[batch_start:batch_end].to(model.device)
        current_batch_size = batch_input_ids.shape[0]

        batch_input_ids_expand = batch_input_ids.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, batch_input_ids.shape[1])
        batch_attention_mask_expand = batch_attention_mask.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, batch_attention_mask.shape[1])

        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch_input_ids_expand,
                attention_mask=batch_attention_mask_expand,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        for i in range(current_batch_size):
            sample_results = []
            sample_reasonings = []
            # num_samples for each sample in the batch
            for j in range(num_samples):
                seq_idx = i * num_samples + j
                if prompt_texts is not None:
                    input_len = len(tokenizer(prompt_texts[batch_start + i], return_tensors="pt")["input_ids"][0])
                else:
                    input_len = (batch_input_ids[i] != tokenizer.pad_token_id).sum().item()
                generated = tokenizer.decode(outputs[seq_idx][input_len:], skip_special_tokens=True).strip()
                generated = truncate_after_final_answer(generated)
                score = extract_score_from_reasoning(generated)
                if score is not None:
                    sample_results.append(score)
                    sample_reasonings.append(generated)

            if sample_results:
                best_result, best_reasoning, vote_info = majority_vote(sample_results, sample_reasonings)
                y_pred.append(best_result)
                reasoning_outputs.append(best_reasoning)
                voting_stats.append(vote_info)
            else:
                y_pred.append(None)
                reasoning_outputs.append("No valid prediction generated")
                voting_stats.append({'votes': [], 'selected': None, 'confidence': 0.0})

    X_test_df = X_test_df.copy()
    X_test_df["generated_reasoning"] = reasoning_outputs
    X_test_df["predicted_label"] = y_pred
    X_test_df["true_label"] = y_true
    X_test_df.to_csv(os.path.join(log_path, "inference_with_reasoning.csv"), index=False, escapechar="\\")
    save_voting_analysis(voting_stats, log_path)
    return y_pred


def majority_vote(scores, reasonings):
    from collections import Counter
    
    if not scores:
        return None, "No valid prediction", {'votes': [], 'selected': None, 'confidence': 0.0}
 
    vote_counts = Counter(scores)
    most_common = vote_counts.most_common()
    
    winner_score = most_common[0][0]
    winner_count = most_common[0][1]
    
    # confidence computed as the proportion of votes for the winning score
    confidence = winner_count / len(scores)
    
    # the reasoning for the winning score
    winner_reasoning = None
    for i, score in enumerate(scores):
        if score == winner_score:
            winner_reasoning = reasonings[i]
            break
    
    vote_info = {
        'votes': scores,
        'vote_counts': dict(vote_counts),
        'selected': winner_score,
        'confidence': confidence,
        'total_samples': len(scores)
    }
    
    return winner_score, winner_reasoning, vote_info


def save_voting_analysis(voting_stats, log_path):
    import json
    
    total_samples = len(voting_stats)
    unanimous_votes = sum(1 for stat in voting_stats if stat.get('confidence', 0) == 1.0)
    high_confidence_votes = sum(1 for stat in voting_stats if stat.get('confidence', 0) >= 0.6)
    
    avg_confidence = sum(stat.get('confidence', 0) for stat in voting_stats) / total_samples if total_samples > 0 else 0
    
    analysis = {
        'total_samples': total_samples,
        'unanimous_votes': unanimous_votes,
        'high_confidence_votes': high_confidence_votes,
        'average_confidence': avg_confidence,
        'unanimous_rate': unanimous_votes / total_samples if total_samples > 0 else 0,
        'high_confidence_rate': high_confidence_votes / total_samples if total_samples > 0 else 0,
        'detailed_stats': voting_stats
    }
    
    with open(os.path.join(log_path, "voting_analysis.json"), 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Voting Analysis Results ===")
    print(f"Total samples: {total_samples}")
    print(f"Unanimous votes: {unanimous_votes} ({unanimous_votes/total_samples:.1%})")
    print(f"High confidence votes: {high_confidence_votes} ({high_confidence_votes/total_samples:.1%})")
    print(f"Average confidence: {avg_confidence:.3f}")
    
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
