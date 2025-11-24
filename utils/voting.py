from tqdm import tqdm
import torch
import re

def predict_voting(test_data, model, tokenizer, args, X_test_df, y_true, log_path, num_samples=10):
    y_pred = []
    reasoning_outputs = []
    voting_stats = []  

    model.eval()
    model.to(model.device)

    for i, sample in enumerate(tqdm(test_data, desc="Predicting")):
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
                num_return_sequences=num_samples,  
                pad_token_id=tokenizer.eos_token_id,
            )
        
        input_len = len(inputs["input_ids"][0])
        sample_results = []
        sample_reasonings = []
        
        for seq_idx in range(num_samples):
            generated = tokenizer.decode(output_ids[seq_idx][input_len:], skip_special_tokens=True).strip()
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
            
            if len(set(sample_results)) > 1: 
                print(f"Sample {i}: votes={sample_results}, selected={best_result} (confidence: {vote_info['confidence']:.2f})")
        else:
            # if all generated results are invalid, append None
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
