# ----------------------------------------------------------------------------------
#  * Two clusters (majority / minority) learnt via EM (default 10 iterations)
#  * After each EM epoch, **validation users are first routed to the lower-perplexity
#    model**, then each cluster model is evaluated **only on它负责的子集**.
#  * Metrics: sklearn `classification_report`, MSE, ROUGE-1/2/L, BLEU-1/2, BERTScore-F1
#  * Saving rule:
#        Cluster-0 → keep checkpoint whose High-class F1 improves
#        Cluster-1 → keep checkpoint whose Low-class  F1 improves
# ----------------------------------------------------------------------------------
from __future__ import annotations
import os, gc, re, math, shutil, functools, multiprocessing as mp, datetime, logging, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "7"                 
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import torch, bitsandbytes as bnb, numpy as np, pandas as pd
from datasets import Dataset
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, classification_report
from rouge_score import rouge_scorer
from transformers import (
    HfArgumentParser, set_seed,
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,BitsAndBytesConfig
)
from peft import LoraConfig, TaskType, get_peft_model,PeftModel, PeftConfig, prepare_model_for_kbit_training
import bitsandbytes as bnb
from CoPeR import generate_coper_prompt_training
from base_template import generate_prompt_for_training, generate_prompt_reasoning_for_training
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # BLEU-1
try:
    from bert_score import score as bert_score_fn                       # BERTScore
    _HAS_BERTSCORE = True
except ImportError:
    _HAS_BERTSCORE = False
    
from datasets import Dataset, concatenate_datasets 
import json, os, numpy as np, torch
from pathlib import Path


# ========== 0. Logging helper ======================================================

def init_logger(out_dir: str, custom_name: Optional[str] = None):
    os.makedirs(out_dir, exist_ok=True)
    if custom_name:
        log_path = os.path.join(out_dir, custom_name if custom_name.endswith(".log") else f"{custom_name}.log")
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(out_dir, f"train_{ts}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.__stdout__),
        ],
    )

    class _Redirect:
        def __init__(self, level): self.level = level
        def write(self, buf):
            for line in buf.rstrip().splitlines():
                logging.log(self.level, line.rstrip())
        def flush(self): ...

    sys.stdout, sys.stderr = _Redirect(logging.INFO), _Redirect(logging.INFO)
    logging.info(f"Logger initialised at {log_path}")
    return logging.getLogger("main")

# ========== 1. CLI arguments =======================================================

@dataclass
class ScriptArguments:
    # --- model paths ---
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    # sft_model_path: str = "../sft/output_coper_8bit/model/Meta-Llama-3-8B-Instruct_finetune_True_steps_lr_0.0001_r_16_alpha_16_dropout_0.1_eos" #coper not llama template
    # sft_model_path: str ="../sft/output_coper_8bit/model/Meta-Llama-3-8B-Instruct_finetune_True_steps_lr_0.0001_r_16_alpha_16_dropout_0.1_eos_llamatemplate" #coper with llama template
    # sft_model_path: str="../sft/output_base_8bit/model/Meta-Llama-3-8B-Instruct_finetune_True_epoch_lr_0.0001_reasoning_False_r_128_alpha_16_dropout_0" #base
    sft_model_path: str="../sft/output_base_8bit/model/Meta-Llama-3-8B-Instruct_finetune_True_epoch_lr_0.0001_reasoning_True_r_16_alpha_16_dropout_0.1_#" #cot
    
    output_dir: str = "./updates/checkpoints_ppl_mauser50_mi20_trainsetall_ep1_cot_retrainlora_sampling0.5/reference_models/"
 
    # --- data paths ---
    majority_train_path: str = "../../../data_split/training_majority.csv"
    minority_train_path: str = "../../../data_split/training_minority.csv"
    majority_valid_path: str = "../../../data_split/valid_majority.csv"
    minority_valid_path: str = "../../../data_split/valid_minority.csv"

    majority_users: int = 50
    minority_users: int = 20
    train_subset: int = 6100   #1100
    eval_subset: int = 800   #800
    eval_generation_samples: int = 200        

    # --- training hyperparams ---
    per_device_train_batch_size: int = 4 #2
    gradient_accumulation_steps: int = 2 #4
    learning_rate: float = 1e-5
    weight_decay: float = 1e-3
    num_train_epochs: int = 1
    warmup_steps: int = 100
    log_freq: int = 10
    lr_scheduler_type: str = "cosine"

    bf16: bool = False
    gradient_checkpointing: bool = True
    use_8bit: bool = True

    # --- LoRA ---
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0

    # --- infra ---
    deepspeed: Optional[str] = None
    local_rank: int = -1

    seed: int = 1103
    max_seq_length: int = 1044 #1280/1044
    label_max_len: int= 20 #256/20
    em_iterations: int = 10

    use_llama_template: bool = False
    log_filename: Optional[str] = None
    template: str = "cot"  # "coper" or "base" or "cot"
    low_sampling: bool = True  # whether to do low sampling for minority cluster
    low_sampling_ratio: float = 0.5

# print script arguments
def print_script_args(args: ScriptArguments):
    print("\n=== Script Arguments ===")
    for field in args.__dataclass_fields__:
        value = getattr(args, field)
        print(f"{field}: {value}")
    print("=========================\n")
    
print_script_args(ScriptArguments())
# ========== 2. Prompt helpers ======================================================

# score_re = re.compile(r"Final score:\s*(\d+)")
score_re= re.compile(r"Final score\s*:\s*([1-5])\s*(?:$|\n)")

def build_generation_prompt(a, ex: Dict, tok, use_chat: bool = False):
    if a. template == "coper":
        p = generate_coper_prompt_training(ex)
    elif a.template == "cot":
        p = generate_prompt_reasoning_for_training(ex)
    else:  # base
        p = generate_prompt_for_training(ex)
    if use_chat:
        return tok.apply_chat_template(
            [{"role": "system", "content": p["instruction"]},
             {"role": "user",   "content": p["input"]}],
            tokenize=False, add_generation_prompt=True)
    return p["instruction"] + "\n\n" + p["input"]

def preprocess_function_llama3(examples, args: ScriptArguments, tok):
    new = {"input_ids": [], "attention_mask": [], "labels": []}
    for i in range(len(examples["input"])):
        row = {k: examples[k][i] for k in examples}
        if args.template == "coper":
            p = generate_coper_prompt_training(row)
        elif args.template == "cot":
            p = generate_prompt_reasoning_for_training(row)
        else:  # base
            p = generate_prompt_for_training(row)
        chat = tok.apply_chat_template(
            [{"role": "system", "content": p["instruction"]},
             {"role": "user",   "content": p["input"]}],
            tokenize=False, add_generation_prompt=True)
        inp_ids = tok(chat, add_special_tokens=False, truncation=True,
                      max_length=args.max_seq_length - args.label_max_len).input_ids
        out_ids = tok(p["output"], add_special_tokens=False, truncation=True,
                      max_length=args.label_max_len - 1).input_ids + [tok.eos_token_id]
        out_ids = out_ids[: args.max_seq_length - len(inp_ids)]
        full = inp_ids + out_ids
        new["input_ids"].append(full)
        new["attention_mask"].append([1] * len(full))
        new["labels"].append([-100] * len(inp_ids) + out_ids)
    return new

def preprocess_function(examples, args: ScriptArguments, tok):
    new = {"input_ids": [], "attention_mask": [], "labels": []}
    for i in range(len(examples["input"])):
        row = {k: examples[k][i] for k in examples}
        if args.template == "coper":
            p = generate_coper_prompt_training(row)
        elif args.template == "cot":
            p = generate_prompt_reasoning_for_training(row)
        else:  # base
            # print("Using base template for training")
            p = generate_prompt_for_training(row)
        chat = p["instruction"] + "\n\n" + p["input"]
        inp_ids = tok(chat, add_special_tokens=True, truncation=True,
                      max_length=args.max_seq_length - args.label_max_len).input_ids
        out_ids = tok(p["output"], add_special_tokens=False, truncation=True,
                      max_length=args.label_max_len - 1).input_ids + [tok.eos_token_id]
        out_ids = out_ids[: args.max_seq_length - len(inp_ids)]
        full = inp_ids + out_ids
        new["input_ids"].append(full)
        new["attention_mask"].append([1] * len(full))
        new["labels"].append([-100] * len(inp_ids) + out_ids)
    return new

# ========== 3. Metric extraction ===================================================

def truncate_after_final_answer(text):
    """
    Truncate text after the first occurrence of 'Final answer: ... satisfaction'.
    Helps avoid repeated template outputs.
    """
    match = re.search(r"(Final answer\s*:\s*(?:High|Low)\s+satisfaction)", text, re.IGNORECASE)
    if match:
        return text[:match.end()]
    return text

def extract_score_from_text(text: str):
    text_lc = text.lower()

    m = re.search(r"final score\s*:\s*([1-5])\s*(?:$|\n)", text_lc)
    if m:
        score = int(m.group(1))
        return score
    return None  # If no score found, return None
    
def extract_category_from_text(text: str):
    text_lc = text.lower()
    m = re.search(r"final answer\s*:\s*(high|low)\s+satisfaction(?!\s+or)", text_lc)
    if m:
        return "High satisfaction" if m.group(1) == "high" else "Low satisfaction"
    
    return None  # If neither pattern matches, return None

# ========== 4. LoRA helpers ========================================================

def print_trainable_parameters(model, model_name="Model"):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params       = sum(p.numel() for p in model.parameters())
    print(f"\n=== {model_name} Parameters ===")
    print(f"Trainable params: {trainable_params:,} || All params: {all_params:,}")
    print(f"Trainable%: {100 * trainable_params / all_params:.2f}%")

def build_lora_cfg(a: ScriptArguments):
    return LoraConfig(task_type=TaskType.CAUSAL_LM, r=a.lora_r,
                      lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                      inference_mode=False,
                      target_modules=["q_proj", "v_proj", "k_proj",
                                      "o_proj", "gate_proj", "up_proj", "down_proj"])

# load base model and train LoRA adapter
def load_model(a: ScriptArguments, ckpt: Optional[str]):
    base = ckpt or a.sft_model_path or a.model_name
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, trust_remote_code=True)
    print_trainable_parameters(model, "Base/SFT Model")
    if not ckpt or not hasattr(model, "peft_config"):
        model = get_peft_model(model, build_lora_cfg(a))
        print_trainable_parameters(model, "Adding LoRA Model")
    else:
        print_trainable_parameters(model, "Loaded Checkpoint Model")
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = not a.gradient_checkpointing
    return model

# ========== 5. Custom Trainer ======================================================
class RefTrainer(Trainer):
    def __init__(self, *args, tokenizer=None, **kw):
        super().__init__(*args, **kw)
        self.tk = tokenizer
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self._smooth = SmoothingFunction().method1

    def evaluate_generation(self, a, model, dataset: Dataset, *,
                            max_new_tokens: int, max_length: int, use_chat: bool):
        model.eval(); tk = self.tk
        ys_true_cls, ys_pred_cls, g_s, t_s, rouges, gens, refs = [], [], [], [], [], [], []
        sample=0
        for ex in dataset:
            prompt = build_generation_prompt(a, ex, tk, use_chat)
            enc = tk(prompt, return_tensors="pt", truncation=True,
                     max_length=max_length).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                     temperature=0.7, do_sample=True,top_p=0.85,
                                     pad_token_id=tk.pad_token_id,min_new_tokens=10)
            resp = tk.decode(out[0][len(enc["input_ids"][0]):], skip_special_tokens=True)
            sample += 1
            # --- metrics ---
            # resp= truncate_after_final_answer(resp)
            # print(f"Sample {sample} | Response: {resp.strip()[:100]}...")
            sc_pred = extract_score_from_text(resp)
            cat_pred = extract_category_from_text(resp)
            if sc_pred is not None:
                g_s.append(sc_pred); t_s.append(float(ex["output"]))
            if cat_pred is not None:
                ys_pred_cls.append(cat_pred)
                ys_true_cls.append("High satisfaction" if ex["label"] == "B" else "Low satisfaction")
            # rouges.append(self.scorer.score(generate_coper_prompt_training(ex)["output"], resp))
            # gens.append(resp); refs.append(generate_coper_prompt_training(ex)["output"])
            groud=generate_coper_prompt_training(ex)["output"] if a.template=="coper" else generate_prompt_reasoning_for_training(ex)["output"] if a.template=="cot" else generate_prompt_for_training(ex)["output"]
            rouges.append(self.scorer.score(groud, resp))
            gens.append(resp); refs.append(groud)

        metrics: Dict[str, float] = {}
        if g_s: metrics["mse"] = mean_squared_error(t_s, g_s)
        if rouges:
            metrics.update({
                "rouge1": float(np.mean([r["rouge1"].fmeasure for r in rouges])),
                "rouge2": float(np.mean([r["rouge2"].fmeasure for r in rouges])),
                "rougeL": float(np.mean([r["rougeL"].fmeasure for r in rouges])),
            })
        if ys_pred_cls:
            rpt = classification_report(ys_true_cls, ys_pred_cls, output_dict=True, zero_division=0)
            metrics.update({
                "cls_report": rpt,
                "f1_high": rpt.get("High satisfaction", {}).get("f1-score", 0.0),
                "f1_low":  rpt.get("Low satisfaction",  {}).get("f1-score", 0.0),
                "macro_f1": rpt.get("macro avg", {}).get("f1-score", 0.0),
            })
        # BLEU
        if gens:
            b1 = [sentence_bleu([r.split()], g.split(), weights=(1, 0, 0, 0),
                                smoothing_function=self._smooth) for r, g in zip(refs, gens)]
            b2 = [sentence_bleu([r.split()], g.split(), weights=(0.5, 0.5, 0, 0),
                                smoothing_function=self._smooth) for r, g in zip(refs, gens)]
            metrics["bleu1"], metrics["bleu2"] = float(np.mean(b1)), float(np.mean(b2))
        # BERTScore
        if _HAS_BERTSCORE and gens:
            _, _, F = bert_score_fn(gens, refs, lang="en", verbose=False)
            metrics["bertscore_f1"] = float(F.mean())
        
        metrics.update({
            "y_true_cls": ys_true_cls,
            "y_pred_cls": ys_pred_cls,
        })
        
        for k in ("mse", "f1_high", "f1_low", "bleu1", "bleu2", "bertscore_f1"):
            metrics.setdefault(k, 0.0)
        return metrics

# ========== 6. Utility =============================================================

def subset_perplexity(model, subset: Dataset, collator):
    dl = torch.utils.data.DataLoader(subset, batch_size=1, collate_fn=collator)
    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            num_tok = (batch["labels"] != -100).sum().item()
            total_loss += out.loss.item() * num_tok
            total_tok += num_tok
    return math.exp(total_loss / max(total_tok, 1))

def _to_python(o):
    # let json.dumps handle these types
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, torch.Tensor):              return o.tolist()
    raise TypeError(f"type {type(o)} not serializable")

def save_metrics(data, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    def clean_data(obj):
        if isinstance(obj, dict):
            return {k: clean_data(v) for k, v in obj.items() 
                    if k not in ("y_true_cls", "y_pred_cls")}
        elif isinstance(obj, list):
            return [clean_data(item) for item in obj]
        else:
            return obj
    
    with open(file_path, "w") as f:
        json.dump(clean_data(data), f, indent=2, default=_to_python)
   
def print_metrics(metrics,name=None):
    logging.info(f"=== Metrics of {name} ===" if name else "=== Metrics ===")
    for k, v in metrics.items():
        if k == "y_true_cls" or k == "y_pred_cls":          
            continue
        if isinstance(v, dict):             
            logging.info(f"{k}:\\n" + json.dumps(v, indent=2, default=lambda x: float(f'{x:.4f}'))) # cls_report
        else:
            logging.info(f"{k}: {v:.4f}")   

# ========== 7. Main ================================================================

def main(a: ScriptArguments):
    _ = init_logger(a.output_dir, a.log_filename)
    os.environ["WANDB_DISABLED"] = "true"
    set_seed(a.seed)

    dev_idx = a.local_rank if a.local_rank >= 0 else 0
    torch.cuda.set_device(dev_idx)
    
    tok = AutoTokenizer.from_pretrained(a.model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = 2

    # ---------- load & subsample csv ------------------------------------------------
    def read_csv(p): return Dataset.from_pandas(pd.read_csv(p))
    train_raw = [read_csv(a.majority_train_path), read_csv(a.minority_train_path)]
    valid_raw = [read_csv(a.majority_valid_path), read_csv(a.minority_valid_path)]

    rng = np.random.default_rng(a.seed)
    for i, ds in enumerate(train_raw):
        train_raw[i] = ds.select(rng.choice(len(ds), size=min(a.train_subset, len(ds)), replace=False))
    for i, ds in enumerate(valid_raw):
        valid_raw[i] = ds.select(rng.choice(len(ds), size=min(a.eval_subset, len(ds)), replace=False))

    # ---------- tokenize -----------------------------------------------------------
    def make_tok(ds):
        fn = functools.partial(
            preprocess_function_llama3 if a.use_llama_template else preprocess_function,
            args=a, tok=tok)
        proc = ds.map(fn, batched=True, num_proc=min(mp.cpu_count(), 8),
                      remove_columns=ds.column_names)
        return proc.filter(lambda x: 0 < len(x["input_ids"]) <= a.max_seq_length)

    train_tok = [make_tok(ds) for ds in train_raw]
    valid_tok = [make_tok(ds) for ds in valid_raw]           # tokenised BUT keep raw separately

    # ---------- split helpers ------------------------------------------------------
    def split(ds, n):
        if n == 0: return []
        base = len(ds) // n; sizes = [base] * n
        for i in range(len(ds) - base * n): sizes[i] += 1
        return torch.utils.data.random_split(ds, sizes, torch.Generator().manual_seed(a.seed))
    
    # train/validation pairs (tok, raw) — ensure pairs are aligned
    def split_pair(tok_ds, raw_ds, n):
        tok_parts = split(tok_ds, n)
        raw_parts = [torch.utils.data.Subset(raw_ds, s.indices) for s in tok_parts]
        return list(zip(tok_parts, raw_parts))

    train_pairs = split_pair(train_tok[0], train_raw[0], a.majority_users) + \
                  split_pair(train_tok[1], train_raw[1], a.minority_users)
    
    train_users=[tok for tok, _ in train_pairs]  
    tok2raw     = {tok: raw for tok, raw in train_pairs} 
    
    valid_pairs = split_pair(valid_tok[0], valid_raw[0], a.majority_users) + \
                  split_pair(valid_tok[1], valid_raw[1], a.minority_users)
           
    collator = DataCollatorForSeq2Seq(tok, pad_to_multiple_of=8,
                                      return_tensors="pt", padding=True)

    base_targs = dict(
        learning_rate=a.learning_rate,
        per_device_train_batch_size=a.per_device_train_batch_size,
        num_train_epochs=a.num_train_epochs,
        weight_decay=a.weight_decay,
        gradient_accumulation_steps=a.gradient_accumulation_steps,
        warmup_steps=a.warmup_steps,
        lr_scheduler_type=a.lr_scheduler_type,
        bf16=False,
        fp16=True,
        logging_strategy="steps",
        logging_steps=a.log_freq,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=a.gradient_checkpointing,
        save_strategy="no",
        deepspeed=a.deepspeed,
        local_rank=a.local_rank,
        ddp_find_unused_parameters=False,
        label_names=["labels"],
    )

    best_high_f1 = best_low_f1 = 0.0
    prev_cluster_models: Optional[Dict[int, torch.nn.Module]] = None   # ==== FIX: cache models

    # ---------- baseline SFT evaluation BEFORE any EM training ----------
    logging.info("=== Baseline SFT evaluation on validation set ===")
    sft_model = load_model(a, None).to("cuda").eval()

    eval_args = TrainingArguments(
        output_dir=os.path.join(a.output_dir, "baseline_eval_tmp"),
        **base_targs)

    trainer = RefTrainer(model=sft_model, args=eval_args, tokenizer=tok)

    # 1) Majority-valid
    raw_major = valid_raw[0]                    
    metrics_major = trainer.evaluate_generation(
        a, sft_model, raw_major,
        max_new_tokens=256, max_length=a.max_seq_length,
        use_chat=a.use_llama_template)
    # logging.info("[Baseline] Majority valid metrics:\n" +
    #             json.dumps(metrics_major, indent=2, default=lambda x: float(f'{x:.4f}')))
    print_metrics(metrics_major, "[Baseline] Majority valid metrics")

    # 2) Minority-valid
    raw_minor = valid_raw[1]
    metrics_minor = trainer.evaluate_generation(
        a, sft_model, raw_minor,
        max_new_tokens=256, max_length=a.max_seq_length,
        use_chat=a.use_llama_template)
    print_metrics(metrics_minor, "[Baseline] Minority valid metrics")
    
    def merge_metrics(met0, met1, n0, n1):
        N = n0 + n1
        merged = {}

        for k in ("mse", "rouge1", "rouge2", "rougeL",
                "bleu1", "bleu2", "bertscore_f1"):
            merged[k] = (met0[k] * n0 + met1[k] * n1) / N if N else 0.0

        y_true = met0["y_true_cls"] + met1["y_true_cls"]
        y_pred = met0["y_pred_cls"] + met1["y_pred_cls"]
        rpt = classification_report(y_true, y_pred,
                                    output_dict=True, zero_division=0)
        merged.update({
            "cls_report": rpt,
            "f1_high": rpt.get("High satisfaction", {}).get("f1-score", 0.0),
            "f1_low":  rpt.get("Low satisfaction", {}).get("f1-score", 0.0),
            "y_true_cls": y_true,
            "y_pred_cls": y_pred,
        })
        return merged

    metrics_all = merge_metrics(metrics_major, metrics_minor,
                            len(raw_major), len(raw_minor))
    print_metrics(metrics_all, "[Baseline] All valid metrics (merged)")

    metrics_dir = os.path.join(a.output_dir, "metrics")
    save_metrics({
        "metrics_major": metrics_major,
        "metrics_minor": metrics_minor,
        "metrics_all" : metrics_all,       
    }, os.path.join(metrics_dir, "baseline.json"))

    sft_model.cpu(); del trainer
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.ipc_collect()
    
    # ================= EM loop =====================================================

    for it in range(a.em_iterations):
        print(f"\n=== EM Iteration {it+1}/{a.em_iterations} ===")

        # ===== E-step ==============================================================
        if prev_cluster_models is None:                       # 1st iter: naive split
            print("No previous cluster models; assigning users to clusters naively")
            assign = [0] * a.majority_users + [1] * a.minority_users
        else:
            print("Previous cluster models found; routing train users to clusters")
            perp = [[0.0] * len(train_users) for _ in range(2)]
            for cid in (0, 1):
                m = prev_cluster_models[cid].to("cuda").eval()
                with torch.no_grad():
                    for uid, subset_tok in enumerate(train_users):
                        perp[cid][uid] = subset_perplexity(m, subset_tok, collator)
                m.cpu(); torch.cuda.empty_cache()
            assign = [0 if perp[0][u] <= perp[1][u] else 1 for u in range(len(train_users))]
            logging.info(f"[E-step] train users → cluster0={assign.count(0)} | cluster1={assign.count(1)}")

        train_groups = {0: [], 1: []}
        for uid, cid in enumerate(assign):
            train_groups[cid].append(train_users[uid])
        

        # ===== M-step ==============================================================
        cluster_models: Dict[int, torch.nn.Module] = {}
        for cid in (0, 1):
            if not train_groups[cid]: continue
            print(f"Training cluster {cid} with {len(train_groups[cid])} user subsets")

            if cid == 1 and a.low_sampling:
                merged_raw = Dataset.from_list([
                            tok2raw[sub].dataset[idx]         
                            for sub in train_groups[cid]
                            for idx in sub.indices])
                low  = merged_raw.filter(lambda x: x["label"] == "A")   # Low satisfaction
                high = merged_raw.filter(lambda x: x["label"] == "B")   # High satisfaction

                sample_size = int(len(low) * a.low_sampling_ratio)
                sampled_low = low.shuffle(seed=a.seed).select(range(sample_size))
                merged_raw = concatenate_datasets([low, sampled_low, high]).shuffle(seed=a.seed) 
            
                merged=make_tok(merged_raw)  # Re-tokenize the merged dataset
            else:
                merged = Dataset.from_list([
                ex
                for sub in train_groups[cid]
                for ex in (sub.dataset[idx] for idx in sub.indices)
            ])
            
            if prev_cluster_models is not None and cid in prev_cluster_models:
                model = prev_cluster_models[cid].to("cuda")      
                print(f"Loaded previous cluster {cid} weights for fine-tuning")
            else:
                model = load_model(a, None)      
            
            train_args = TrainingArguments(
                output_dir=os.path.join(a.output_dir, f"tmp_c{cid}_iter{it}"),
                **base_targs)
            trainer = RefTrainer(
                model=model, args=train_args, train_dataset=merged,
                tokenizer=tok, data_collator=collator)
            trainer.train()
            print_trainable_parameters(model, f"Cluster {cid} Model")
            cluster_models[cid] = model.to("cpu")  
            del trainer
            torch.cuda.empty_cache(); gc.collect(); torch.cuda.ipc_collect()

        if len(cluster_models) != 2:
            print("One cluster empty; skipping validation routing")
            # prev_cluster_models = cluster_models
            continue

        # ===== Routing validation users ============================================
        val_perp = [[0.0] * len(valid_pairs) for _ in range(2)]

        for cid in (0, 1):
            m = cluster_models[cid].to("cuda").eval()
            with torch.no_grad():
                for uid, (tok_sub, _) in enumerate(valid_pairs):
                    val_perp[cid][uid] = subset_perplexity(m, tok_sub, collator)
            m.cpu()

        val_assign = [0 if val_perp[0][u] <= val_perp[1][u] else 1 for u in range(len(valid_pairs))]
        logging.info(f"[routing-val] valid users → cluster0={val_assign.count(0)} | cluster1={val_assign.count(1)}")

        # ===== Evaluation ==========================================================
        eval_args = TrainingArguments(output_dir=os.path.join(a.output_dir, "eval_tmp"),
                                      **base_targs)
        metrics_cluster = {0: {}, 1: {}}
        cluster_sizes   = {0: 0, 1: 0} 
        for cid in (0, 1):
            raw_ds = Dataset.from_list([
                ex
                for uid, (tok_sub, raw_sub) in enumerate(valid_pairs)
                if val_assign[uid] == cid
                for ex in (raw_sub.dataset[idx] for idx in raw_sub.indices)
            ])
            cluster_sizes[cid] = len(raw_ds) 
            trainer = RefTrainer(model=cluster_models[cid], args=eval_args, tokenizer=tok)
            metrics = trainer.evaluate_generation(a,
                cluster_models[cid], raw_ds,
                max_new_tokens=256, max_length=a.max_seq_length,
                use_chat=a.use_llama_template)
            metrics_cluster[cid] = metrics
            logging.info(
                f"Cluster {cid} | size={len(raw_ds)} | "
                f"f1_high={metrics.get('f1_high', 0.0):.4f} | "
                f"f1_low={metrics.get('f1_low', 0.0):.4f}")
                    
        # ===== Saving best checkpoints =========================================================
        hi_id, lo_id = (0, 1) if metrics_cluster[0].get("f1_high", 0.0) >= \
                                 metrics_cluster[1].get("f1_high", 0.0) else (1, 0)

        cur_high_f1, cur_low_f1 = metrics_cluster[hi_id]["f1_high"], metrics_cluster[lo_id]["f1_low"]

        if cur_high_f1 > best_high_f1:
            best_high_dir = os.path.join(a.output_dir, "best_majority")
            shutil.rmtree(best_high_dir, ignore_errors=True)
            cluster_models[hi_id].save_pretrained(best_high_dir)
            tok.save_pretrained(best_high_dir)
            best_high_f1 = cur_high_f1
            logging.info(f"★ New best HIGH saved (cluster {hi_id}) | F1={best_high_f1:.4f}")

        if cur_low_f1 > best_low_f1:
            best_low_dir = os.path.join(a.output_dir, "best_minority")
            shutil.rmtree(best_low_dir, ignore_errors=True)
            cluster_models[lo_id].save_pretrained(best_low_dir)
            tok.save_pretrained(best_low_dir)
            best_low_f1 = cur_low_f1
            logging.info(f"★ New best LOW  saved (cluster {lo_id}) | F1={best_low_f1:.4f}")

        print_metrics(metrics_cluster[0], "Cluster 0 metrics")
        print_metrics(metrics_cluster[1], "Cluster 1 metrics")
        
        # ---------- overall (weighted-by-samples) ------------------------------
        total_val = cluster_sizes[0] + cluster_sizes[1]
        overall_metrics = {}

        def _weighted(k):
            return (metrics_cluster[0].get(k, 0.0) * cluster_sizes[0] +
                    metrics_cluster[1].get(k, 0.0) * cluster_sizes[1]) / total_val

        for key in ["precision", "recall", "mse",
                    "rouge1", "rouge2", "rougeL",
                    "bleu1", "bleu2", "bertscore_f1"]:
            if key in metrics_cluster[0] and isinstance(metrics_cluster[0][key], (int, float)):
                overall_metrics[key] = _weighted(key)

        y_true_all   = metrics_cluster[0]["y_true_cls"]   + metrics_cluster[1]["y_true_cls"]
        y_pred_all   = metrics_cluster[0]["y_pred_cls"]   + metrics_cluster[1]["y_pred_cls"]
        
        overall_cls = classification_report(
            y_true_all, y_pred_all,
            target_names=["High satisfaction", "Low satisfaction"],
            output_dict=True, zero_division=0
        )
        overall_metrics["cls_report"] = overall_cls
        
        logging.info(f"[EM_iter{it}] Overall weighted metrics:\n" +
            json.dumps(overall_metrics, indent=2,
                        default=lambda x: float(f'{x:.4f}')))
        
        iter_tag = f"iter{it}"         
        metrics_dir = os.path.join(a.output_dir, "metrics")     
        
        save_metrics({
            "cluster0": metrics_cluster[0],
            "cluster1": metrics_cluster[1],
            "overall" : overall_metrics,
            "sizes"   : cluster_sizes           
        }, os.path.join(metrics_dir, f"{iter_tag}.json"))

        # ===== Cleaning & Preparing for next iteration ====================================================
        del trainer; torch.cuda.empty_cache(); gc.collect()
        prev_cluster_models = {cid: m.cpu() for cid, m in cluster_models.items()}
        torch.cuda.empty_cache(); gc.collect();torch.cuda.ipc_collect()

    # ===== Final cluster mapping w.r.t. BEST checkpoints =====================
    # load best_majority / best_minority for routing
    best_major_path = os.path.join(a.output_dir, "best_majority")
    best_minor_path = os.path.join(a.output_dir, "best_minority")
    best_major = load_model(a, best_major_path).to("cuda").eval()
    best_minor = load_model(a, best_minor_path).to("cuda").eval()

    cluster_map_best: Dict[str, int] = {}

    def route_subset(tok_subset, raw_subset):
        ppl_maj = subset_perplexity(best_major, tok_subset, collator)
        ppl_min = subset_perplexity(best_minor, tok_subset, collator)
        cid = 0 if ppl_maj <= ppl_min else 1
        for idx in raw_subset.indices:
            dlg_id = raw_subset.dataset[idx]["dialogue_id"]
            cluster_map_best[dlg_id] = cid

    # —— train users
    for tok_sub, raw_sub in train_pairs:
        route_subset(tok_sub, raw_sub)
    # —— valid users
    for tok_sub, raw_sub in valid_pairs:
        route_subset(tok_sub, raw_sub)

    map_path = os.path.join(a.output_dir, "dialogue_cluster_map_best.json")
    with open(map_path, "w") as f:
        json.dump(cluster_map_best, f, indent=2)
    logging.info(f"Saved BEST cluster_map to {map_path} (N={len(cluster_map_best)})")

    # ------------------ done -------------------------------------------------------
    print("\n=== EM finished ===")
    logging.info(f"Training finished. best_high_f1={best_high_f1:.4f} | best_low_f1={best_low_f1:.4f}")

# ========== 8. Entry ===============================================================
if __name__ == "__main__":
    args: ScriptArguments = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]
    main(args)
