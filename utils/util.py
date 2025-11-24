import torch    
import torch.nn as nn
import torch.nn.functional as F
from trl_custom import SFTTrainer
from transformers import Trainer

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(gamma=2, alpha=class_weights.to(self.args.device))

    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        mask = labels != -100
        loss = self.focal_loss(logits[mask], labels[mask])
        return (loss, outputs) if return_outputs else loss


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided to WeightedLossTrainer")
        self._tokenizer = tokenizer  # 避免使用弃用接口 Trainer.tokenizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # 计算 token-level loss（不平均）
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # === 提升 "high", "low", "satisfaction" token 的 loss 权重 ===
        id_high = self._tokenizer.convert_tokens_to_ids("high")
        id_low = self._tokenizer.convert_tokens_to_ids("low")
        # id_satisfaction = self._tokenizer.convert_tokens_to_ids("satisfaction")

        shift_labels_flat = shift_labels.view(-1)
        is_high_or_low = (shift_labels_flat == id_high) | (shift_labels_flat == id_low)
        # is_satisfaction = (shift_labels_flat == id_satisfaction)

        weight = torch.ones_like(loss)
        weight[is_high_or_low] = 3.0
        # weight[is_satisfaction] = 2.0

        weighted_loss = (loss * weight).mean()
        return (weighted_loss, outputs) if return_outputs else weighted_loss

import random
import numpy as np
import torch
import signal
import json

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def timeout_handler(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

def is_numeric(value):
    try:
        value = float(value)
        return True
    except Exception as e:
        return False

def floatify(s):
    try:
        return float(s)
    except:
        return None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def write_data(file: str, data) -> None:
    with open(file, "w", encoding="utf-8") as write_file:
        json.dump(data, write_file, ensure_ascii=False, indent=4)


from torch.distributed import all_reduce, ReduceOp
def do_gather(var):
    var = torch.FloatTensor(var).cuda()
    all_reduce(var, op=ReduceOp.SUM)
    var = var.cpu().numpy().tolist()
    return var

def allgather(tensor, group=None):
    """smantic sugar for torch.distributed.all_gather.

    Args:
        tensor: (bs, ...)
        group:

    Returns:
        All gathered tensor (world_size, bs, ...)
    """
    if group is None:
        group = torch.distributed.group.WORLD
    allgather_tensor = [torch.zeros_like(tensor) for _ in range(group.size())]
    torch.distributed.all_gather(allgather_tensor, tensor, group=group)
    allgather_tensor = torch.stack(allgather_tensor, dim=0)
    return allgather_tensor

from trl.core import masked_mean, masked_var
def allgather_masked_whiten(values, mask, shift_mean=False):
    """Whiten values with all-gathered masked values.

    Args:
        values: (bs, ...)
        mask: (bs, ...)
        shift_mean: bool

    Returns:
        whitened values, (bs, ...)
    """
    allgather_values = allgather(values)  # (n_proc, bs, ...)
    # accelerator.print(f'allgather_values {allgather_values.shape}, {allgather_values[0, 0:3]}')

    allgather_mask = allgather(mask)  # (n_proc, bs, ...)
    # accelerator.print(f'allgather_mask {allgather_mask.shape}, {allgather_mask[0, 0:3]}')

    global_mean = masked_mean(allgather_values, allgather_mask)
    global_var = masked_var(allgather_values, allgather_mask)
    whitened = (values - global_mean) * torch.rsqrt(global_var + 1e-8)
    if shift_mean:
        whitened += global_mean
    return whitened


import scipy.signal as scipy_signal
def discount_cumsum(rewards, discount):
    return scipy_signal.lfilter([1], [1, -discount], x=rewards[::-1])[::-1]

from datetime import timedelta
def compute_ETA(tqdm_t, num_period=1):
    # elapsed = tqdm_t.format_dict["elapsed"]
    rate = tqdm_t.format_dict["rate"]
    time_per_period = tqdm_t.total / rate if rate and tqdm_t.total else 0  # Seconds*
    period_remaining = (tqdm_t.total - tqdm_t.n) / rate if rate and tqdm_t.total else 0  # Seconds*
    remaining = time_per_period*(num_period-1) + period_remaining
    return timedelta(seconds=remaining)

