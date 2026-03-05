
import torch
import torch.nn.functional as F
import math

def kd_loss_logits(student_logits, teacher_logits, T=2.0, alpha=0.7):
    # Shape: (seq_len, vocab_size)
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)
    teacher_probs = F.softmax(teacher_logits / T, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return alpha * (T ** 2) * kl

def kd_loss_mixed(student_logits, teacher_logits, teacher_targets, T=2.0, alpha=0.7):
    ce_loss = F.cross_entropy(student_logits, teacher_targets)
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)
    teacher_probs = F.softmax(teacher_logits / T, dim=-1)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return alpha * (T ** 2) * kl_loss + (1 - alpha) * ce_loss

def syn_ant_loss_no_norm(target, syns=None, ants=None, margin=2.0, pos_w=1.0, neg_w=1.0):
    t = target.float()
    loss = t.sum() * 0.0

    if syns is not None and syns.numel() > 0:
        d2p = (syns.float() - t).pow(2).mean(-1)   # mean over d (4096)
        loss = loss + pos_w * d2p.mean()

    if ants is not None and ants.numel() > 0:
        d2n = (ants.float() - t).pow(2).mean(-1)   # mean over d
        loss = loss + neg_w * torch.relu(margin - d2n).mean()

    return loss

