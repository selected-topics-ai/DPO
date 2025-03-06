import torch
import torch.nn.functional as F

def mean_kl_divergence(logits_left: torch.Tensor, logits_right: torch.Tensor) -> torch.Tensor:

    if logits_left.shape != logits_right.shape:
        raise ValueError('logits_left and logits_right must have same shape')

    logits_left = logits_left.cpu().detach()
    logits_right = logits_right.cpu().detach()

    log_probs_left = F.log_softmax(logits_left, dim=-1)
    log_probs_right = F.log_softmax(logits_right, dim=-1)

    log_ratio = log_probs_left - log_probs_right
    left_prob = F.softmax(logits_left, dim=-1)

    kl = (left_prob * log_ratio).mean()

    return kl