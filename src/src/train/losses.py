import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    """
    Batch-hard triplet loss (Hermans et al. 2017).
    Assumes `embeddings` are L2-normalized; uses squared Euclidean distance.
    For each anchor: hardest positive (max same-class distance) and hardest
    negative (min different-class distance).
    """
    pairwise = torch.cdist(embeddings, embeddings, p=2).pow(2)
    labels = labels.view(-1, 1)
    mask_pos = (labels == labels.t()).float()
    mask_pos.fill_diagonal_(0)
    mask_neg = (labels != labels.t()).float()

    pos_dist = pairwise * mask_pos
    hardest_pos = pos_dist.max(dim=1).values

    neg_dist = pairwise + (1 - mask_neg) * 1e9
    hardest_neg = neg_dist.min(dim=1).values

    has_pos = mask_pos.sum(1) > 0
    loss = F.relu(hardest_pos - hardest_neg + margin)
    loss = loss[has_pos]
    if loss.numel() == 0:
        return torch.zeros((), device=embeddings.device, requires_grad=True)
    return loss.mean()


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al. 2020).
    Expects L2-normalised embeddings.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        N = features.shape[0]

        sim = torch.mm(features, features.T) / self.temperature

        labels_col = labels.view(-1, 1)
        pos_mask = (labels_col == labels_col.T).float()
        pos_mask.fill_diagonal_(0.0)

        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        eye = torch.eye(N, device=device, dtype=torch.bool)
        exp_sim = torch.exp(sim).masked_fill(eye, 0.0)

        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        per_sample = -(pos_mask * log_prob).sum(dim=1)
        per_sample[valid] = per_sample[valid] / pos_count[valid]

        return per_sample[valid].mean() if valid.any() else per_sample.mean()
