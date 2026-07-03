import torch
import torch.nn as nn
import torch.nn.functional as F


class CapsuleHardClassHead(nn.Module):
    def __init__(
        self,
        in_channels,
        num_hard_classes,
        primary_caps=8,
        capsule_dim=8,
        routing_iters=3,
    ):
        super().__init__()
        self.num_hard_classes = int(num_hard_classes)
        self.capsule_dim = int(capsule_dim)
        self.routing_iters = int(routing_iters)
        self.primary = nn.Conv2d(
            in_channels,
            primary_caps * capsule_dim,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.proj = nn.Linear(capsule_dim, self.num_hard_classes * capsule_dim)

    @staticmethod
    def squash(x, dim=-1, eps=1e-8):
        squared_norm = (x * x).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1.0 + squared_norm)
        return scale * x / torch.sqrt(squared_norm + eps)

    def forward(self, feature_map):
        b = feature_map.shape[0]
        caps = self.primary(feature_map)
        caps = caps.view(b, -1, self.capsule_dim, caps.shape[-2], caps.shape[-1])
        caps = caps.permute(0, 3, 4, 1, 2).reshape(b, -1, self.capsule_dim)
        caps = self.squash(caps)
        votes = self.proj(caps).view(b, caps.shape[1], self.num_hard_classes, self.capsule_dim)
        logits = votes.new_zeros(b, caps.shape[1], self.num_hard_classes)
        for _ in range(max(1, self.routing_iters)):
            weights = F.softmax(logits, dim=2)
            outputs = self.squash((weights.unsqueeze(-1) * votes).sum(dim=1), dim=-1)
            agreement = (votes * outputs.unsqueeze(1)).sum(dim=-1)
            logits = logits + agreement
        return torch.linalg.norm(outputs, dim=-1)


def capsule_margin_loss(capsule_logits, target, hard_class_ids, m_pos=0.9, m_neg=0.1, lambda_neg=0.5):
    hard_class_ids = list(hard_class_ids)
    if capsule_logits.numel() == 0 or not hard_class_ids:
        return capsule_logits.new_tensor(0.0)
    local_target = torch.full_like(target, -1)
    for local_idx, class_idx in enumerate(hard_class_ids):
        local_target = torch.where(target == int(class_idx), torch.full_like(local_target, local_idx), local_target)
    mask = local_target >= 0
    if not mask.any():
        return capsule_logits.new_tensor(0.0)
    labels = F.one_hot(local_target[mask], num_classes=len(hard_class_ids)).float()
    logits = capsule_logits[mask]
    positive = labels * F.relu(m_pos - logits).pow(2)
    negative = (1.0 - labels) * F.relu(logits - m_neg).pow(2)
    return (positive + lambda_neg * negative).sum(dim=1).mean()
