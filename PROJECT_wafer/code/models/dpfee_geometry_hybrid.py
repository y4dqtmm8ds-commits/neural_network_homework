import torch
import torch.nn as nn

from models.capsule_head import CapsuleHardClassHead


class DPFEEGeometryHybrid(nn.Module):
    def __init__(
        self,
        dpfee_backbone,
        num_classes,
        geo_feature_dim,
        geo_mlp_hidden=64,
        geo_dropout=0.1,
        dropout=0.25,
        use_scratchness_head=False,
        use_capsule_head=False,
        capsule_hard_class_count=4,
        capsule_dim=8,
        capsule_routing_iters=3,
    ):
        super().__init__()
        self.backbone = dpfee_backbone
        self.use_scratchness_head = bool(use_scratchness_head)
        self.use_capsule_head = bool(use_capsule_head)
        self.geo_mlp = nn.Sequential(
            nn.Linear(int(geo_feature_dim), int(geo_mlp_hidden)),
            nn.BatchNorm1d(int(geo_mlp_hidden)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(geo_dropout)),
        )
        fused_dim = int(self.backbone.feature_dim) + int(geo_mlp_hidden)
        self.main_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(fused_dim, num_classes))
        self.aux_head = nn.Sequential(
            nn.Dropout(min(dropout + 0.1, 0.6)),
            nn.Linear(fused_dim, num_classes),
        )
        self.scratchness_head = nn.Linear(fused_dim, 1) if self.use_scratchness_head else None
        if self.use_capsule_head:
            self.capsule_head = CapsuleHardClassHead(
                in_channels=int(self.backbone.feature_dim),
                num_hard_classes=int(capsule_hard_class_count),
                capsule_dim=int(capsule_dim),
                routing_iters=int(capsule_routing_iters),
            )
        else:
            self.capsule_head = None

    def forward(self, x, geo):
        feature_map = self.backbone.forward_feature_map(x)
        image_feature = self.backbone.flatten(self.backbone.pool(feature_map))
        geo_feature = self.geo_mlp(geo.float())
        fused = torch.cat([image_feature, geo_feature], dim=1)
        payload = {}
        if self.scratchness_head is not None:
            payload["scratchness_logits"] = self.scratchness_head(fused).squeeze(1)
        if self.capsule_head is not None:
            payload["capsule_logits"] = self.capsule_head(feature_map)
        if payload:
            return self.main_head(fused), self.aux_head(fused), payload
        return self.main_head(fused), self.aux_head(fused)
