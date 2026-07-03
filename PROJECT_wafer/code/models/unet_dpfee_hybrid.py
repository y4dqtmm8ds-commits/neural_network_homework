# models/unet_dpfee_hybrid.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetStructureBranch(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, mask_channels=4):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)
        self.mask_head = nn.Conv2d(c1, mask_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.up3(e4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.mask_head(d1)


def mask_regularization_loss(masks, min_activation=0.05, entropy_weight=1.0):
    activation = masks.mean(dim=(0, 2, 3))
    non_empty = F.relu(float(min_activation) - activation).mean()
    flat = masks.flatten(2).clamp_min(1e-6)
    prob = flat / flat.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    entropy = -(prob * prob.log()).sum(dim=-1)
    entropy = entropy / torch.log(torch.tensor(float(flat.shape[-1]), device=masks.device))
    return non_empty + float(entropy_weight) * entropy.mean()


def mask_structure_features(mask_prob):
    area = mask_prob.mean(dim=(2, 3))
    defect = mask_prob[:, 0].clamp_min(1e-6)
    edge_ratio = (mask_prob[:, 1] * defect).sum(dim=(1, 2)) / defect.sum(dim=(1, 2)).clamp_min(1e-6)
    scratch_ratio = (mask_prob[:, 2] * defect).sum(dim=(1, 2)) / defect.sum(dim=(1, 2)).clamp_min(1e-6)
    center_ring_ratio = (mask_prob[:, 3] * defect).sum(dim=(1, 2)) / defect.sum(dim=(1, 2)).clamp_min(1e-6)
    return torch.cat([area, edge_ratio[:, None], scratch_ratio[:, None], center_ring_ratio[:, None]], dim=1)


class UNetDPFEEHybrid(nn.Module):
    def __init__(
        self,
        dpfee_backbone,
        num_classes,
        dropout=0.25,
        in_channels=3,
        unet_base_channels=32,
        unet_entropy_weight=1.0,
        mask_channels=4,
        struct_hidden=16,
    ):
        super().__init__()
        self.dpfee_backbone = dpfee_backbone
        self.unet_entropy_weight = float(unet_entropy_weight)
        self.structure_branch = UNetStructureBranch(
            in_channels=in_channels,
            base_channels=unet_base_channels,
            mask_channels=mask_channels,
        )
        self.struct_mlp = nn.Sequential(
            nn.Linear(mask_channels + 3, struct_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
        )
        feature_dim = int(dpfee_backbone.feature_dim) + struct_hidden
        self.main_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, num_classes))
        self.aux_head = nn.Sequential(
            nn.Dropout(min(dropout + 0.1, 0.6)),
            nn.Linear(feature_dim, num_classes),
        )

    def freeze_dpfee_backbone(self):
        for param in self.dpfee_backbone.parameters():
            param.requires_grad_(False)

    def freeze_unet_branch(self):
        for param in self.structure_branch.parameters():
            param.requires_grad_(False)
        for param in self.struct_mlp.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        dpfee_feature = self.dpfee_backbone.forward_features(x)
        mask_logits = self.structure_branch(x)
        masks = torch.sigmoid(mask_logits)
        struct_feature = self.struct_mlp(mask_structure_features(masks))
        feature = torch.cat([dpfee_feature, struct_feature], dim=1)
        aux = {
            "masks": masks,
            "mask_logits": mask_logits,
            "mask_loss": mask_regularization_loss(masks, entropy_weight=self.unet_entropy_weight),
            "mask_activation_mean": masks.mean(),
        }
        return self.main_head(feature), self.aux_head(feature), aux
