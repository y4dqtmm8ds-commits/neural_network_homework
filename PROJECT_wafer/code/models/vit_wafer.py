# models/vit_wafer.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class TransformerTokenHead(nn.Module):
    def __init__(self, embed_dim, num_classes, dropout=0.1, dual_head=True):
        super().__init__()
        self.dual_head = bool(dual_head)
        self.norm = nn.LayerNorm(embed_dim)
        self.main_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))
        self.aux_head = nn.Sequential(
            nn.Dropout(min(dropout + 0.1, 0.5)),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, tokens):
        pooled = self.norm(tokens).mean(dim=1)
        if self.dual_head:
            return self.main_head(pooled), self.aux_head(pooled)
        return self.main_head(pooled)


def make_encoder(embed_dim=192, depth=4, heads=4, mlp_ratio=4.0, dropout=0.1):
    layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=heads,
        dim_feedforward=int(embed_dim * mlp_ratio),
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)


class ViTTinyWafer(nn.Module):
    def __init__(
        self,
        num_classes=8,
        in_channels=3,
        image_size=64,
        patch_size=8,
        embed_dim=192,
        depth=6,
        heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
        dual_head=True,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        num_tokens = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.encoder = make_encoder(embed_dim, depth, heads, mlp_ratio, dropout)
        self.head = TransformerTokenHead(embed_dim, num_classes, dropout, dual_head)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1], :]
        tokens = self.encoder(tokens)
        return self.head(tokens)


class CNNStemViTWafer(nn.Module):
    def __init__(
        self,
        num_classes=8,
        in_channels=3,
        embed_dim=192,
        depth=4,
        heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
        dual_head=True,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, 16 * 16, embed_dim))
        self.encoder = make_encoder(embed_dim, depth, heads, mlp_ratio, dropout)
        self.head = TransformerTokenHead(embed_dim, num_classes, dropout, dual_head)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        tokens = self.stem(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1], :]
        tokens = self.encoder(tokens)
        return self.head(tokens)


# Backward-compatible alias used by earlier smoke tests.
ViTWafer = CNNStemViTWafer
