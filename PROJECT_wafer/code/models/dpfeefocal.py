# models/dpfeefocal.py
# -*- coding: utf-8 -*-

import torch.nn as nn


class GeoInputProjection(nn.Module):
    """Project geometric-prior inputs back to the 3-channel DPFEE input space."""

    def __init__(self, in_chans):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, 3, kernel_size=1, bias=False) if int(in_chans) > 3 else nn.Identity()

    def forward(self, x):
        return self.proj(x)
