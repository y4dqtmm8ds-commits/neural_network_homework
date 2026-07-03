# models/base.py
# -*- coding: utf-8 -*-

import torch.nn as nn


class BaseModel(nn.Module):
    """Minimal compatibility base class for project models."""

    def __init__(self):
        super().__init__()
