# coding=utf-8
"""
Predictor model during dvrl training
"""

import torch
import torch.nn as nn


class Predictor(nn.Module):
    """
    Predictor.
    """
    def __init__(self, input_feature=2048):
        super().__init__()
        self.linear_layer = nn.Linear(input_feature, 2)

    def forward(self, x):
        return self.linear_layer(x)
