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
        self.linear_layer = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.linear_layer(x)
