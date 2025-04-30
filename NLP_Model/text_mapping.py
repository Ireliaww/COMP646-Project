# File: model/text_mapping.py

import torch
import torch.nn as nn

class StyleMapping(nn.Module):
    """
    Simple MLP that maps CLIP text features back to the same dimension for reconstruction.
    """
    def __init__(self, clip_dim=512, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, clip_dim)
        )

    def forward(self, x):
        return self.net(x)
