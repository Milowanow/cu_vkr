
import torch
import torch.nn as nn

class MRAELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # pred, target: (B, C, H, W)
        return torch.mean(torch.abs(pred - target) / (target + self.eps))

