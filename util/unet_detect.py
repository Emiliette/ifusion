from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetDetector2D(nn.Module):
    """
    U-Net style encoder + classification head for slice-level detection.

    Input:  (N,1,H,W)
    Output: logits (N,) for BCEWithLogitsLoss
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        b = int(base_channels)
        self.inc = _DoubleConv(int(in_channels), b)
        self.down1 = _DoubleConv(b, b * 2)
        self.down2 = _DoubleConv(b * 2, b * 4)
        self.down3 = _DoubleConv(b * 4, b * 8)
        self.bottleneck = _DoubleConv(b * 8, b * 16)

        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.head = nn.Linear(b * 16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.inc(x)
        x1 = self.down1(F.max_pool2d(x0, kernel_size=2))
        x2 = self.down2(F.max_pool2d(x1, kernel_size=2))
        x3 = self.down3(F.max_pool2d(x2, kernel_size=2))
        xb = self.bottleneck(F.max_pool2d(x3, kernel_size=2))

        feat = torch.mean(xb, dim=(-2, -1))  # global average pool -> (N,C)
        feat = self.dropout(feat)
        logits = self.head(feat).squeeze(1)  # (N,)
        return logits

