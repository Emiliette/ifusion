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


class UNet2D(nn.Module):
    """
    Lightweight 2D U-Net for segmentation.

    - Works with variable H/W by interpolating upsampled features to skip sizes.
    - Output are logits with shape (N, num_classes, H, W).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        b = int(base_channels)
        self.inc = _DoubleConv(int(in_channels), b)
        self.down1 = _DoubleConv(b, b * 2)
        self.down2 = _DoubleConv(b * 2, b * 4)
        self.down3 = _DoubleConv(b * 4, b * 8)
        self.bottleneck = _DoubleConv(b * 8, b * 16)

        self.up3 = _DoubleConv(b * 16 + b * 8, b * 8)
        self.up2 = _DoubleConv(b * 8 + b * 4, b * 4)
        self.up1 = _DoubleConv(b * 4 + b * 2, b * 2)
        self.up0 = _DoubleConv(b * 2 + b, b)

        self.outc = nn.Conv2d(b, int(num_classes), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.inc(x)
        x1 = self.down1(F.max_pool2d(x0, kernel_size=2))
        x2 = self.down2(F.max_pool2d(x1, kernel_size=2))
        x3 = self.down3(F.max_pool2d(x2, kernel_size=2))
        xb = self.bottleneck(F.max_pool2d(x3, kernel_size=2))

        u3 = F.interpolate(xb, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        u3 = self.up3(torch.cat([u3, x3], dim=1))

        u2 = F.interpolate(u3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, x2], dim=1))

        u1 = F.interpolate(u2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, x1], dim=1))

        u0 = F.interpolate(u1, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        u0 = self.up0(torch.cat([u0, x0], dim=1))
        return self.outc(u0)

