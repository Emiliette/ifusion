from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(ch: int) -> nn.Module:
    # GroupNorm is stable for small batch sizes.
    g = 8
    while ch % g != 0 and g > 1:
        g //= 2
    return nn.GroupNorm(num_groups=max(1, g), num_channels=ch)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetBBox(nn.Module):
    """
    U-Net style encoder/decoder, with a bbox head.

    Input:  x (N,C,H,W)
    Output:
      obj_logit: (N,) raw logit
      bbox_xyxy: (N,4) normalized xyxy with x1/y1 exclusive, clamped to [0,1]
    """

    def __init__(self, *, in_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.down1 = _ConvBlock(in_channels, c1)
        self.down2 = _ConvBlock(c1, c2)
        self.down3 = _ConvBlock(c2, c3)
        self.down4 = _ConvBlock(c3, c4)

        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = _ConvBlock(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = _ConvBlock(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = _ConvBlock(c1 + c1, c1)

        # Detection heads from the last decoder features.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, c1),
            nn.SiLU(inplace=True),
        )
        self.obj = nn.Linear(c1, 1)
        self.box = nn.Linear(c1, 4)

    def forward(self, x: torch.Tensor):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))

        y = self.up3(x4)
        if y.shape[-2:] != x3.shape[-2:]:
            x3m = _center_crop_like(x3, y)
        else:
            x3m = x3
        y = self.dec3(torch.cat([y, x3m], dim=1))
        y = self.up2(y)
        if y.shape[-2:] != x2.shape[-2:]:
            x2m = _center_crop_like(x2, y)
        else:
            x2m = x2
        y = self.dec2(torch.cat([y, x2m], dim=1))
        y = self.up1(y)
        if y.shape[-2:] != x1.shape[-2:]:
            x1m = _center_crop_like(x1, y)
        else:
            x1m = x1
        y = self.dec1(torch.cat([y, x1m], dim=1))

        f = self.head(y)
        obj_logit = self.obj(f).squeeze(1)  # (N,)

        b = torch.sigmoid(self.box(f))  # (N,4) in [0,1]
        # enforce xyxy ordering
        x0 = torch.minimum(b[:, 0], b[:, 2])
        y0 = torch.minimum(b[:, 1], b[:, 3])
        x1 = torch.maximum(b[:, 0], b[:, 2])
        y1 = torch.maximum(b[:, 1], b[:, 3])
        bbox = torch.stack([x0, y0, x1, y1], dim=1)
        bbox = torch.clamp(bbox, 0.0, 1.0)
        return obj_logit, bbox


def _center_crop_like(skip: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Center-crop skip features to match ref spatial size.
    """
    _, _, hs, ws = skip.shape
    _, _, hr, wr = ref.shape
    if hs == hr and ws == wr:
        return skip
    if hs < hr or ws < wr:
        # If upsample is larger (can happen with odd sizes), pad skip.
        pad_h = max(0, hr - hs)
        pad_w = max(0, wr - ws)
        # pad: (left,right,top,bottom)
        skip = F.pad(skip, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        _, _, hs, ws = skip.shape
    top = (hs - hr) // 2
    left = (ws - wr) // 2
    return skip[:, :, top : top + hr, left : left + wr]


def bbox_loss(
    *,
    obj_logit: torch.Tensor,
    bbox_pred: torch.Tensor,
    has_tumor: torch.Tensor,
    bbox_true: torch.Tensor,
    w_obj: float = 1.0,
    w_box: float = 5.0,
) -> torch.Tensor:
    """
    has_tumor: (N,) float 0/1
    bbox_true/bbox_pred: (N,4) normalized xyxy
    """
    has_tumor = has_tumor.float()
    obj = F.binary_cross_entropy_with_logits(obj_logit, has_tumor)

    pos = has_tumor > 0.5
    if pos.any():
        box = F.smooth_l1_loss(bbox_pred[pos], bbox_true[pos])
    else:
        box = obj.detach() * 0.0
    return w_obj * obj + w_box * box
