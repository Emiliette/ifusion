import torch
import torch.nn as nn


class FlexibleFusionNet(nn.Module):
    """
    A lightweight flexible-input fusion model.

    Input:
      x: (N, K, 1, H, W) where K is 2..4 (selected modalities for this batch)
    Output:
      fused: (N, 1, H, W) in [0,1]
    """

    def __init__(self, feat_channels: int = 32):
        super().__init__()
        c = int(feat_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.attn = nn.Sequential(
            nn.Conv2d(c, c // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, 1, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected (N,K,1,H,W), got {tuple(x.shape)}")
        n, k, ch, h, w = x.shape
        if ch != 1:
            raise ValueError("Expected single-channel modalities.")
        if k < 2 or k > 4:
            raise ValueError("Expected K in [2,4].")

        x_flat = x.view(n * k, 1, h, w)
        feat = self.encoder(x_flat)  # (N*K,C,H,W)
        c = feat.shape[1]
        feat = feat.view(n, k, c, h, w)  # (N,K,C,H,W)

        attn_logits = self.attn(feat.view(n * k, c, h, w)).view(n, k, 1, h, w)
        attn_w = torch.softmax(attn_logits, dim=1)
        agg = (attn_w * feat).sum(dim=1)  # (N,C,H,W)
        return self.decoder(agg)

