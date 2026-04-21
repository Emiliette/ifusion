import torch
import torch.nn as nn


class FlexibleFusionNet(nn.Module):
    """
    Attention-only fusion model for 2D grayscale BraTS slices.

    Input:
      x: (N, K, 1, H, W)
      mod_ids: optional modality ids in {0,1,2,3}. If omitted, uses [0, ..., K-1].
    Output:
      fused: (N, 1, H, W) in [0,1]
    """

    def __init__(
        self,
        feat_channels: int = 32,
        *,
        fuse_mode: str = "attn",
        use_modality_emb: bool = True,
        num_modalities: int = 4,
    ) -> None:
        super().__init__()
        if str(fuse_mode).lower().strip() != "attn":
            raise ValueError("FlexibleFusionNet only supports attention fusion.")
        if not bool(use_modality_emb):
            raise ValueError("FlexibleFusionNet requires modality embeddings.")

        channels = int(feat_channels)
        self.num_modalities = int(num_modalities)
        if self.num_modalities <= 0:
            raise ValueError("num_modalities must be positive.")

        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mod_emb = nn.Embedding(self.num_modalities, channels)
        self.mod_gamma = nn.Linear(channels, channels)
        self.mod_beta = nn.Linear(channels, channels)
        self.attn = nn.Sequential(
            nn.Conv2d(channels, max(1, channels // 2), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // 2), 1, kernel_size=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def _resolve_mod_ids(self, *, mod_ids: torch.Tensor | None, n: int, k: int, device: torch.device) -> torch.Tensor:
        if mod_ids is None:
            mod_ids_t = torch.arange(k, device=device, dtype=torch.long).view(1, k).expand(n, k)
        elif mod_ids.dim() == 1:
            mod_ids_t = mod_ids.view(1, k).expand(n, k).to(device=device, dtype=torch.long)
        elif mod_ids.dim() == 2:
            mod_ids_t = mod_ids.to(device=device, dtype=torch.long)
        else:
            raise ValueError("mod_ids must have shape (K,) or (N,K).")

        if mod_ids_t.shape != (n, k):
            raise ValueError(f"mod_ids must broadcast to (N,K)=({n},{k}), got {tuple(mod_ids_t.shape)}")
        if (mod_ids_t.min() < 0) or (mod_ids_t.max() >= self.num_modalities):
            raise ValueError(f"mod_ids values must be in [0,{self.num_modalities - 1}]")
        return mod_ids_t

    def forward(self, x: torch.Tensor, mod_ids: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected (N,K,1,H,W), got {tuple(x.shape)}")
        n, k, ch, h, w = x.shape
        if ch != 1:
            raise ValueError("Expected single-channel modalities.")
        if k < 1 or k > self.num_modalities:
            raise ValueError(f"Expected K in [1,{self.num_modalities}].")

        x_flat = x.view(n * k, 1, h, w)
        feat = self.encoder(x_flat).view(n, k, -1, h, w)
        channels = feat.shape[2]

        mod_ids_t = self._resolve_mod_ids(mod_ids=mod_ids, n=n, k=k, device=x.device)
        emb = self.mod_emb(mod_ids_t)
        gamma = self.mod_gamma(emb).view(n, k, channels, 1, 1)
        beta = self.mod_beta(emb).view(n, k, channels, 1, 1)
        feat = feat * (1.0 + gamma) + beta

        attn_logits = self.attn(feat.view(n * k, channels, h, w)).view(n, k, 1, h, w)
        attn_w = torch.softmax(attn_logits, dim=1)
        fused_feat = (attn_w * feat).sum(dim=1)
        return self.decoder(fused_feat)

