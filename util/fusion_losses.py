from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def grad_mag(x: torch.Tensor) -> torch.Tensor:
    # x: (N,1,H,W)
    dx = torch.zeros_like(x)
    dy = torch.zeros_like(x)
    dx[..., :, :-1] = (x[..., :, 1:] - x[..., :, :-1]).abs()
    dy[..., :-1, :] = (x[..., 1:, :] - x[..., :-1, :]).abs()
    return dx + dy


def soft_assign(x: torch.Tensor, bins: int, sigma: float) -> torch.Tensor:
    """
    x: (N, P) in [0,1]
    returns: (N, bins, P) soft assignment weights that sum to 1 over bins (per pixel)
    """
    centers = torch.linspace(0.0, 1.0, steps=bins, device=x.device, dtype=x.dtype).view(1, bins, 1)
    x = x.unsqueeze(1)  # (N,1,P)
    w = torch.exp(-0.5 * ((x - centers) / max(sigma, 1e-6)) ** 2)
    w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
    return w


def entropy_soft(x01: torch.Tensor, *, bins: int = 64, sigma: float = 0.02) -> torch.Tensor:
    """
    Differentiable entropy approximation in bits (per batch item).
    x01: (N,1,H,W) in [0,1]
    returns: (N,)
    """
    n = x01.shape[0]
    x = x01.view(n, -1)
    a = soft_assign(x, bins=bins, sigma=sigma)  # (N,bins,P)
    p = a.mean(dim=2)  # (N,bins)
    return -(p * (p + 1e-12).log2()).sum(dim=1)


def mutual_information_soft(
    x01: torch.Tensor,
    y01: torch.Tensor,
    *,
    bins: int = 32,
    sigma: float = 0.04,
) -> torch.Tensor:
    """
    Differentiable MI approximation in bits (per batch item).
    x01,y01: (N,1,H,W) in [0,1]
    returns: (N,)
    """
    n = x01.shape[0]
    x = x01.view(n, -1)
    y = y01.view(n, -1)
    ax = soft_assign(x, bins=bins, sigma=sigma)  # (N,bins,P)
    ay = soft_assign(y, bins=bins, sigma=sigma)  # (N,bins,P)
    joint = torch.bmm(ax, ay.transpose(1, 2)) / ax.shape[2]  # (N,bins,bins)
    px = joint.sum(dim=2, keepdim=True)  # (N,bins,1)
    py = joint.sum(dim=1, keepdim=True)  # (N,1,bins)
    denom = px * py + 1e-12
    mi = (joint * (joint / denom + 1e-12).log2()).sum(dim=(1, 2))
    return mi


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10(1.0 / (mse + 1e-12))


def ssim_torch(x01: torch.Tensor, y01: torch.Tensor, *, win: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """
    Differentiable SSIM (per batch item), with Gaussian window.
    x01,y01: (N,1,H,W) in [0,1]
    returns: (N,)
    """
    if x01.shape != y01.shape:
        raise ValueError("SSIM expects same shapes.")
    n, c, h, w = x01.shape
    if c != 1:
        raise ValueError("SSIM expects single channel.")

    coords = torch.arange(win, device=x01.device, dtype=x01.dtype) - (win - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_1d = g.view(1, 1, win)
    kernel_2d = (kernel_1d.transpose(1, 2) @ kernel_1d).view(1, 1, win, win)

    mu_x = F.conv2d(x01, kernel_2d, padding=win // 2)
    mu_y = F.conv2d(y01, kernel_2d, padding=win // 2)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x01 * x01, kernel_2d, padding=win // 2) - mu_x2
    sigma_y2 = F.conv2d(y01 * y01, kernel_2d, padding=win // 2) - mu_y2
    sigma_xy = F.conv2d(x01 * y01, kernel_2d, padding=win // 2) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12)
    return ssim_map.mean(dim=(1, 2, 3))


@dataclass
class FusionLossWeights:
    w_en: float = 1.0
    w_mi: float = 1.0
    w_psnr: float = 1.0
    w_ssim: float = 1.0
    w_sd: float = 1.0
    w_ag: float = 1.0


def fusion_loss_unsupervised(
    fused01: torch.Tensor,
    mods01: torch.Tensor,
    *,
    weights: FusionLossWeights,
    mi_bins: int = 32,
    mi_sigma: float = 0.04,
    en_bins: int = 64,
    en_sigma: float = 0.02,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    fused01: (N,1,H,W) in [0,1]
    mods01: (N,K,1,H,W) in [0,1]
    returns: (loss scalar, logs dict)
    """
    n, k, _, _, _ = mods01.shape
    fused = torch.clamp(fused01, 0.0, 1.0)

    en = entropy_soft(fused, bins=en_bins, sigma=en_sigma).mean()
    sd = fused.std(dim=(1, 2, 3)).mean()
    ag = grad_mag(fused).mean()

    mi_list = []
    ssim_list = []
    mse_list = []
    for i in range(k):
        m = torch.clamp(mods01[:, i], 0.0, 1.0)
        mi_list.append(mutual_information_soft(fused, m, bins=mi_bins, sigma=mi_sigma))
        ssim_list.append(ssim_torch(fused, m))
        mse_list.append(((fused - m) ** 2).mean(dim=(1, 2, 3)))

    mi = torch.stack(mi_list, dim=1).mean()
    ssim = torch.stack(ssim_list, dim=1).mean()
    mse = torch.stack(mse_list, dim=1).mean()
    psnr = psnr_from_mse(mse)

    # Minimize loss: encourage high EN/MI/SSIM/PSNR/SD/AG.
    loss = (
        -weights.w_en * en
        -weights.w_mi * mi
        + weights.w_ssim * (1.0 - ssim)
        + weights.w_psnr * mse
        -weights.w_sd * sd
        -weights.w_ag * ag
    )

    logs = {
        "loss": float(loss.detach().cpu()),
        "en": float(en.detach().cpu()),
        "mi": float(mi.detach().cpu()),
        "ssim": float(ssim.detach().cpu()),
        "mse": float(mse.detach().cpu()),
        "psnr": float(psnr.detach().cpu()),
        "sd": float(sd.detach().cpu()),
        "ag": float(ag.detach().cpu()),
        "k": float(k),
    }
    return loss, logs

