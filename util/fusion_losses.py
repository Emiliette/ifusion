from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def grad_mag(x: torch.Tensor) -> torch.Tensor:
    dx = torch.zeros_like(x)
    dy = torch.zeros_like(x)
    dx[..., :, :-1] = (x[..., :, 1:] - x[..., :, :-1]).abs()
    dy[..., :-1, :] = (x[..., 1:, :] - x[..., :-1, :]).abs()
    return dx + dy


def ssim_torch(x01: torch.Tensor, y01: torch.Tensor, *, win: int = 11, sigma: float = 1.5) -> torch.Tensor:
    if x01.shape != y01.shape:
        raise ValueError("SSIM expects tensors with the same shape.")
    n, c, _, _ = x01.shape
    if c != 1:
        raise ValueError("SSIM expects single-channel inputs.")

    coords = torch.arange(win, device=x01.device, dtype=x01.dtype) - (win - 1) / 2.0
    gauss_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel_1d = gauss_1d.view(1, 1, win)
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
    return ssim_map.mean(dim=(1, 2, 3)).view(n)


def _weighted_mean(values: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    if weights is None:
        return values.mean()
    if weights.dim() == 1:
        weights_t = weights.view(1, -1).to(values.device, values.dtype)
    else:
        weights_t = weights.to(values.device, values.dtype)
    weights_t = weights_t / (weights_t.sum(dim=1, keepdim=True) + 1e-12)
    return (values * weights_t).sum(dim=1).mean()


def fusion_loss_source_guided(
    fused01: torch.Tensor,
    mods01: torch.Tensor,
    *,
    w_mod: Optional[torch.Tensor] = None,
    lam_ssim: float = 0.5,
    lam_grad: float = 0.3,
    lam_l1: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Source-guided fusion loss without fused ground-truth.

    fused01: (N,1,H,W) in [0,1]
    mods01: (N,K,1,H,W) in [0,1]
    """
    fused = torch.clamp(fused01, 0.0, 1.0)
    _, k, _, _, _ = mods01.shape

    ssim_vals = []
    grad_vals = []
    l1_vals = []
    fused_grad = grad_mag(fused)
    for i in range(k):
        modality = torch.clamp(mods01[:, i], 0.0, 1.0)
        ssim_vals.append(ssim_torch(fused, modality))
        grad_vals.append((fused_grad - grad_mag(modality)).abs().mean(dim=(1, 2, 3)))
        l1_vals.append((fused - modality).abs().mean(dim=(1, 2, 3)))

    ssim_vals_t = torch.stack(ssim_vals, dim=1)
    grad_vals_t = torch.stack(grad_vals, dim=1)
    l1_vals_t = torch.stack(l1_vals, dim=1)

    loss_ssim = _weighted_mean(1.0 - ssim_vals_t, w_mod)
    loss_grad = _weighted_mean(grad_vals_t, w_mod)
    loss_l1 = _weighted_mean(l1_vals_t, w_mod)
    loss = float(lam_ssim) * loss_ssim + float(lam_grad) * loss_grad + float(lam_l1) * loss_l1

    logs = {
        "loss": float(loss.detach().cpu()),
        "loss_ssim": float(loss_ssim.detach().cpu()),
        "loss_grad": float(loss_grad.detach().cpu()),
        "loss_l1": float(loss_l1.detach().cpu()),
        "ssim": float(ssim_vals_t.mean().detach().cpu()),
    }
    return loss, logs

