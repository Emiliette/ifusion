from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import torch


def _grad_mag(x: torch.Tensor) -> torch.Tensor:
    """
    x: (N, 1, H, W)
    returns: (N, 1, H, W) gradient magnitude (L1-ish)
    """
    dx = torch.zeros_like(x)
    dy = torch.zeros_like(x)
    dx[..., :, :-1] = (x[..., :, 1:] - x[..., :, :-1]).abs()
    dy[..., :-1, :] = (x[..., 1:, :] - x[..., :-1, :]).abs()
    return dx + dy


def _to_01(x: torch.Tensor) -> torch.Tensor:
    # expects (N,1,H,W)
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + 1e-12)


def _entropy_u8(x01: torch.Tensor, bins: int = 256) -> torch.Tensor:
    """
    Approx Shannon entropy in bits, per batch element.
    x01: (N,1,H,W) in [0,1]
    returns: (N,)
    """
    x = torch.clamp((x01 * (bins - 1)).round(), 0, bins - 1).to(torch.int64)
    n = x.shape[0]
    ent = []
    for i in range(n):
        hist = torch.bincount(x[i].view(-1), minlength=bins).float()
        p = hist / (hist.sum() + 1e-12)
        ent.append(-(p * (p + 1e-12).log2()).sum())
    return torch.stack(ent, dim=0)


def _mutual_information_u8(x01: torch.Tensor, y01: torch.Tensor, bins: int = 64) -> torch.Tensor:
    """
    Approx mutual information in bits, per batch element (histogram-based).
    x01,y01: (N,1,H,W) in [0,1]
    returns: (N,)
    """
    x = torch.clamp((x01 * (bins - 1)).round(), 0, bins - 1).to(torch.int64)
    y = torch.clamp((y01 * (bins - 1)).round(), 0, bins - 1).to(torch.int64)
    n = x.shape[0]
    mi = []
    for i in range(n):
        xi = x[i].view(-1)
        yi = y[i].view(-1)
        joint = torch.bincount(xi * bins + yi, minlength=bins * bins).float().view(bins, bins)
        pxy = joint / (joint.sum() + 1e-12)
        px = pxy.sum(dim=1, keepdim=True)
        py = pxy.sum(dim=0, keepdim=True)
        denom = px @ py + 1e-12
        mi.append((pxy * (pxy / denom + 1e-12).log2()).sum())
    return torch.stack(mi, dim=0)


def _psnr_like(x01: torch.Tensor, y01: torch.Tensor) -> torch.Tensor:
    """
    PSNR (dB) assuming inputs are in [0,1]. Returns (N,).
    """
    mse = torch.mean((x01 - y01) ** 2, dim=(-3, -2, -1))
    return 10.0 * torch.log10(1.0 / (mse + 1e-12))


def _ssim_like(x01: torch.Tensor, y01: torch.Tensor) -> torch.Tensor:
    """
    Simplified global SSIM (no window), returns (N,).
    """
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = x01.mean(dim=(-3, -2, -1))
    mu_y = y01.mean(dim=(-3, -2, -1))
    var_x = ((x01 - mu_x.view(-1, 1, 1, 1)) ** 2).mean(dim=(-3, -2, -1))
    var_y = ((y01 - mu_y.view(-1, 1, 1, 1)) ** 2).mean(dim=(-3, -2, -1))
    cov = ((x01 - mu_x.view(-1, 1, 1, 1)) * (y01 - mu_y.view(-1, 1, 1, 1))).mean(dim=(-3, -2, -1))
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return num / (den + 1e-12)


def flex_fuse_onestep(
    f_pre: torch.Tensor,
    modalities: List[torch.Tensor],
    state: Dict,
    *,
    lamb: float = 0.5,
    rho: float = 0.001,
    objective: str = "edge",
    metric_weights: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Flexible fusion step for 2..4 modalities.

    This replaces the original EM_onestep(I, V, img_3, ...) which was hard-coded
    to <=3 inputs. Here we treat each modality equally and compute a per-pixel
    edge-aware weighted combination across all provided modalities, then blend
    with the diffusion prediction `f_pre`.

    f_pre: (N, 1, H, W) predicted fused luminance (from diffusion model)
    modalities: list of (N, 1, H, W) modality images
    lamb: blend weight for modalities vs f_pre (higher -> trust modalities more)
    rho: temperature for softmax over gradient magnitudes (smaller -> sharper selection)
    """
    if len(modalities) < 2:
        raise ValueError("flex_fuse_onestep requires at least 2 modalities.")

    device = f_pre.device
    mods = [m.to(device) for m in modalities]
    for m in mods:
        if m.shape != f_pre.shape:
            raise ValueError(f"All modalities must match f_pre shape. Got {m.shape} vs {f_pre.shape}")

    objective = objective.lower()
    if objective not in {"edge", "metrics"}:
        raise ValueError(f"Unknown objective: {objective}")

    stack = torch.stack(mods, dim=1)  # (N, K, 1, H, W)
    grads = torch.stack([_grad_mag(m) for m in mods], dim=1)  # (N, K, 1, H, W)

    temp = float(max(rho, 1e-6))
    edge_weights = torch.softmax(grads / temp, dim=1)  # (N,K,1,H,W)

    if objective == "edge":
        fused_from_modalities = (edge_weights * stack).sum(dim=1)
        a = float(lamb)
        return a * fused_from_modalities + (1.0 - a) * f_pre, state

    # objective == "metrics"
    # Global modality scores using requested metrics; local detail is still guided by edge_weights.
    w = {
        "EN": 1.0,
        "MI": 1.0,
        "PSNR": 1.0,
        "SSIM": 1.0,
        "SD": 1.0,
        "AG": 1.0,
    }
    if metric_weights:
        for k, v in metric_weights.items():
            if k in w:
                w[k] = float(v)

    f01 = _to_01(f_pre)
    scores = []
    for m in mods:
        m01 = _to_01(m)
        en = _entropy_u8(m01)
        mi = _mutual_information_u8(f01, m01)
        psnr = _psnr_like(f01, m01)
        ssim = _ssim_like(f01, m01)
        sd = m01.std(dim=(-3, -2, -1))
        ag = _grad_mag(m01).mean(dim=(-3, -2, -1))
        score = w["EN"] * en + w["MI"] * mi + w["PSNR"] * psnr + w["SSIM"] * ssim + w["SD"] * sd + w["AG"] * ag
        scores.append(score)

    score_t = torch.stack(scores, dim=1)  # (N,K)
    score_w = torch.softmax(score_t / max(temp, 1e-3), dim=1)  # (N,K)
    score_w = score_w.view(score_w.shape[0], score_w.shape[1], 1, 1, 1)

    weights = edge_weights * score_w
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)
    fused_from_modalities = (weights * stack).sum(dim=1)

    a = float(lamb)
    fused = a * fused_from_modalities + (1.0 - a) * f_pre
    return fused, state
