from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


def _grad_mag(x: torch.Tensor) -> torch.Tensor:
    dx = torch.zeros_like(x)
    dy = torch.zeros_like(x)
    dx[..., :, :-1] = (x[..., :, 1:] - x[..., :, :-1]).abs()
    dy[..., :-1, :] = (x[..., 1:, :] - x[..., :-1, :]).abs()
    return dx + dy


@dataclass
class EMNModalState:
    step: int = 0


def em_fuse_onestep(
    f_pre: torch.Tensor,
    mods: torch.Tensor,
    state: Optional[EMNModalState] = None,
    *,
    eta: float = 0.2,
    rho: float = 0.001,
    alpha_grad: float = 1.0,
    beta_residual: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, EMNModalState]:
    """
    N-modal "EM-like" fusion refinement (1 iteration).

    Inputs:
      f_pre: (B,1,H,W) fused prior/prediction (e.g., from diffusion or FlexibleFusionNet)
      mods: (B,K,1,H,W) modality stack in [0,1]

    E-step (reliability):
      Compute per-pixel weights that depend on BOTH:
        - local detail (gradient magnitude): prefer sharper modalities
        - consistency w.r.t current fused estimate (residual): prefer modalities closer to f_pre

      score = alpha_grad * grad(mod) - beta_residual * |mod - f_pre|
      w = softmax(score / rho) over K.

    M-step (update):
      fused_from_mods = sum_k w_k * mod_k
      fused = eta * f_pre + (1-eta) * fused_from_mods

    Notes:
      - eta is "keep prior" strength: higher eta keeps more of f_pre.
      - This is a lightweight, differentiable refinement suitable for training loops.
    """
    if f_pre.dim() != 4 or f_pre.shape[1] != 1:
        raise ValueError(f"f_pre must have shape (N,1,H,W), got {tuple(f_pre.shape)}")
    if mods.dim() != 5 or mods.shape[2] != 1:
        raise ValueError(f"mods must have shape (N,K,1,H,W), got {tuple(mods.shape)}")
    if mods.shape[0] != f_pre.shape[0] or mods.shape[-2:] != f_pre.shape[-2:]:
        raise ValueError(f"mods spatial/batch must match f_pre. mods={tuple(mods.shape)} f_pre={tuple(f_pre.shape)}")

    n, k, _, h, w = mods.shape
    if k < 2:
        raise ValueError("em_fuse_onestep expects at least 2 modalities.")

    device = f_pre.device
    mods = mods.to(device)
    rho_t = float(max(float(rho), 1e-6))

    grads = torch.stack([_grad_mag(mods[:, i]) for i in range(k)], dim=1)  # (B,K,1,H,W)
    resid = (mods - f_pre.unsqueeze(1)).abs()  # (B,K,1,H,W)
    score = float(alpha_grad) * grads - float(beta_residual) * resid
    weights = torch.softmax(score / rho_t, dim=1)  # (B,K,1,H,W)

    fused_from_mods = (weights * mods).sum(dim=1)  # (N,1,H,W)

    eta_f = float(eta)
    if not (0.0 <= eta_f <= 1.0):
        raise ValueError("--em_eta/eta must be in [0,1].")
    fused = eta_f * f_pre + (1.0 - eta_f) * fused_from_mods

    st = state or EMNModalState()
    st.step = int(st.step) + 1
    return fused, weights, st


def em_fuse_refine(
    f_init: torch.Tensor,
    mods: torch.Tensor,
    *,
    steps: int = 1,
    eta: float = 0.2,
    rho: float = 0.001,
    alpha_grad: float = 1.0,
    beta_residual: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Run multiple EM refinement steps.

    Returns:
      fused: (N,1,H,W)
      aux: {"weights": (N,K,1,H,W), "last_fused_from_mods": (N,1,H,W)}
    """
    fused = f_init
    state = EMNModalState()
    weights = None
    last_fused_from_mods = None

    for _ in range(int(steps)):
        fused, weights, state = em_fuse_onestep(
            fused,
            mods,
            state,
            eta=float(eta),
            rho=float(rho),
            alpha_grad=float(alpha_grad),
            beta_residual=float(beta_residual),
        )
        # Derive last fused-from-mods for debugging/analysis (cheap).
        last_fused_from_mods = (weights * mods).sum(dim=1)

    assert weights is not None
    assert last_fused_from_mods is not None
    return fused, {"weights": weights, "last_fused_from_mods": last_fused_from_mods}
