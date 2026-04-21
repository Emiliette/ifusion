from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-12


def _gaussian_kernel1d(sigma: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if sigma <= 0:
        return torch.tensor([1.0], device=device, dtype=dtype)
    radius = int(3.0 * float(sigma) + 0.5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * float(sigma) * float(sigma)))
    k = k / (torch.sum(k) + 1e-12)
    return k


def _gaussian_blur2d(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur for (N,1,H,W) using separable conv + reflect padding."""
    if sigma <= 0:
        return img
    device, dtype = img.device, img.dtype
    k1 = _gaussian_kernel1d(sigma, device=device, dtype=dtype)
    kx = k1.view(1, 1, 1, -1)
    ky = k1.view(1, 1, -1, 1)
    pad = int((k1.numel() - 1) // 2)
    x = F.pad(img, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, kx)
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    x = F.conv2d(x, ky)
    return x


def _gaussian_filter_np(img: np.ndarray, sigma: float) -> np.ndarray:
    t = torch.from_numpy(np.asarray(img, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    out = _gaussian_blur2d(t, float(sigma))
    return out.squeeze(0).squeeze(0).cpu().numpy().astype(np.float64)


def _conv2d_reflect_np(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D conv 'same' with reflect padding (like filter2(...,'same') up to flips)."""
    k = torch.from_numpy(np.asarray(kernel, dtype=np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,kh,kw)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    pad_y = (k.shape[-2] - 1) // 2
    pad_x = (k.shape[-1] - 1) // 2
    x = F.pad(x, (pad_x, pad_x, pad_y, pad_y), mode="reflect")
    y = F.conv2d(x, k)
    return y.squeeze(0).squeeze(0).cpu().numpy().astype(np.float64)


def normalize1_u8(data: np.ndarray) -> np.ndarray:
    """Match normalize1.m from imageFusionMetrics (Z. Liu).

    Scales array to [0,255] (per-image min/max) and rounds to integer.
    """
    data = np.asarray(data, dtype=np.float64)
    da = float(np.max(data))
    mi = float(np.min(data))
    if da == 0.0 and mi == 0.0:
        out = data
    else:
        out = (data - mi) / (da - mi + EPS)
        out = np.round(out * 255.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def mutual_information_u8(x_u8: np.ndarray, y_u8: np.ndarray, bins: int = 256, normalize_log_base: int = 256) -> float:
    """Mutual information between two uint8 images.

    - bins=256 uses raw intensities.
    - Returns MI normalized by log2(normalize_log_base) (default 256 -> in [0,1] roughly).
    """
    x_u8 = np.asarray(x_u8, dtype=np.uint8)
    y_u8 = np.asarray(y_u8, dtype=np.uint8)
    if x_u8.shape != y_u8.shape:
        raise ValueError("x_u8 and y_u8 must have same shape")

    # Quantize to bins.
    if bins != 256:
        x = (x_u8.astype(np.int64) * bins) // 256
        y = (y_u8.astype(np.int64) * bins) // 256
    else:
        x = x_u8.astype(np.int64)
        y = y_u8.astype(np.int64)

    joint = np.bincount((x.reshape(-1) * bins + y.reshape(-1)), minlength=bins * bins).astype(np.float64).reshape(bins, bins)
    pxy = joint / (joint.sum() + EPS)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = px @ py + EPS
    nz = pxy > 0
    mi_bits = float((pxy[nz] * np.log2((pxy[nz] / denom[nz]) + EPS)).sum())
    return float(mi_bits / (np.log2(float(normalize_log_base)) + EPS))


def ncc_u8(x_u8: np.ndarray, y_u8: np.ndarray, b: int = 256) -> float:
    """Nonlinear correlation coefficient (NCC) from NCC.m (Z. Liu).

    NCC here equals (Hx + Hy - Hxy), where entropies are normalized by log2(b).
    """
    x_u8 = np.asarray(x_u8, dtype=np.uint8)
    y_u8 = np.asarray(y_u8, dtype=np.uint8)
    if x_u8.shape != y_u8.shape:
        raise ValueError("x_u8 and y_u8 must have same shape")

    x = x_u8.astype(np.int64)
    y = y_u8.astype(np.int64)
    joint = np.bincount((x.reshape(-1) * 256 + y.reshape(-1)), minlength=256 * 256).astype(np.float64).reshape(256, 256)
    h = joint / (joint.sum() + EPS)
    px = h.sum(axis=0)  # im1_marg in NCC.m (sum each column)
    py = h.sum(axis=1)  # im2_marg in NCC.m (sum each row)

    Hx = -float(np.sum(px * np.log2(px + (px == 0))))
    Hy = -float(np.sum(py * np.log2(py + (py == 0))))
    Hxy = -float(np.sum(h * np.log2(h + (h == 0))))
    denom = np.log2(float(b)) + EPS
    return float((Hx / denom) + (Hy / denom) - (Hxy / denom))


def q_ncie_wang(sources_u8: Sequence[np.ndarray], fused_u8: np.ndarray, b: int = 256) -> float:
    """Generalized Q_NCIE (Wang) for N source images + 1 fused.

    Original metricWang.m uses 2 sources + fused (K=3). Here we use K = len(sources)+1.
    """
    imgs = [np.asarray(s, dtype=np.uint8) for s in sources_u8] + [np.asarray(fused_u8, dtype=np.uint8)]
    if len(imgs) < 3:
        raise ValueError("Need at least 2 sources for Q_NCIE.")
    shape = imgs[0].shape
    if any(im.shape != shape for im in imgs):
        raise ValueError("All images must have same shape for Q_NCIE.")

    k = len(imgs)
    R = np.eye(k, dtype=np.float64)
    for i in range(k):
        for j in range(i + 1, k):
            v = ncc_u8(imgs[i], imgs[j], b=b)
            R[i, j] = v
            R[j, i] = v

    r = np.linalg.eigvalsh(R)
    r = np.clip(r, EPS, None)
    HR = float(np.sum(r * np.log2(r / float(k)) / float(k)))
    HR = -HR / (np.log2(float(b)) + EPS)
    return float(1.0 - HR)


def psnr01(x01: np.ndarray, y01: np.ndarray) -> float:
    x01 = np.asarray(x01, dtype=np.float64)
    y01 = np.asarray(y01, dtype=np.float64)
    mse = float(np.mean((x01 - y01) ** 2))
    return float(10.0 * np.log10(1.0 / (mse + EPS)))


def _ssim_map_and_vars(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sigma: float = 1.5,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian-window SSIM map + local variances (similar to ssim_index.m)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("SSIM inputs must have same shape.")

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu_x = _gaussian_filter_np(x, sigma=float(sigma))
    mu_y = _gaussian_filter_np(y, sigma=float(sigma))

    sigma_x_sq = _gaussian_filter_np(x * x, sigma=float(sigma)) - mu_x * mu_x
    sigma_y_sq = _gaussian_filter_np(y * y, sigma=float(sigma)) - mu_y * mu_y
    sigma_xy = _gaussian_filter_np(x * y, sigma=float(sigma)) - mu_x * mu_y

    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = num / (den + EPS)
    return float(np.mean(ssim_map)), ssim_map, sigma_x_sq, sigma_y_sq


def q_s_piella(sources01: Sequence[np.ndarray], fused01: np.ndarray) -> float:
    """Generalized Piella Q_s (weighted fusion quality index, sw=2 in metricPeilla.m).

    For >2 sources, uses per-pixel variance weights across all sources.
    """
    if len(sources01) < 2:
        raise ValueError("Need at least 2 sources for Q_s.")
    fused01 = np.asarray(fused01, dtype=np.float64)
    sources01 = [np.asarray(s, dtype=np.float64) for s in sources01]
    shape = fused01.shape
    if any(s.shape != shape for s in sources01):
        raise ValueError("All images must have same shape for Q_s.")

    # Local variances per source.
    sigmas = []
    for s in sources01:
        mu = _gaussian_filter_np(s, sigma=1.5)
        sig = _gaussian_filter_np(s * s, sigma=1.5) - mu * mu
        sigmas.append(sig)
    sigmas = np.stack(sigmas, axis=0)  # (K,H,W)

    denom = np.sum(sigmas, axis=0)
    # Handle all-zero variance windows -> uniform weights.
    uniform = (denom <= 0).astype(np.float64)
    denom = denom + uniform  # make denom=1 where all zero
    lambdas = sigmas / (denom[None, ...] + EPS)
    if np.any(uniform > 0):
        lambdas[:, uniform > 0] = 1.0 / float(len(sources01))

    ssim_maps = []
    for s in sources01:
        _, ssim_map, _, _ = _ssim_map_and_vars(fused01, s, sigma=1.5, data_range=1.0)
        ssim_maps.append(ssim_map)
    ssim_maps = np.stack(ssim_maps, axis=0)  # (K,H,W)

    q_map = np.sum(lambdas * ssim_maps, axis=0)  # (H,W)

    cw = np.max(sigmas, axis=0)
    cw_sum = float(np.sum(cw))
    if cw_sum <= 0:
        return float(np.mean(q_map))
    cw = cw / (cw_sum + EPS)
    return float(np.sum(cw * q_map))


def q_cv_chen_varshney(sources01: Sequence[np.ndarray], fused01: np.ndarray, *, window_size: int = 16, alpha: int = 5) -> float:
    """Generalized Chen-Varshney Q_CV (metricChen.m).

    Lower is better.
    """
    if len(sources01) < 2:
        raise ValueError("Need at least 2 sources for Q_CV.")
    sources_u8 = [normalize1_u8(s) for s in sources01]
    fused_u8 = normalize1_u8(fused01)

    # Edge magnitude for saliency.
    flt1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    flt2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    def grad_mag(img_u8: np.ndarray) -> np.ndarray:
        img = img_u8.astype(np.float64)
        gx = _conv2d_reflect_np(img, flt1)
        gy = _conv2d_reflect_np(img, flt2)
        return np.sqrt(gx * gx + gy * gy)

    grads = [grad_mag(s) for s in sources_u8]

    # Crop to full blocks (metricChen assumes exact blocks).
    h, w = fused_u8.shape
    H = h // window_size
    L = w // window_size
    hh = H * window_size
    ww = L * window_size
    if hh == 0 or ww == 0:
        return float("nan")

    grads = [g[:hh, :ww] for g in grads]
    fused_u8_c = fused_u8[:hh, :ww].astype(np.float64)

    # Saliency per block: sum(sum(G^alpha))
    sal = []
    for g in grads:
        blk = (g ** float(alpha)).reshape(H, window_size, L, window_size)
        sal.append(blk.sum(axis=(1, 3)))
    sal = np.stack(sal, axis=0)  # (K,H,L)

    # Difference images in spatial domain.
    diffs = []
    for s_u8 in sources_u8:
        d = (s_u8[:hh, :ww].astype(np.float64) - fused_u8_c).astype(np.float64)
        diffs.append(d)

    # Frequency grid like freqspace(...,'meshgrid') then scaling by /8.
    u = np.linspace(-1.0, 1.0, ww, endpoint=True, dtype=np.float64)
    v = np.linspace(-1.0, 1.0, hh, endpoint=True, dtype=np.float64)
    U, V = np.meshgrid(u, v)
    U = (ww / 8.0) * U
    V = (hh / 8.0) * V
    r = np.sqrt(U * U + V * V)
    theta_m = 2.6 * (0.0192 + 0.144 * r) * np.exp(-np.power(0.144 * r, 1.1))

    def csf_filter(img: np.ndarray) -> np.ndarray:
        ff = np.fft.fft2(img)
        out = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(ff) * theta_m))
        return np.real(out)

    Ds = []
    for d in diffs:
        Df = csf_filter(d)
        blk = (Df * Df).reshape(H, window_size, L, window_size)
        Ds.append(blk.mean(axis=(1, 3)))
    Ds = np.stack(Ds, axis=0)  # (K,H,L)

    num = float(np.sum(sal * Ds))
    den = float(np.sum(sal)) + EPS
    return float(num / den)


def vifp(x01: np.ndarray, y01: np.ndarray, *, sigma_nsq: float = 2.0) -> float:
    """Visual Information Fidelity (pixel domain) approximation (VIFp).

    Inputs are float images in [0,1]. Returns a ratio (can be > 1).
    """
    x = np.asarray(x01, dtype=np.float64)
    y = np.asarray(y01, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("VIF inputs must have same shape.")

    num = 0.0
    den = 0.0
    for scale in range(1, 5):
        n = 2 ** (5 - scale) + 1  # 17, 9, 5, 3
        sd = n / 5.0

        if scale > 1:
            x = _gaussian_filter_np(x, sigma=sd)[::2, ::2]
            y = _gaussian_filter_np(y, sigma=sd)[::2, ::2]

        mu1 = _gaussian_filter_np(x, sigma=sd)
        mu2 = _gaussian_filter_np(y, sigma=sd)
        sigma1_sq = _gaussian_filter_np(x * x, sigma=sd) - mu1 * mu1
        sigma2_sq = _gaussian_filter_np(y * y, sigma=sd) - mu2 * mu2
        sigma12 = _gaussian_filter_np(x * y, sigma=sd) - mu1 * mu2

        sigma1_sq = np.maximum(0.0, sigma1_sq)
        sigma2_sq = np.maximum(0.0, sigma2_sq)

        g = sigma12 / (sigma1_sq + EPS)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < EPS] = 0.0
        sv_sq[sigma1_sq < EPS] = sigma2_sq[sigma1_sq < EPS]
        sigma1_sq[sigma1_sq < EPS] = 0.0

        g[sigma2_sq < EPS] = 0.0
        sv_sq[sigma2_sq < EPS] = 0.0

        neg = g < 0
        g[neg] = 0.0
        sv_sq[neg] = sigma2_sq[neg]

        sv_sq = np.maximum(sv_sq, EPS)

        num += float(np.sum(np.log10(1.0 + (g * g) * sigma1_sq / (sv_sq + sigma_nsq))))
        den += float(np.sum(np.log10(1.0 + sigma1_sq / sigma_nsq)))

    return float(num / (den + EPS))


@dataclass(frozen=True)
class FSIMConfig:
    scales: int = 4
    orientations: int = 4
    wavelength: float = 6.0
    factor: float = 2.0
    sigma_f: float = 0.5978
    sigma_theta: float = 0.6545
    t1: float = 0.85
    t2: float = 0.0024605920799692428  # for value_range=1.0
    eps: float = 1e-8


_FSIM_FILTER_CACHE: dict[tuple[int, int, FSIMConfig], torch.Tensor] = {}
_SCHARR_CACHE: dict[tuple[str, str], torch.Tensor] = {}


def _scharr_kernels(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (str(device), str(dtype))
    cached = _SCHARR_CACHE.get(key)
    if cached is not None:
        return cached
    kx = torch.tensor([[3.0, 0.0, -3.0], [10.0, 0.0, -10.0], [3.0, 0.0, -3.0]], device=device, dtype=dtype) / 16.0
    ky = kx.t()
    kernel = torch.stack([kx, ky], dim=0).unsqueeze(1)  # (2,1,3,3)
    _SCHARR_CACHE[key] = kernel
    return kernel


def _log_gabor_filters(
    h: int,
    w: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    cfg: FSIMConfig,
) -> torch.Tensor:
    """Returns filters with shape (S, O, H, W) in frequency domain."""
    # Frequency grid in [-0.5,0.5) with fftshift-style coordinates.
    fy = torch.fft.fftshift(torch.fft.fftfreq(h, d=1.0, device=device, dtype=dtype)).reshape(h, 1)
    fx = torch.fft.fftshift(torch.fft.fftfreq(w, d=1.0, device=device, dtype=dtype)).reshape(1, w)
    y, x = torch.meshgrid(fy.squeeze(1), fx.squeeze(0), indexing="ij")
    radius = torch.sqrt(x * x + y * y)
    radius = torch.clamp(radius, min=cfg.eps)

    theta = torch.atan2(-y, x)
    sintheta = torch.sin(theta)
    costheta = torch.cos(theta)

    filters = []
    for s in range(cfg.scales):
        fo = 1.0 / (cfg.wavelength * (cfg.factor ** s))
        log_rad = torch.log(radius / fo)
        log_gabor = torch.exp(-(log_rad * log_rad) / (2.0 * (torch.log(torch.tensor(cfg.sigma_f, device=device, dtype=dtype)) ** 2 + cfg.eps)))
        log_gabor[radius < cfg.eps * 10] = 0.0

        ori_filters = []
        for o in range(cfg.orientations):
            angl = o * np.pi / cfg.orientations
            ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
            dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
            dtheta = torch.abs(torch.atan2(ds, dc))
            spread = torch.exp(-(dtheta * dtheta) / (2.0 * (cfg.sigma_theta ** 2) + cfg.eps))
            ori_filters.append(log_gabor * spread)
        filters.append(torch.stack(ori_filters, dim=0))
    return torch.stack(filters, dim=0)  # (S,O,H,W)


def _phase_congruency(x01: torch.Tensor, filters: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Simplified phase congruency over orientations (N,H,W)."""
    if x01.ndim != 4 or x01.shape[1] != 1:
        raise ValueError("phase_congruency expects (N,1,H,W)")
    n, _, h, w = x01.shape
    X = torch.fft.fftshift(torch.fft.fft2(x01.squeeze(1)), dim=(-2, -1))  # (N,H,W)
    pc_sum = torch.zeros((n, h, w), device=x01.device, dtype=x01.dtype)
    for o in range(filters.shape[1]):
        sum_e = torch.zeros((n, h, w), device=x01.device, dtype=x01.dtype)
        sum_o = torch.zeros((n, h, w), device=x01.device, dtype=x01.dtype)
        sum_an = torch.zeros((n, h, w), device=x01.device, dtype=x01.dtype)
        for s in range(filters.shape[0]):
            Fso = filters[s, o]  # (H,W)
            resp = torch.fft.ifft2(torch.fft.ifftshift(X * Fso, dim=(-2, -1)))  # (N,H,W) complex
            even = resp.real
            odd = resp.imag
            an = torch.abs(resp)
            sum_e = sum_e + even
            sum_o = sum_o + odd
            sum_an = sum_an + an
        energy = torch.sqrt(sum_e * sum_e + sum_o * sum_o + eps)
        pc = energy / (sum_an + eps)
        pc_sum = pc_sum + pc
    return pc_sum / float(filters.shape[1])


def fsim01(x01: np.ndarray, y01: np.ndarray, *, cfg: FSIMConfig = FSIMConfig()) -> float:
    """FSIM (grayscale) on float01 arrays, using simplified phase congruency."""
    x = torch.from_numpy(np.asarray(x01, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    y = torch.from_numpy(np.asarray(y01, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    if x.shape != y.shape:
        raise ValueError("FSIM inputs must have same shape.")

    device = torch.device("cpu")
    x = x.to(device)
    y = y.to(device)

    _, _, h, w = x.shape
    cache_key = (h, w, cfg)
    filters = _FSIM_FILTER_CACHE.get(cache_key)
    if filters is None:
        filters = _log_gabor_filters(h, w, device=device, dtype=x.dtype, cfg=cfg)
        _FSIM_FILTER_CACHE[cache_key] = filters
    pc_x = _phase_congruency(x, filters, eps=cfg.eps)
    pc_y = _phase_congruency(y, filters, eps=cfg.eps)

    kernel = _scharr_kernels(device, x.dtype)
    gm_x = torch.sqrt(torch.sum(F.conv2d(x, kernel, padding=1) ** 2, dim=1) + cfg.eps)  # (1,H,W)
    gm_y = torch.sqrt(torch.sum(F.conv2d(y, kernel, padding=1) ** 2, dim=1) + cfg.eps)

    t1 = float(cfg.t1)
    t2 = float(cfg.t2)
    s_pc = (2.0 * pc_x * pc_y + t1) / (pc_x * pc_x + pc_y * pc_y + t1 + cfg.eps)
    s_g = (2.0 * gm_x * gm_y + t2) / (gm_x * gm_x + gm_y * gm_y + t2 + cfg.eps)
    s_l = s_pc * s_g
    pc_m = torch.maximum(pc_x, pc_y)

    num = torch.sum(s_l * pc_m, dim=(-2, -1))
    den = torch.sum(pc_m, dim=(-2, -1)) + cfg.eps
    return float((num / den).item())
