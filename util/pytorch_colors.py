import torch
import torch.cuda
from torch.autograd import Variable
try:
    from skimage.color import (
        rgb2lab,
        rgb2yuv,
        rgb2ycbcr,
        lab2rgb,
        yuv2rgb,
        ycbcr2rgb,
        rgb2hsv,
        hsv2rgb,
        rgb2xyz,
        xyz2rgb,
        rgb2hed,
        hed2rgb,
    )

    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover
    # Minimal fallback: implement only YCbCr conversions without scikit-image.
    _HAS_SKIMAGE = False


def _convert(input_, type_):
    return {
        'float': input_.float(),
        'double': input_.double(),
    }.get(type_, input_)


def _generic_transform_sk_4d(transform, in_type='', out_type=''):
    def apply_transform(input_):
        if not _HAS_SKIMAGE:
            raise ModuleNotFoundError("scikit-image is required for this color conversion.")
        to_squeeze = (input_.dim() == 3)
        device = input_.device
        input_ = input_.cpu()
        input_ = _convert(input_, in_type)

        if to_squeeze:
            input_ = input_.unsqueeze(0)

        input_ = input_.permute(0, 2, 3, 1).numpy()
        transformed = transform(input_)
        output = torch.from_numpy(transformed).float().permute(0, 3, 1, 2)
        if to_squeeze:
            output = output.squeeze(0)
        output = _convert(output, out_type)
        return output.to(device)
    return apply_transform


def _generic_transform_sk_3d(transform, in_type='', out_type=''):
    def apply_transform_individual(input_):
        if not _HAS_SKIMAGE:
            raise ModuleNotFoundError("scikit-image is required for this color conversion.")
        device = input_.device
        input_ = input_.cpu()
        input_ = _convert(input_, in_type)

        input_ = input_.permute(1, 2, 0).numpy()
        transformed = transform(input_)
        output = torch.from_numpy(transformed).float().permute(2, 0, 1)
        output = _convert(output, out_type)
        return output.to(device)

    def apply_transform(input_):
        to_stack = []
        for image in input_:
            to_stack.append(apply_transform_individual(image))
        return torch.stack(to_stack)
    return apply_transform


if _HAS_SKIMAGE:
    # --- Cie*LAB ---
    rgb_to_lab = _generic_transform_sk_4d(rgb2lab)
    lab_to_rgb = _generic_transform_sk_3d(lab2rgb, in_type='double', out_type='float')
    # --- YUV ---
    rgb_to_yuv = _generic_transform_sk_4d(rgb2yuv)
    yuv_to_rgb = _generic_transform_sk_4d(yuv2rgb)
    # --- YCbCr ---
    rgb_to_ycbcr = _generic_transform_sk_4d(rgb2ycbcr)
    ycbcr_to_rgb = _generic_transform_sk_4d(ycbcr2rgb, in_type='double', out_type='float')
    # --- HSV ---
    rgb_to_hsv = _generic_transform_sk_3d(rgb2hsv)
    hsv_to_rgb = _generic_transform_sk_3d(hsv2rgb)
    # --- XYZ ---
    rgb_to_xyz = _generic_transform_sk_4d(rgb2xyz)
    xyz_to_rgb = _generic_transform_sk_3d(xyz2rgb, in_type='double', out_type='float')
    # --- HED ---
    rgb_to_hed = _generic_transform_sk_4d(rgb2hed)
    hed_to_rgb = _generic_transform_sk_3d(hed2rgb, in_type='double', out_type='float')
else:
    def rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
        """
        Fallback BT.601 conversion.
        Expects x in [-1, 1] with shape (N, 3, H, W). Returns YCbCr in [0, 255].
        """
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(f"Expected (N,3,H,W), got {tuple(x.shape)}")
        x255 = (x + 1.0) * 127.5
        r, g, b = x255[:, 0:1], x255[:, 1:2], x255[:, 2:3]
        y = 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
        cb = 128.0 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255.0
        cr = 128.0 + (112.0 * r - 93.786 * g - 18.214 * b) / 255.0
        return torch.cat([y, cb, cr], dim=1)

    def ycbcr_to_rgb(x: torch.Tensor) -> torch.Tensor:
        """
        Fallback BT.601 inverse conversion.
        Expects x in [0, 255] with shape (N, 3, H, W). Returns RGB in [-1, 1].
        """
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(f"Expected (N,3,H,W), got {tuple(x.shape)}")
        y, cb, cr = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y = y - 16.0
        cb = cb - 128.0
        cr = cr - 128.0
        r = 1.164 * y + 1.596 * cr
        g = 1.164 * y - 0.392 * cb - 0.813 * cr
        b = 1.164 * y + 2.017 * cb
        rgb255 = torch.cat([r, g, b], dim=1)
        rgb255 = torch.clamp(rgb255, 0.0, 255.0)
        return rgb255 / 127.5 - 1.0

    def _missing(*_args, **_kwargs):
        raise ModuleNotFoundError("scikit-image is required for this color conversion.")

    rgb_to_lab = _missing
    lab_to_rgb = _missing
    rgb_to_yuv = _missing
    yuv_to_rgb = _missing
    rgb_to_hsv = _missing
    hsv_to_rgb = _missing
    rgb_to_xyz = _missing
    xyz_to_rgb = _missing
    rgb_to_hed = _missing
    hed_to_rgb = _missing


def err(type_):
    raise NotImplementedError('Color space conversion %s not implemented yet' % type_)


def convert(input_, type_):
    return {
        'rgb2lab': rgb_to_lab(input_),
        'lab2rgb': lab_to_rgb(input_),
        'rgb2yuv': rgb_to_yuv(input_),
        'yuv2rgb': yuv_to_rgb(input_),
        'rgb2xyz': rgb_to_xyz(input_),
        'xyz2rgb': xyz_to_rgb(input_),
        'rgb2hsv': rgb_to_hsv(input_),
        'hsv2rgb': hsv_to_rgb(input_),
        'rgb2ycbcr': rgb_to_ycbcr(input_),
        'ycbcr2rgb': ycbcr_to_rgb(input_)
    }.get(type_, err(type_))
