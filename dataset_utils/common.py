import random
import torch
import numpy as np
import skimage.color as sc


def get_patch(*args, patch_size, scale):
    """
    args[0] : LR image  (H, W, C)
    args[1:] : HR image(s) (H, W, C)
    patch_size : HR patch size
    scale      : upscaling factor
    """

    img_lr = args[0]
    ih, iw = img_lr.shape[:2]

    tp = patch_size
    ip = tp // scale
    ip = min(ip, ih, iw)
    tp = ip * scale

    if ip <= 0 or tp <= 0:
        ip = ih
        tp = ih * scale

    max_iy = max(0, ih - ip)
    max_ix = max(0, iw - ip)

    iy = random.randint(0, max_iy) if max_iy > 0 else 0
    ix = random.randint(0, max_ix) if max_ix > 0 else 0

    ty = iy * scale
    tx = ix * scale

    lr_patch = img_lr[iy:iy + ip, ix:ix + ip, :]

    hr_patches = []
    for a in args[1:]:
        h, w = a.shape[:2]

        if ty + tp > h:
            ty_safe = max(0, h - tp)
        else:
            ty_safe = ty

        if tx + tp > w:
            tx_safe = max(0, w - tp)
        else:
            tx_safe = tx

        patch = a[ty_safe:ty_safe + tp, tx_safe:tx_safe + tp, :]
        hr_patches.append(patch)

    if lr_patch.shape[0] == 0 or lr_patch.shape[1] == 0:
        cy = ih // 2
        cx = iw // 2
        half = ip // 2
        y0 = max(0, cy - half)
        x0 = max(0, cx - half)
        lr_patch = img_lr[y0:y0 + ip, x0:x0 + ip, :]

    fixed_hr_patches = []
    for a, patch in zip(args[1:], hr_patches):
        h, w = a.shape[:2]
        if patch.shape[0] == 0 or patch.shape[1] == 0:
            c_y = h // 2
            c_x = w // 2
            half_t = tp // 2
            y0 = max(0, c_y - half_t)
            x0 = max(0, c_x - half_t)
            patch = a[y0:y0 + tp, x0:x0 + tp, :]
        fixed_hr_patches.append(patch)

    ret = [lr_patch, *fixed_hr_patches]
    return ret


def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]


def np2Tensor(*args, rgb_range):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose.copy()).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]


def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(a) for a in args]

