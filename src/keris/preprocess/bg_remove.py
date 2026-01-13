from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from rembg import remove

from keris.utils.io import ensure_dir


def rgba_bytes_to_rgba_array(png_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    arr = np.array(img)  # H x W x 4 (RGBA)
    return arr


def mask_from_alpha(rgba: np.ndarray, thr: int = 1) -> np.ndarray:
    alpha = rgba[:, :, 3]
    mask = (alpha >= thr).astype(np.uint8) * 255
    return mask


def refine_mask(mask: np.ndarray, ksize: int = 3, do_open: bool = True, do_close: bool = True) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    out = mask
    if do_open:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    if do_close:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return out


def composite_white(rgba: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Composite RGBA to RGB with white background using a binary mask.
    Returns uint8 RGB.
    """
    rgb = rgba[:, :, :3].copy()
    bg = np.full_like(rgb, 255)
    m = (mask > 0)[:, :, None]
    out = np.where(m, rgb, bg)
    return out.astype(np.uint8)


def crop_bbox_from_mask(img: np.ndarray, mask: np.ndarray, margin: int = 8) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h, w = img.shape[:2]
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(h - 1, y1 + margin)
    x1 = min(w - 1, x1 + margin)
    return img[y0 : y1 + 1, x0 : x1 + 1]


def remove_bg_white(
    in_path: str | Path,
    out_path: str | Path,
    alpha_thr: int = 1,
    ksize: int = 3,
    margin: int = 8,
    crop: bool = True,
) -> None:
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    png_bytes = remove(in_path.read_bytes())
    rgba = rgba_bytes_to_rgba_array(png_bytes)

    mask = mask_from_alpha(rgba, thr=alpha_thr)
    mask = refine_mask(mask, ksize=ksize, do_open=True, do_close=True)
    rgb = composite_white(rgba, mask)

    if crop:
        rgb = crop_bbox_from_mask(rgb, mask, margin=margin)

    # Save as 3-channel PNG
    Image.fromarray(rgb).save(out_path, format="PNG")


def batch_remove_bg(
    input_dir: str | Path,
    output_dir: str | Path,
    pattern: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    alpha_thr: int = 1,
    ksize: int = 3,
    margin: int = 8,
    crop: bool = True,
    keep_structure: bool = True,
) -> int:
    input_dir = Path(input_dir)
    output_dir = ensure_dir(output_dir)

    n = 0
    for p in input_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in pattern:
            continue

        rel = p.relative_to(input_dir)
        out_path = output_dir / rel.with_suffix(".png")
        if not keep_structure:
            out_path = output_dir / p.name
            out_path = out_path.with_suffix(".png")

        remove_bg_white(
            p, out_path,
            alpha_thr=alpha_thr,
            ksize=ksize,
            margin=margin,
            crop=crop,
        )
        n += 1
    return n
