from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from sklearn.decomposition import PCA

from keris.utils.io import ensure_dir


def pad_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    size = int(np.ceil(np.hypot(h, w)))  # diagonal
    pad_v = (size - h) // 2
    pad_h = (size - w) // 2
    pad_top = pad_v
    pad_bottom = size - h - pad_top
    pad_left = pad_h
    pad_right = size - w - pad_left
    if img.ndim == 3 and img.shape[2] == 4:
        border = (0, 0, 0, 0)
    else:
        border = (255, 255, 255)
    return cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=border)


def correct_handle_down(img: np.ndarray) -> np.ndarray:
    """
    Notebook heuristic: if mean y of object mask is above center, flip vertically.
    """
    h = img.shape[0]
    if img.ndim == 3 and img.shape[2] == 4:
        mask = img[:, :, 3] > 0
    else:
        # if white bg: object pixels are those not near-white
        mask = np.any(img < 250, axis=2)
    ys, _ = np.where(mask)
    if len(ys) and ys.mean() < h / 2:
        return cv2.flip(img, 0)
    return img


def align_and_crop_pca(img: np.ndarray, tol_angle: float = 1.0) -> np.ndarray:
    """
    PCA-based vertical alignment, followed by contour crop and handle-down correction.
    Works for RGBA (alpha mask) or RGB/BGR on white background.
    """
    # initial mask
    if img.ndim == 3 and img.shape[2] == 4:
        mask = img[:, :, 3] > 0
    else:
        mask = np.any(img < 250, axis=2)

    ys, xs = np.where(mask)
    if len(xs) < 20:
        # fallback crop
        if len(xs) == 0:
            return img
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        crop = img[y0 : y1 + 1, x0 : x1 + 1]
        return correct_handle_down(crop)

    coords = np.column_stack([xs, ys])
    vx, vy = PCA(n_components=2).fit(coords).components_[0]
    angle = np.degrees(np.arctan2(vx, vy))

    if abs(angle) < tol_angle:
        rotated = img.copy()
    else:
        padded = pad_to_square(img)
        size = padded.shape[0]
        M = cv2.getRotationMatrix2D((size // 2, size // 2), -angle, 1.0)
        border = (0, 0, 0, 0) if (img.ndim == 3 and img.shape[2] == 4) else (255, 255, 255)
        rotated = cv2.warpAffine(
            padded,
            M,
            (size, size),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )

    # contour crop after rotation
    if rotated.ndim == 3 and rotated.shape[2] == 4:
        mask2 = rotated[:, :, 3] > 0
    else:
        mask2 = np.any(rotated < 250, axis=2)

    mask2_u8 = (mask2.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask2_u8 = cv2.morphologyEx(mask2_u8, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask2_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        crop = rotated[y : y + h, x : x + w]
    else:
        crop = rotated

    return correct_handle_down(crop)


def contour_crop(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 4:
        mask = img[:, :, 3] > 0
    else:
        mask = np.any(img < 250, axis=2)
    mask = (mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return img[y : y + h, x : x + w]
    return img


def resize_and_pad(img: np.ndarray, target_size: int = 512) -> np.ndarray:
    """
    Resize to target_size height, preserve aspect ratio, pad horizontally to 512x512.
    Background is white for RGB, transparent for RGBA.
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    scale = target_size / float(h)
    new_w = max(1, int(w * scale))
    resized = cv2.resize(img, (new_w, target_size), interpolation=cv2.INTER_AREA)

    pad_total = target_size - new_w
    if pad_total < 0:
        # too wide after scaling: center-crop horizontally
        x0 = (-pad_total) // 2
        return resized[:, x0 : x0 + target_size]

    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    border = (0, 0, 0, 0) if (resized.ndim == 3 and resized.shape[2] == 4) else (255, 255, 255)
    return cv2.copyMakeBorder(resized, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=border)


def process_tree_pca(
    input_dir: str | Path,
    output_dir: str | Path,
    target_size: int = 512,
) -> int:
    input_dir = Path(input_dir)
    output_dir = ensure_dir(output_dir)

    n = 0
    for p in input_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        rel = p.relative_to(input_dir)
        out_path = output_dir / rel.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        proc = align_and_crop_pca(img)
        proc = contour_crop(proc)
        proc = resize_and_pad(proc, target_size=target_size)

        # If RGBA, composite to white RGB for consistency (optional)
        if proc.ndim == 3 and proc.shape[2] == 4:
            alpha = proc[:, :, 3:4] / 255.0
            rgb = proc[:, :, :3].astype(np.float32)
            bg = np.full_like(rgb, 255.0)
            proc = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)

        cv2.imwrite(str(out_path), proc)
        n += 1
    return n
