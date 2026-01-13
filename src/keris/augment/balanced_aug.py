from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import albumentations as A
import cv2
import numpy as np
from collections import Counter

from keris.utils.io import ensure_dir


def to_int_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim > 1 and y.shape[-1] > 1:
        return np.argmax(y, axis=1).astype(np.int32)
    return y.reshape(-1).astype(np.int32)


def to_one_hot(y_int: np.ndarray, num_classes: int) -> np.ndarray:
    y_int = y_int.astype(int).reshape(-1)
    return np.eye(num_classes, dtype=np.float32)[y_int]


def mask_from_white_bg(img_u8: np.ndarray, white_thr: int = 245) -> np.ndarray:
    """
    Create foreground mask from (almost) white background.
    Returns boolean mask (H,W).
    """
    if img_u8.ndim != 3:
        raise ValueError("Expected HxWx3 image")
    bg = np.all(img_u8 >= white_thr, axis=2)
    mask = ~bg
    # morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return mask_u8 > 0


def photometric_on_mask(img_u8: np.ndarray, mask_bool: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    out = img_u8.copy()

    # brightness/contrast
    if np.random.rand() < float(cfg.get("bc_p", 0.5)) and mask_bool.any():
        brightness_limit = float(cfg.get("brightness_limit", 0.2))
        contrast_limit = float(cfg.get("contrast_limit", 0.2))
        alpha = 1.0 + np.random.uniform(-contrast_limit, contrast_limit)
        beta = np.random.uniform(-brightness_limit, brightness_limit) * 255.0
        adj = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        out[mask_bool] = adj[mask_bool]

    # gaussian blur
    if np.random.rand() < float(cfg.get("blur_p", 0.2)) and mask_bool.any():
        k = int(cfg.get("blur_k", 3))
        if k % 2 == 0:
            k += 1
        blur = cv2.GaussianBlur(out, (k, k), 0)
        out[mask_bool] = blur[mask_bool]

    # gaussian noise
    if np.random.rand() < float(cfg.get("noise_p", 0.2)) and mask_bool.any():
        vmin, vmax = cfg.get("noise_var", [10.0, 50.0])
        var = float(np.random.uniform(vmin, vmax))
        noise = np.random.normal(0, np.sqrt(var), out.shape).astype(np.float32)
        noisy = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        out[mask_bool] = noisy[mask_bool]

    # enforce background white
    out[~mask_bool] = 255
    return out


def build_geo_transform(cfg: Dict[str, Any]) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=float(cfg.get("hflip_p", 0.5))),
            A.ShiftScaleRotate(
                shift_limit=float(cfg.get("shift_limit", 0.06)),
                scale_limit=float(cfg.get("scale_limit", 0.06)),
                rotate_limit=float(cfg.get("rotate_limit", 5)),
                border_mode=cv2.BORDER_CONSTANT,
                value=255,
                mask_value=0,
                p=float(cfg.get("ssr_p", 0.7)),
            ),
            A.Perspective(
                scale=(float(cfg.get("persp_scale_min", 0.02)), float(cfg.get("persp_scale_max", 0.04))),
                keep_size=True,
                pad_val=255,
                mask_pad_val=0,
                p=float(cfg.get("persp_p", 0.2)),
            ),
        ]
    )


def augment_and_balance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (X_aug, y_aug) where y_aug is one-hot (float32).
    """
    num_classes = int(cfg["num_classes"])
    variants_per_image = int(cfg.get("variants_per_image", 2))
    white_thr = int(cfg.get("white_thr", 245))

    y_int = to_int_labels(y_train)
    geo = build_geo_transform(cfg.get("geo", {}))
    photo_cfg = cfg.get("photo", {})

    # Convert X to uint8 if needed
    if X_train.dtype != np.uint8:
        X_u8 = (np.clip(X_train, 0, 1) * 255.0).astype(np.uint8) if X_train.max() <= 1.5 else X_train.astype(np.uint8)
    else:
        X_u8 = X_train

    # Build list per class for oversampling
    idx_by_class = {c: np.where(y_int == c)[0].tolist() for c in range(num_classes)}
    counts = {c: len(v) for c, v in idx_by_class.items()}
    max_count = max(counts.values()) if counts else 0

    X_out = []
    y_out = []

    # Always include originals
    for i in range(len(X_u8)):
        X_out.append(X_u8[i])
        y_out.append(y_int[i])

    # Augment each image a few times
    for i in range(len(X_u8)):
        img = X_u8[i]
        label = y_int[i]

        for _ in range(variants_per_image):
            mask = mask_from_white_bg(img, white_thr=white_thr)
            aug = geo(image=img, mask=mask.astype(np.uint8) * 255)
            img2 = aug["image"]
            mask2 = aug["mask"] > 0

            img2 = photometric_on_mask(img2, mask2, photo_cfg)

            # resize enforce (if any drift)
            if int(cfg.get("out_size", 512)) != img2.shape[0] or int(cfg.get("out_size", 512)) != img2.shape[1]:
                out_size = int(cfg.get("out_size", 512))
                img2 = cv2.resize(img2, (out_size, out_size), interpolation=cv2.INTER_AREA)

            X_out.append(img2)
            y_out.append(label)

    # Balance to max_count (optional)
    balance = cfg.get("balance", {"enabled": True, "target": "max"})
    if bool(balance.get("enabled", True)) and max_count > 0:
        # recompute after initial augmentation
        y_tmp = np.array(y_out, dtype=np.int32)
        idx_by_class = {c: np.where(y_tmp == c)[0].tolist() for c in range(num_classes)}
        counts = {c: len(v) for c, v in idx_by_class.items()}
        target = max(counts.values()) if balance.get("target", "max") == "max" else int(balance["target"])

        rng = np.random.default_rng(int(cfg.get("seed", 42)))

        for c in range(num_classes):
            cur = counts.get(c, 0)
            if cur == 0:
                continue
            need = target - cur
            if need <= 0:
                continue

            candidates = idx_by_class[c]
            for _ in range(need):
                j = rng.choice(candidates)
                base_img = X_out[j]
                mask = mask_from_white_bg(base_img, white_thr=white_thr)
                aug = geo(image=base_img, mask=mask.astype(np.uint8) * 255)
                img2 = aug["image"]
                mask2 = aug["mask"] > 0
                img2 = photometric_on_mask(img2, mask2, photo_cfg)

                X_out.append(img2)
                y_out.append(c)

    X_aug = np.stack(X_out, axis=0).astype(np.uint8)
    y_aug = to_one_hot(np.array(y_out, dtype=np.int32), num_classes=num_classes)

    return X_aug, y_aug
