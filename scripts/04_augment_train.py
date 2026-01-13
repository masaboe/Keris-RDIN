#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np

from keris.augment.balanced_aug import augment_and_balance
from keris.utils.config import build_argparser, load_config_with_overrides
from keris.utils.io import ensure_dir, save_npy
from keris.utils.seeds import seed_everything


def main():
    ap = build_argparser("Stage 04: Mask-aware augmentation + class balancing (NPY)")
    args = ap.parse_args()
    cfg = load_config_with_overrides(args)

    seed_everything(int(cfg.get("seed", 42)))

    x_train = np.load(cfg["data"]["x_train"])
    y_train = np.load(cfg["data"]["y_train"])

    X_aug, y_aug = augment_and_balance(x_train, y_train, cfg["augment"])

    out_dir = ensure_dir(cfg["run"]["out_dir"])
    save_npy(out_dir / cfg["outputs"]["x_train_aug"], X_aug)
    save_npy(out_dir / cfg["outputs"]["y_train_aug"], y_aug)

    print(f"[DONE] Saved X_aug={X_aug.shape}, y_aug={y_aug.shape} to {out_dir}")


if __name__ == "__main__":
    main()
