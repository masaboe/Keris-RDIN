#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from keris.preprocess.pca_align import process_tree_pca
from keris.utils.config import build_argparser, load_config_with_overrides
from keris.utils.seeds import seed_everything


def main():
    ap = build_argparser("Stage 02: PCA-based vertical alignment + baseline normalization")
    args = ap.parse_args()
    cfg = load_config_with_overrides(args)

    seed_everything(int(cfg.get("seed", 42)))

    n = process_tree_pca(
        input_dir=cfg["data"]["input_dir"],
        output_dir=cfg["data"]["output_dir"],
        target_size=int(cfg["params"].get("target_size", 512)),
    )
    print(f"[DONE] Processed {n} images.")


if __name__ == "__main__":
    main()
