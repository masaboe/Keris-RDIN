#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from keris.preprocess.bg_remove import batch_remove_bg
from keris.utils.config import build_argparser, load_config_with_overrides
from keris.utils.seeds import seed_everything


def main():
    ap = build_argparser("Stage 01: Background removal (rembg) + white compositing")
    args = ap.parse_args()
    cfg = load_config_with_overrides(args)

    seed_everything(int(cfg.get("seed", 42)))

    n = batch_remove_bg(
        input_dir=cfg["data"]["input_dir"],
        output_dir=cfg["data"]["output_dir"],
        alpha_thr=int(cfg["params"].get("alpha_thr", 1)),
        ksize=int(cfg["params"].get("morph_ksize", 3)),
        margin=int(cfg["params"].get("crop_margin", 8)),
        crop=bool(cfg["params"].get("crop", True)),
        keep_structure=bool(cfg["params"].get("keep_structure", True)),
    )
    print(f"[DONE] Processed {n} images.")


if __name__ == "__main__":
    main()
