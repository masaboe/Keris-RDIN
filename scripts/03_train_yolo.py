#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from keris.yolo.train import train_yolov8
from keris.utils.config import build_argparser, load_config_with_overrides
from keris.utils.seeds import seed_everything


def main():
    ap = build_argparser("Stage 03a: Train YOLOv8 (bilah detector)")
    args = ap.parse_args()
    cfg = load_config_with_overrides(args)

    seed_everything(int(cfg.get("seed", 42)))

    train_yolov8(cfg)
    print("[DONE] YOLO training finished.")


if __name__ == "__main__":
    main()
