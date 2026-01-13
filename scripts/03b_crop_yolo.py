#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from keris.yolo.crop import yolo_crop_dir
from keris.utils.config import build_argparser, load_config_with_overrides
from keris.utils.seeds import seed_everything


def main():
    ap = build_argparser("Stage 03b: YOLOv8 crop (bilah) -> 512x512 white background")
    args = ap.parse_args()
    cfg = load_config_with_overrides(args)

    seed_everything(int(cfg.get("seed", 42)))

    n = yolo_crop_dir(
        input_dir=cfg["data"]["input_dir"],
        output_dir=cfg["data"]["output_dir"],
        model_path=cfg["yolo"]["model_path"],
        target_class_names=cfg["yolo"].get("target_classes", ["bilah"]),
        conf_thres=float(cfg["yolo"].get("conf_thres", 0.10)),
        iou_thres=float(cfg["yolo"].get("iou_thres", 0.45)),
        out_size=int(cfg["yolo"].get("out_size", 512)),
    )
    print(f"[DONE] Cropped {n} images.")


if __name__ == "__main__":
    main()
