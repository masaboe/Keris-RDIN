#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from keris.models.rdin import build_keris_rdin
from keris.train.classifier import train_classifier
from keris.utils.config import build_argparser, load_config_with_overrides
from keris.utils.io import ensure_dir, save_json
from keris.utils.seeds import seed_everything


def main():
    ap = build_argparser("Stage 05: Train Keris-RDIN classifier (NPY)")
    args = ap.parse_args()
    cfg = load_config_with_overrides(args)

    seed_everything(int(cfg.get("seed", 42)))

    out_dir = ensure_dir(cfg["run"]["out_dir"])

    model = build_keris_rdin(
        input_shape=tuple(cfg["model"].get("input_shape", [512, 512, 3])),
        num_classes=int(cfg["model"]["num_classes"]),
        backbone_trainable=bool(cfg["model"].get("backbone_trainable", False)),
        block_filters=int(cfg["model"].get("block_filters", 768)),
        block_repeats=int(cfg["model"].get("block_repeats", 2)),
    )

    model, hist, (X_test, y_test) = train_classifier(model, cfg)

    # save weights
    weights_path = out_dir / cfg["outputs"].get("weights", "weights/keris_rdin.weights.h5")
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(weights_path)

    # training history
    hist_path = out_dir / cfg["outputs"].get("history_json", "artifacts/train_history.json")
    save_json(hist_path, hist.history)

    print(f"[DONE] Saved weights: {weights_path}")
    print(f"[DONE] Saved history: {hist_path}")


if __name__ == "__main__":
    main()
