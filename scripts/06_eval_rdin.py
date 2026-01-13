#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from keris.eval.classifier import evaluate_multiclass, save_eval_outputs
from keris.models.rdin import build_keris_rdin
from keris.utils.config import build_argparser, load_config_with_overrides
from keris.utils.io import ensure_dir, save_json
from keris.utils.seeds import seed_everything


def main():
    ap = build_argparser("Stage 06: Evaluate Keris-RDIN classifier (NPY)")
    args = ap.parse_args()
    cfg = load_config_with_overrides(args)

    seed_everything(int(cfg.get("seed", 42)))

    out_dir = ensure_dir(cfg["run"]["out_dir"])

    # load data
    paths = cfg["data"]["npy"]
    X_test = np.load(paths["x_test"])
    y_test = np.load(paths["y_test"])

    model = build_keris_rdin(
        input_shape=tuple(cfg["model"].get("input_shape", [512, 512, 3])),
        num_classes=int(cfg["model"]["num_classes"]),
        backbone_trainable=bool(cfg["model"].get("backbone_trainable", False)),
        block_filters=int(cfg["model"].get("block_filters", 768)),
        block_repeats=int(cfg["model"].get("block_repeats", 2)),
    )
    model.load_weights(cfg["weights"]["path"])

    y_prob = model.predict(X_test, batch_size=int(cfg.get("eval", {}).get("batch_size", 8)), verbose=1)
    results = evaluate_multiclass(y_test, y_prob)

    # save structured JSON + CSVs
    save_json(out_dir / "artifacts/eval_metrics.json", results["metrics"])
    save_eval_outputs(out_dir / "artifacts", results)

    print(f"[DONE] Evaluation artifacts saved under: {out_dir/'artifacts'}")


if __name__ == "__main__":
    main()
