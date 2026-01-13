from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO


def train_yolov8(cfg: Dict[str, Any]) -> None:
    """
    Thin wrapper around Ultralytics training, driven by YAML config.
    """
    ycfg = cfg["yolo"]
    model_name = ycfg.get("model", "yolov8n.pt")
    data_yaml = ycfg["data_yaml"]

    yolo = YOLO(model_name)
    yolo.train(
        data=data_yaml,
        epochs=int(ycfg.get("epochs", 300)),
        imgsz=int(ycfg.get("imgsz", 512)),
        batch=int(ycfg.get("batch", 16)),
        optimizer=ycfg.get("optimizer", "AdamW"),
        lr0=float(ycfg.get("lr0", 5e-4)),
        lrf=float(ycfg.get("lrf", 0.01)),
        dropout=float(ycfg.get("dropout", 0.1)),
        seed=int(ycfg.get("seed", 42)),
        project=ycfg.get("project", "runs"),
        name=ycfg.get("name", "train"),
        device=ycfg.get("device", None),
    )
