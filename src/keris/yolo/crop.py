from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from keris.utils.io import ensure_dir


def crop_largest_bbox(
    img: np.ndarray,
    bboxes_xyxy: np.ndarray,
    conf: np.ndarray,
) -> Optional[np.ndarray]:
    if bboxes_xyxy is None or len(bboxes_xyxy) == 0:
        return None
    areas = (bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]) * (bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1])
    # Prefer highest confidence; tie-break by area
    idx = np.lexsort((areas, conf))[-1]
    x1, y1, x2, y2 = bboxes_xyxy[idx].astype(int).tolist()
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1 : y2, x1 : x2]


def yolo_crop_dir(
    input_dir: str | Path,
    output_dir: str | Path,
    model_path: str | Path,
    target_class_names: Optional[Sequence[str]] = ("bilah",),
    conf_thres: float = 0.10,
    iou_thres: float = 0.45,
    out_size: int = 512,
) -> int:
    input_dir = Path(input_dir)
    output_dir = ensure_dir(output_dir)
    model = YOLO(str(model_path))

    n = 0
    for p in input_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue

        res = model.predict(
            source=img,
            conf=conf_thres,
            iou=iou_thres,
            verbose=False,
        )[0]

        if res.boxes is None or len(res.boxes) == 0:
            continue

        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)

        # filter by class name (if provided)
        if target_class_names is not None:
            names = res.names  # id->name
            keep = []
            for i, c in enumerate(cls):
                if names.get(int(c), str(c)) in set(target_class_names):
                    keep.append(i)
            if len(keep) == 0:
                continue
            boxes = boxes[keep]
            scores = scores[keep]

        crop = crop_largest_bbox(img, boxes, scores)
        if crop is None:
            continue

        crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)

        rel = p.relative_to(input_dir)
        out_path = output_dir / rel.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), crop)
        n += 1

    return n
