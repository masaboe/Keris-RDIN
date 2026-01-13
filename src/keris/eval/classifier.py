from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ensure_int_labels(y: np.ndarray) -> np.ndarray:
    if y.ndim > 1 and y.shape[1] > 1:
        return np.argmax(y, axis=1).astype(int)
    return y.reshape(-1).astype(int)


def evaluate_multiclass(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    y_true_i = ensure_int_labels(y_true)
    y_pred_i = np.argmax(y_prob, axis=1).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true_i, y_pred_i)),
        "precision_macro": float(precision_score(y_true_i, y_pred_i, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true_i, y_pred_i, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true_i, y_pred_i, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true_i, y_pred_i, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true_i, y_pred_i, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true_i, y_pred_i, average="weighted", zero_division=0)),
    }

    # multiclass AUC (OvR)
    try:
        # y_true one-hot required
        if y_true.ndim == 1:
            # convert to one-hot
            num_classes = y_prob.shape[1]
            y_true_oh = np.eye(num_classes)[y_true_i]
        else:
            y_true_oh = y_true
        metrics["auc_ovr_macro"] = float(
            roc_auc_score(y_true_oh, y_prob, multi_class="ovr", average="macro")
        )
    except Exception:
        metrics["auc_ovr_macro"] = None

    cm = confusion_matrix(y_true_i, y_pred_i)
    report = classification_report(y_true_i, y_pred_i, output_dict=True, zero_division=0)

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_true": y_true_i,
        "y_pred": y_pred_i,
    }


def save_eval_outputs(out_dir: str | Path, results: Dict[str, Any]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # metrics summary
    pd.DataFrame([results["metrics"]]).to_csv(out_dir / "metrics_summary.csv", index=False)

    # confusion matrix
    cm = results["confusion_matrix"]
    pd.DataFrame(cm).to_csv(out_dir / "confusion_matrix.csv", index=False)

    # classification report
    rep = pd.DataFrame(results["classification_report"]).transpose()
    rep.to_csv(out_dir / "classification_report.csv")

    # y_true/y_pred
    pd.DataFrame({"y_true": results["y_true"], "y_pred": results["y_pred"]}).to_csv(out_dir / "predictions.csv", index=False)
