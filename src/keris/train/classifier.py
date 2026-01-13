from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Precision, Recall, TopKCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from keris.models.losses import multiclass_focal_loss
from keris.utils.io import ensure_dir


class TimeHistory(Callback):
    def on_train_begin(self, logs=None):
        import time
        self._t0 = time.time()

    def on_train_end(self, logs=None):
        import time
        dt = time.time() - self._t0
        print(f"[INFO] Total training time: {dt:.2f} seconds")


def load_npy_splits(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = cfg["data"]["npy"]
    X_train = np.load(paths["x_train"])
    X_val   = np.load(paths["x_val"])
    X_test  = np.load(paths["x_test"])
    y_train = np.load(paths["y_train"])
    y_val   = np.load(paths["y_val"])
    y_test  = np.load(paths["y_test"])
    return X_train, X_val, X_test, y_train, y_val, y_test


def to_one_hot_if_needed(y: np.ndarray, num_classes: int) -> np.ndarray:
    if y.ndim == 1:
        return tf.keras.utils.to_categorical(y.astype(int), num_classes=num_classes)
    return y


def compute_class_weights_from_one_hot(y_train_onehot: np.ndarray) -> Dict[int, float]:
    y_labels = np.argmax(y_train_onehot, axis=1)
    classes = np.unique(y_labels)
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_labels,
    )
    return dict(zip(classes.tolist(), weights.tolist()))


def compile_model(model: tf.keras.Model, cfg: Dict[str, Any]) -> None:
    train_cfg = cfg["train"]
    lr_cfg = train_cfg.get("lr_schedule", {})
    lr_schedule = ExponentialDecay(
        initial_learning_rate=float(lr_cfg.get("initial", 5e-5)),
        decay_steps=int(lr_cfg.get("decay_steps", 100_000)),
        decay_rate=float(lr_cfg.get("decay_rate", 0.96)),
        staircase=bool(lr_cfg.get("staircase", True)),
    )
    optimizer = Adam(
        learning_rate=lr_schedule,
        clipnorm=float(train_cfg.get("clipnorm", 1.0)),
        beta_1=float(train_cfg.get("beta_1", 0.9)),
        beta_2=float(train_cfg.get("beta_2", 0.999)),
        epsilon=float(train_cfg.get("epsilon", 1e-7)),
    )

    loss_fn = multiclass_focal_loss(
        alpha=float(train_cfg.get("focal_alpha", 0.25)),
        gamma=float(train_cfg.get("focal_gamma", 2.0)),
    )

    model.compile(
        loss=loss_fn,
        optimizer=optimizer,
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="auc"),
            TopKCategoricalAccuracy(k=int(train_cfg.get("topk", 3)), name=f"top_{int(train_cfg.get('topk', 3))}_acc"),
        ],
    )


def build_callbacks(cfg: Dict[str, Any]) -> list:
    cb_cfg = cfg.get("callbacks", {})
    lr_cfg = cb_cfg.get("reduce_on_plateau", {})
    es_cfg = cb_cfg.get("early_stopping", {})

    lr_reduction = ReduceLROnPlateau(
        monitor=lr_cfg.get("monitor", "val_loss"),
        mode=lr_cfg.get("mode", "min"),
        patience=int(lr_cfg.get("patience", 4)),
        factor=float(lr_cfg.get("factor", 0.5)),
        min_lr=float(lr_cfg.get("min_lr", 1e-6)),
        cooldown=int(lr_cfg.get("cooldown", 2)),
        verbose=int(lr_cfg.get("verbose", 1)),
    )
    early_stop = EarlyStopping(
        monitor=es_cfg.get("monitor", "val_loss"),
        mode=es_cfg.get("mode", "min"),
        patience=int(es_cfg.get("patience", 8)),
        restore_best_weights=bool(es_cfg.get("restore_best_weights", True)),
        verbose=int(es_cfg.get("verbose", 1)),
    )
    return [lr_reduction, early_stop, TimeHistory()]


def train_classifier(model: tf.keras.Model, cfg: Dict[str, Any]):
    X_train, X_val, X_test, y_train, y_val, y_test = load_npy_splits(cfg)

    num_classes = int(cfg["model"]["num_classes"])
    y_train = to_one_hot_if_needed(y_train, num_classes)
    y_val   = to_one_hot_if_needed(y_val, num_classes)
    y_test  = to_one_hot_if_needed(y_test, num_classes)

    # class weights
    class_weights = compute_class_weights_from_one_hot(y_train)

    compile_model(model, cfg)
    callbacks = build_callbacks(cfg)

    hist = model.fit(
        X_train,
        y_train.astype("float32"),
        batch_size=int(cfg["train"].get("batch_size", 8)),
        epochs=int(cfg["train"].get("epochs", 100)),
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=int(cfg["train"].get("verbose", 1)),
    )

    return model, hist, (X_test, y_test)
