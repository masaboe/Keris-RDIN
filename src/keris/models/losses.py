from __future__ import annotations

import tensorflow as tf


def multiclass_focal_loss(alpha: float = 0.25, gamma: float = 2.0):
    """
    Manual multiclass focal loss (one-hot y_true), as used in the notebook.
    """
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        fl = weight * cross_entropy
        return tf.reduce_sum(fl, axis=-1)

    return loss_fn
