from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Lambda,
    Multiply,
    Reshape,
    SeparableConv2D,
)


def inception_residual_dilated_block(
    x: tf.Tensor,
    filters: int,
    reduction: int = 16,
    drop_rate: float = 0.2,
    scale: float = 0.1,
) -> tf.Tensor:
    """
    Residual Dilated Inception Block + Squeeze-and-Excitation (SE),
    aligned with the Keris-RD-Net notebook.

    Branches:
      (1x1, dil=1), (3x3, dil=2), (3x3, dil=3), (5x5, dil=4)
    """
    # pre-activation
    preact = BatchNormalization()(x)
    preact = Activation("relu")(preact)

    branches = []
    for kernel_size, rate in [(1, 1), (3, 2), (3, 3), (5, 4)]:
        b = SeparableConv2D(
            filters // 4,
            kernel_size,
            dilation_rate=rate,
            padding="same",
            use_bias=False,
        )(preact)
        b = BatchNormalization()(b)
        b = Activation("relu")(b)
        branches.append(b)

    merged = Concatenate()(branches)
    merged = Dropout(drop_rate)(merged)
    merged = Lambda(lambda z: z * scale)(merged)

    # skip connection (projection if needed)
    skip = x
    if x.shape[-1] != filters:
        skip = Conv2D(filters, 1, padding="same", use_bias=False)(preact)
        skip = BatchNormalization()(skip)

    r = Add()([skip, merged])

    # SE block
    se = GlobalAveragePooling2D()(r)
    se = Dense(filters // reduction, activation="relu", use_bias=False)(se)
    se = Dense(filters, activation="sigmoid", use_bias=False)(se)
    se = Reshape((1, 1, filters))(se)
    r = Multiply()([r, se])

    # final BN + ReLU
    r = BatchNormalization()(r)
    return Activation("relu")(r)


def build_keris_rdin(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    num_classes: int = 27,
    backbone_trainable: bool = False,
    block_filters: int = 768,
    block_repeats: int = 2,
    head_units: Tuple[int, int] = (512, 256),
    head_dropout: float = 0.5,
) -> Model:
    base = InceptionResNetV2(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base.trainable = backbone_trainable

    inp = Input(shape=input_shape)
    x = base(inp, training=False)
    for _ in range(block_repeats):
        x = inception_residual_dilated_block(
            x, block_filters, reduction=16, drop_rate=0.2, scale=0.1
        )
    x = GlobalAveragePooling2D()(x)

    for units in head_units:
        x = Dense(units, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(head_dropout)(x)

    out = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=inp, outputs=out, name="Keris-RDIN")
