import os
import random
from typing import Optional

import numpy as np


def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # TensorFlow seed (optional)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Best-effort deterministic ops; may impact performance
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    except Exception:
        pass
