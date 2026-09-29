"""The model under test.

Deliberately small. The point of this repository is the harness around a model,
not the model itself, so we use an 8x8 digit classifier that trains on a CPU in
seconds. That keeps the whole benchmark reproducible on a laptop and free to run
in CI.

Data: ``sklearn.datasets.load_digits`` ships with scikit-learn, so there is no
download and the benchmark works offline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

INPUT_SHAPE = (1, 8, 8)
NUM_CLASSES = 10


@dataclass(frozen=True)
class DataSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": int(len(self.y_train)),
            "val": int(len(self.y_val)),
            "test": int(len(self.y_test)),
        }


def load_digits_split(
    seed: int,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> DataSplit:
    """Load and split the digits dataset.

    The split is stratified and seeded, so the same seed always yields the same
    partition. Normalisation statistics come from the training split only -
    computing them over the full dataset would leak test information into the
    model and quietly inflate every reported number.
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    x = digits.images.astype(np.float32)[:, None, :, :]  # (N, 1, 8, 8)
    y = digits.target.astype(np.int64)

    # load_digits pixel range is 0-16.
    x = x / 16.0

    x_train, x_rest, y_train, y_rest = train_test_split(
        x, y, test_size=val_size + test_size, random_state=seed, stratify=y
    )
    # Split the remainder evenly into val and test.
    relative_test = test_size / (val_size + test_size)
    x_val, x_test, y_val, y_test = train_test_split(
        x_rest, y_rest, test_size=relative_test, random_state=seed, stratify=y_rest
    )
    return DataSplit(x_train, y_train, x_val, y_val, x_test, y_test)


class DigitCNN(nn.Module):
    """Two conv layers plus an MLP head.

    Kept simple so that ``torch.jit.script`` can compile it directly and so the
    ONNX graph is legible when you inspect it with netron.
    """

    def __init__(self, conv1: int = 16, conv2: int = 32, fc: int = 64, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, conv1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv1, conv2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # Two 3x3 convs with padding keep the 8x8 spatial size.
        flat = conv2 * INPUT_SHAPE[1] * INPUT_SHAPE[2]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flat, fc),
            nn.ReLU(),
            nn.Linear(fc, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.head(self.features(x))


def build_model(cfg: dict) -> DigitCNN:
    model_cfg = cfg.get("model", {})
    return DigitCNN(
        conv1=int(model_cfg.get("conv1", 16)),
        conv2=int(model_cfg.get("conv2", 32)),
        fc=int(model_cfg.get("fc", 64)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
