"""Configuration loading and reproducibility helpers.

Every module reads its settings from a YAML file under ``configs/`` and every
module seeds the RNGs through :func:`set_seed` before doing anything random.
That pairing is what makes a rerun reproduce a previous run.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# src/mlserve/config.py -> repo root is three levels up.
ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
ARTIFACTS = Path(os.environ.get("MLSERVE_ARTIFACTS", ROOT / "artifacts"))
RESULTS = Path(os.environ.get("MLSERVE_RESULTS", ROOT / "results"))


def load_config(name: str) -> dict[str, Any]:
    """Load ``configs/<name>.yaml``.

    ``name`` may be given with or without the ``.yaml`` suffix.
    """
    stem = name[:-5] if name.endswith(".yaml") else name
    path = CONFIG_DIR / f"{stem}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing config: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def set_seed(seed: int) -> None:
    """Seed every RNG that can affect a result.

    ``PYTHONHASHSEED`` cannot be set after interpreter start, so it is only
    effective when exported before launch - the Dockerfile and CI do that.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Single-threaded keeps timings comparable across runs; the benchmark is
        # measuring runtime overhead, not our ability to saturate the CPU.
        torch.set_num_threads(int(os.environ.get("MLSERVE_TORCH_THREADS", "1")))
    except ImportError:  # torch is optional for pure drift/backup work
        pass


def ensure_dirs() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


@dataclass(frozen=True)
class RuntimeSpec:
    """One serving runtime under test.

    ``kind`` selects the loader in :mod:`mlserve.registry`; ``file`` is resolved
    against the artifacts directory.
    """

    name: str
    kind: str
    file: str


#: The three runtimes the benchmark compares. All three are produced from the
#: same trained weights, which is what makes the comparison fair.
RUNTIMES: tuple[RuntimeSpec, ...] = (
    RuntimeSpec("torchscript", "torchscript", "model.ts"),
    RuntimeSpec("onnx-fp32", "onnx", "model.onnx"),
    RuntimeSpec("onnx-int8", "onnx", "model.int8.onnx"),
)
RUNTIME_BY_NAME = {spec.name: spec for spec in RUNTIMES}
