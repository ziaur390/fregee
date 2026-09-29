"""Training entrypoint.

Run: ``python tasks.py train``  (or ``make train``)

Writes ``artifacts/model.pt`` and ``artifacts/train_metrics.json``. The metrics
file records the config hash, the seed and the per-split scores, so a result in
the report can always be traced back to the configuration that produced it.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import nn

from mlserve import model as model_mod
from mlserve.config import ARTIFACTS, CONFIG_DIR, ensure_dirs, load_config, set_seed, write_json
from mlserve.model import build_model, load_digits_split


def config_hash(*names: str) -> str:
    """Short digest of the config files that define an experiment."""
    digest = hashlib.sha256()
    for name in names:
        digest.update((CONFIG_DIR / f"{name}.yaml").read_bytes())
    return digest.hexdigest()[:12]


def _batches(x: np.ndarray, y: np.ndarray, batch_size: int, seed: int):
    """Shuffled mini-batches. Seeded so the batch order is reproducible."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        yield torch.from_numpy(x[idx]), torch.from_numpy(y[idx])


def _evaluate(net: nn.Module, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    net.eval()
    with torch.no_grad():
        logits = net(torch.from_numpy(x))
        preds = logits.argmax(dim=1).numpy()
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro")),
    }


def train_model(cfg: dict) -> tuple[nn.Module, dict]:
    """Train and return the model plus a metrics payload."""
    seed = int(cfg["seed"])
    set_seed(seed)

    split = load_digits_split(
        seed=seed,
        val_size=float(cfg["data"]["val_size"]),
        test_size=float(cfg["data"]["test_size"]),
    )
    net = build_model(cfg)
    train_cfg = cfg["train"]

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    best_state: dict | None = None
    best_val = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        net.train()
        running = 0.0
        seen = 0
        for xb, yb in _batches(
            split.x_train, split.y_train, int(train_cfg["batch_size"]), seed + epoch
        ):
            optimizer.zero_grad()
            loss = criterion(net(xb), yb)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(yb)
            seen += len(yb)

        val = _evaluate(net, split.x_val, split.y_val)
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1), **val})
        if val["accuracy"] > best_val:
            best_val = val["accuracy"]
            # deepcopy by value: state_dict tensors are views onto the live model.
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict(best_state)

    metrics = {
        "seed": seed,
        "config_hash": config_hash("model"),
        "parameters": model_mod.parameter_count(net),
        "split_sizes": split.sizes,
        "best_val_accuracy": best_val,
        "test": _evaluate(net, split.x_test, split.y_test),
        "history": history,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    return net, metrics


def main() -> int:
    cfg = load_config("model")
    ensure_dirs()

    net, metrics = train_model(cfg)

    weights = ARTIFACTS / "model.pt"
    torch.save(net.state_dict(), weights)
    write_json(ARTIFACTS / "train_metrics.json", metrics)

    test = metrics["test"]
    print(f"seed            {metrics['seed']}  (config {metrics['config_hash']})")
    print(f"parameters      {metrics['parameters']:,}")
    print(f"splits          {metrics['split_sizes']}")
    print(f"best val acc    {metrics['best_val_accuracy']:.4f}")
    print(f"test accuracy   {test['accuracy']:.4f}")
    print(f"test macro F1   {test['macro_f1']:.4f}")
    print(f"wrote           {weights.relative_to(ARTIFACTS.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
