"""Export one trained model into three serving runtimes.

Run: ``python tasks.py export``  (or ``make export``)

Artifacts written to ``artifacts/``:
  model.ts         TorchScript, from ``torch.jit.script``
  model.onnx       ONNX opset 17, fp32, dynamic batch axis
  model.int8.onnx  ONNX int8, weights quantised with onnxruntime dynamic quantisation

Two things make this an experiment rather than a build step:

1.  All three runtimes come from the *same* ``model.pt``. A parity check then
    asserts the three agree on the test set, so a later latency difference can
    be attributed to the runtime and not to a divergent model.
2.  The parity tolerance is a real assertion, not a warning. A silently broken
    export is exactly the failure mode that would make the whole benchmark
    report nonsense, so it fails the build instead of printing a footnote.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from mlserve import registry
from mlserve.config import (
    ARTIFACTS,
    RUNTIME_BY_NAME,
    ensure_dirs,
    load_config,
    set_seed,
    write_json,
)
from mlserve.model import INPUT_SHAPE, build_model, load_digits_split
from mlserve.train import config_hash

#: Max acceptable share of test samples where a runtime disagrees with PyTorch.
#: fp32 ONNX should be exact; int8 is allowed a little room because quantisation
#: is lossy by construction. If either is exceeded, the export failed.
PARITY_TOLERANCE = {"torchscript": 0.0, "onnx-fp32": 0.0, "onnx-int8": 0.02}


def load_trained_model(cfg: dict) -> torch.nn.Module:
    net = build_model(cfg)
    weights = ARTIFACTS / "model.pt"
    if not weights.exists():
        raise FileNotFoundError(
            f"{weights} is missing. Run `python tasks.py train` (or `make train`) first."
        )
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    return net


def export_torchscript(net: torch.nn.Module, path: Path) -> None:
    scripted = torch.jit.script(net)
    # Round-trip once so a broken script fails here rather than at serve time.
    scripted.save(str(path))
    torch.jit.load(str(path)).eval()


def export_onnx(net: torch.nn.Module, path: Path, opset: int = 17) -> None:
    dummy = torch.zeros(1, *INPUT_SHAPE, dtype=torch.float32)
    torch.onnx.export(
        net,
        (dummy,),
        str(path),
        input_names=["input"],
        output_names=["logits"],
        # Dynamic batch axis: without this the graph is frozen at batch 1 and
        # the batch-size sweep in the benchmark would be measuring nothing.
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(str(path))


def export_int8(src: Path, dst: Path) -> dict[str, object]:
    """Dynamic (weight-only) int8 quantisation, validated by actually running it.

    Two things this function learned the hard way:

    1.  ``quantize_dynamic`` happily emits ``ConvInteger`` nodes when you ask it
        to quantise ``Conv``, and onnxruntime's CPU provider then refuses to
        load the file. The quantiser succeeding is therefore *not* evidence the
        artifact works, so every candidate is loaded and executed here and the
        result is discarded if it does not run.
    2.  That means the int8 model in this repo quantises ``MatMul``/``Gemm``
        only. Convolution weights stay fp32, so the size reduction is smaller
        than a static-quantisation pipeline would give. That is recorded in the
        metadata and called out in the report rather than glossed over.
    """
    dummy = np.zeros((1, *INPUT_SHAPE), dtype=np.float32)
    attempts: tuple[dict[str, object], ...] = (
        {"label": "matmul+gemm+conv", "op_types_to_quantize": ["MatMul", "Gemm", "Conv"]},
        {"label": "matmul+gemm", "op_types_to_quantize": ["MatMul", "Gemm"]},
    )

    errors: list[str] = []
    chosen: dict[str, object] | None = None

    for attempt in attempts:
        candidates = {
            "label": attempt["label"],
            "op_types_to_quantize": attempt["op_types_to_quantize"],
        }
        try:
            quantize_dynamic(
                model_input=str(src),
                model_output=str(dst),
                weight_type=QuantType.QInt8,
                op_types_to_quantize=attempt["op_types_to_quantize"],
            )
        except Exception as exc:  # noqa: BLE001 - record and try the next candidate
            errors.append(f"{attempt['label']}: quantise failed: {exc}")
            continue

        # The load-and-run check is the whole point of this loop.
        try:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                str(dst), sess_options=options, providers=["CPUExecutionProvider"]
            )
            out = session.run(None, {session.get_inputs()[0].name: dummy})[0]
            if not np.isfinite(out).all():
                raise ValueError("non-finite output from quantised model")
        except Exception as exc:  # noqa: BLE001 - record and try the next candidate
            errors.append(
                f"{attempt['label']}: quantised model unusable: {type(exc).__name__}: {exc}"
            )
            dst.unlink(missing_ok=True)
            continue

        chosen = candidates
        break

    if chosen is None:
        raise RuntimeError("int8 quantisation produced no usable model:\n  " + "\n  ".join(errors))

    graph = onnx.load(str(dst)).graph
    dst_ops = {node.op_type for node in graph.node}
    src_ops = {node.op_type for node in onnx.load(str(src)).graph.node}
    quant_ops = sorted(op for op in dst_ops if "QuantizeLinear" in op or "DequantizeLinear" in op)

    meta: dict[str, object] = {
        **chosen,
        "weight_type": "QInt8",
        "mode": "dynamic (weight-only)",
        "source_ops": sorted(src_ops),
        "quantised_graph_ops": quant_ops,
        "rejected_attempts": errors,
    }
    (ARTIFACTS / "model.int8.meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def parity_report(cfg: dict, runtimes: list[str]) -> dict[str, dict[str, float]]:
    """Measure agreement between each runtime and PyTorch on the test split."""
    from sklearn.metrics import accuracy_score, f1_score

    seed = int(cfg["seed"])
    set_seed(seed)
    split = load_digits_split(
        seed=seed,
        val_size=float(cfg["data"]["val_size"]),
        test_size=float(cfg["data"]["test_size"]),
    )

    registry.clear_cache()
    reference = registry.load("torchscript").predict(split.x_test)
    reference_labels = reference.argmax(axis=1)

    report: dict[str, dict[str, float]] = {}
    for name in runtimes:
        started = time.perf_counter()
        logits = registry.load(name).predict(split.x_test)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        labels = logits.argmax(axis=1)
        disagreement = float(np.mean(labels != reference_labels))
        report[name] = {
            "test_accuracy": float(accuracy_score(split.y_test, labels)),
            "test_macro_f1": float(f1_score(split.y_test, labels, average="macro")),
            "disagreement_vs_torchscript": disagreement,
            "inference_ms_total": round(elapsed_ms, 3),
            "size_bytes": registry.artifact_size_bytes(name),
            "load_time_ms": round(registry.load_time_ms(name), 3),
        }
    return report


def main() -> int:
    cfg = load_config("model")
    ensure_dirs()
    set_seed(int(cfg["seed"]))

    net = load_trained_model(cfg)

    ts_path = ARTIFACTS / RUNTIME_BY_NAME["torchscript"].file
    onnx_path = ARTIFACTS / RUNTIME_BY_NAME["onnx-fp32"].file
    int8_path = ARTIFACTS / RUNTIME_BY_NAME["onnx-int8"].file

    export_torchscript(net, ts_path)
    print(f"wrote {ts_path.name}")
    export_onnx(net, onnx_path)
    print(f"wrote {onnx_path.name}")
    export_int8(onnx_path, int8_path)
    meta = json.loads((ARTIFACTS / "model.int8.meta.json").read_text(encoding="utf-8"))
    print(f"wrote {int8_path.name}")
    print(f"      quantised ops: {meta['quantised_graph_ops']}  (mode: {meta['mode']})")
    for rejected in meta["rejected_attempts"]:
        print(f"      rejected: {rejected}")

    runtimes = list(RUNTIME_BY_NAME)
    parity = parity_report(cfg, runtimes)

    failures: list[str] = []
    print()
    print(f"{'runtime':<14} {'accuracy':>9} {'disagree':>9} {'size KiB':>10} {'load ms':>9}")
    for name in runtimes:
        row = parity[name]
        print(
            f"{name:<14} {row['test_accuracy']:>9.4f} "
            f"{row['disagreement_vs_torchscript']:>9.4f} "
            f"{row['size_bytes'] / 1024:>10.1f} "
            f"{row['load_time_ms']:>9.2f}"
        )
        if row["disagreement_vs_torchscript"] > PARITY_TOLERANCE[name]:
            failures.append(
                f"{name} disagrees with torchscript on "
                f"{row['disagreement_vs_torchscript']:.2%} of samples "
                f"(tolerance {PARITY_TOLERANCE[name]:.2%})"
            )

    write_json(
        ARTIFACTS / "export_metrics.json",
        {
            "config_hash": config_hash("model"),
            "seed": cfg["seed"],
            "parity_tolerance": PARITY_TOLERANCE,
            "runtimes": parity,
        },
    )

    if failures:
        print()
        for line in failures:
            print(f"PARITY FAILURE: {line}")
        return 1

    print()
    print("parity OK - all runtimes agree within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
