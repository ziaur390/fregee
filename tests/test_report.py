"""Report generator tests.

The important one here is the interpretation-section behaviour. `results/REPORT.md`
is generated output, so a human who writes their conclusions into it loses them on
the next `tasks.py report` run. That is a trap, and the point of these tests is that
the escape hatch from it actually works.
"""

from __future__ import annotations

import csv
import importlib

import pytest

from mlserve.bench import report as report_mod
from mlserve.config import ROOT

#: A tiny but structurally valid raw.csv, so the report can be built without
#: running the full benchmark.
FAKE_ROWS = [
    {
        "mode": mode,
        "runtime": runtime,
        "batch_size": str(batch),
        "concurrency": "1",
        "repeat": "0",
        "call_index": str(i),
        "n_samples": str(batch),
        "latency_ms": f"{1.0 + i * 0.1:.4f}",
        "samples_per_second": f"{batch / ((1.0 + i * 0.1) / 1000):.2f}",
    }
    for mode in ("inproc", "http")
    for runtime in ("torchscript", "onnx-fp32", "onnx-int8")
    for batch in (1, 64)
    for i in range(5)
]


@pytest.fixture
def with_raw_csv(tmp_path, monkeypatch):
    """Point RESULTS at a temp dir containing a minimal raw.csv."""
    results = tmp_path / "results"
    results.mkdir()
    with (results / "raw.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FAKE_ROWS[0]))
        writer.writeheader()
        writer.writerows(FAKE_ROWS)
    monkeypatch.setattr(report_mod, "RESULTS", results)
    return results


# ------------------------------------------------------------------ interpretation


def test_placeholder_is_used_when_no_interpretation_exists(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(report_mod, "INTERPRETATION_SOURCE", tmp_path / "missing.md")
    text, authored = report_mod.interpretation_section()
    assert authored is False
    assert "empty on purpose" in text
    assert "docs/interpretation.md" in text, "the placeholder must say where to write it"
    assert "Do **not** edit `results/REPORT.md`" in text, (
        "the placeholder must warn about the regeneration footgun"
    )


def test_placeholder_is_used_while_todo_markers_remain(tmp_path, monkeypatch) -> None:
    """A half-finished template must not be inlined as if it were an answer."""
    source = tmp_path / "interpretation.md"
    source.write_text("Some prose.\n\n<!-- TODO: finish this -->\n", encoding="utf-8")
    monkeypatch.setattr(report_mod, "INTERPRETATION_SOURCE", source)

    text, authored = report_mod.interpretation_section()
    assert authored is False
    assert text == report_mod.INTERPRETATION_PLACEHOLDER


def test_authored_interpretation_is_inlined(tmp_path, monkeypatch) -> None:
    source = tmp_path / "interpretation.md"
    source.write_text(
        "## Which runtime\n\nI would deploy onnx-int8 because traffic arrives at batch 1.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(report_mod, "INTERPRETATION_SOURCE", source)

    text, authored = report_mod.interpretation_section()
    assert authored is True
    assert "I would deploy onnx-int8" in text
    assert text.startswith("## Interpretation")


def test_empty_interpretation_file_falls_back(tmp_path, monkeypatch) -> None:
    """An empty file is not an answer. Falling through to the placeholder keeps the
    report honest instead of emitting a bare heading."""
    source = tmp_path / "interpretation.md"
    source.write_text("   \n\n", encoding="utf-8")
    monkeypatch.setattr(report_mod, "INTERPRETATION_SOURCE", source)

    _, authored = report_mod.interpretation_section()
    assert authored is False


def test_the_shipped_template_has_either_todos_or_a_draft_marker() -> None:
    """Guards against a shipped interpretation that looks authoritative.

    Two acceptable states, and one that is not:

    * the answer prompts are still there (a TODO marker), so the report falls back
      to the placeholder; or
    * a human has written real answers.

    The unacceptable state is an interpretation that reads as finished while
    actually being generated text. `docs/interpretation.md` therefore has to either
    contain a TODO marker or declare itself a draft, so the report never presents
    unreviewed prose as a conclusion.
    """
    shipped = ROOT / "docs" / "interpretation.md"
    assert shipped.exists(), "the interpretation template is missing"
    text = shipped.read_text(encoding="utf-8")
    has_todo = "TODO" in text
    declares_draft = "DRAFT" in text.upper()
    assert has_todo or declares_draft, (
        "docs/interpretation.md reads as a finished conclusion. Either leave a TODO "
        "marker, or state clearly that it is a draft awaiting the author's own words."
    )


# ------------------------------------------------------------------ report build


def test_report_builds_from_a_raw_csv(with_raw_csv, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(report_mod, "INTERPRETATION_SOURCE", tmp_path / "missing.md")
    out = report_mod.build_report()
    assert out.exists()
    body = out.read_text(encoding="utf-8")
    assert "# Serving Runtime Benchmark" in body
    assert "## Limitations" in body
    assert "## Interpretation" in body


def test_report_contains_the_latency_table(with_raw_csv, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(report_mod, "INTERPRETATION_SOURCE", tmp_path / "missing.md")
    body = report_mod.build_report().read_text(encoding="utf-8")
    assert "| mode | runtime | batch | conc | calls |" in body
    # Every runtime appears as a row.
    for runtime in ("torchscript", "onnx-fp32", "onnx-int8"):
        assert runtime in body


def test_report_reports_tail_power_per_percentile(with_raw_csv, monkeypatch, tmp_path) -> None:
    """With 5 observations per cell, nothing beyond the p50 is powered, and the
    report must say so rather than printing a number for the p99."""
    monkeypatch.setattr(report_mod, "INTERPRETATION_SOURCE", tmp_path / "missing.md")
    body = report_mod.build_report().read_text(encoding="utf-8")
    assert "### Tail power" in body
    assert "NOT powered" in body
    assert "p99" in body


def test_report_includes_the_config_hash(with_raw_csv, monkeypatch, tmp_path) -> None:
    """Without it the report cannot be traced to the experiment that produced it."""
    monkeypatch.setattr(report_mod, "INTERPRETATION_SOURCE", tmp_path / "missing.md")
    body = report_mod.build_report().read_text(encoding="utf-8")
    assert "Config hash" in body
    assert body.rstrip().count("`") >= 2


def test_missing_raw_csv_raises_a_helpful_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(report_mod, "RESULTS", tmp_path / "nothing")
    with pytest.raises(FileNotFoundError, match="tasks.py bench"):
        report_mod.build_report()


def test_results_directory_is_the_generated_output_location() -> None:
    """Guards against the report being written beside the source instead."""
    assert report_mod.RESULTS.name == "results"
    assert report_mod.INTERPRETATION_SOURCE.parent.name == "docs", (
        "the interpretation source must live outside results/, which is regenerated"
    )


def _reload() -> None:
    importlib.reload(report_mod)
