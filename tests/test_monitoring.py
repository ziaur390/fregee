"""Monitoring configuration tests.

The important test here is :func:`test_every_alert_expression_references_a_real_metric`.

An alert rule naming a metric that does not exist is valid YAML, loads fine, and
never fires. That is the normal way alerting rots: someone renames a metric,
the dashboard is updated, the rule is forgotten, and the service loses its
monitoring silently. Parsing the expressions and checking every metric name
against the ones the code actually registers turns that into a test failure.

The dashboard JSON is checked the same way, plus a check that every panel queries
something - a panel with no target renders as a permanent "No data" box.
"""

from __future__ import annotations

import json
import re

import pytest
import yaml

from mlserve.config import ROOT

MONITORING = ROOT / "monitoring"
PROM_DIR = MONITORING / "prometheus"
GRAFANA_DIR = MONITORING / "grafana"

# Metric names look like mlserve_something. Deliberately excludes PromQL functions
# (rate, sum, histogram_quantile) because the regex requires the mlserve_ prefix.
METRIC_RE = re.compile(r"\b(mlserve_[a-z0-9_]+)\b")


def app_metric_names() -> set[str]:
    """Every metric name the FastAPI app registers."""
    from prometheus_client import REGISTRY

    from mlserve.app import app  # noqa: F401  (import registers the collectors)

    names: set[str] = set()
    for collector in REGISTRY.collect():
        names.add(collector.name)
        # Histogram and Summary expose derived series with suffixes.
        for metric in collector.samples:
            names.add(metric.name)
    return {name for name in names if name.startswith("mlserve_")}


def drift_metric_names() -> set[str]:
    """Every metric name the drift job can write to its textfile."""
    import numpy as np

    from mlserve.config import load_config
    from mlserve.drift.detect import DriftReport

    report = DriftReport(
        status="stable",
        n_requests=0,
        n_features=1,
        n_informative=1,
        n_uninformative=0,
        uninformative_features=[],
        bins=1,
        max_psi=0.0,
        mean_psi=0.0,
        n_watched=0,
        n_alerted=0,
        fraction_watched=0.0,
        features=[],
        thresholds={},
        pruned_rows=0,
        reference_samples=0,
    )
    names = set(METRIC_RE.findall(report.to_prometheus()))
    assert names, "drift job exposes no metrics"
    # These are written once by the job and are not part of DriftReport's output.
    names |= {"mlserve_drift_last_run_unixtime"}
    _ = load_config, np
    return names


@pytest.fixture(scope="module")
def known_metrics() -> set[str]:
    return app_metric_names() | drift_metric_names()


@pytest.fixture(scope="module")
def alerts_config() -> dict:
    with (PROM_DIR / "alerts.yml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _all_expressions(alerts: dict) -> list[tuple[str, str]]:
    """(rule_name, expression) for every alert and recording rule."""
    found: list[tuple[str, str]] = []
    for group in alerts.get("groups", []):
        for rule in group.get("rules", []):
            label = rule.get("alert") or rule.get("record") or "unnamed"
            if "expr" in rule:
                found.append((label, str(rule["expr"])))
    return found


# ------------------------------------------------------------------ structure


def test_alerts_file_is_valid_yaml(alerts_config) -> None:
    assert "groups" in alerts_config
    assert alerts_config["groups"], "no rule groups defined"


def test_alerts_group_have_names(alerts_config) -> None:
    for group in alerts_config["groups"]:
        assert group.get("name"), "a rule group has no name"
        assert group.get("rules"), f"group {group['name']} has no rules"


def test_every_alert_has_a_severity(alerts_config) -> None:
    """Without a severity the alertmanager route cannot match it, so it is routed
    to the default receiver and its urgency is invisible."""
    for group in alerts_config["groups"]:
        for rule in group.get("rules", []):
            if "alert" not in rule:
                continue
            labels = rule.get("labels", {})
            assert labels.get("severity") in {"critical", "warning", "info"}, (
                f"{rule['alert']} has no usable severity: {labels.get('severity')!r}"
            )


def test_every_alert_has_annotations(alerts_config) -> None:
    """An alert with no summary is a page with no explanation."""
    for group in alerts_config["groups"]:
        for rule in group.get("rules", []):
            if "alert" not in rule:
                continue
            annotations = rule.get("annotations", {})
            assert annotations.get("summary"), f"{rule['alert']} has no summary"
            assert annotations.get("description"), f"{rule['alert']} has no description"


def test_alerts_are_unique(alerts_config) -> None:
    names = [
        rule["alert"]
        for group in alerts_config["groups"]
        for rule in group.get("rules", [])
        if "alert" in rule
    ]
    assert len(names) == len(set(names)), f"duplicate alert names: {names}"


def test_threshold_alerts_have_a_for_clause(alerts_config) -> None:
    """A threshold comparison without `for` fires on a single scrape.

    Informational alerts and zero-threshold detectors are allowed to be
    immediate; anything comparing against a latency or rate threshold is not.
    """
    immediate_ok = {
        "MLServeModelNotLoaded",
        "MLServeDriftDetected",
        "MLServeInferenceErrors",
        "MLServeDbWriteFailures",
    }
    for group in alerts_config["groups"]:
        for rule in group.get("rules", []):
            if "alert" not in rule or rule["alert"] in immediate_ok:
                continue
            assert "for" in rule, (
                f"{rule['alert']} has no `for` clause and will fire on one bad scrape"
            )


# ------------------------------------------------------------------ the real check


def test_every_alert_expression_references_a_real_metric(
    alerts_config, known_metrics: set[str]
) -> None:
    """Every metric named in a rule must be one the code actually exposes.

    This is the test that stops alerting from rotting. It has already earned its
    place: the drift metrics are written by a separate job to a textfile and
    scraped from a different target, so a typo there would never surface as an
    error anywhere else.
    """
    unknown: list[str] = []
    for name, expression in _all_expressions(alerts_config):
        for metric in set(METRIC_RE.findall(expression)):
            # Histogram/summary derived series: strip the suffix and check the base.
            base = metric
            for suffix in ("_bucket", "_sum", "_count", "_created", "_total"):
                if metric.endswith(suffix):
                    base = metric[: -len(suffix)]
                    break
            if metric in known_metrics or base in known_metrics or f"{base}_total" in known_metrics:
                continue
            unknown.append(f"{name}: {metric}")
    assert not unknown, "alert rules reference metrics that do not exist:\n  " + "\n  ".join(
        unknown
    )


def test_prometheus_scrapes_the_api(alerts_config) -> None:
    with (PROM_DIR / "prometheus.yml").open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}
    assert "mlserve-api" in jobs
    assert jobs["mlserve-api"]["metrics_path"] == "/metrics"


def test_prometheus_loads_the_alert_rules() -> None:
    with (PROM_DIR / "prometheus.yml").open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    assert config["rule_files"], "no rule_files configured, so alerts.yml is never loaded"
    assert any("alerts.yml" in str(path) for path in config["rule_files"])


def test_prometheus_points_at_alertmanager() -> None:
    with (PROM_DIR / "prometheus.yml").open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    targets = config["alerting"]["alertmanagers"][0]["static_configs"][0]["targets"]
    assert targets == ["alertmanager:9093"]


def test_alertmanager_has_a_receiver_and_routes() -> None:
    with (MONITORING / "alertmanager" / "alertmanager.yml").open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    receivers = {r["name"] for r in config["receivers"]}
    assert config["route"]["receiver"] in receivers
    for route in config["route"].get("routes", []):
        assert route["receiver"] in receivers


def test_alertmanager_inhibits_symptoms_of_a_dead_model() -> None:
    """The inhibition rule that stops one root cause producing five alerts."""
    with (MONITORING / "alertmanager" / "alertmanager.yml").open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    rules = config.get("inhibit_rules", [])
    sources = [str(rule.get("source_matchers", "")) for rule in rules]
    assert any("MLServeModelNotLoaded" in source for source in sources), (
        "a model-load failure will also trip latency and error alerts with nothing to suppress the noise"
    )


# ------------------------------------------------------------------ grafana


def test_dashboard_is_valid_json() -> None:
    payload = json.loads((GRAFANA_DIR / "dashboards" / "mlserve.json").read_text(encoding="utf-8"))
    assert payload["title"]
    assert payload["panels"]


def test_every_dashboard_panel_has_targets() -> None:
    """A panel with no query renders a permanent 'No data' box."""
    payload = json.loads((GRAFANA_DIR / "dashboards" / "mlserve.json").read_text(encoding="utf-8"))
    for panel in payload["panels"]:
        assert panel.get("targets"), (
            f"panel {panel.get('id')} ({panel.get('title')}) has no targets"
        )
        for target in panel["targets"]:
            assert target.get("expr"), f"panel {panel.get('id')} has a target with no expr"


def test_every_dashboard_panel_has_a_unique_id_and_grid_position() -> None:
    payload = json.loads((GRAFANA_DIR / "dashboards" / "mlserve.json").read_text(encoding="utf-8"))
    ids = [panel["id"] for panel in payload["panels"]]
    assert len(ids) == len(set(ids)), f"duplicate panel ids: {ids}"
    for panel in payload["panels"]:
        grid = panel.get("gridPos", {})
        assert {"h", "w", "x", "y"} <= set(grid), f"panel {panel['id']} has an incomplete gridPos"
        assert grid["x"] + grid["w"] <= 24, f"panel {panel['id']} overflows the 24-column grid"


def test_dashboard_queries_only_real_metrics(known_metrics: set[str]) -> None:
    payload = json.loads((GRAFANA_DIR / "dashboards" / "mlserve.json").read_text(encoding="utf-8"))
    unknown: list[str] = []
    for panel in payload["panels"]:
        for target in panel["targets"]:
            for metric in set(METRIC_RE.findall(target["expr"])):
                base = metric
                for suffix in ("_bucket", "_sum", "_count", "_created", "_total"):
                    if metric.endswith(suffix):
                        base = metric[: -len(suffix)]
                        break
                if metric in known_metrics or base in known_metrics:
                    continue
                unknown.append(f"panel {panel['id']}: {metric}")
    assert not unknown, "dashboard queries metrics that do not exist:\n  " + "\n  ".join(unknown)


def test_grafana_provisioning_points_at_the_dashboard_folder() -> None:
    path = GRAFANA_DIR / "provisioning" / "dashboards" / "dashboards.yml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    options = config["providers"][0]["options"]
    assert options["path"] == "/var/lib/grafana/dashboards"


def test_grafana_datasource_targets_the_prometheus_service() -> None:
    path = GRAFANA_DIR / "provisioning" / "datasources" / "prometheus.yml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["datasources"][0]["url"] == "http://prometheus:9090"
    assert config["datasources"][0]["isDefault"] is True
