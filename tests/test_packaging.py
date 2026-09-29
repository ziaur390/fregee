"""Packaging and CI tests: Dockerfile, docker-compose, workflows.

These catch the class of failure that only shows up on the day you need it — a
`.dockerignore` that excludes a script the image needs, an image that runs as
root, a workflow that runs the same commands a human would but with a different
environment so it passes in CI and fails locally.
"""

from __future__ import annotations

import re

import pytest
import yaml

from mlserve.config import ROOT

DOCKERFILE = ROOT / "Dockerfile"
COMPOSE = ROOT / "docker-compose.yml"
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))


# ------------------------------------------------------------------ Dockerfile


def test_dockerfile_uses_multiple_stages() -> None:
    """The builder carries compilers; the runtime must not inherit them."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    stages = re.findall(r"^FROM .*? AS (\w+)", text, re.MULTILINE)
    assert len(stages) >= 2, f"expected a multi-stage build, found stages: {stages}"
    assert "builder" in stages and "runtime" in stages


def test_runtime_stage_switches_to_a_non_root_user() -> None:
    """An image that runs as root is the finding this project can actually control."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    runtime = text.split("AS runtime", 1)[1]
    assert re.search(r"^USER (?!root)\S+", runtime, re.MULTILINE), (
        "the runtime stage never drops privileges with USER"
    )
    assert "useradd" in runtime, "no service user is created"


def test_torch_is_installed_from_the_cpu_index() -> None:
    """The default PyPI torch wheel pulls ~2.5 GB of CUDA libraries this service
    never loads. That is the difference between a 400 MB image and a 3 GB one."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "download.pytorch.org/whl/cpu" in text


def test_healthchecks_live_in_compose_not_in_the_image() -> None:
    """The bug this test exists for, which actually happened.

    The Dockerfile carried a HEALTHCHECK hitting /healthz. Five services are built
    from that image and only one speaks HTTP, so init, drift and backup all
    backup all reported unhealthy and `docker compose up --wait` never returned.

    An image used for several roles cannot have a meaningful image-level
    healthcheck. It belongs per service, checking what that service does.
    """
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    # Match the directive, not the word. The file explains why there is no
    # image-level healthcheck, and a substring check flags that explanation.
    directives = [
        line for line in dockerfile.splitlines() if line.strip().upper().startswith("HEALTHCHECK")
    ]
    assert not directives, (
        f"the image carries a HEALTHCHECK directive, but it is used by non-HTTP roles: {directives}"
    )

    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    assert "healthcheck" in compose["services"]["api"], "the api has no healthcheck"
    api_check = str(compose["services"]["api"]["healthcheck"]["test"])
    assert "/healthz" in api_check, "the api healthcheck should hit the liveness endpoint"


def test_dockerfile_keeps_curl_for_container_healthchecks() -> None:
    """Container healthchecks use curl, so removing it from the runtime image would
    break every healthcheck in compose."""
    runtime = DOCKERFILE.read_text(encoding="utf-8").split("AS runtime", 1)[1]
    assert "curl" in runtime


def test_dockerfile_does_not_copy_the_whole_context() -> None:
    """A blanket COPY . . invalidates the dependency layer on every source edit and
    is how a .env ends up inside an image."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert not re.search(r"^COPY\s+\.\s+\.", text, re.MULTILINE), (
        "COPY . . is present; copy explicit paths so the layer cache survives edits"
    )


def test_dockerignore_excludes_generated_and_sensitive_paths() -> None:
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for pattern in (".git", "artifacts", "results", "backups", ".venv", "__pycache__"):
        assert pattern in ignore, f".dockerignore does not exclude {pattern}"


def test_dockerignore_does_not_exclude_what_the_image_needs() -> None:
    """The bug this test exists for: `monitoring/` was excluded while a compose
    service ran a script from it, so the service started and immediately died with
    'no such file'. Verify every path the image runs actually exists in the build
    context."""
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    ignored = {
        line.strip() for line in ignore.splitlines() if line.strip() and not line.startswith("#")
    }
    # Paths the Dockerfile COPYs must not be ignored.
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    copied = set(re.findall(r"^COPY\s+(?:--\S+\s+)?(\S+)", dockerfile, re.MULTILINE))
    for path in copied:
        assert path not in ignored, f"Dockerfile copies {path} but .dockerignore excludes it"


def test_dockerfile_sets_deterministic_thread_limits() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "OMP_NUM_THREADS" in text and "MLSERVE_TORCH_THREADS" in text, (
        "thread counts are not pinned, so latency becomes a function of host cores"
    )


def test_every_writable_path_env_var_is_set_and_chowned() -> None:
    """The bug this test exists for, which actually happened.

    The Dockerfile created and chowned /results but never exported MLSERVE_RESULTS,
    so config.py fell back to <repo>/results = /app/results. WORKDIR creates /app as
    root, so the non-root service user could not write to it and `docker compose up`
    failed with `PermissionError: '/app/results'`.

    It passed everywhere except in the container, because on a developer machine
    the repository directory happens to be writable. That asymmetry is exactly what
    makes it worth a test.
    """
    text = DOCKERFILE.read_text(encoding="utf-8")
    writable = {
        "MLSERVE_ARTIFACTS": "/artifacts",
        "MLSERVE_RESULTS": "/results",
        "MLSERVE_DATABASE_URL": "/data",
    }
    for var, directory in writable.items():
        assert var in text, (
            f"{var} is not set in the image, so the code falls back to a repo-relative "
            f"path that the non-root service user cannot create"
        )
        assert directory in text, f"{directory} is not mentioned in the Dockerfile"

    # And each directory must be made writable by the service user, not just created.
    assert re.search(r"chown[^\n]*mlserve[^\n]*(/data|/artifacts|/results)", text), (
        "the writable directories are not chowned to the service user"
    )


def _code_only(text: str) -> str:
    """Strip comments and docstrings, so a check matches code rather than prose.

    This exists because the first version of the test below matched the docstring
    that *describes* the bug it searches for, and so failed on the file that had
    already been fixed. A test that reads the comments is testing the comments.
    """
    lines: list[str] = []
    in_docstring = False
    delimiter = ""
    for line in text.splitlines():
        stripped = line.strip()
        if in_docstring:
            if delimiter in stripped:
                in_docstring = False
            continue
        if stripped.startswith(('"""', "'''")):
            delimiter = stripped[:3]
            # A one-line docstring opens and closes on the same line.
            if stripped.count(delimiter) < 2:
                in_docstring = True
            continue
        if stripped.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


def test_output_paths_resolve_under_the_overridable_results_directory() -> None:
    """The structural check for the class of bug that broke the container twice.

    The container failed because code joined an output path onto ``ROOT``, which
    inside the image is ``/app`` — created as root by WORKDIR, so the non-root
    service user could not create ``/app/results``. On a developer machine the same
    code works, which is why the tests passed and only the container failed.

    Every writable location must be derived from ``RESULTS`` or ``ARTIFACTS``,
    which are overridable by ``MLSERVE_RESULTS`` and ``MLSERVE_ARTIFACTS``.
    """
    from mlserve.config import ARTIFACTS, RESULTS
    from mlserve.drift.detect import resolve_output_path

    # Both spellings resolve into the same overridable directory.
    assert resolve_output_path("drift.json").parent == RESULTS
    assert resolve_output_path("results/drift.json").parent == RESULTS
    assert resolve_output_path("drift.json") == resolve_output_path("results/drift.json")

    # No module may build a result path by joining onto ROOT.
    #
    # config.py is excluded because it is where RESULTS is *defined*, and
    # `ROOT / "results"` there is the correct default for the environment override.
    # The rule is about code deriving output paths elsewhere.
    offenders: list[str] = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name == "config.py":
            continue
        code = _code_only(path.read_text(encoding="utf-8"))
        for pattern in ('ROOT / "results"', "ROOT / 'results'", 'ROOT / cfg["outputs"]'):
            if pattern in code:
                offenders.append(f"{path.relative_to(ROOT)} uses {pattern}")
    assert not offenders, (
        "these derive an output path from ROOT rather than the overridable RESULTS "
        "directory, so they will fail in a container:\n  " + "\n  ".join(offenders)
    )

    # And the directories the code writes to are the overridable ones.
    assert ARTIFACTS.name == "artifacts"
    assert RESULTS.name == "results"


def test_config_falls_back_to_writable_locations() -> None:
    """Every path the code writes to must be overridable by an environment
    variable, or a container cannot redirect it to somewhere writable.

    This is the requirement that the container failure was an instance of: the
    fallback was a repo-relative path, and inside the image that path is owned by
    root.
    """
    source = (ROOT / "src" / "mlserve" / "config.py").read_text(encoding="utf-8")
    code = _code_only(source)

    for var, fallback in (("MLSERVE_ARTIFACTS", "artifacts"), ("MLSERVE_RESULTS", "results")):
        assert var in code, f"config.py does not honour {var}"
        assert fallback in code, f"config.py has no fallback directory named {fallback}"
        assert f'os.environ.get("{var}"' in code, (
            f"{var} is mentioned but not actually read from the environment"
        )

    # And the values in the running process match those names.
    from mlserve.config import ARTIFACTS, RESULTS

    assert ARTIFACTS.name == "artifacts"
    assert RESULTS.name == "results"
    assert ARTIFACTS.is_absolute() and RESULTS.is_absolute(), (
        "the resolved paths must be absolute, or a container's working directory "
        "becomes part of the answer"
    )


# ------------------------------------------------------------------ compose


@pytest.fixture(scope="module")
def compose() -> dict:
    with COMPOSE.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_compose_parses_and_has_services(compose) -> None:
    assert compose["services"], "no services defined"


def test_api_waits_for_the_artefact_build(compose) -> None:
    """`init` runs train -> export -> reference. Without the completion gate the API
    races the model build and readiness flaps."""
    depends = compose["services"]["api"]["depends_on"]
    assert "init" in depends, "the api does not depend on the artefact build"
    assert depends["init"]["condition"] == "service_completed_successfully", (
        "the api only waits for init to *start*, not to finish"
    )


def test_init_does_not_restart_forever(compose) -> None:
    """A one-shot build job with restart: always becomes a rebuild loop."""
    assert compose["services"]["init"].get("restart") == "no"


def test_monitoring_depends_on_api_health(compose) -> None:
    for name in ("prometheus", "grafana"):
        depends = compose["services"][name]["depends_on"]
        assert any(cfg.get("condition") == "service_healthy" for cfg in depends.values()), (
            f"{name} does not wait for a healthy dependency"
        )


def test_every_service_that_others_wait_on_has_a_healthcheck(compose) -> None:
    """`service_healthy` against a service with no healthcheck is a compose error;
    against a service with a *wrong* healthcheck it is a silent hang."""
    waited_on = set()
    for service in compose["services"].values():
        for name, cfg in (service.get("depends_on") or {}).items():
            if isinstance(cfg, dict) and cfg.get("condition") == "service_healthy":
                waited_on.add(name)
    for name in waited_on:
        assert "healthcheck" in compose["services"][name], (
            f"{name} is waited on for health but defines no healthcheck"
        )


def test_compose_volumes_are_declared(compose) -> None:
    declared = set(compose.get("volumes") or {})
    used: set[str] = set()
    for service in compose["services"].values():
        for mount in service.get("volumes") or []:
            if isinstance(mount, str) and not mount.startswith((".", "/", "~")):
                used.add(mount.split(":")[0])
    missing = used - declared
    assert not missing, f"compose references undeclared named volumes: {sorted(missing)}"


def test_monitoring_config_is_mounted_read_only(compose) -> None:
    """A container that can rewrite its own alert rules can silence itself."""
    for name in ("prometheus", "alertmanager", "grafana"):
        for mount in compose["services"][name].get("volumes") or []:
            if isinstance(mount, str) and "./monitoring" in mount:
                assert mount.endswith(":ro"), f"{name} mounts {mount} read-write"


def test_alertmanager_points_at_the_alert_sink_service(compose) -> None:
    sink_hosts = set(compose["services"])
    alertmanager = (ROOT / "monitoring" / "alertmanager" / "alertmanager.yml").read_text("utf-8")
    url = re.search(r'url:\s*"http://([\w-]+):', alertmanager)
    assert url, "no webhook url found in alertmanager.yml"
    assert url.group(1) in sink_hosts, (
        f"alertmanager posts to {url.group(1)}, which is not a compose service"
    )


def test_alert_sink_runs_a_module_that_exists(compose) -> None:
    """The failure this catches already happened once: a script under monitoring/
    was excluded from the build context, so the service started and died."""
    command = compose["services"]["alert-sink"]["command"]
    joined = " ".join(command) if isinstance(command, list) else str(command)
    assert "alert_sink" in joined
    assert (ROOT / "src" / "mlserve" / "alert_sink.py").exists(), (
        "alert-sink runs a module that does not exist in the image"
    )
    # And it must not live somewhere .dockerignore removes.
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "src/mlserve" not in ignore


def test_grafana_dashboard_path_matches_the_mount(compose) -> None:
    """Grafana provisioning points at /var/lib/grafana/dashboards. If the compose
    mount disagrees, Grafana starts healthy with zero dashboards."""
    provisioning = yaml.safe_load(
        (
            ROOT / "monitoring" / "grafana" / "provisioning" / "dashboards" / "dashboards.yml"
        ).read_text("utf-8")
    )
    target = provisioning["providers"][0]["options"]["path"]
    mounts = compose["services"]["grafana"]["volumes"]
    assert any(str(m).split(":")[1].startswith(target) for m in mounts), (
        f"no compose mount targets {target}"
    )


# ------------------------------------------------------------------ workflows


def test_workflows_parse() -> None:
    assert WORKFLOWS, "no workflows found"
    for path in WORKFLOWS:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert data, f"{path} is empty"
        # `on` is parsed as the boolean True by YAML 1.1, which is why this
        # accesses it as a key of either type.
        triggers = data.get("on") or data.get(True)
        assert triggers, f"{path} has no trigger"


def test_every_workflow_job_runs_the_commands_locally_available_targets() -> None:
    """CI must not be the only place a command works.

    Every `python tasks.py <target>` invoked in a workflow must be a target that
    exists, so a typo fails here rather than after a push.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("tasks", ROOT / "tasks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    known = set(module.TARGETS)

    offenders: list[str] = []
    for path in WORKFLOWS:
        text = path.read_text(encoding="utf-8")
        for target in set(re.findall(r"tasks\.py\s+([a-z][\w-]*)", text)):
            if target not in known:
                offenders.append(f"{path.name}: tasks.py {target}")
    assert not offenders, "workflows invoke targets that do not exist:\n  " + "\n  ".join(offenders)


def test_bench_workflow_pins_thread_counts() -> None:
    """A benchmark that lets the runner choose its own thread count measures the
    runner, not the runtime."""
    data = yaml.safe_load((ROOT / ".github" / "workflows" / "bench.yml").read_text("utf-8"))
    env = data.get("env", {})
    assert env.get("OMP_NUM_THREADS") == "1"
    assert env.get("MLSERVE_TORCH_THREADS") == "1"
    assert env.get("PYTHONHASHSEED"), (
        "PYTHONHASHSEED is not pinned, so string hashing differs between runs"
    )


def test_ci_installs_torch_from_the_cpu_index() -> None:
    """Otherwise CI pulls 2.5 GB of CUDA wheels per matrix leg and times out."""
    for path in WORKFLOWS:
        text = path.read_text(encoding="utf-8")
        if "torch" in text:
            assert "download.pytorch.org/whl/cpu" in text, (
                f"{path.name} installs torch without the CPU index"
            )


def test_security_workflow_runs_on_a_schedule() -> None:
    """Dependency CVEs are published after commits. A scan that only runs on push
    reports a clean repository that has been vulnerable for weeks."""
    data = yaml.safe_load((ROOT / ".github" / "workflows" / "security.yml").read_text("utf-8"))
    triggers = data.get("on") or data.get(True)
    assert "schedule" in triggers, "the security workflow has no schedule trigger"


def test_workflows_have_concurrency_where_appropriate() -> None:
    """ci.yml runs on every push; without a concurrency group, a rapid series of
    pushes queues redundant runs against a shared runner pool."""
    data = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8"))
    assert "concurrency" in data
