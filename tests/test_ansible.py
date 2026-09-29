"""Ansible playbook tests.

These check the two properties that make a playbook usable rather than merely
correct once:

1. **Idempotency.** A `command` or `shell` task with no guard runs on every
   invocation and reports changed every time. That breaks the property the
   playbook is built on — a second run must report zero changes, otherwise it
   cannot be used to recover from a partial failure.

2. **Handler references resolve.** A `notify` naming a handler that does not exist
   is not an error in Ansible. It is a silent no-op, and the service quietly keeps
   running the old configuration after a successful-looking deploy.

Both are checked by parsing the YAML, so they run without Ansible installed.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from mlserve.config import ROOT

ANSIBLE = ROOT / "ansible"
PLAYBOOK = ANSIBLE / "playbook.yml"

# Modules that are idempotent by declaration. Everything else that runs a command
# needs an explicit guard.
GUARD_MODULES = {
    "ansible.builtin.apt",
    "ansible.builtin.apt_repository",
    "ansible.builtin.copy",
    "ansible.builtin.template",
    "ansible.builtin.file",
    "ansible.builtin.git",
    "ansible.builtin.group",
    "ansible.builtin.user",
    "ansible.builtin.pip",
    "ansible.builtin.get_url",
    "ansible.builtin.lineinfile",
    "ansible.builtin.uri",
    "ansible.builtin.service",
    "ansible.builtin.systemd_service",
    "ansible.builtin.systemd",
    "ansible.builtin.timezone",
    "ansible.builtin.debug",
    "ansible.builtin.set_fact",
    "ansible.builtin.include_tasks",
    "ansible.builtin.import_tasks",
    "ansible.posix.sysctl",
    "ansible.posix.authorized_key",
    "community.general.ufw",
}

#: Guards that make a command/shell task idempotent or at least honestly reported.
IDEMPOTENCY_GUARDS = {"creates", "removes", "changed_when", "when", "failed_when"}


def task_files() -> list[Path]:
    return sorted(ANSIBLE.rglob("tasks/*.yml")) + sorted(ANSIBLE.rglob("tasks/*.yaml"))


def handler_files() -> list[Path]:
    return sorted(ANSIBLE.rglob("handlers/*.yml")) + sorted(ANSIBLE.rglob("handlers/*.yaml"))


def load_tasks(path: Path) -> list[dict]:
    """Flatten a task file into a list of task dicts (top-level only)."""
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return []
    if not isinstance(data, list):
        raise AssertionError(f"{path} should contain a list of tasks")
    return [task for task in data if isinstance(task, dict)]


def module_name(task: dict) -> str | None:
    return next(
        (
            key
            for key in task
            if key
            not in {
                "name",
                "when",
                "tags",
                "register",
                "notify",
                "listen",
                "become",
                "environment",
                "args",
                "loop",
                "loop_control",
                "until",
                "retries",
                "delay",
                "changed_when",
                "failed_when",
                "no_log",
                "vars",
                "block",
                "rescue",
                "always",
                "delegate_to",
                "run_once",
                "ignore_errors",
                "check_mode",
                "with_items",
                "set_fact",
            }
        ),
        None,
    )


# ------------------------------------------------------------------ structure


def test_playbook_exists_and_parses() -> None:
    with PLAYBOOK.open(encoding="utf-8") as fh:
        plays = yaml.safe_load(fh)
    assert isinstance(plays, list) and plays, "playbook.yml should contain at least one play"
    play = plays[0]
    assert play["hosts"], "the play targets no hosts"
    assert play.get("become") is True, "provisioning tasks need privilege escalation"


def test_every_role_in_the_playbook_exists() -> None:
    with PLAYBOOK.open(encoding="utf-8") as fh:
        play = yaml.safe_load(fh)[0]
    roles = [entry["role"] if isinstance(entry, dict) else entry for entry in play.get("roles", [])]
    assert roles, "the playbook declares no roles"
    for role in roles:
        role_dir = ANSIBLE / "roles" / role
        assert role_dir.is_dir(), f"role {role} has no directory"
        assert (role_dir / "tasks" / "main.yml").exists(), (
            f"role {role} has no tasks/main.yml - Ansible would silently do nothing"
        )


def test_every_ansible_yaml_file_parses() -> None:
    files = list(ANSIBLE.rglob("*.yml")) + list(ANSIBLE.rglob("*.yaml"))
    assert files, "no Ansible YAML found"
    for path in files:
        with path.open(encoding="utf-8") as fh:
            list(yaml.safe_load_all(fh))


# ------------------------------------------------------------------ idempotency


def test_task_files_are_non_empty() -> None:
    files = task_files()
    assert files, "no role task files found"
    for path in files:
        tasks = load_tasks(path)
        assert tasks, f"{path} defines no tasks"


def test_command_and_shell_tasks_have_an_idempotency_guard() -> None:
    """The property that makes the playbook re-runnable.

    A bare `command:` runs every time and reports changed every time, so the
    second run is indistinguishable from the first and a partial failure cannot be
    recovered by re-running. Every command/shell task here carries a `creates:`,
    or an explicit `changed_when:`/`when:` that states why it is safe.
    """
    offenders: list[str] = []
    for path in task_files():
        for task in load_tasks(path):
            module = module_name(task)
            if module not in {
                "ansible.builtin.command",
                "ansible.builtin.shell",
                "ansible.builtin.raw",
            }:
                continue
            keys = set(task)
            args = task.get("args") or {}
            if isinstance(args, dict):
                keys |= set(args)
            # The module's own arguments. `creates:` is normally written nested
            # inside `ansible.builtin.command:`, which is the idiomatic form, so a
            # check that only inspects the task level reports every correct task as
            # an offender.
            module_args = task.get(module)
            if isinstance(module_args, dict):
                keys |= set(module_args)
            if not (keys & IDEMPOTENCY_GUARDS):
                offenders.append(f"{path.relative_to(ROOT)}: {task.get('name', '<unnamed>')}")
    assert not offenders, (
        "command/shell tasks with no creates/changed_when/when guard - these are not "
        "idempotent:\n  " + "\n  ".join(offenders)
    )


def test_deploy_roles_guards_the_model_build() -> None:
    """The model build is the most expensive task in the playbook.

    Guarded on reference.npz, the last artefact the pipeline writes, so it runs
    once. Without the guard every deploy retrains from scratch.
    """
    tasks = load_tasks(ANSIBLE / "roles" / "deploy" / "tasks" / "main.yml")
    build = next(
        (
            t
            for t in tasks
            if "pipeline" in str((t.get("ansible.builtin.command") or {}).get("cmd", ""))
        ),
        None,
    )
    assert build is not None, "no task invoking `tasks.py pipeline` found in the deploy role"
    module_args = build["ansible.builtin.command"]
    creates = module_args.get("creates") or (build.get("args") or {}).get("creates")
    assert creates, "the model build has no `creates:` guard, so it runs on every deploy"
    assert "reference.npz" in str(creates), (
        f"the guard should be the last artefact the pipeline writes, not {creates!r}"
    )


def test_no_shell_used_where_a_module_exists() -> None:
    """A shell task that reimplements a module is both less idempotent and less
    auditable. This is a canary against that creeping in."""
    allowed = {
        "Install the Docker engine",  # placeholder if ever added
    }
    offenders: list[str] = []
    for path in task_files():
        for task in load_tasks(path):
            module = module_name(task)
            if module != "ansible.builtin.shell":
                continue
            if task.get("name") in allowed:
                continue
            offenders.append(f"{path.relative_to(ROOT)}: {task.get('name', '<unnamed>')}")
    assert not offenders, (
        "shell tasks present; prefer a module unless there is a documented reason:\n  "
        + "\n  ".join(offenders)
    )


# ------------------------------------------------------------------ handlers


def test_notified_handlers_are_defined() -> None:
    """A `notify` naming a handler that does not exist is a silent no-op.

    Ansible does not error. The deploy reports success and the service keeps
    running the old configuration, which is the worst possible failure mode:
    successful-looking and wrong.
    """
    defined: set[str] = set()
    for path in handler_files():
        for task in load_tasks(path):
            if task.get("name"):
                defined.add(str(task["name"]))
            listen = task.get("listen")
            if listen:
                defined.update([listen] if isinstance(listen, str) else listen)

    assert defined, "no handlers defined anywhere"

    missing: list[str] = []
    for path in task_files() + [PLAYBOOK]:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        for task in data if isinstance(data, list) else []:
            if not isinstance(task, dict):
                continue
            notify = task.get("notify")
            if not notify:
                continue
            for name in [notify] if isinstance(notify, str) else notify:
                if name not in defined:
                    missing.append(f"{path.relative_to(ROOT)} notifies {name!r}")
    assert not missing, "notify references a handler that is not defined:\n  " + "\n  ".join(
        missing
    )


def test_handler_names_are_unique_within_a_role() -> None:
    for path in handler_files():
        names = [t.get("name") for t in load_tasks(path) if t.get("name")]
        assert len(names) == len(set(names)), f"{path} has duplicate handler names: {names}"


def test_sshd_handler_validates_before_restarting() -> None:
    """The one config file where a mistake locks you out of the machine.

    sshd -t must run before the reload, so a malformed config fails the play rather
    than the login.
    """
    handlers = load_tasks(ANSIBLE / "roles" / "users" / "handlers" / "main.yml")
    validate = [
        h
        for h in handlers
        if "sshd -t" in str(h.get("command", ""))
        or "sshd -t" in str(h.get("ansible.builtin.command", ""))
    ]
    assert validate, "no sshd -t validation handler found"
    assert any("restart sshd" in str(h.get("listen", "")) for h in validate), (
        "the sshd -t check does not share the restart sshd listen topic, so it will not run before the reload"
    )


# ------------------------------------------------------------------ firewall parity


def test_ufw_ports_match_the_cloud_security_list() -> None:
    """Two firewalls that disagree produce a bug that looks like an application
    failure: the port is open on the host and closed at the cloud layer, so curl
    times out and the logs show nothing at all.

    ``allowed_tcp_ports`` holds a Jinja reference to ``api_port``, so entries are
    resolved against the playbook's own vars rather than read literally.
    """
    import re

    playbook = yaml.safe_load(PLAYBOOK.read_text(encoding="utf-8"))[0]
    play_vars = playbook["vars"]

    def resolve(entry) -> int:
        text = str(entry).strip()
        match = re.fullmatch(r"\{\{\s*(\w+)\s*\}\}", text)
        return int(play_vars[match.group(1)]) if match else int(text)

    ufw_ports = {resolve(port) for port in play_vars["allowed_tcp_ports"]}
    terraform = (ROOT / "terraform" / "envs" / "free-tier" / "main.tf").read_text(encoding="utf-8")
    match = re.search(
        r'variable "ingress_ports"\s*\{.*?default\s*=\s*\[([^\]]+)\]', terraform, re.DOTALL
    )
    assert match, "could not find the ingress_ports variable in the free-tier environment"
    cloud_ports = {int(p.strip()) for p in match.group(1).split(",") if p.strip().isdigit()}
    assert ufw_ports == cloud_ports, (
        f"firewall mismatch: ufw allows {sorted(ufw_ports)}, the cloud security list allows "
        f"{sorted(cloud_ports)}. A port open on one and closed on the other presents as an "
        f"application failure with nothing in the logs."
    )


def test_default_deny_is_set_before_allow_rules() -> None:
    """Ordering matters: if the play fails midway, a host with default-allow and no
    allow rules yet is an open host."""
    tasks = load_tasks(ANSIBLE / "roles" / "firewall" / "tasks" / "main.yml")
    deny_index = next(
        i
        for i, t in enumerate(tasks)
        if isinstance(t.get("community.general.ufw"), dict)
        and t["community.general.ufw"].get("policy") == "deny"
    )
    allow_index = next(
        i
        for i, t in enumerate(tasks)
        if isinstance(t.get("community.general.ufw"), dict)
        and t["community.general.ufw"].get("rule") in {"allow", "limit"}
    )
    assert deny_index < allow_index, "allow rules are applied before the default-deny policy"
