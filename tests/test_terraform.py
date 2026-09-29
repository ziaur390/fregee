"""Terraform configuration tests.

Checked by parsing the HCL as text, because Terraform is not installed where the
rest of this suite runs — and because the two things worth checking do not need a
plan:

1. **No credentials in source.** A tenancy OCID, a user OCID, a private key or an
   API key fingerprint committed to a repository is the whole incident. `.gitignore`
   covers `*.tfvars`, but it cannot cover what someone typed into a `.tf` file.

2. **Everything referenced is declared.** A variable used but never declared, or a
   resource referenced that does not exist, fails at `apply` — which is the worst
   moment to find out. Catching it here costs nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mlserve.config import ROOT

TERRAFORM = ROOT / "terraform"
TF_FILES = sorted(TERRAFORM.rglob("*.tf"))


def read_all() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in TF_FILES)


def files_in(env: str) -> list[Path]:
    return [p for p in TF_FILES if env in p.parts]


# ------------------------------------------------------------------ structure


def test_terraform_files_exist() -> None:
    assert TF_FILES, "no .tf files found"


def test_every_tf_file_declares_a_terraform_block_or_is_part_of_one() -> None:
    """Each environment needs its own required_providers block.

    Without it Terraform guesses the provider source from the resource prefix, which
    works until the day it does not — and the error is at `init`, not at review.
    """
    for env in ("local", "free-tier"):
        main = next((p for p in files_in(env) if p.name == "main.tf"), None)
        assert main is not None, f"terraform/envs/{env} has no main.tf"
        text = main.read_text(encoding="utf-8")
        assert re.search(r"^terraform\s*\{", text, re.MULTILINE), f"{main} has no terraform block"
        assert "required_providers" in text, f"{main} has no required_providers"
        assert "required_version" in text, f"{main} does not pin a Terraform version"


def test_module_declares_required_version() -> None:
    module = TERRAFORM / "modules" / "cloud-init" / "main.tf"
    assert "required_version" in module.read_text(encoding="utf-8")


def test_every_environment_has_outputs() -> None:
    """Terraform creates the host and Ansible configures it. The hand-off is the
    output, so an environment with no outputs leaves the operator to guess the IP."""
    for env in ("local", "free-tier"):
        outputs = [p for p in files_in(env) if p.name == "outputs.tf"]
        assert outputs, f"terraform/envs/{env} has no outputs.tf"
        text = outputs[0].read_text(encoding="utf-8")
        assert "next_command" in text, f"{env} does not output the Ansible hand-off command"
        assert "inventory_line" in text, f"{env} does not output an inventory line"


# ------------------------------------------------------------------ secrets


def test_no_credential_literals_in_committed_files() -> None:
    """The check that matters most. A leaked tenancy OCID plus a key is a full
    account compromise, and it is committed far more often than people expect."""
    patterns = {
        "tenancy OCID": r"ocid1\.tenancy\.oc1\.[a-z0-9]{20,}",
        "user OCID": r"ocid1\.user\.oc1\.[a-z0-9]{20,}",
        "compartment OCID": r"ocid1\.compartment\.oc1\.[a-z0-9]{20,}",
        "private key block": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        "OCI API key fingerprint": r'fingerprint\s*=\s*"(?:[0-9a-f]{2}:){15}[0-9a-f]{2}"',
        "hardcoded password": r'(?i)password\s*=\s*"[^"$]{4,}"',
    }
    hits: list[str] = []
    for path in TF_FILES:
        text = path.read_text(encoding="utf-8")
        for label, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                line = text[: match.start()].count("\n") + 1
                hits.append(f"{path.relative_to(ROOT)}:{line} looks like a {label}")
    assert not hits, "credential material in Terraform source:\n  " + "\n  ".join(hits)


def test_example_vars_contains_only_placeholders() -> None:
    """terraform.tfvars.example is the one vars file that reaches git, so it must
    not contain anything real."""
    example = TERRAFORM / "envs" / "free-tier" / "terraform.tfvars.example"
    assert example.exists(), "no tfvars.example - the next person will commit a real tfvars instead"
    text = example.read_text(encoding="utf-8")
    assert "example" in text.lower() or "replace" in text.lower(), (
        "the example vars file does not look like a placeholder"
    )
    assert not re.search(r"ocid1\.compartment\.oc1\.[a-z0-9]{40,}", text), (
        "the example file contains a real-looking compartment OCID"
    )


def test_gitignore_excludes_state_and_vars() -> None:
    """State files contain every secret a provider returned, in plaintext."""
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("*.tfstate", "*.tfvars", ".terraform/"):
        assert pattern in ignore, f".gitignore does not exclude {pattern}"
    assert "!*.tfvars.example" in ignore, (
        ".gitignore excludes *.tfvars but does not re-include the example file"
    )


# ------------------------------------------------------------------ references resolve


def test_referenced_variables_are_declared() -> None:
    """An undeclared variable fails at plan time. Catching it here is free."""
    for env in ("local", "free-tier"):
        env_files = files_in(env)
        declared: set[str] = set()
        for path in env_files:
            declared.update(re.findall(r'variable\s+"(\w+)"', path.read_text(encoding="utf-8")))

        missing: list[str] = []
        for path in env_files:
            text = path.read_text(encoding="utf-8")
            for name in set(re.findall(r"\bvar\.(\w+)", text)):
                if name not in declared:
                    missing.append(f"{path.relative_to(ROOT)}: var.{name}")
        assert not missing, f"undefined variables in terraform/envs/{env}:\n  " + "\n  ".join(
            missing
        )


def test_referenced_outputs_and_resources_exist() -> None:
    """An outputs.tf referencing a resource or module output that does not exist
    fails at apply, after the host has been created."""
    for env in ("local", "free-tier"):
        env_files = files_in(env)
        declared_resources: set[str] = set()
        declared_modules: set[str] = set()
        declared_outputs: set[str] = set()
        for path in env_files:
            text = path.read_text(encoding="utf-8")
            declared_resources.update(
                f"{kind}.{name}"
                for kind, name in re.findall(r'resource\s+"([\w]+)"\s+"([\w-]+)"', text)
            )
            declared_modules.update(re.findall(r'module\s+"(\w+)"', text))
            if path.name == "outputs.tf":
                declared_outputs.update(re.findall(r'output\s+"(\w+)"', text))

        problems: list[str] = []
        for path in env_files:
            if path.name != "outputs.tf":
                continue
            text = path.read_text(encoding="utf-8")
            for ref in set(re.findall(r"\b((?:\w+)\.(?:\w+-\w+|\w+))\.\w+", text)):
                prefix = ref.split(".")[0]
                if prefix in {"var", "local", "path", "data", "each", "count"}:
                    continue
                if ref not in declared_resources and prefix not in declared_modules:
                    problems.append(f"{path.relative_to(ROOT)} references {ref}")
        assert not problems, f"unresolvable references in terraform/envs/{env}:\n  " + "\n  ".join(
            problems
        )


def test_module_inputs_match_module_outputs() -> None:
    """The cloud-init module declares variables; both environments pass values. A
    typo in a passed variable name is silently ignored by Terraform."""
    module = TERRAFORM / "modules" / "cloud-init" / "main.tf"
    declared = set(re.findall(r'variable\s+"(\w+)"', module.read_text(encoding="utf-8")))
    assert declared, "the cloud-init module declares no variables"

    for env in ("local", "free-tier"):
        text = "\n".join(p.read_text(encoding="utf-8") for p in files_in(env))
        block = re.search(r'module\s+"cloud_init"\s*\{(.*?)\n\}', text, re.DOTALL)
        assert block, f"terraform/envs/{env} does not instantiate the cloud-init module"
        passed = set(re.findall(r"^\s*(\w+)\s*=", block.group(1), re.MULTILINE))
        setup = {"source"}
        unknown = passed - declared - setup
        assert not unknown, (
            f"terraform/envs/{env} passes undeclared inputs to the cloud-init module: "
            f"{sorted(unknown)}. Terraform ignores these silently."
        )


def test_terraform_lock_files_are_committed() -> None:
    """Provider versions must be pinned in version control.

    HashiCorp recommends committing ``.terraform.lock.hcl``, and here it matters more
    than usual: the local environment depends on a **third-party** provider
    (``larstobi/multipass``) whose resource schema is not guaranteed stable. The
    configuration in this repo was validated against a specific provider version;
    without the lock file, a future release could change that schema and silently
    invalidate the check.
    """
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    # Check for an actual ignore PATTERN, not the substring. The .gitignore explains
    # in a comment why this file is not ignored, and a substring check matches that
    # explanation - the same trap that bit the Dockerfile HEALTHCHECK assertion.
    patterns = [
        line.strip()
        for line in ignore.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert ".terraform.lock.hcl" not in patterns, (
        ".gitignore excludes the Terraform lock file, so provider versions are not pinned"
    )

    locks = list(TERRAFORM.rglob(".terraform.lock.hcl"))
    if not locks:
        pytest.skip("run `terraform init` in each env to generate the lock files")

    for lock in locks:
        assert 'provider "' in lock.read_text(encoding="utf-8"), f"{lock} pins no providers"

    # The third-party provider is the one that actually needs pinning.
    local_lock = TERRAFORM / "envs" / "local" / ".terraform.lock.hcl"
    if local_lock.exists():
        assert "larstobi/multipass" in local_lock.read_text(encoding="utf-8"), (
            "the multipass provider is not pinned"
        )


def test_free_tier_limits_are_validated() -> None:
    """The Always Free allowance is a hard boundary, not a suggestion. Beyond it
    the instance bills, and the failure arrives as an invoice."""
    text = (TERRAFORM / "envs" / "free-tier" / "main.tf").read_text(encoding="utf-8")
    assert "validation" in text, "no validation on the free-tier resource sizing"
    assert "4 OCPUs" in text or "4" in text
    assert "Always Free" in text


def test_free_tier_uses_the_ampere_shape() -> None:
    """VM.Standard.A1.Flex is the Always Free ARM shape. Any other shape in this
    environment is a billing mistake."""
    text = (TERRAFORM / "envs" / "free-tier" / "main.tf").read_text(encoding="utf-8")
    assert "VM.Standard.A1.Flex" in text


def test_free_tier_warns_about_cost() -> None:
    """The output that stops someone leaving a paid instance running."""
    outputs = (TERRAFORM / "envs" / "free-tier" / "outputs.tf").read_text(encoding="utf-8")
    assert "cost_warning" in outputs
    assert "home region" in outputs


def test_local_env_still_hands_off_to_ansible() -> None:
    """Both environments must hand off rather than invoking Ansible themselves.
    Merging provisioning with configuration means re-applying configuration risks
    recreating the machine."""
    for path in TF_FILES:
        text = path.read_text(encoding="utf-8")
        assert "local-exec" not in text or "ansible-playbook" not in text, (
            f"{path} appears to invoke Ansible; the hand-off should be an output instead"
        )
