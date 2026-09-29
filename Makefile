# Thin shim over tasks.py so Linux, macOS and CI get the familiar `make <target>`.
# All logic lives in tasks.py; there are no duplicate command definitions here.
#
#   make up      make bench      make test      make verify
#
PYTHON ?= python

TARGETS = install lint format train export reference serve test test-all cov \
          bench report drift pipeline verify backup restore-verify \
          up down nuke logs smoke clean

.PHONY: help $(TARGETS)

help:
	@$(PYTHON) tasks.py

$(TARGETS):
	@$(PYTHON) tasks.py $@
