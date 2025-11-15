SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

PY ?= python3
PIP ?= pip
VENV ?= .venv
ACT := source $(VENV)/bin/activate

help:
	@echo "GCE targets:"
	@echo "  make init        Create venv + pip upgrade"
	@echo "  make install     Install GCE (editable) + dev deps"
	@echo "  make install-cc  Also bring in cc-framework (Git URL)"
	@echo "  make fmt|lint    Code quality"
	@echo "  make type        mypy"
	@echo "  make test        pytest"
	@echo "  make info        Print backend info (CC vs fallback)"

$(VENV)/bin/activate:
	$(PY) -m venv $(VENV)
	$(ACT); $(PIP) install -U pip wheel setuptools

init: $(VENV)/bin/activate

install: init
	$(ACT); $(PIP) install -e '.[dev]'
	@echo "Tip: to prefer CC backend, export GCE_PREFER_CC=1"

install-cc: init
	$(ACT); $(PIP) install -e '.[dev,cc]'
	@echo "Installed cc-framework via Git optional extra."

fmt:
	$(ACT); ruff check --fix .
	$(ACT); black .

lint:
	$(ACT); ruff check .
	$(ACT); black --check .

type:
	$(ACT); mypy src/gce

test:
	$(ACT); pytest

info:
	$(ACT); gce-backend-info
