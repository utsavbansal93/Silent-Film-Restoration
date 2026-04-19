set dotenv-load := false

PYTHON := ".venv/bin/python"
PYTEST := ".venv/bin/pytest"

default:
    @just --list

bootstrap:
    test -d .venv || /opt/homebrew/bin/python3.11 -m venv .venv
    {{PYTHON}} -m pip install --upgrade pip
    {{PYTHON}} -m pip install -e ".[dev]"

fetch-source:
    bash scripts/fetch_source.sh

regenerate-fixture:
    bash scripts/regenerate_fixture.sh

run profile="modern_smooth":
    {{PYTHON}} -m pipeline.run --config configs/{{profile}}.yaml

stage name profile="modern_smooth":
    {{PYTHON}} -m pipeline.stages.{{name}} --config configs/{{profile}}.yaml

test:
    {{PYTEST}} tests/

test-unit:
    {{PYTEST}} tests/unit/ -v

test-golden:
    {{PYTEST}} tests/golden/ -v

test-integration:
    {{PYTEST}} tests/integration/ -v -m integration

viewer:
    {{PYTHON}} -m viewer.server

bench:
    {{PYTHON}} -m pipeline.bench_all --config configs/test_30sec.yaml

fetch-east-model:
    bash scripts/fetch_east_model.sh

clean-runs:
    @echo "Refusing to auto-clean runs/ — remove manually if you really want to."

lint:
    .venv/bin/ruff check pipeline/ viewer/ tests/
