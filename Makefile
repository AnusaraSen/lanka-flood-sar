# Windows note: `make` is not installed by default. Install it (for example with
# Chocolatey: `choco install make`) or run the underlying commands by hand.

.PHONY: setup setup-ml lint format test up down

setup:            ## Install core + dev dependencies and git hooks
	uv sync
	uv run pre-commit install

setup-ml:         ## Add the deep learning stack (PyTorch CUDA wheels)
	uv sync --extra ml

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

up:               ## Start Postgres, Prometheus and Grafana
	docker compose up -d

down:
	docker compose down

# Targets added as milestones are built: data, chips, baselines, train,
# calibrate, eval, case-study, api, all
