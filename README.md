# LankaFlood-SAR

Flood extent mapping from Sentinel-1 radar with deep learning, calibrated uncertainty and
domain-shift testing, with a May 2017 Sri Lanka case study.

> Status: Milestone 1 (setup and data checks). Individual hobby and portfolio project built on
> free, open data. Maps are model estimates and must not be used for emergency decisions.

## Stack
Python 3.11 (uv), rasterio / geopandas, Earth Engine and Copernicus Data Space, PyTorch +
segmentation-models-pytorch, FastAPI, React + MapLibre, Postgres + Prometheus + Grafana, Docker.

## Quick start (Windows, PowerShell)
```powershell
copy .env.example .env      # then fill in your values
uv sync                     # core + dev dependencies
uv run pre-commit install
uv run pytest
docker compose up -d        # Postgres, Prometheus, Grafana
```

Deep learning stack (PyTorch with CUDA wheels), when you reach Week 2:
```powershell
uv sync --extra ml
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## Layout
- `src/lankaflood/` Python package (data, features, baselines, models, train, uncertainty, eval, infer, api)
- `configs/` data, training and inference settings
- `frontend/` React + MapLibre app (Week 6)
- `monitoring/`, `db/` Grafana, Prometheus and Postgres setup
- `docs/` data card, evaluation, case study, limitations

## Milestones
See the project plan: environment and baselines, first U-Net, leave-one-event-out, uncertainty,
Sri Lanka case study, service and web map, evaluation write-up, polish.

## Licence
To be chosen (for example MIT). Dataset licences are kept separate and recorded in `docs/data_card.md`.

*This project is for educational and portfolio purposes. Flood maps are model estimates with
uncertainty and must not be used for emergency or safety decisions.*
