from pathlib import Path

import yaml

import lankaflood

ROOT = Path(__file__).resolve().parents[1]


def test_package_imports():
    assert lankaflood.__version__ == "0.1.0"


def test_configs_parse():
    for name in ("data", "train", "infer"):
        with open(ROOT / "configs" / f"{name}.yaml", encoding="utf-8") as f:
            assert isinstance(yaml.safe_load(f), dict)


def test_channel_count_matches_model_input():
    with open(ROOT / "configs" / "data.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    with open(ROOT / "configs" / "train.yaml", encoding="utf-8") as f:
        train = yaml.safe_load(f)
    assert len(data["channels"]) == train["model"]["in_channels"]
