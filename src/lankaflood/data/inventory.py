"""Per-chip inventory of the Sen1Floods11 hand-labelled subset.

Run from the repo root:

    uv run python -m lankaflood.data.inventory

Writes results/chip_inventory.csv (one row per chip) and prints a per-event summary.

Notes on the data (from the dataset layout; confirm in the exploration notebook):
- S1Hand chips have two bands, VV and VH, in dB. Pixels outside the radar swath are NaN.
- LabelHand chips have one band: -1 = no data, 0 = not water, 1 = water.
  The label marks *water*, which includes permanent water. Separating flood water from
  permanent water is done later with a permanent-water layer.
- The event name is the first part of the file name, for example Sri-Lanka_101973_S1Hand.tif.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "valid": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",
}


def event_from_name(file_name: str) -> str:
    """Event name is the text before the first underscore."""
    return file_name.split("_")[0]


def read_official_splits(splits_dir: Path) -> dict[str, str]:
    """Map S1 chip file name -> official split name (train, valid, test, bolivia)."""
    mapping: dict[str, str] = {}
    for split, file_name in SPLIT_FILES.items():
        path = splits_dir / file_name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                mapping[line.split(",")[0]] = split
    return mapping


def chip_stats(s1_path: Path, label_path: Path) -> dict:
    """Basic statistics for one chip."""
    with rasterio.open(s1_path) as src:
        s1 = src.read().astype("float32")  # (bands, H, W)
    with rasterio.open(label_path) as src:
        label = src.read(1)

    valid = label != -1
    n_valid = int(valid.sum())
    water_share = float((label[valid] == 1).mean()) if n_valid else float("nan")

    with np.errstate(all="ignore"):
        vv_mean = float(np.nanmean(s1[0])) if s1.shape[0] > 0 else float("nan")
        vh_mean = float(np.nanmean(s1[1])) if s1.shape[0] > 1 else float("nan")

    return {
        "height": int(s1.shape[1]),
        "width": int(s1.shape[2]),
        "s1_nan_share": float(np.isnan(s1).any(axis=0).mean()),
        "vv_mean_db": vv_mean,
        "vh_mean_db": vh_mean,
        "label_valid_share": float(valid.mean()),
        "water_share": water_share,
    }


def build_inventory(root: Path) -> pd.DataFrame:
    """Build the inventory table from a Sen1Floods11 download folder."""
    s1_dir = root / "HandLabeled" / "S1Hand"
    label_dir = root / "HandLabeled" / "LabelHand"
    official = read_official_splits(root / "splits")

    s1_paths = sorted(s1_dir.glob("*_S1Hand.tif"))
    if not s1_paths:
        raise FileNotFoundError(f"No *_S1Hand.tif files found in {s1_dir}")

    rows = []
    for s1_path in s1_paths:
        label_path = label_dir / s1_path.name.replace("_S1Hand.tif", "_LabelHand.tif")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {s1_path.name}: {label_path}")
        row = {
            "chip_id": s1_path.name.replace("_S1Hand.tif", ""),
            "event": event_from_name(s1_path.name),
            "official_split": official.get(s1_path.name, "none"),
        }
        row.update(chip_stats(s1_path, label_path))
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-event summary: chip counts, mean water share, and official split counts."""
    summary = df.groupby("event").agg(
        chips=("chip_id", "count"),
        mean_water_share=("water_share", "mean"),
        mean_s1_nan_share=("s1_nan_share", "mean"),
    )
    splits = pd.crosstab(df["event"], df["official_split"])
    return summary.join(splits).sort_values("chips", ascending=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/raw/sen1floods11"))
    parser.add_argument("--out", type=Path, default=Path("results/chip_inventory.csv"))
    args = parser.parse_args(argv)

    df = build_inventory(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Wrote {len(df)} chips to {args.out}\n")
    print(summarise(df).round(3).to_string())


if __name__ == "__main__":
    main()
