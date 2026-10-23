import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from lankaflood.data.inventory import (
    build_inventory,
    event_from_name,
    read_official_splits,
    summarise,
)


def _write_tif(path, array):
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[1],
        width=array.shape[2],
        count=array.shape[0],
        dtype=array.dtype,
        crs="EPSG:4326",
        transform=from_origin(0, 0, 0.0001, 0.0001),
    ) as dst:
        dst.write(array)


@pytest.fixture
def fake_root(tmp_path):
    s1_dir = tmp_path / "HandLabeled" / "S1Hand"
    label_dir = tmp_path / "HandLabeled" / "LabelHand"
    splits_dir = tmp_path / "splits"
    for d in (s1_dir, label_dir, splits_dir):
        d.mkdir(parents=True)

    # Chip 1: 4x4. 3 water, 9 land, 4 no-data pixels. 2 NaN pixels in the radar image.
    s1 = np.full((2, 4, 4), -10.0, dtype="float32")
    s1[:, 0, 0] = np.nan
    s1[0, 0, 1] = np.nan  # only VV missing here, still counts as one NaN pixel
    label = np.zeros((4, 4), dtype="int16")
    label[0, :] = -1
    label[1, :3] = 1
    _write_tif(s1_dir / "Aaa_1_S1Hand.tif", s1)
    _write_tif(label_dir / "Aaa_1_LabelHand.tif", label)

    # Chip 2: all water, no missing data, different event.
    _write_tif(s1_dir / "Bbb_2_S1Hand.tif", np.full((2, 4, 4), -20.0, dtype="float32"))
    _write_tif(label_dir / "Bbb_2_LabelHand.tif", np.ones((4, 4), dtype="int16"))

    (splits_dir / "flood_train_data.csv").write_text(
        "Aaa_1_S1Hand.tif,Aaa_1_LabelHand.tif\n", encoding="utf-8"
    )
    (splits_dir / "flood_test_data.csv").write_text(
        "Bbb_2_S1Hand.tif,Bbb_2_LabelHand.tif\n", encoding="utf-8"
    )
    return tmp_path


def test_event_from_name():
    assert event_from_name("Sri-Lanka_101973_S1Hand.tif") == "Sri-Lanka"
    assert event_from_name("USA_5_S1Hand.tif") == "USA"


def test_read_official_splits(fake_root):
    mapping = read_official_splits(fake_root / "splits")
    assert mapping == {"Aaa_1_S1Hand.tif": "train", "Bbb_2_S1Hand.tif": "test"}


def test_build_inventory_values(fake_root):
    df = build_inventory(fake_root).set_index("chip_id")
    assert list(df.index) == ["Aaa_1", "Bbb_2"]

    a = df.loc["Aaa_1"]
    assert a["event"] == "Aaa"
    assert a["official_split"] == "train"
    assert a["label_valid_share"] == pytest.approx(12 / 16)
    assert a["water_share"] == pytest.approx(3 / 12)  # water pixels / valid pixels
    assert a["s1_nan_share"] == pytest.approx(2 / 16)
    assert a["vh_mean_db"] == pytest.approx(-10.0)

    b = df.loc["Bbb_2"]
    assert b["official_split"] == "test"
    assert b["water_share"] == pytest.approx(1.0)
    assert b["s1_nan_share"] == 0.0


def test_summarise(fake_root):
    summary = summarise(build_inventory(fake_root))
    assert set(summary.index) == {"Aaa", "Bbb"}
    assert summary.loc["Aaa", "chips"] == 1
    assert summary.loc["Aaa", "train"] == 1
    assert summary.loc["Bbb", "test"] == 1


def test_missing_label_raises(fake_root):
    (fake_root / "HandLabeled" / "LabelHand" / "Bbb_2_LabelHand.tif").unlink()
    with pytest.raises(FileNotFoundError):
        build_inventory(fake_root)
