from pathlib import Path

import matplotlib
from shapely.geometry import LineString, MultiLineString, Polygon

matplotlib.use('Agg')
import geopandas as gpd

from dfastbe.cmd import run


def test_bank_lines():
    test_r_dir = Path("tests/data/bank_lines")
    language = "UK"
    run_mode = "BANKLINES"
    config_file = test_r_dir / "Meuse_manual.cfg"
    run(language, run_mode, str(config_file))

    # check the detected banklines
    file_1 = test_r_dir / "output/banklines/raw_detected_bankline_fragments.shp"
    assert file_1.exists()
    fragments = gpd.read_file(str(file_1))
    assert len(fragments) == 1
    assert all(fragments.columns == ["FID", "geometry"])
    geom = fragments.loc[0, "geometry"]
    assert isinstance(geom, MultiLineString)
    assert len(geom.geoms) == 22

    # check the bank areas
    file_2 = test_r_dir / "output/banklines/bank_areas.shp"
    assert file_2.exists()
    bank_areas = gpd.read_file(str(file_2))
    assert len(bank_areas) == 2
    assert all(bank_areas.columns == ["FID", "geometry"])
    assert all(isinstance(bank_areas.loc[i, "geometry"], Polygon) for i in range(2))

    # check the bank_line fragments per bank area
    file_3 = test_r_dir / "output/banklines/bankline_fragments_per_bank_area.shp"
    assert file_3.exists()
    fragments_per_bank_area = gpd.read_file(str(file_3))
    assert len(fragments_per_bank_area) == 2
    fragments_per_bank_area.loc[0, "geometry"]
    assert all(
        isinstance(fragments_per_bank_area.loc[i, "geometry"], MultiLineString)
        for i in range(2)
    )

    # check the bankfile
    file_4 = test_r_dir / "output/banklines/bankfile.shp"
    assert file_4.exists()
    bankfile = gpd.read_file(str(file_4))
    assert len(bankfile) == 2
    assert all(bankfile.columns == ["FID", "geometry"])
    assert all(isinstance(bankfile.loc[i, "geometry"], LineString) for i in range(2))

    # check the bankline plotted image
    fig_1 = test_r_dir / r"output/figures/1_banklinedetection.png"
    assert fig_1.exists()
