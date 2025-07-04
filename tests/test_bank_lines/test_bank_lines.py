from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import matplotlib
import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString, Polygon

from dfastbe.bank_lines.bank_lines import BankLines
from dfastbe.bank_lines.plotter import BankLinesPlotter
from dfastbe.cmd import run
from dfastbe.io.data_models import BaseSimulationData, LineGeometry
from dfastbe.io.config import ConfigFile

matplotlib.use('Agg')


@pytest.mark.e2e
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


class TestBankLines:
    @pytest.fixture
    def mock_simulation_data(self):
        """Fixture to create a mock BaseSimulationData object."""
        mock_data = MagicMock(spec=BaseSimulationData)
        mock_data.face_node = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
        mock_data.x_node = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        mock_data.y_node = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        mock_data.water_level_face = np.array([1.0, 2.0, 1.0])
        mock_data.bed_elevation_values = np.ma.masked_array(
            [0.9, 0.9, 1.1, 1.1, 1.1, 0.9, 1.1, 0.9]
        )
        mock_data.n_nodes = np.array([3, 3, 3])
        return mock_data

    @pytest.fixture
    def mock_config_file(self):
        """Fixture to create a mock ConfigFile object."""
        mock_config = MagicMock(spec=ConfigFile)
        mock_config.crs = "EPSG:4326"
        return mock_config

    def test_max_river_width(self, mock_simulation_data):
        """Test the _max_river_width method."""
        with patch(
            "dfastbe.bank_lines.bank_lines.BankLinesRiverData"
        ) as mock_river_data:
            mock_river_data.return_value.simulation_data.return_value = (
                mock_simulation_data,
                0.3,
            )
            bank_lines = BankLines(MagicMock())
        assert bank_lines.max_river_width == 1000

    @patch("dfastbe.bank_lines.bank_lines.BankLinesRiverData")
    def test_detect(self, mock_river_data_class):
        """Test the detect method of the BankLines class."""
        mock_config_file = MagicMock(spec=ConfigFile)
        mock_config_file.get_output_dir.return_value = "mock_output_dir"
        mock_config_file.get_plotting_flags.return_value = {"plot_data": False}

        mock_river_data = MagicMock()
        mock_river_data.river_center_line.stations_bounds = (0, 100)
        mock_river_data.river_center_line.values = MagicMock()
        mock_river_data.river_center_line.as_array.return_value = np.array(
            [[0, 0], [100, 100]]
        )
        mock_river_data.search_lines.to_polygons.return_value = [
            Polygon([(0, 0), (1, 1), (1, 0)])
        ]
        mock_river_data.search_lines.size = 1
        mock_river_data.search_lines.values = [LineString([(0, 0), (1, 1)])]
        mock_river_data.simulation_data.return_value = (MagicMock(), 0.8)
        mock_river_data_class.return_value = mock_river_data

        bank_lines = BankLines(mock_config_file)
        bank_lines.detect_bank_lines = MagicMock(return_value=MagicMock())
        bank_lines.mask = MagicMock(return_value=MagicMock())
        bank_lines.save = MagicMock()
        bank_lines.plot = MagicMock()

        with patch(
            "dfastbe.bank_lines.bank_lines.sort_connect_bank_lines"
        ) as mock_sort:
            mock_sort.return_value = [LineString([(0, 0), (1, 1)])]
            bank_lines.detect()
            bank_lines.plot()

        bank_lines.detect_bank_lines.assert_called_once_with(
            bank_lines.simulation_data,
            bank_lines.critical_water_depth,
            mock_config_file,
        )
        bank_lines.save.assert_called_once()
        if mock_config_file.get_plotting_flags.return_value["plot_data"]:
            bank_lines.plot.assert_called_once()

    def test_calculate_water_depth_per_node(self, mock_simulation_data):
        """Test the calculate_water_depth_per_node method."""
        h_node = BankLines._calculate_water_depth(mock_simulation_data)
        expected_h_node = np.array(
            [
                [0.1, 0.6, 0.2333333333],
                [0.6, 0.2333333333, 0.4],
                [0.2333333333, 0.4, -0.1],
            ]
        )
        assert h_node.shape == mock_simulation_data.face_node.shape
        assert np.allclose(h_node, expected_h_node)

    def test_generate_bank_lines(self, mock_simulation_data):
        """Test the _generate_bank_lines method."""
        wet_node = np.array(
            [[True, False, True], [False, True, True], [True, False, True]]
        )
        n_wet_arr = np.ma.masked_array([2, 2, 2])
        h_node = np.array([[0.9, 1.1, 0.9], [0.9, 0.5, 1.1], [0.9, 0.5, 1.1]])
        h0 = 0.3

        lines = BankLines._generate_bank_lines(
            mock_simulation_data, wet_node, n_wet_arr, h_node, h0
        )
        expected = [
            LineString([(-2.999999999, -2.999999999), (5.0, 5.0)]),
            LineString([(-4.999999999, -4.999999999), (2.5, 2.5)]),
            LineString([(3.5, 3.5), (2.6666666666, 2.6666666666)]),
        ]
        assert all(
            [
                line.equals_exact(expected[i], tolerance=1e-8)
                for i, line in enumerate(lines)
            ]
        )

    @pytest.mark.parametrize(
        "face_node, n_nodes, expected",
        [
            (
                np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]]),
                np.array([3, 3, 3]),
                LineString(
                    [
                        (0.4, 0.4),
                        (1.818181818, 1.818181818),
                        (2.4, 2.4),
                        (3.199999999, 3.199999999),
                    ]
                ),
            ),
            (
                np.array([[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]]),
                np.array([4, 4, 4]),
                MultiLineString(
                    [
                        [
                            (2.0, 2.0),
                            (1.666666666, 1.66666666),
                            (1.333333333, 1.33333333),
                        ],
                        [
                            (4.399999999, 4.399999999),
                            (4.999999999, 4.999999999),
                            (5.428571428, 5.428571428),
                        ],
                    ],
                ),
            ),
            (
                np.ma.masked_array(
                    [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]],
                    mask=[
                        [False, False, False, False],
                        [False, False, False, False],
                        [True, True, True, True],
                    ],
                ),
                np.array([4, 4, 4]),
                MultiLineString(
                    [
                        [
                            (2.0, 2.0),
                            (1.666666666, 1.66666666),
                            (1.333333333, 1.33333333),
                        ],
                        [
                            (5.333333333, 5.33333333),
                            (5.727272727, 5.72727272),
                            (6, 6),
                        ],
                    ]
                ),
            ),
        ],
        ids=["triangle faces", "polygonal faces", "masked faces"],
    )
    def test_detect_bank_lines(
        self, mock_simulation_data, mock_config_file, face_node, n_nodes, expected
    ):
        """Test the detect_bank_lines method.

        Test the detect_bank_lines method with different face_node inputs.
        triangle faces: a simple triangle face with 3 nodes.
        polygonal faces: a polygon with 4 nodes.
        masked faces: a masked array with 4 nodes, where the last row is masked.
        """
        mock_simulation_data.face_node = face_node
        mock_simulation_data.n_nodes = n_nodes
        h0 = 0.3
        result = BankLines.detect_bank_lines(mock_simulation_data, h0, mock_config_file)
        assert isinstance(result, gpd.GeoSeries)
        assert result.iloc[0].equals_exact(expected, tolerance=1e-8)
        assert len(result) == 1

    def test_progress_bar(self, capsys):
        """Test the _progress_bar method."""
        total = 1000
        for i in range(total):
            BankLines._progress_bar(i, total)
        captured = capsys.readouterr()
        assert "Progress: 100.00%" in captured.out

    def test_save(self, mock_config_file, tmp_path: Path):
        """Test the save method of the BankLines class."""
        bank = [LineString([(0, 0), (1, 1)])]
        banklines = gpd.GeoSeries([LineString([(0, 0), (1, 1)])], crs="EPSG:4326")
        clipped_banklines = [MultiLineString([LineString([(0, 0), (1, 1)])])]
        bank_areas = [Polygon([(0, 0), (1, 1), (1, 0)])]

        mock_config_file.get_str.return_value = "bank_file"
        mock_config_file.get_output_dir.return_value = tmp_path
        with patch(
            "dfastbe.bank_lines.bank_lines.BankLinesRiverData"
        ) as mock_river_data:
            mock_river_data.return_value.simulation_data.return_value = (
                0.3,
                MagicMock(),
            )
            bank_lines = BankLines(mock_config_file)

        bank_lines.save(
            bank, banklines, clipped_banklines, bank_areas, mock_config_file
        )

        assert (tmp_path / "bank_file.shp").exists()
        assert (tmp_path / "raw_detected_bankline_fragments.shp").exists()
        assert (tmp_path / "bank_areas.shp").exists()
        assert (tmp_path / "bankline_fragments_per_bank_area.shp").exists()

    def test_plot(self, mock_config_file, tmp_path: Path):
        """Test the plot method of the BankLines class."""
        xy_km_numpy = LineGeometry(line=np.array([[0, 0, 0], [1, 1, 0]]))
        n_search_lines = 1
        bank = [LineString([(0, 0), (1, 1)])]
        km_bounds = (0, 1)
        bank_areas = [Polygon([(0, 0), (1, 1), (1, 0)])]

        mock_simulation_data = MagicMock()
        mock_simulation_data.face_node = np.array([[0, 1, 2]])
        mock_simulation_data.n_nodes = np.array([3])
        mock_simulation_data.x_node = np.array([0, 1, 2])
        mock_simulation_data.y_node = np.array([0, 1, 2])
        mock_simulation_data.water_depth_face = np.array([1.0, 2.0, 1.5])

        with patch(
            "dfastbe.bank_lines.bank_lines.BankLinesRiverData"
        ) as mock_river_data:
            mock_river_data.return_value.simulation_data.return_value = (
                mock_simulation_data,
                0.3,
            )
            bank_lines = BankLines(mock_config_file)
            bank_lines.plot_flags = {
                "save_plot": True,
                "save_plot_zoomed": True,
                "close_plot": True,
                "zoom_km_step": 0.1,
                "fig_dir": str(tmp_path),
                "plot_ext": ".png",
            }

        with patch(
            "dfastbe.bank_lines.plotter.BankLinesPlotter.plot_detect1"
        ) as mock_plot_detect1, patch("matplotlib.pyplot.show") as mock_show, patch(
            "matplotlib.pyplot.close"
        ) as mock_close, patch(
            "dfastbe.plotting.BasePlot.zoom_xy_and_save"
        ) as mock_zoom_xy_and_save:
            mock_plot_detect1.return_value = (MagicMock(), MagicMock())

            bank_lines_plotter = BankLinesPlotter(
                False, bank_lines.plot_flags, mock_config_file.crs, mock_simulation_data, xy_km_numpy, km_bounds,
            )
            bank_lines_plotter.plot(
                n_search_lines,
                bank,
                bank_areas,
            )

            mock_plot_detect1.assert_called_once_with(
                bank_areas,
                bank,
                "x-coordinate [m]",
                "y-coordinate [m]",
                "water depth and detected bank lines",
                "water depth [m]",
                "bank search area",
                "detected bank line",
            )
            mock_zoom_xy_and_save.assert_called_once()
            mock_show.assert_called_once()
            mock_close.assert_called_once()

    def test_mask(self):
        """Test the mask method of the BankLines class."""
        # Mock banklines as a GeoSeries
        banklines = gpd.GeoSeries(
            [
                LineString([(0, 0), (2, 2)]),
                LineString([(1, 1), (3, 3)]),
                LineString([(2, 0), (2, 3)]),
            ],
            crs="EPSG:4326",
        )
        bank_area = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])

        clipped_banklines = BankLines.mask(banklines, bank_area)

        expected_clipped = LineString([(0, 0), (2, 2)])
        assert isinstance(clipped_banklines, LineString)
        assert clipped_banklines.equals(expected_clipped)
