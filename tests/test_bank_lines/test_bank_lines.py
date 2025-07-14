from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import matplotlib
import numpy as np
import pytest
from matplotlib.testing.compare import compare_images
from shapely.geometry import LineString, MultiLineString, Polygon

from dfastbe.bank_lines.bank_lines import BankLines
from dfastbe.bank_lines.plotter import BankLinesPlotter
from dfastbe.cmd import run
from dfastbe.io.config import ConfigFile
from dfastbe.io.data_models import BaseSimulationData, LineGeometry

matplotlib.use('Agg')


@pytest.mark.e2e
def test_bank_lines():
    """End-to-end test for the bank lines detection.

    Use a real world case to run the bank lines detection module and verify the output.
    This test checks the detection of bank lines, bank areas, and the generation of
    bankline fragments per bank area. It also verifies the plotting of the detected
    bank lines and compares the generated image with a reference image.

    The test uses a configuration file located in the `tests/data/bank_lines` directory
    and runs the bank lines detection in the "BANKLINES" mode for the "UK" language.

    Asserts:
        - The detected bankline fragments are correctly formed as a MultiLineString.
        - The bank areas are correctly formed as Polygons.
        - The bankline fragments per bank area are correctly formed as MultiLineStrings.
        - The bankfile is correctly formed as LineStrings.
        - The generated plot image matches the reference image within a tolerance.
    """
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
    fig_1 = test_r_dir / "output/figures/1_bankline-detection.png"
    reference = test_r_dir / "reference/figures/1_bankline-detection.png"
    assert fig_1.exists()
    assert compare_images(str(fig_1), str(reference), 0.0001) is None


class TestBankLines:
    @pytest.fixture
    def mock_simulation_data(self):
        """Fixture to create a mock BaseSimulationData object.

        This mock creates 3 triangles as a small river. It can be used for basic tests
        of the BankLines class methods that require simulation data.

        Returns:
            MagicMock: A mock object simulating BaseSimulationData.
        """
        mock_data = MagicMock(spec=BaseSimulationData)
        mock_data.face_node = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
        mock_data.x_node = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        mock_data.y_node = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        mock_data.water_level_face = np.array([1.0, 2.0, 1.0])
        mock_data.water_depth_face = np.array([0.1, 0.6, 0.4])
        mock_data.bed_elevation_values = np.ma.masked_array(
            [0.9, 0.9, 1.1, 1.1, 1.1, 0.9, 1.1, 0.9]
        )
        mock_data.n_nodes = np.array([3, 3, 3])
        return mock_data

    @pytest.fixture
    def mock_config_file(self):
        """Fixture to create a mock ConfigFile object.

        This mock simulates a configuration file with a specific CRS.

        Returns:
            MagicMock: A mock object simulating ConfigFile.
        """
        mock_config = MagicMock(spec=ConfigFile)
        mock_config.crs = "EPSG:4326"
        return mock_config

    @pytest.mark.unit
    def test_max_river_width(self, mock_simulation_data):
        """Test the max_river_width property.

        Asserts:
            The max_river_width is correctly calculated based on the simulation data.
        """
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
    @pytest.mark.unit
    def test_detect(self, mock_river_data_class):
        """Test the detect method of the BankLines class.

        This test mocks the BankLinesRiverData class and its methods to ensure
        that the detect method of BankLines works as expected.

        Args:
            mock_river_data_class (MagicMock): Mocked class for BankLinesRiverData.

        Mocks:
            BankLinesRiverData:
                Mocked to return a MagicMock data object with
                predefined values matching a BankLinesRiverData class object.
            detect_bank_lines:
                Mocked to simulate the detection of bank lines.
            mask:
                Mocked to simulate the masking of bank lines.
            save:
                Mocked to simulate saving the detected bank lines.
            plot:
                Mocked to simulate plotting the detected bank lines.

        Asserts:
            The detect_bank_lines method is called with the correct parameters.
            The save method is called.
            The plot method is called if plotting is enabled in the config file.
        """
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
        ) as mock_sort, patch("dfastbe.bank_lines.bank_lines.log_text"):
            mock_sort.return_value = [LineString([(0, 0), (1, 1)])]
            bank_lines.detect()
            bank_lines.plot()
            bank_lines.save()

        bank_lines.detect_bank_lines.assert_called_once_with(
            bank_lines.simulation_data,
            bank_lines.critical_water_depth,
            mock_config_file,
        )
        bank_lines.save.assert_called_once()
        if mock_config_file.get_plotting_flags.return_value["plot_data"]:
            bank_lines.plot.assert_called_once()

    @pytest.mark.unit
    def test_calculate_water_depth_per_node(self, mock_simulation_data):
        """Test the calculate_water_depth_per_node method.

        This method calculates the water depth at each node based on the face node
        water levels and bed elevations. It uses the face_node, water_level_face, and
        bed_elevation_values attributes of the BaseSimulationData object.

        Args:
            mock_simulation_data (MagicMock):
                Mocked BaseSimulationData object with predefined attributes.

        Mocks:
            BaseSimulationData:
                A MagicMock data object with predefined values matching a BaseSimulationData class object.

        Asserts:
            The shape of the resulting water depth array matches the face_node shape.
            The calculated water depth matches the expected values.
        """
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

    @pytest.mark.unit
    def test_generate_bank_lines(self, mock_simulation_data):
        """Test the _generate_bank_lines method.

        This method generates bank lines based on the wet nodes, number of wet nodes,
        water depth at nodes, and critical water depth (h0). It calculates the bank lines
        by determining the start and end points of each line segment based on the water
        depth and wet nodes.

        Args:
            mock_simulation_data (MagicMock):
                Mocked BaseSimulationData object with predefined attributes.

        Mocks:
            BaseSimulationData:
                A MagicMock data object with predefined values matching a BaseSimulationData class object.

        Asserts:
            The generated bank lines match the expected LineString objects.
            The method correctly handles the wet nodes and water depth to create
                appropriate bank lines.
        """
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
    @pytest.mark.unit
    def test_detect_bank_lines(
        self, mock_simulation_data, mock_config_file, face_node, n_nodes, expected
    ):
        """Test the detect_bank_lines method.

        Test the detect_bank_lines method with different face_node inputs.
        triangle faces: a simple triangle face with 3 nodes.
        polygonal faces: a polygon with 4 nodes.
        masked faces: a masked array with 4 nodes, where the last row is masked.

        Args:
            mock_simulation_data (MagicMock):
                Mocked BaseSimulationData object with predefined attributes.
            mock_config_file (MagicMock):
                Mocked ConfigFile object with predefined attributes.
            face_node (np.ndarray):
                The face node array to test.
            n_nodes (np.ndarray):
                The number of nodes per face to test.
            expected (LineString or MultiLineString):
                The expected result for the given face_node and n_nodes.

        Mocks:
            BaseSimulationData:
                A MagicMock data object with predefined values matching a BaseSimulationData class object.
            ConfigFile:
                A MagicMock data object with predefined values matching a ConfigFile class object.

        Asserts:
            The resulting GeoSeries contains the expected LineString or MultiLineString.
            The length of the GeoSeries matches the expected number of bank lines.
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

    @pytest.mark.unit
    def test_save(self, mock_config_file, tmp_path: Path):
        """Test the save method of the BankLines class.

        This test checks if the save method correctly saves the bank lines, bank areas,
        and bankline fragments per bank area to the specified output directory.

        Args:
            mock_config_file (MagicMock): Mocked ConfigFile object with predefined attributes.
            tmp_path (Path): Temporary path to save the output files.

        Mocks:
            BankLinesRiverData:
                Mocked to return a MagicMock data object with
                predefined values matching a BankLinesRiverData class object.

        Asserts:
            The saved files exist in the output directory.
        """
        bank = [LineString([(0, 0), (1, 1)])]
        banklines = gpd.GeoSeries([LineString([(0, 0), (1, 1)])], crs="EPSG:4326")
        clipped_banklines = [MultiLineString([LineString([(0, 0), (1, 1)])])]
        bank_areas = [Polygon([(0, 0), (1, 1), (1, 0)])]

        mock_config_file.get_str.return_value = "bank_file"
        mock_config_file.get_output_dir.return_value = tmp_path
        mock_config_file.crs = "EPSG:4326"
        with patch(
            "dfastbe.bank_lines.bank_lines.BankLinesRiverData"
        ) as mock_river_data:
            mock_river_data.return_value.simulation_data.return_value = (
                MagicMock(),
                0.3,
            )
            bank_lines = BankLines(mock_config_file)
        bank_lines.results = {
            "bank": bank,
            "banklines": banklines,
            "masked_bank_lines": clipped_banklines,
            "bank_areas": bank_areas,
        }

        with patch("dfastbe.bank_lines.bank_lines.log_text"):
            bank_lines.save()

        assert (tmp_path / "bank_file.shp").exists()
        assert (tmp_path / "raw_detected_bankline_fragments.shp").exists()
        assert (tmp_path / "bank_areas.shp").exists()
        assert (tmp_path / "bankline_fragments_per_bank_area.shp").exists()

    @pytest.mark.unit
    def test_plot(self, mock_config_file, mock_simulation_data, tmp_path: Path):
        """Test the plot method of the BankLines class.

        This test checks if the plot method correctly plots the bank lines and saves the
        plot to the specified directory.

        Args:
            mock_config_file (MagicMock):
                Mocked ConfigFile object with predefined attributes.
            mock_simulation_data (MagicMock):
                Mocked BaseSimulationData object with predefined attributes.
            tmp_path (Path):
                Temporary path to save the output plot.

        Mocks:
            BankLinesRiverData:
                Mocked to return a MagicMock data object with
                predefined values matching a BankLinesRiverData class object.
            pyplot.show:
                Mocked to prevent displaying the plot on screen during tests,
                ensuring compatibility with non-GUI environments.
            pyplot.close:
                Mocked to prevent closing the plot during tests,
                ensuring compatibility with non-GUI environments.
            BasePlot.zoom_xy_and_save:
                Mocked to simulate the zooming and saving of the plot.

        Asserts:
            The plot is saved to the specified directory.
            The zoom_xy_and_save method is called.
            The show and close methods of matplotlib.pyplot are called.
        """
        xy_km_numpy = LineGeometry(line=np.array([[0, 0, 0], [1, 1, 0]]))
        n_search_lines = 1
        bank = [LineString([(0, 0), (1, 1)])]
        km_bounds = (0, 1)
        bank_areas = [Polygon([(0, 0), (1, 1), (1, 0)])]

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

        with patch("matplotlib.pyplot.show") as mock_show, patch(
            "matplotlib.pyplot.close"
        ) as mock_close, patch(
            "dfastbe.plotting.BasePlot.zoom_xy_and_save"
        ) as mock_zoom_xy_and_save, patch(
            "dfastbe.bank_lines.plotter.log_text"
        ):

            bank_lines_plotter = BankLinesPlotter(
                False, bank_lines.plot_flags, mock_config_file.crs, mock_simulation_data, xy_km_numpy, km_bounds,
            )
            bank_lines_plotter.plot(
                n_search_lines,
                bank,
                bank_areas,
            )

            mock_zoom_xy_and_save.assert_called_once()
            mock_show.assert_called_once()
            mock_close.assert_called_once()
            assert (tmp_path / "1_bankline-detection.png").exists()

    @pytest.mark.unit
    def test_mask(self):
        """Test the mask method of the BankLines class.

        This test checks if the mask method correctly clips bank lines to a specified area.

        Asserts:
            - The clipped bank lines are of type LineString.
            - The clipped bank lines match the expected geometry.
        """
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
