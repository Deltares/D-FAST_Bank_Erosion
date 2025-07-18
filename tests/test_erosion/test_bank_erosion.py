from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np
import pytest
from matplotlib.testing.compare import compare_images
from pyfakefs.fake_filesystem import FakeFilesystem

import dfastbe.io.logger
from dfastbe.bank_erosion.bank_erosion import Erosion, calculate_alpha
from dfastbe.bank_erosion.data_models.calculation import (
    BankData,
    ErosionInputs,
    ErosionResults,
    FairwayData,
    SingleBank,
    SingleDischargeLevel,
    SingleLevelParameters,
    SingleParameters,
)
from dfastbe.bank_erosion.data_models.inputs import ErosionSimulationData
from dfastbe.bank_erosion.erosion_calculator import ErosionCalculator
from dfastbe.cmd import run
from dfastbe.io.config import ConfigFile
from dfastbe.io.data_models import LineGeometry

matplotlib.use('Agg')


@pytest.fixture
def image_list() -> List[str]:
    """Fixture to provide a list of all expected resulting images from an erosion run."""
    return [
        "1_bankline-detection.png",
        "1_banklines.png",
        "2_erosion_sensitivity.png",
        "3_eroded_volume.png",
        "4_eroded_volume_per_discharge.png",
        "5_eroded_volume_per_bank.png",
        "6_eroded_volume_eq.png",
        "7_levels_bank_1.png",
        "8_levels_bank_2.png",
        "9_velocity_bank_1.png",
        "10_velocity_bank_2.png",
        "11_banktype.png",
        "12_erodis.png",
    ]


@pytest.mark.e2e
def test_bank_erosion(image_list: List[str]):
    file = "erosion"
    language = "UK"
    config_file = f"tests/data/{file}/meuse_manual.cfg"
    run(language, "BANKLINES", config_file)
    print("Banklines done")
    run(language, "BANKEROSION", config_file)
    print("Bank erosion done")
    test_path = Path(f"./tests/data/{file}")

    output_dir = test_path / "output/figures"
    reference_dir = test_path / "reference/figures"

    for image in image_list:
        reference_img = reference_dir / image
        output_img = output_dir / image
        assert output_img.exists()
        assert compare_images(str(output_img), str(reference_img), 0.0001) is None


class TestErosion:

    @pytest.fixture
    def shipping_data(self) -> Dict[str, list]:
        """Fixture to create mock shipping data.

        Returns:
            Dict[str, list]: A dictionary with mock shipping data arrays.
                vship0 (np.array):
                    initial ship velocities for two banks.
                Nship0 (np.array):
                    initial number of ships for two banks.
                nwave0 (np.array):
                    initial amount of waves per ship for two banks.
                Tship0 (np.array):
                    initial ship periods for two banks.
                ship0 (np.array):
                    initial ship heights for two banks.
                parslope0 (np.array):
                    initial bank slope parameters for two banks.
                parreed0 (np.array):
                    initial bank vegetation parameters for two banks.
        """
        return {
            "vship0": [np.array([5.0, 5.0, 5.0]), np.array([5.0, 5.0, 5.0])],
            "Nship0": [
                np.array([20912, 20912, 20912]),
                np.array([20912, 20912, 20912]),
            ],
            "nwave0": [np.array([5.0, 5.0, 5.0]), np.array([5.0, 5.0, 5.0])],
            "Tship0": [np.array([1.2, 1.2, 1.2]), np.array([1.2, 1.2, 1.2])],
            "ship0": [np.array([2.0, 2.0, 2.0]), np.array([2.0, 2.0, 2.0])],
            "parslope0": [np.array([20.0, 20.0, 20.0]), np.array([20.0, 20.0, 20.0])],
            "parreed0": [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])],
        }

    @pytest.fixture
    def mock_erosion(self):
        """Fixture to patch and mock the __init__ method of the Erosion class.

        This allows us to create an instance of Erosion without executing the original __init__ method,
        which may involve file system operations or other side effects that are not suitable for unit tests.

        Yields:
            Erosion: A mock instance of the Erosion class.
        """
        with patch(
            "dfastbe.bank_erosion.bank_erosion.Erosion.__init__", return_value=None
        ):
            erosion_instance = Erosion(MagicMock(), MagicMock())

            erosion_instance.root_dir = Path("mock_root_dir")
            erosion_instance.logger = MagicMock()
            erosion_instance._config_file = MagicMock()
            erosion_instance.gui = False
            erosion_instance.river_data = MagicMock()
            erosion_instance.river_center_line_arr = MagicMock()
            erosion_instance.simulation_data = MagicMock()
            erosion_instance.sim_files = MagicMock()
            erosion_instance.p_discharge = MagicMock()
            erosion_instance.bl_processor = MagicMock()
            erosion_instance.debugger = MagicMock()
            erosion_instance.erosion_calculator = MagicMock()

            yield erosion_instance

    @pytest.fixture
    def mock_config_file(self):
        """Fixture to create a mock ConfigFile.

        Creates a mock ConfigFile instance with predefined parameters.

        Returns:
            MagicMock: A mocked ConfigFile instance.
        """
        mock_config = MagicMock(spec=ConfigFile)
        mock_config.get_parameter.side_effect = (
            lambda section, key, num_stations, **kwargs: [
                np.array([1.0] * n) for n in num_stations
            ]
        )
        mock_config.crs = "EPSG:28992"
        return mock_config

    @pytest.fixture
    def mock_debug(self):
        with patch.object(dfastbe.io.logger, "PROGTEXTS", {}, create=True):
            yield

    @pytest.mark.unit
    def test_get_ship_parameters(self, mock_erosion: Erosion, mock_config_file):
        """Test the get_ship_parameters method.

        This method retrieves ship parameters based on the number of stations per bank.
        Leverage the mock_config_file to simulate the configuration file behavior.

        Args:
            mock_erosion (Erosion): The Erosion instance to test.
            mock_config_file (MagicMock): A mocked ConfigFile instance.

        Mocks:
            Erosion:
                The Erosion instance without executing the original __init__ method.
            ConfigFile:
                The behavior of the get_parameter method to return predefined numpy arrays.

        Asserts:
            The returned ship parameters are a dictionary with expected keys.
            Each value in the dictionary is a list of numpy arrays,
                each array's length matches the number of stations per bank.
        """
        num_stations_per_bank = [10, 15]
        mock_erosion._config_file = mock_config_file

        ship_parameters = mock_erosion.get_ship_parameters(num_stations_per_bank)

        expected_keys = [
            "vship0",
            "Nship0",
            "nwave0",
            "Tship0",
            "ship0",
            "parslope0",
            "parreed0",
        ]
        mock_config_file.get_parameter.assert_called()
        assert isinstance(ship_parameters, dict)
        assert set(ship_parameters.keys()) == set(expected_keys)

        for key, value in ship_parameters.items():
            assert isinstance(value, list)
            assert len(value) == len(num_stations_per_bank)
            for arr, n in zip(value, num_stations_per_bank):
                assert isinstance(arr, np.ndarray)
                assert len(arr) == n

    @pytest.mark.integration
    def test_prepare_initial_conditions(
        self, mock_erosion: Erosion, shipping_data, mock_config_file
    ):
        """Test the _prepare_initial_conditions method.

        This method tests the prepare initial conditions for bank erosion calculations.

        Args:
            mock_erosion (Erosion):
                The Erosion instance to test.
            shipping_data (Dict[str, list]):
                Mocked shipping data to be used in the test.
            mock_config_file (MagicMock):
                A mocked ConfigFile instance.

        Mocks:
            FairwayData:
                A mocked FairwayData instance to simulate fairway data retrieval.
            ConfigFile:
                A mocked ConfigFile instance to simulate configuration file behavior.
            get_ship_parameters:
                A mocked method to return predefined shipping data.

        Asserts:
            The returned erosion inputs contain expected shipping data and parameters.
            The bank protection levels, wave fairway distances, and taucls are correctly set.
            The mocked config file methods are called.
        """
        num_stations_per_bank = [3, 3]
        mock_fairway_data = MagicMock(spec=FairwayData)
        mock_fairway_data.fairway_initial_water_levels = [
            np.array([10, 20, 30]),
            np.array([10, 20, 30]),
        ]
        taucls = np.array([1, 1, 1])
        taucls_str = (
            "protected",
            "vegetation",
            "good clay",
            "moderate/bad clay",
            "sand",
        )
        mock_config_file.get_parameter.side_effect = [
            [np.array([150.0, 150.0, 150.0]), np.array([150.0, 150.0, 150.0])],  # Wave0
            [np.array([110.0, 110.0, 110.0]), np.array([110.0, 110.0, 110.0])],  # Wave1
            [np.array([1.0, 1.0, 1.0]), np.array([0.18, 0.18, 0.18])],  # BankType
            [
                np.array([-13.0, -13.0, -13.0]),
                np.array([-13.0, -13.0, -13.0]),
            ],  # ProtectionLevel
        ]

        mock_config_file.get_bool.return_value = False
        mock_erosion._config_file = mock_config_file

        with patch(
            "dfastbe.bank_erosion.bank_erosion.Erosion.get_ship_parameters",
            return_value=shipping_data,
        ):
            erosion_inputs = mock_erosion._prepare_initial_conditions(
                num_stations_per_bank, mock_fairway_data
            )

        mock_config_file.get_parameter.assert_called()
        mock_config_file.get_bool.assert_called_once()

        assert np.array_equal(
            erosion_inputs.shipping_data["vship0"][0], np.array([5.0, 5.0, 5.0])
        )
        assert np.array_equal(
            erosion_inputs.left.wave_fairway_distance_0, np.array([150, 150, 150])
        )
        assert np.array_equal(
            erosion_inputs.left.wave_fairway_distance_1, np.array([110, 110, 110])
        )
        assert np.array_equal(
            erosion_inputs.bank_protection_level[0], np.array([-13, -13, -13])
        )
        assert np.array_equal(erosion_inputs.tauc[0], taucls)
        assert erosion_inputs.taucls_str == taucls_str
        assert len(erosion_inputs.bank_type) == 4

    @pytest.mark.unit
    def test_process_river_axis_by_center_line(self, mock_erosion: Erosion, mock_debug):
        """Test the _process_river_axis_by_center_line method.

        This method processes the river axis based on the center line of the river.

        Args:
            mock_erosion (Erosion): The Erosion instance to test.

        Mocks:
            LineGeometry:
                A mocked LineGeometry instance to simulate line geometry operations.
            Erosion:
                The Erosion instance without executing the original __init__ method.

        Asserts:
            The river axis is processed correctly based on the center line.
            The mocked LineGeometry methods are called.
        """
        mock_center_line = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        mock_erosion.river_center_line_arr = mock_center_line

        with patch(
            "dfastbe.bank_erosion.bank_erosion.LineGeometry"
        ) as mock_line_geometry:
            mock_line_geometry.return_value = MagicMock()
            mock_line_geometry.return_value.as_array.return_value = np.array(
                [
                    [118594.085937, 414471.53125],
                    [118608.34068032, 414475.92354911],
                    [118622.59542364, 414480.31584821],
                    [118636.85016696, 414484.70814732],
                    [118651.10491029, 414489.10044643],
                ]
            )
            mock_line_geometry.return_value.intersect_with_line.return_value = np.array(
                [128.0, 128.0, 128.0, 128.0, 128.0]
            )
            river_axis = mock_erosion._process_river_axis_by_center_line()

        mock_line_geometry.return_value.as_array.assert_called_once()
        mock_line_geometry.return_value.intersect_with_line.assert_called_once()
        river_axis.add_data.assert_called_with(data={"stations": np.array([128.0])})

    def test_get_fairway_data(
        self, mock_erosion: Erosion, mock_config_file, mock_debug
    ):
        """Test the _get_fairway_data method.

        This method retrieves fairway data by intersecting the river center line with the mesh.

        Args:
            mock_erosion (Erosion): The Erosion instance to test.
            mock_config_file (MagicMock): A mocked ConfigFile instance.

        Mocks:
            FairwayData:
                A mocked FairwayData instance to simulate fairway data retrieval.
            intersect_line_mesh:
                A mocked function to simulate the intersection of a line with a mesh.
            GeoDataFrame:
                A mocked GeoDataFrame to simulate the saving of fairway data to a file.

        Asserts:
            The fairway data contains the expected intersection coordinates and face indices.
        """
        mock_erosion.river_data.debug = True
        mock_erosion._config_file = mock_config_file
        with patch("dfastbe.bank_erosion.data_models.calculation.FairwayData"), patch(
            "dfastbe.bank_erosion.bank_erosion.intersect_line_mesh"
        ) as line_mock, patch("dfastbe.io.data_models.GeoDataFrame") as gdf_mock:
            fairway_intersection_coords = np.array(
                [
                    [209186.621094, 389659.99609375],
                    [209187.69800938, 389665.38986148],
                    [209189.26657398, 389673.24607124],
                    [209189.367188, 389673.75],
                    [209192.19921925, 389687.4921875],
                    [209195.0312505, 389701.234375],
                    [209195.96700092, 389705.77502325],
                ]
            )
            fairway_face_indices = np.array([59166, 59167, 62557, 62557, 62557, 62557])
            line_mock.return_value = (fairway_intersection_coords, fairway_face_indices)
            fairway_data = mock_erosion._get_fairway_data(MagicMock(), MagicMock())
            gdf_mock.return_value.to_file.assert_called_once()
        assert np.allclose(fairway_data.fairway_face_indices, fairway_face_indices)
        assert np.allclose(
            fairway_data.intersection_coords, fairway_intersection_coords
        )

    @pytest.fixture
    def read_grid_data(self):
        """Fixture to read grid data from files.

        Reads nodes, x_coords, and y_coords from specified text files using np.loadtxt.
        First 30 entries are fairway squares, second 30 is the right bank, third 30 is the left bank.

        Returns:
            - np.ma.array: Masked array of node indices.
                - (0,29) entries are fairway squares,
                - (30,59) entries are the right bank squares,
                - (60,89) entries are the left bank squares.
        """
        nodes = np.loadtxt(
            "tests/data/input/meuse_cropped_data/erosion_simulation_data/grid_nodes.txt",
            dtype=int,
            comments="#",
            delimiter=",",
        )
        return np.ma.array(nodes)

    @pytest.fixture
    def read_line_data(self):
        """Fixture to read line data from a text file.

        Reads points from a specified text file using np.loadtxt.
        The 30 entries intersects the fairway, the second 30 entries are the right bank,
        and the third 30 entries are the left bank.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - fairway_points:
                    Fairway line data corresponds to the first 2 columns (0, 1),
                    containing (x,y) coordinates.
                - left_bank:
                    Left bank data corresponds to the second 2 columns (2, 3) (x,y),
                    containing (x,y) coordinates.
                - right_bank:
                    Right bank data corresponds to the third 2 columns (4, 5) (x,y),
                    containing (x,y) coordinates.
        """
        data = np.loadtxt(
            "tests/data/input/meuse_cropped_data/bank_data/line_points.txt",
            comments="#",
            delimiter=",",
        )
        fairway_points = np.array([(entry[0], entry[1]) for entry in data])
        left_bank = np.array([(entry[1], entry[2]) for entry in data])
        right_bank = np.array([(entry[2], entry[3]) for entry in data])
        return fairway_points, left_bank, right_bank

    @pytest.fixture
    def read_face_values(self):
        """Fixture to read face values from a text file.

        These values are coupled to the nodes in the grid data.

        Returns:
            np.ndarray: A multidemensional numpy array of values for each face read from a file.
                - velocity_x_face: first row,
                - velocity_y_face: second row,
                - water_depth_face: third row,
                - water_level_face: fourth row,
                - chezy_face: fifth row.
        """
        return np.loadtxt(
            "tests/data/input/meuse_cropped_data/erosion_simulation_data/grid_node_values.txt",
            comments="#",
            delimiter=",",
        )

    @pytest.fixture
    def read_coord_values(self):
        """Fixture to read coordinate values from a text file.

        These are coordinate values for each point in the grid a (x, y) with bed_elevation as z.

        Returns:
            Tuple[np.ma.array, np.ma.array, np.array]: A tuple containing:
                - x_coords: Masked array of x coordinates.
                - y_coords: Masked array of y coordinates.
                - bed_elevation: Array of bed elevation values.
        """
        data = np.loadtxt(
            "tests/data/input/meuse_cropped_data/erosion_simulation_data/grid_coordinates.txt",
            comments="#",
            delimiter=",",
        )
        return np.ma.array(data[0]), np.ma.array(data[1]), np.array(data[2])

    @pytest.fixture
    def mock_bank_data(self, read_line_data):
        """Fixture to create a mock BankData.

        Creates two SingleBank instances representing left and right banks,
        each with specific coordinates, face indices, and chainage midpoints.

        Args:
            read_line_data (np.ndarray): The points read from the input file.

        Returns:
            BankData: A BankData instance containing the left and right banks.
        """
        mock_left_bank = SingleBank(
            is_right_bank=False,
            bank_line_coords=read_line_data[1],
            bank_face_indices=np.arange(30, 59),
            bank_chainage_midpoints=np.array([123.00166634401488] * 29),
            fairway_face_indices=np.arange(0, 29),
        )

        mock_right_bank = SingleBank(
            is_right_bank=True,
            bank_line_coords=read_line_data[2],
            bank_face_indices=np.arange(60, 89),
            bank_chainage_midpoints=np.array([123.00943873095339] * 29),
            fairway_face_indices=np.arange(0, 29),
        )

        mock_bank_data = BankData(
            left=mock_left_bank, right=mock_right_bank, n_bank_lines=2
        )
        return mock_bank_data

    @pytest.fixture
    def mock_fairway_data(self, read_line_data):
        """Fixture to create a mock FairwayData.

        Creates a FairwayData instance with intersection coordinates and face indices.

        Args:
            read_line_data (np.ndarray): The points read from the input file.

        Returns:
            FairwayData: A FairwayData instance containing intersection coordinates and face indices.
        """
        points = read_line_data
        mock_fairway_data = FairwayData(
            intersection_coords=points[0],
            fairway_face_indices=np.arange(0, 29),
        )
        return mock_fairway_data

    @pytest.fixture
    def mock_single_level_parameters(self):
        """Fixture to create a mock SingleLevelParameters.

        Creates a SingleLevelParameters instance with predefined parameters for two banks.

        Returns:
            SingleLevelParameters: A SingleLevelParameters instance with predefined parameters.
                - id: 0
                - ship_velocity: [5.0, 5.0] for both banks
                - num_ship: [15613, 15613] for both banks
                - num_waves_per_ship: [5.0, 5.0] for both banks
                - ship_draught: [1.2, 1.2] for both banks
                - ship_type: [2, 2] for both banks
                - par_slope: [2.0, 2.0] for both banks
                - par_reed: [0.0, 0.0] for both banks
                - mu_slope: [0.5, 0.5] for both banks
                - mu_reed: [0.0, 0.0] for both banks
        """
        return SingleLevelParameters.from_column_arrays(
            {
                "id": 0,
                "ship_velocity": [
                    np.array([5.0] * 29),
                    np.array([5.0] * 29),
                ],
                "num_ship": [
                    np.array([15613] * 29),
                    np.array([15613] * 29),
                ],
                "num_waves_per_ship": [
                    np.array([5.0] * 29),
                    np.array([5.0] * 29),
                ],
                "ship_draught": [
                    np.array([1.2] * 29),
                    np.array([1.2] * 29),
                ],
                "ship_type": [
                    np.array([2] * 29),
                    np.array([2] * 29),
                ],
                "par_slope": [
                    np.array([2.0] * 29),
                    np.array([2.0] * 29),
                ],
                "par_reed": [
                    np.array([0.0] * 29),
                    np.array([0.0] * 29),
                ],
                "mu_slope": [
                    np.array([0.5] * 29),
                    np.array([0.5] * 29),
                ],
                "mu_reed": [
                    np.array([0.0] * 29),
                    np.array([0.0] * 29),
                ],
            },
            SingleParameters,
        )

    @pytest.fixture
    def mock_simulation_data(self, read_grid_data, read_face_values, read_coord_values):
        """Fixture to create a mock ErosionSimulationData.

        Creates an ErosionSimulationData instance with bed elevation values, chezy face values,
        dry-wet threshold, face-node indices, number of nodes, velocity components,
        water depth, water level, and node coordinates.

        Args:
            read_grid_data (tuple): A tuple containing nodes, x_coords, and y_coords.
            read_face_values (np.ndarray): The face values read from the input file.
            read_coord_values (np.ndarray): The coordinate values read from the input file.

        Returns:
            ErosionSimulationData: An ErosionSimulationData instance with the following attributes:
                - bed_elevation_location: "node"
                - bed_elevation_values: np.ma.array(read_coord_values)
                - chezy_face: np.ma.array(read_face_values[4])
                - dry_wet_threshold: 0.1
                - face_node: nodes
                - n_nodes: np.array([4] * 90)
                - velocity_x_face: np.ma.array(read_face_values[0])
                - velocity_y_face: np.ma.array(read_face_values[1])
                - water_depth_face: np.ma.array(read_face_values[2])
                - water_level_face: np.ma.array(read_face_values[3])
                - x_node: x_coords
                - y_node: y_coords
        """
        return ErosionSimulationData(
            bed_elevation_location="node",
            bed_elevation_values=np.ma.array(read_coord_values[2]),
            chezy_face=np.ma.array(read_face_values[4]),
            dry_wet_threshold=0.1,
            face_node=read_grid_data,
            n_nodes=np.array([4] * 90),
            velocity_x_face=np.ma.array(read_face_values[0]),
            velocity_y_face=np.ma.array(read_face_values[1]),
            water_depth_face=np.ma.array(read_face_values[2]),
            water_level_face=np.ma.array(read_face_values[3]),
            x_node=read_coord_values[0],
            y_node=read_coord_values[1],
        )

    def test_calculate_fairway_bank_line_distance(
        self,
        mock_erosion: Erosion,
        mock_config_file,
        mock_bank_data,
        mock_fairway_data,
        mock_simulation_data,
        mock_debug,
    ):
        """Test the calculate_fairway_bank_line_distance method.

        This test, tests the calculation of the distance between the fairway and bank lines.

        Args:
            mock_erosion (Erosion):
                The Erosion instance to test.
            mock_config_file (MagicMock):
                A mocked ConfigFile instance.
            mock_bank_data (BankData):
                A BankData instance containing left and right banks from real data.
            mock_fairway_data (FairwayData):
                A FairwayData instance containing intersection points from real data.
            mock_simulation_data (ErosionSimulationData):
                An ErosionSimulationData instance containing a section of the real test case data.

        Mocks:
            Erosion:
                The Erosion instance without executing the original __init__ method.
            ConfigFile:
                The behavior of the ConfigFile to return predefined parameters.
            GeoDataFrame:
                A mocked GeoDataFrame to simulate the saving of fairway data to a file.

        Asserts:
            The fairway initial water levels are calculated correctly based on the bank data and fairway data.
            The calculated fairway initial water levels match the expected values.
        """
        mock_erosion._config_file = mock_config_file

        with patch("dfastbe.io.data_models.LineGeometry.to_file"):
            mock_erosion.calculate_fairway_bank_line_distance(
                mock_bank_data, mock_fairway_data, mock_simulation_data
            )

        assert np.allclose(
            mock_fairway_data.fairway_initial_water_levels,
            np.array(
                [
                    [
                        [
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                            1.152048969268798828e01,
                        ],
                        [
                            1.152041339874267578e01,
                            1.152041339874267578e01,
                            1.151991367340087891e01,
                            1.151930999755859375e01,
                            1.151847076416015625e01,
                            1.151847076416015625e01,
                            1.151850986480712891e01,
                            1.151728820800781250e01,
                            1.151728820800781250e01,
                            1.151696395874023438e01,
                            1.151589488983154297e01,
                            1.151629066467285156e01,
                            1.151629066467285156e01,
                            1.151542472839355469e01,
                            1.151440048217773438e01,
                            1.151440048217773438e01,
                            1.151371192932128906e01,
                            1.151291656494140625e01,
                            1.151322746276855469e01,
                            1.151289367675781250e01,
                            1.151226520538330078e01,
                            1.151273822784423828e01,
                            1.151253795623779297e01,
                            1.151153373718261719e01,
                            1.151172065734863281e01,
                            1.151096057891845703e01,
                            1.151048469543457031e01,
                            1.151048469543457031e01,
                            1.151048469543457031e01,
                        ],
                    ],
                ]
            ),
        )

    @pytest.mark.unit
    def test_process_discharge_levels(self, mock_erosion: Erosion, mock_debug):
        """Test the _process_discharge_levels method.

        This method processes discharge levels and calculates erosion results.

        Args:
            mock_erosion (Erosion): The Erosion instance to test.

        Mocks:
            _read_discharge_parameters:
                Mocks the read method to return a SingleLevelParameters instance with predefined parameters.
            ErosionSimulationData.read:
                Mocks the read method to return an instance containing a section of the real test case data.
            _calculate_bank_height:
                Mocks the method to return a MagicMock instance simulating bank height calculations.
            compute_erosion_per_level:
                Mocks the method to return two MagicMock instances simulating erosion calculations.
            write_km_eroded_volumes:
                Mocks the method to simulate writing eroded volumes to a file.
            DischargeLevels:
                Mocks the DischargeLevels class to simulate the accumulation of erosion distributions.

        Asserts:
            The write_km_eroded_volumes method is called, indicating that eroded volumes are written to a file.
            The method returns a WaterLevelData instance and an ErosionResults instance.
            The erosion results contain flow erosion distribution, ship erosion distribution,
            total erosion distribution, and total eroded volume arrays that match the expected values.
        """
        km_bin = np.array([123.0, 128.0, 0.1])
        km_mid = np.array(
            [123.05, 123.15, 123.25, 123.35, 123.45, 123.55, 123.65, 123.75]
        )
        erosion_inputs = MagicMock()
        bank_data = MagicMock()
        fairway_data = MagicMock()
        mock_erosion.river_data.erosion_time = 1

        with patch(
            "dfastbe.bank_erosion.bank_erosion.Erosion._read_discharge_parameters"
        ) as mock_read_discharge_parameters, patch(
            "dfastbe.bank_erosion.bank_erosion.ErosionSimulationData.read"
        ) as mock_read_data, patch(
            "dfastbe.bank_erosion.bank_erosion.Erosion._calculate_bank_height"
        ) as mock_calculate_bank_height, patch(
            "dfastbe.bank_erosion.bank_erosion.Erosion.compute_erosion_per_level"
        ) as mock_compute_erosion, patch(
            "dfastbe.bank_erosion.bank_erosion.write_km_eroded_volumes"
        ) as mock_write_km_eroded_volumes, patch(
            "dfastbe.bank_erosion.bank_erosion.DischargeLevels"
        ) as mock_discharge_levels:
            flow_erosion_dist = [
                np.array([7.06542424e-02, 6.75617155e-02, 7.01268742e-02]),
                np.array([0.10222567, 0.10100284, 0.09953936]),
            ]
            ship_erosion_dist = [
                np.array([0.02990375, 0.02993445, 0.02986104]),
                np.array([0.15359159, 0.15620181, 0.15601963]),
            ]
            total_erosion_dist = [
                np.array([1.00557989e-01, 9.74961629e-02, 9.99879152e-02]),
                np.array([0.25581725, 0.25720465, 0.25555899]),
            ]
            total_eroded_vol = [
                np.ma.array(data=[2.52356124e00, 1.00578263e01, 1.35324671e01]),
                np.ma.array(data=[5.56431863e00, 1.27198016e01, 1.34650868e01]),
            ]
            discharge_levels_instance = mock_discharge_levels.return_value
            discharge_levels_instance.accumulate.side_effect = [
                flow_erosion_dist,
                ship_erosion_dist,
                total_erosion_dist,
                total_eroded_vol,
            ]
            discharge_levels_instance.get_attr_both_sides_level.side_effect = [
                [
                    np.ma.array(data=[9.304824589232581, 9.272062777155057]),
                    np.ma.array(data=[8.946751000892306, 8.986074721759323]),
                ],
                [
                    np.ma.array(data=[78.08261970217333, 328.12862224567755]),
                    np.ma.array(data=[115.70074874647455, 268.66556572232156]),
                ],
            ]
            mock_discharge_levels.get_water_level_data.return_value = MagicMock()
            mock_read_discharge_parameters.return_value = MagicMock()
            mock_read_data.return_value = MagicMock()
            mock_calculate_bank_height.return_value = MagicMock()
            mock_compute_erosion.return_value = (MagicMock(), MagicMock())

            water_level_data, erosion_results = mock_erosion._process_discharge_levels(
                km_mid, km_bin, erosion_inputs, bank_data, fairway_data
            )

            assert mock_write_km_eroded_volumes.called

        assert isinstance(water_level_data, MagicMock)  # WaterLevelData
        assert isinstance(erosion_results, ErosionResults)  # ErosionResults
        assert np.allclose(erosion_results.flow_erosion_dist, flow_erosion_dist)
        assert np.allclose(erosion_results.ship_erosion_dist, ship_erosion_dist)
        assert np.allclose(erosion_results.total_erosion_dist, total_erosion_dist)
        assert np.allclose(erosion_results.total_eroded_vol, total_eroded_vol)

    def test_read_discharge_parameters(self, mock_erosion: Erosion, shipping_data):
        """Test the _read_discharge_parameters method.

        This method reads discharge parameters for a specific discharge level.

        Args:
            mock_erosion (Erosion):
                The Erosion instance to test.
            shipping_data (dict):
                The shipping data to use for testing.

        Mocks:
            Erosion._get_param:
                Mocks the method to return predefined values for testing.
            Erosion:
                The Erosion instance without executing the original __init__ method.

        Asserts:
            The returned discharge parameters are an instance of SingleLevelParameters.
            The parameters match the expected values from the shipping data.
        """
        with patch(
            "dfastbe.bank_erosion.bank_erosion.Erosion._get_param"
        ) as mock_get_param:
            mock_get_param.side_effect = [
                shipping_data["vship0"],
                shipping_data["Nship0"],
                shipping_data["nwave0"],
                shipping_data["Tship0"],
                shipping_data["ship0"],
                shipping_data["parslope0"],
                shipping_data["parreed0"],
            ]
            discharge_parameters = mock_erosion._read_discharge_parameters(
                1, shipping_data, [13]
            )
        assert isinstance(discharge_parameters, SingleLevelParameters)
        assert discharge_parameters.id == 1
        assert np.allclose(
            discharge_parameters.left.ship_velocity, shipping_data["vship0"]
        )
        assert np.allclose(discharge_parameters.left.num_ship, shipping_data["Nship0"])
        assert np.allclose(
            discharge_parameters.left.num_waves_per_ship, shipping_data["nwave0"]
        )

    def test_compute_erosion_per_level(self, mock_erosion: Erosion, mock_debug):
        """Test the compute_erosion_per_level method.

        This method computes the erosion for a specific discharge level.

        Args:
            mock_erosion (Erosion): The Erosion instance to test.

        Mocks:
            ErosionSimulationData:
                A mocked ErosionSimulationData instance to simulate simulation data.
            BankData:
                A mocked BankData instance to simulate bank data.
            FairwayData:
                A mocked FairwayData instance to simulate fairway data.
            SingleLevelParameters:
                A mocked SingleLevelParameters instance to simulate parameters for a single discharge level.
            ErosionInputs:
                A mocked ErosionInputs instance to simulate erosion inputs.
            get_km_eroded_volume:
                A mocked function to simulate the retrieval of eroded volume per kilometer.

        Asserts:
            The method returns a SingleDischargeLevel instance and a 2D numpy array of eroded volumes.
            The SingleDischargeLevel instance has the expected maximum water level.
            The eroded volume array is correctly initialized with zeros.
        """
        mock_erosion.river_data.num_discharge_levels = 2
        bank_data = MagicMock(spec=BankData)
        bank_data.__iter__.return_value = [MagicMock(), MagicMock()]
        simulation_data = MagicMock()
        simulation_data.calculate_bank_velocity.return_value = np.array(
            [0.036150366339510215, 0.035504928556677105, 0.04197925796741744]
        )
        simulation_data.get_fairway_data.return_value = {
            "water_depth": np.ma.array(data=[5.55804443, 5.47807884, 5.47807884]),
            "water_level": np.ma.array(data=[11.10304451, 11.10307884, 11.10307884]),
            "chezy": np.ma.array(data=[79.57007898, 79.57007898, 79.57007898]),
        }
        fairway_data = MagicMock(spec=FairwayData)
        fairway_data.fairway_initial_water_levels = [
            np.array([10.0, 10.3, 10.4]),
            np.array([10.2, 10.7, 10.3]),
        ]
        single_parameters = MagicMock(spec=SingleLevelParameters)
        erosion_inputs = MagicMock(spec=ErosionInputs)
        km_bin = np.array([123.0, 128.0, 0.1])
        with patch(
            "dfastbe.bank_erosion.bank_erosion.get_km_eroded_volume"
        ) as mock_get_km_eroded_volume:
            mock_get_km_eroded_volume.return_value = np.array([0.0] * 50)
            level_calculation, dvol_bank = mock_erosion.compute_erosion_per_level(
                0,
                bank_data,
                simulation_data,
                fairway_data,
                single_parameters,
                erosion_inputs,
                km_bin,
                50,
            )
        assert isinstance(level_calculation, SingleDischargeLevel)
        assert level_calculation.hfw_max == pytest.approx(5.55804443)
        assert np.allclose(dvol_bank, np.array([[0.0] * 2] * 50))

    def test_run(
        self,
        mock_erosion: Erosion,
        mock_fairway_data,
        mock_bank_data,
        shipping_data,
        mock_single_level_parameters,
        mock_simulation_data,
        mock_debug,
        fs: FakeFilesystem,
    ):
        """Test the run method.

        This test simulates running the bank erosion workflow and verifies,
        that output files are generated as expected. Input data and dependencies are mocked
        to isolate the behavior of the Erosion class, avoiding execution
        of the original __init__ method and file I/O.

        Args:
            mock_erosion (Erosion):
                The Erosion instance to test.
            mock_fairway_data (FairwayData):
                A FairwayData instance.
            mock_bank_data (BankData):
                A BankData instance.
            shipping_data (dict):
                A dictionary containing shipping data.
            mock_single_level_parameters (SingleLevelParameters):
                A SingleLevelParameters instance.
            mock_simulation_data (ErosionSimulationData):
                A ErosionSimulationData instance.

        Mocks:
            Erosion:
                The Erosion instance without executing the original __init__ method.
            LineGeometry:
                A mocked LineGeometry instance to simulate line geometry operations.
            GeoDataFrame:
                A mocked GeoDataFrame to simulate the saving of fairway data to a file.
            _get_fairway_data:
                Return the mock_fairway_data instance instead of executing the function.
            get_km_bins:
                A mocked function to return a predefined array of km midpoints.
            _process_river_axis_by_center_line:
                Return a mocked center line instead of executing the function.
            get_ship_parameters:
                A mocked function to return the shipping_data dictionary.
            _read_discharge_parameters:
                Return the mock_single_level_parameters instance instead of executing the function.
            ErosionSimulationData.read:
                Return the mock_simulation_data instance instead of executing the function.
            _write_bankline_shapefiles:
                A mocked function to simulate writing bankline shapefiles.
            _generate_plots:
                A mocked function to simulate generating plots.

        Asserts:
            - All mocked methods are called as expected.
            The run method processes the bank erosion simulation correctly.
            The output files are generated with the expected content.
        """
        mock_km_mid = MagicMock()
        mock_km_mid.return_value = np.arange(123.0, 123.6, 0.02)
        mock_erosion.bl_processor.intersect_with_mesh.return_value = mock_bank_data
        mock_erosion.simulation_data = mock_simulation_data
        mock_erosion.config_file.get_parameter.side_effect = [
            [np.array([150.0] * 29), np.array([150.0] * 29)],  # wave0
            [np.array([110.0] * 29), np.array([110.0] * 29)],  # wave1
            [np.array([1.0] * 29), np.array([0.18] * 29)],  # BankType
            [np.array([-13.0] * 29), np.array([-13.0] * 29)],  # ProtectionLevel
        ]
        mock_erosion.config_file.get_bool.return_value = False  # Classes
        # write km output filenames
        mock_erosion.config_file.get_str.side_effect = [
            "erovolQ1.evo",
            "erovolQ2.evo",
            "erovolQ3.evo",
            "erovol.evo",
            "erovol_eq.evo",
        ]

        mock_erosion.river_data.zb_dx = 0.3
        mock_erosion.river_data.vel_dx = 0.3
        mock_erosion.river_data.output_intervals = 0.02
        mock_erosion.river_data.erosion_time = 1.0
        mock_erosion.river_data.output_dir = Path("mock_output_dir")
        mock_erosion.river_data.num_discharge_levels = 3
        fs.create_dir(mock_erosion.river_data.output_dir)

        mock_erosion.erosion_calculator = ErosionCalculator()
        mock_erosion.p_discharge = np.array([0.311, 0.2329, 0.2055])
        center_line_mock = MagicMock(spec=LineGeometry)
        center_line_mock.data["stations"].min.return_value = 123.0
        center_line_mock.data["stations"].max.return_value = 123.61
        with (patch(
            "dfastbe.bank_erosion.bank_erosion.Erosion._get_fairway_data",
            return_value=mock_fairway_data,
        ) as mock_get_fairway_data, patch(
            "dfastbe.bank_erosion.bank_erosion.get_km_bins", mock_km_mid
        ) as mock_get_km_bins, patch(
            "dfastbe.bank_erosion.bank_erosion.Erosion._process_river_axis_by_center_line",
            return_value=center_line_mock,
        ) as mock_process_river_axis_by_center_line, patch(
            "dfastbe.io.data_models.LineGeometry.to_file",
        ), patch(
            "dfastbe.bank_erosion.bank_erosion.Erosion.get_ship_parameters",
            return_value=shipping_data,
        ) as mock_get_ship_parameters, patch(
            "dfastbe.bank_erosion.bank_erosion.Erosion._read_discharge_parameters",
            return_value=mock_single_level_parameters,
        ) as mock_read_discharge_parameters, patch(
            "dfastbe.bank_erosion.bank_erosion.ErosionSimulationData.read",
            return_value=mock_simulation_data,
        ) as mock_read_simulation_data,
            patch(
            "dfastbe.bank_erosion.bank_erosion.ErosionSimulationData.compute_mesh_topology"
        ) as mock_compute_mesh_topology):
            mock_erosion.run()

        mock_get_fairway_data.assert_called_once()
        mock_get_km_bins.assert_called_once()
        mock_process_river_axis_by_center_line.assert_called_once()
        mock_get_ship_parameters.assert_called_once()
        mock_read_discharge_parameters.assert_called()
        mock_read_simulation_data.assert_called()
        mock_compute_mesh_topology.assert_called_once()
        with open(mock_erosion.river_data.output_dir / "erovolQ3.evo", "r") as file:
            content = file.read()
            assert "123.0" in content
            assert "123.58" in content
            assert "0.0" in content


def test_calculate_alpha():
    """Test the calculate_alpha method."""
    coords = np.array(
        [
            [209186.621094, 389659.99609375],
            [209187.69800938, 389665.38986148],
            [209189.26657398, 389673.24607124],
            [209189.367188, 389673.75],
            [209192.19921925, 389687.4921875],
        ]
    )
    ind_1 = 1
    ind_2 = 0
    bp = np.array([209118.40334772525, 389682.4659760762])
    alpha = calculate_alpha(coords, ind_1, ind_2, bp)

    # Assert the result
    assert alpha == pytest.approx(1.5778075234167066)
