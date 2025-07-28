from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from geopandas import GeoDataFrame
from shapely.geometry import LineString

from dfastbe.bank_erosion.data_models.calculation import (
    BankData,
    ErosionInputs,
    ErosionResults,
    FairwayData,
    MeshData,
    SingleBank,
    SingleErosion,
    SingleLevelParameters,
    WaterLevelData,
)
from dfastbe.bank_erosion.data_models.inputs import (
    ErosionRiverData,
    ErosionSimulationData,
    ShipsParameters,
)
from dfastbe.io.config import ConfigFile


class TestErosionInputs:
    @pytest.mark.parametrize(
        "bank_order, result",
        [(("right", "left"), (1, 2)), (("left", "right"), (1, 2))],
        ids=[
            "right-left bank order",
            "left-right bank order",
        ],
    )
    def test_erosion_inputs(self, bank_order, result):
        """Test instantiation of the ErosionInputs dataclass."""
        shipping_data = {"ship1": [np.array([1.0, 1.0])]}
        bank_type = np.array([1.0, 1.0])
        data = dict(
            wave_fairway_distance_0=[np.array([1.0, 1.0]), np.array([2.0, 2.0])],
            wave_fairway_distance_1=[np.array([1.0, 1.0]), np.array([2.0, 2.0])],
            bank_protection_level=[np.array([1.0, 1.0]), np.array([2.0, 2.0])],
            tauc=[np.array([1.0, 1.0]), np.array([2.0, 2.0])],
        )
        erosion_inputs = ErosionInputs.from_column_arrays(
            data,
            SingleErosion,
            shipping_data=shipping_data,
            bank_order=bank_order,
            bank_type=bank_type,
        )
        assert (
            erosion_inputs.right.wave_fairway_distance_0[0]
            == result[bank_order.index("right")]
        )
        assert (
            erosion_inputs.left.wave_fairway_distance_0[0]
            == result[bank_order.index("left")]
        )
        assert erosion_inputs.shipping_data["ship1"][0] == pytest.approx(1.0)
        assert erosion_inputs.taucls[1] == 95
        assert erosion_inputs.taucls_str[0] == "protected"


def test_water_level_data():
    """Test instantiation of the WaterLevelData dataclass."""
    water_level_data = WaterLevelData(
        hfw_max=5.0,
        water_level=[[np.array([1.0, 2.0])]],
        ship_wave_max=[[np.array([0.5, 1.0])]],
        ship_wave_min=[[np.array([0.2, 0.4])]],
        velocity=[[np.array([0.1, 0.2])]],
        chezy=[[np.array([30.0, 40.0])]],
        vol_per_discharge=[[np.array([0.9, 1.0])]],
    )
    assert water_level_data.hfw_max == pytest.approx(5.0)
    assert water_level_data.water_level[0][0][1] == pytest.approx(2.0)


def test_mesh_data():
    """Test instantiation of the MeshData dataclass."""
    mesh_data = MeshData(
        x_face_coords=np.array([1.0, 2.0]),
        y_face_coords=np.array([3.0, 4.0]),
        x_edge_coords=np.array([5.0, 6.0]),
        y_edge_coords=np.array([7.0, 8.0]),
        face_node=np.array([[0, 1], [1, 2]]),
        n_nodes=np.array([3, 3]),
        edge_node=np.array([[0, 1], [1, 2]]),
        edge_face_connectivity=np.array([[0, 1], [1, -1]]),
        face_edge_connectivity=np.array([[0, 1], [1, 2]]),
        boundary_edge_nrs=np.array([0, 1]),
    )
    assert mesh_data.x_face_coords[0] == pytest.approx(1.0)
    assert mesh_data.face_node[1][1] == 2
    assert mesh_data.boundary_edge_nrs[1] == 1


class TestBankData:
    @pytest.mark.parametrize(
        "bank_order, result",
        [(("right", "left"), (1, 2)), (("left", "right"), (1, 2))],
        ids=[
            "right-left bank order",
            "left-right bank order",
        ],
    )
    def test_default_parameters(self, bank_order, result):
        """Test instantiation of the BankData dataclass."""
        bank_lines = GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])])
        n_bank_lines = 1
        bank_data = dict(
            is_right_bank=[True, False],
            bank_line_coords=[
                np.array([[1.0, 1.0], [1.0, 1.0]]),
                np.array([[2.0, 2.0], [2.0, 2.0]]),
            ],
            bank_face_indices=[np.array([1, 1]), np.array([2, 2])],
            bank_line_size=[np.array([1.0, 1.0]), np.array([2, 2])],
            fairway_distances=[np.array([1.0, 1.0]), np.array([2, 2])],
            fairway_face_indices=[np.array([1, 1]), np.array([2, 2])],
            bank_chainage_midpoints=[np.array([1.0, 1.0]), np.array([2.0, 2.0])],
        )
        bank_data = BankData.from_column_arrays(
            bank_data,
            SingleBank,
            bank_lines=bank_lines,
            n_bank_lines=n_bank_lines,
            bank_order=bank_order,
        )
        assert (
            bank_data.right.bank_line_coords[0, 0] == result[bank_order.index("right")]
        )
        assert bank_data.left.bank_line_coords[0, 0] == result[bank_order.index("left")]


def test_fairway_data():
    """Test instantiation of the FairwayData dataclass."""
    fairway_data = FairwayData(
        fairway_face_indices=np.array([0, 1]),
        intersection_coords=np.array([[0.0, 0.0], [1.0, 1.0]]),
        fairway_initial_water_levels=[np.array([1.0, 2.0])],
    )
    assert fairway_data.fairway_face_indices[1] == 1
    assert fairway_data.intersection_coords[1][1] == pytest.approx(1.0)
    assert fairway_data.fairway_initial_water_levels[0][1] == pytest.approx(2.0)


def test_erosion_results():
    """Test instantiation of the ErosionResults dataclass."""
    erosion_results = ErosionResults(
        eq_erosion_dist=[np.array([0.1, 0.2])],
        total_erosion_dist=[np.array([0.3, 0.4])],
        flow_erosion_dist=[np.array([0.5, 0.6])],
        ship_erosion_dist=[np.array([0.7, 0.8])],
        eq_eroded_vol=[np.array([1.1, 1.2])],
        total_eroded_vol=[np.array([1.3, 1.4])],
        erosion_time=10,
        avg_erosion_rate=np.array([0.1, 0.2]),
        eq_eroded_vol_per_km=np.array([0.3, 0.4]),
        total_eroded_vol_per_km=np.array([0.5, 0.6]),
    )
    assert erosion_results.eq_erosion_dist[0][1] == pytest.approx(0.2)
    assert erosion_results.total_erosion_dist[0][0] == pytest.approx(0.3)
    assert erosion_results.erosion_time == 10


class TestSimulationData:

    @pytest.fixture
    def simulation_data(self) -> ErosionSimulationData:
        x_node = np.array([194949.796875, 194966.515625, 194982.8125, 195000.0])
        y_node = np.array([361366.90625, 361399.46875, 361431.03125, 361450.0])
        n_nodes = np.array([4, 4])
        face_node = np.ma.masked_array(
            data=[[0, 1, 2, 3], [1, 2, 3, 0]],
            mask=[[False, False, False, False], [False, False, False, False]],
        )
        bed_elevation_location = "node"
        bed_elevation_values = np.array([10.0, 20.0, 30.0, 40.0])
        water_level_face = np.array([1.0, 2.0])
        water_depth_face = np.array([0.5, 1.0])
        velocity_x_face = np.array([0.1, 0.2])
        velocity_y_face = np.array([0.4, 0.5])
        chezy_face = np.array([30.0, 40.0])
        dry_wet_threshold = 0.1

        sim_data = ErosionSimulationData(
            x_node=x_node,
            y_node=y_node,
            n_nodes=n_nodes,
            face_node=face_node,
            bed_elevation_location=bed_elevation_location,
            bed_elevation_values=bed_elevation_values,
            water_level_face=water_level_face,
            water_depth_face=water_depth_face,
            velocity_x_face=velocity_x_face,
            velocity_y_face=velocity_y_face,
            chezy_face=chezy_face,
            dry_wet_threshold=dry_wet_threshold,
        )
        return sim_data

    def test_compute_mesh_topology(self, simulation_data: ErosionSimulationData):
        """
        Test the compute_mesh_topology method of SimulationData.
        """
        # Call the method to compute the mesh topology
        mesh_data = simulation_data.compute_mesh_topology()

        assert isinstance(mesh_data, MeshData)

        assert np.array_equal(
            mesh_data.edge_face_connectivity, np.array([[0, 1], [0, 1], [0, 1], [0, 1]])
        )
        assert np.array_equal(
            mesh_data.face_edge_connectivity, np.array([[1, 0, 2, 3], [0, 2, 3, 1]])
        )
        assert np.allclose(
            mesh_data.x_edge_coords,
            np.array(
                [
                    [194949.796875, 194966.515625],
                    [194949.796875, 195000.0],
                    [194966.515625, 194982.8125],
                    [194982.8125, 195000.0],
                ]
            ),
        )
        assert np.allclose(
            mesh_data.x_face_coords.data,
            np.array(
                [
                    [194949.796875, 194966.515625, 194982.8125, 195000.0],
                    [194966.515625, 194982.8125, 195000.0, 194949.796875],
                ]
            ),
        )
        assert np.allclose(
            mesh_data.y_edge_coords,
            np.array(
                [
                    [361366.90625, 361399.46875],
                    [361366.90625, 361450.0],
                    [361399.46875, 361431.03125],
                    [361431.03125, 361450.0],
                ]
            ),
        )
        assert np.allclose(
            mesh_data.y_face_coords.data,
            np.array(
                [
                    [361366.90625, 361399.46875, 361431.03125, 361450.0],
                    [361399.46875, 361431.03125, 361450.0, 361366.90625],
                ]
            ),
        )

    @patch("dfastbe.bank_lines.data_models.BankLinesRiverData")
    @patch("dfastbe.io.data_models.LineGeometry")
    def test_simulation_data(self, mock_center_line, mock_simulation_data):
        """Test the simulation_data method of the BaseRiverData class with a mocked SimulationData."""
        # Mock the SimulationData instance
        mock_simulation_data_class = MagicMock()
        mock_simulation_data_class.dry_wet_threshold = 0.1
        mock_simulation_data.read.return_value = mock_simulation_data_class

        # Mock the ConfigFile
        mock_config_file = MagicMock()
        mock_config_file.get_sim_file.return_value = "mock_sim_file.nc"
        mock_config_file.get_float.return_value = 0.5  # Critical water depth
        mock_config_file.get_start_end_stations.return_value = (0.0, 10.0)
        mock_config_file.get_search_lines.return_value = [
            LineString([(0, 0), (1, 1)]),
            LineString([(2, 2), (3, 3)]),
        ]
        mock_config_file.get_bank_search_distances.return_value = [1.0, 2.0]

        mock_center_line_class = MagicMock()
        mock_center_line_class.values = LineString([(0, 0), (1, 1), (2, 2)])
        mock_center_line.return_value = mock_center_line_class

        # Create a BaseRiverData instance
        with patch(
            "dfastbe.bank_erosion.data_models.inputs.ErosionRiverData._get_bank_line_dir"
        ) as mock_get_bank_line_dir, patch(
            "dfastbe.bank_erosion.data_models.inputs.ErosionRiverData._read_river_axis"
        ) as mock_read_river_axis:
            mock_get_bank_line_dir.return_value = Path("tests/data/erosion/inputs")
            mock_read_river_axis.return_value = LineString([(0, 0), (1, 1), (2, 2)])

            river_data = ErosionRiverData(mock_config_file)

        # Call the simulation_data method
        with patch(
            "dfastbe.bank_erosion.data_models.inputs.ErosionSimulationData.read",
        ) as mock_read_simulation_data:
            mock_read_simulation_data.return_value = mock_simulation_data_class
            # Call the simulation_data method
            simulation_data = river_data.simulation_data()

        assert simulation_data == mock_simulation_data_class


class TestErosionRiverData:

    @pytest.fixture
    def river_data(self) -> ErosionRiverData:
        path = "tests/data/erosion/meuse_manual/meuse_manual.cfg"
        config_file = ConfigFile.read(path)
        river_data = ErosionRiverData(config_file)
        return river_data

    @patch("dfastbe.io.config.XYCModel.read")
    def test_read_river_axis(self, mock_read, river_data):
        """Test the read_river_axis method by mocking XYCModel.read."""
        mock_river_axis = LineString([(0, 0), (1, 1), (2, 2)])
        mock_read.return_value = mock_river_axis
        expected_path = Path(
            "tests/data/erosion/meuse_manual/inputs/maas_rivieras_mod.xyc"
        )

        river_axis = river_data._read_river_axis()

        mock_read.assert_called_once_with(str(expected_path.resolve()))
        assert isinstance(river_axis, LineString)
        assert river_axis.equals(mock_river_axis)


class TestShipsParameters:

    @pytest.fixture
    def mock_config_file(self, shipping_dict):
        """Fixture to create a mock ConfigFile instance for testing ShipsParameters."""
        config_file = MagicMock(spec=ConfigFile)
        config_file.get_parameter.side_effect = [
            shipping_dict["velocity"],
            shipping_dict["number"],
            shipping_dict["num_waves"],
            shipping_dict["draught"],
            shipping_dict["type"],
            shipping_dict["slope"],
            shipping_dict["reed"],
        ]
        return config_file

    @pytest.mark.unit
    def test_read_discharge_parameters(self, mock_config_file, shipping_dict):
        """Test the _read_discharge_parameters method.

        This method reads discharge parameters for a specific discharge level.

        Args:
            mock_config_file (ConfigFile):
                The ConfigFile instance to get parameter data from.
            shipping_data (dict):
                The shipping data to use for testing.

        Mocks:
            ConfigFile:
                The behavior of the get_parameter method to return predefined numpy arrays.

        Asserts:
            The returned discharge parameters are an instance of SingleLevelParameters.
            The parameters match the expected values from the shipping data.
        """
        shipping_data = ShipsParameters(mock_config_file, **shipping_dict)
        discharge_parameters = shipping_data.read_discharge_parameters(1, [13])
        assert isinstance(discharge_parameters, SingleLevelParameters)
        assert discharge_parameters.id == 1
        assert np.allclose(
            discharge_parameters.left.ship_velocity, shipping_dict["velocity"]
        )
        assert np.allclose(discharge_parameters.left.num_ship, shipping_dict["number"])
        assert np.allclose(
            discharge_parameters.left.num_waves_per_ship, shipping_dict["num_waves"]
        )

    @pytest.mark.unit
    def test_get_ship_data(self, mock_config_file):
        """Test the get_ship_data method.

        This method retrieves ship parameters based on the number of stations per bank.
        Leverage the mock_config_file to simulate the configuration file behavior.

        Args:
            mock_config_file (MagicMock):
                A mocked ConfigFile instance to get parameter data from.

        Mocks:
            ConfigFile:
                The behavior of the get_parameter method to return predefined numpy arrays.

        Asserts:
            The returned ship parameters are a dictionary with expected keys.
            Each value in the dictionary is a list of numpy arrays,
                each array's length matches the number of stations per bank.
        """
        num_stations_per_bank = [3, 3]

        ship_data = ShipsParameters.get_ship_data(
            num_stations_per_bank, mock_config_file
        )

        assert isinstance(ship_data, ShipsParameters)
        assert ship_data.velocity[0].shape[0] == num_stations_per_bank[0]
        assert ship_data.number[0].shape[0] == num_stations_per_bank[0]
        assert ship_data.num_waves[0].shape[0] == num_stations_per_bank[0]
        assert ship_data.draught[0].shape[0] == num_stations_per_bank[0]
        assert ship_data.type[0].shape[0] == num_stations_per_bank[0]
        assert ship_data.slope[0].shape[0] == num_stations_per_bank[0]
        assert ship_data.reed[0].shape[0] == num_stations_per_bank[0]

    @pytest.mark.unit
    def test_calculate_ship_derived_parameters(self, mock_config_file, shipping_dict):
        """Test the ship_derived_parameters method.

        This method retrieves derived ship parameters based on the shipping data.

        Args:
            mock_config_file (MagicMock):
                A mocked ConfigFile instance to get parameter data from.
            shipping_dict (dict):
                The shipping data to use for testing.

        Mocks:
            ConfigFile:
                The behavior of the get_parameter method to return predefined numpy arrays.

        Asserts:
            The returned parameters are a list of Parameters instances with expected values.
        """
        shipping_data = ShipsParameters(mock_config_file, **shipping_dict)
        mu_slope, mu_reed = shipping_data._calculate_ship_derived_parameters(
            shipping_dict["slope"], shipping_dict["reed"]
        )

        assert np.allclose(mu_slope[0], np.array([0.05, 0.05, 0.05]))
        assert np.allclose(mu_reed[0], np.array([0.0, 0.0, 0.0]))
