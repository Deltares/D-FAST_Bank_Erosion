import pytest
from pathlib import Path
import numpy as np
from unittest.mock import patch
from geopandas import GeoDataFrame
from shapely.geometry import LineString
from dfastbe.io import ConfigFile
from dfastbe.erosion.data_models import (
    ErosionRiverData,
    ErosionInputs,
    WaterLevelData,
    MeshData,
    BankData,
    FairwayData,
    ErosionResults,
    ErosionSimulationData
)


def test_erosion_inputs():
    """Test instantiation of the ErosionInputs dataclass."""
    erosion_inputs = ErosionInputs(
        shipping_data={"ship1": np.array([1.0, 2.0])},
        wave_fairway_distance_0=[np.array([10.0, 20.0])],
        wave_fairway_distance_1=[np.array([15.0, 25.0])],
        bank_protection_level=[np.array([1, 0])],
        tauc=[np.array([0.5, 0.6])],
        bank_type=[np.array([1, 2])],
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
        bank_height=[np.array([3.0, 4.0])],
        chezy=[[np.array([30.0, 40.0])]],
    )
    assert water_level_data.hfw_max == pytest.approx(5.0)
    assert water_level_data.water_level[0][0][1] == pytest.approx(2.0)
    assert water_level_data.bank_height[0][1] == pytest.approx(4.0)


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


def test_bank_data():
    """Test instantiation of the BankData dataclass."""
    bank_data = BankData(
        is_right_bank=[True, False],
        bank_chainage_midpoints=[np.array([0.0, 1.0])],
        bank_line_coords=[np.array([[0.0, 0.0], [1.0, 1.0]])],
        bank_face_indices=[np.array([0, 1])],
        bank_lines=GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])]),
        n_bank_lines=1,
        bank_line_size=[np.array([10.0, 20.0])],
        fairway_distances=[np.array([5.0, 10.0])],
        fairway_face_indices=[np.array([0, 1])],
    )
    assert bank_data.is_right_bank[0] is True
    assert bank_data.bank_chainage_midpoints[0][1] == pytest.approx(1.0)
    assert len(bank_data.bank_lines) == 1


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
        vol_per_discharge=[[np.array([0.9, 1.0])]],
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


class TestErosionRiverData:

    @pytest.fixture
    def river_data(self) -> ErosionRiverData:
        path = "tests/data/erosion/meuse_manual.cfg"
        config_file = ConfigFile.read(path)
        river_data = ErosionRiverData(config_file)
        return river_data

    @patch("dfastbe.io.XYCModel.read")
    def test_read_river_axis(self, mock_read, river_data):
        """Test the read_river_axis method by mocking XYCModel.read."""
        mock_river_axis = LineString([(0, 0), (1, 1), (2, 2)])
        mock_read.return_value = mock_river_axis
        expected_path = Path("tests/data/erosion/inputs/maas_rivieras_mod.xyc")

        river_axis = river_data.read_river_axis()

        mock_read.assert_called_once_with(str(expected_path.resolve()))
        assert isinstance(river_axis, LineString)
        assert river_axis.equals(mock_river_axis)