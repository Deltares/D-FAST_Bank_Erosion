from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np
import pytest

import dfastbe.io.logger
from dfastbe.bank_erosion.bank_erosion import Erosion, calculate_alpha
from dfastbe.bank_erosion.data_models.calculation import (
    BankData,
    ErosionResults,
    FairwayData,
    SingleBank,
    SingleLevelParameters,
)
from dfastbe.cmd import run
from dfastbe.io.config import ConfigFile

matplotlib.use('Agg')


@pytest.mark.e2e
def test_bank_erosion():
    file = "erosion"
    language = "UK"
    config_file = f"tests/data/{file}/meuse_manual.cfg"
    run(language, "BANKLINES", config_file)
    print("Banklines done")
    run(language, "BANKEROSION", config_file)
    print("Bank erosion done")


class TestErosion:

    @pytest.fixture
    def shipping_data(self) -> Dict[str, list]:
        """Fixture to create mock shipping data."""
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
        """Fixture to patch and mock the __init__ method of the Erosion class."""
        with patch(
            "dfastbe.bank_erosion.bank_erosion.Erosion.__init__", return_value=None
        ):
            # Create an instance of Erosion with the patched __init__
            erosion_instance = Erosion(MagicMock())

            # Mock attributes that would normally be initialized in __init__
            erosion_instance.root_dir = Path("mock_root_dir")
            erosion_instance._config_file = MagicMock()
            erosion_instance.gui = False
            erosion_instance.river_data = MagicMock()
            erosion_instance.river_center_line_arr = MagicMock()
            erosion_instance.simulation_data = MagicMock()
            erosion_instance.sim_files = MagicMock()
            erosion_instance.p_discharge = MagicMock()
            erosion_instance.bl_processor = MagicMock()
            erosion_instance.debugger = MagicMock()

            yield erosion_instance

    @pytest.fixture
    def mock_config_file(self):
        """Fixture to create a mock ConfigFile."""
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

    def test_get_ship_parameters(self, mock_erosion, mock_config_file):
        """Test the get_ship_parameters method."""
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
        assert isinstance(ship_parameters, dict)
        assert set(ship_parameters.keys()) == set(expected_keys)

        for key, value in ship_parameters.items():
            assert isinstance(value, list)
            assert len(value) == len(num_stations_per_bank)
            for arr, n in zip(value, num_stations_per_bank):
                assert isinstance(arr, np.ndarray)
                assert len(arr) == n

    def test_prepare_initial_conditions(self, mock_erosion: Erosion, shipping_data):
        """Test the _prepare_initial_conditions method."""
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
        config_file = MagicMock(spec=ConfigFile)
        config_file.get_parameter.side_effect = [
            [np.array([150.0, 150.0, 150.0]), np.array([150.0, 150.0, 150.0])],
            [np.array([110.0, 110.0, 110.0]), np.array([110.0, 110.0, 110.0])],
            [np.array([1.0, 1.0, 1.0]), np.array([0.18, 0.18, 0.18])],
            [np.array([-13.0, -13.0, -13.0]), np.array([-13.0, -13.0, -13.0])],
        ]

        config_file.get_bool.return_value = False

        with patch(
            "dfastbe.bank_erosion.bank_erosion.Erosion.get_ship_parameters"
        ) as mock_get_ship_params:
            mock_get_ship_params.return_value = shipping_data
            erosion_inputs = mock_erosion._prepare_initial_conditions(
                config_file, num_stations_per_bank, mock_fairway_data
            )

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

    def test_process_river_axis_by_center_line(self, mock_erosion, mock_debug):
        """Test the _process_river_axis_by_center_line method."""
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

        river_axis.add_data.assert_called_with(data={"stations": np.array([128.0])})

    def test_get_fairway_data(self, mock_erosion, mock_config_file, mock_debug):
        mock_erosion.river_data.debug = True
        mock_erosion._config_file = mock_config_file
        with patch(
            "dfastbe.bank_erosion.data_models.calculation.FairwayData"
        ) as fairway_mock, patch(
            "dfastbe.bank_erosion.bank_erosion.intersect_line_mesh"
        ) as line_mock, patch(
            "dfastbe.io.data_models.GeoDataFrame"
        ) as gdf_mock:
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

    def test_calculate_fairway_bank_line_distance(
        self, mock_erosion, mock_config_file, mock_debug
    ):
        """Test the calculate_fairway_bank_line_distance method."""
        mock_erosion._config_file = mock_config_file
        mock_bank_data = MagicMock()
        mock_fairway_data = MagicMock()
        mock_fairway_data.intersection_coords = np.array(
            [
                [209186.621094, 389659.99609375],
                [209187.69800938, 389665.38986148],
                [209189.26657398, 389673.24607124],
                [209189.367188, 389673.75],
            ]
        )
        mock_fairway_data.face_indices = np.array([59166, 59167, 62557])
        mock_left_bank = MagicMock(spec=SingleBank)
        mock_left_bank.is_right_bank = False
        mock_left_bank.bank_line_coords = np.array(
            [
                [209117.80853726, 389680.26397752],
                [209118.99815819, 389684.66797463],
                [209118.7134304, 389703.72840232],
                [209127.20673008, 389727.17509686],
            ]
        )
        mock_left_bank.bank_chainage_midpoints = np.array(
            [
                123.00166634401488,
                123.01335778611656,
                123.03520808078332,
            ]
        )

        mock_right_bank = MagicMock(spec=SingleBank)
        mock_right_bank.is_right_bank = True
        mock_right_bank.bank_line_coords = np.array(
            [
                [209266.44709443, 389650.16238121],
                [209267.30013864, 389654.45330198],
                [209269.67183787, 389664.217019],
                [209271.7614607, 389674.70572161],
            ]
        )
        mock_right_bank.bank_chainage_midpoints = np.array(
            [
                123.00943873095339,
                123.01990543424606,
                123.04044936886122,
            ]
        )

        mock_bank_data = MagicMock(spec=BankData)
        mock_bank_data.left = mock_left_bank
        mock_bank_data.right = mock_right_bank
        mock_bank_data.__iter__.return_value = iter([mock_left_bank, mock_right_bank])

        mock_simulation_data = MagicMock()
        mock_simulation_data.fairway_face_indices = np.array(
            [
                23.954999923706055,
                21.69499969482422,
                21.69499969482422,
                21.809999465942383,
                21.545000076293945,
                21.134998321533203,
            ]
        )

        with patch("dfastbe.io.data_models.GeoDataFrame") as gdf_mock:
            mock_erosion.calculate_fairway_bank_line_distance(
                mock_bank_data, mock_fairway_data, mock_simulation_data
            )

        assert mock_fairway_data.fairway_initial_water_levels == []

    def test_process_discharge_levels(self, mock_erosion, mock_debug):
        km_bin = np.array([123.0, 128.0, 0.1])
        km_mid = np.array(
            [123.05, 123.15, 123.25, 123.35, 123.45, 123.55, 123.65, 123.75]
        )
        config_file = MagicMock()
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
                km_mid, km_bin, config_file, erosion_inputs, bank_data, fairway_data
            )

            assert mock_write_km_eroded_volumes.called

        assert isinstance(water_level_data, MagicMock)  # WaterLevelData
        assert isinstance(erosion_results, ErosionResults)  # ErosionResults
        assert np.allclose(erosion_results.flow_erosion_dist, flow_erosion_dist)
        assert np.allclose(erosion_results.ship_erosion_dist, ship_erosion_dist)
        assert np.allclose(erosion_results.total_erosion_dist, total_erosion_dist)
        assert np.allclose(erosion_results.total_eroded_vol, total_eroded_vol)

    def test_read_discharge_parameters(self, mock_erosion, shipping_data):
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
                1, shipping_data, 13
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


def test_calculate_alpha():
    """Test the calculate_alpha method."""
    # Mock the bank data and fairway data

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
