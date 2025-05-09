from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np
import pytest

from dfastbe.bank_erosion.bank_erosion import Erosion
from dfastbe.bank_erosion.data_models import BankData, FairwayData
from dfastbe.cmd import run
from dfastbe.io import ConfigFile

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
    def config_file(self) -> ConfigFile:
        """Fixture to create a mock config file."""
        return ConfigFile.read("tests/data/erosion/meuse_manual.cfg")

    @pytest.fixture
    def erosion_instance(self, config_file) -> Erosion:
        """Fixture to create an instance of the Erosion class."""
        return Erosion(config_file)

    def test_prepare_initial_conditions(
        self, erosion_instance: Erosion, config_file: ConfigFile
    ):
        """Test the _prepare_initial_conditions method."""
        num_stations_per_bank = [3, 3]
        mock_fairway_data = MagicMock(spec=FairwayData)
        mock_fairway_data.fairway_initial_water_levels = [np.array([10, 20, 30]), np.array([10, 20, 30])]
        taucls = np.array([1, 1, 1])
        taucls_str = (
            "protected",
            "vegetation",
            "good clay",
            "moderate/bad clay",
            "sand",
        )

        erosion_inputs = erosion_instance._prepare_initial_conditions(
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

    @pytest.fixture
    def mock_erosion(self):
        """Fixture to patch and mock the __init__ method of the Erosion class."""
        with patch(
            "dfastbe.bank_erosion.bank_erosion.Erosion.__init__", return_value=None
        ):
            # Create an instance of Erosion with the patched __init__
            erosion_instance = Erosion(MagicMock())

            # Mock attributes that would normally be initialized in __init__
            erosion_instance.root_dir = "mock_root_dir"
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

    def test_get_ship_parameters(self, mock_erosion, mock_config_file):
        """Test the get_ship_parameters method."""
        # Arrange
        num_stations_per_bank = [10, 15]
        mock_erosion._config_file = mock_config_file

        # Act
        ship_parameters = mock_erosion.get_ship_parameters(num_stations_per_bank)

        # Assert
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

    def test_process_river_axis_by_center_line(self, mock_erosion):
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
