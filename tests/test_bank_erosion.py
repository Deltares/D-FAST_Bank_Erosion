import matplotlib
import numpy as np
import pytest
from unittest.mock import MagicMock

from dfastbe.bank_erosion import Erosion
from dfastbe.cmd import run
from dfastbe.io import ConfigFile
from dfastbe.structures import BankData, FairwayData

matplotlib.use('Agg')


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
        mock_bank_data = MagicMock(type=BankData)
        mock_bank_data.bank_chainage_midpoints = [np.array([3.0, 3.0, 3.0])]
        mock_fairway_data = MagicMock(type=FairwayData)
        mock_fairway_data.fairway_initial_water_levels = [np.array([10, 20, 30])]
        taucls = np.array([1, 1, 1])
        taucls_str = (
            "protected",
            "vegetation",
            "good clay",
            "moderate/bad clay",
            "sand",
        )

        erosion_inputs = erosion_instance._prepare_initial_conditions(
            config_file, mock_bank_data, mock_fairway_data
        )

        assert np.array_equal(
            erosion_inputs.shipping_data["vship0"][0], np.array([5.0, 5.0, 5.0])
        )
        assert np.array_equal(
            erosion_inputs.wave_fairway_distance_0[0], np.array([150, 150, 150])
        )
        assert np.array_equal(
            erosion_inputs.wave_fairway_distance_1[0], np.array([110, 110, 110])
        )
        assert np.array_equal(
            erosion_inputs.bank_protection_level[0], np.array([-13, -13, -13])
        )
        assert np.array_equal(erosion_inputs.tauc[0], taucls)
        assert erosion_inputs.taucls_str == taucls_str
        assert len(erosion_inputs.bank_type) == 4
