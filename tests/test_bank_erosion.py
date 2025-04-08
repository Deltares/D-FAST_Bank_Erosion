import matplotlib
import numpy as np
import pytest

from dfastbe.bank_erosion import Erosion
from dfastbe.cmd import run
from dfastbe.io import ConfigFile

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
        bank_km_mid = [np.array([3.0, 3.0, 3.0])]
        zfw_ini = [np.array([10, 20, 30])]
        taucls = np.array([1, 1, 1])
        taucls_str = (
            "protected",
            "vegetation",
            "good clay",
            "moderate/bad clay",
            "sand",
        )

        erosion_inputs = erosion_instance._prepare_initial_conditions(
            config_file, bank_km_mid, zfw_ini
        )

        assert np.array_equal(
            erosion_inputs.ship_data["vship0"][0], np.array([5.0, 5.0, 5.0])
        )
        assert np.array_equal(erosion_inputs.wave_0[0], np.array([150, 150, 150]))
        assert np.array_equal(erosion_inputs.wave_1[0], np.array([110, 110, 110]))
        assert np.array_equal(
            erosion_inputs.bank_protection_level[0], np.array([-13, -13, -13])
        )
        assert np.array_equal(erosion_inputs.tauc[0], taucls)
        assert erosion_inputs.taucls_str == taucls_str
        assert len(erosion_inputs.bank_type) == 4
