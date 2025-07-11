from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import matplotlib
import numpy as np
import pytest
from matplotlib.testing.compare import compare_images

from dfastbe.bank_erosion.bank_erosion import Erosion
from dfastbe.bank_erosion.data_models.calculation import FairwayData
from dfastbe.cmd import run
from dfastbe.io.config import ConfigFile

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
