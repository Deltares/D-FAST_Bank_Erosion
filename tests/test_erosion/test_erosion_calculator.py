from unittest.mock import MagicMock

import numpy as np
import pytest

from dfastbe.bank_erosion.data_models.calculation import (
    SingleCalculation,
    SingleErosion,
    SingleParameters,
)
from dfastbe.bank_erosion.erosion_calculator import ErosionCalculator


class TestErosionCalculator:
    @pytest.fixture
    def single_parameters(self) -> SingleParameters:
        single_parameters= MagicMock(spec=SingleParameters)
        single_parameters.ship_velocity = np.array([5.0, 5.0, 5.0])
        single_parameters.num_ship= np.array([15613, 15613, 15613])
        single_parameters.num_waves_per_ship= np.array([5.0, 5.0, 5.0])
        single_parameters.ship_draught= np.array([1.2, 1.2, 1.2])
        single_parameters.ship_type= np.array([2.0, 2.0, 2.0])
        single_parameters.mu_slope= np.array([0.05, 0.05, 0.05])
        return single_parameters

    @pytest.fixture
    def single_calculation(self) -> SingleCalculation:
        single_calculation = MagicMock(spec=SingleCalculation)
        single_calculation.bank_velocity = np.array([0.1, 0.1, 0.1])
        single_calculation.water_level = np.array([1.0, 1.0, 1.0])
        single_calculation.water_depth = np.array([3.0, 3.0, 3.0])
        single_calculation.chezy = np.array([50.0, 50.0, 50.0])
        return single_calculation

    @pytest.fixture
    def erosion_inputs(self) -> SingleErosion:
        erosion_inputs = MagicMock(spec=SingleErosion)
        erosion_inputs.wave_fairway_distance_0 = np.array([150.0, 150.0, 150.0])
        erosion_inputs.wave_fairway_distance_1 = np.array([110.0, 110.0, 110.0])
        erosion_inputs.bank_protection_level = np.array([-13.0, -13.0, -13.0])
        erosion_inputs.tauc = np.array([1.0, 1.0, 1.0])
        return erosion_inputs

    @pytest.fixture
    def shared_input(self) -> dict:
        return {
            "bank_height": np.array(
                [12.671252954404748, 12.689950029452636, 12.660388714966068]
            ),
            "segment_length": np.array(
                [4.561840481043876, 19.06255422681582, 24.937594592804164]
            ),
            "water_level_fairway_ref": np.array(
                [11.296967506408691, 11.297088623046875, 11.297088623046875]
            ),
            "bank_fairway_dist": np.array(
                [71.29687009540487, 73.18743494969097, 73.45747203868702]
            ),
            "water_depth_fairway": np.ma.array(
                [10.305103302001953, 10.225088119506836, 10.225088119506836]
            ),
        }

    def test_comp_erosion_eq(self, single_parameters, erosion_inputs, shared_input):
        eq_erosion_distance, eq_erosion_volume = (
            ErosionCalculator.comp_erosion_eq(
                bank_height=shared_input["bank_height"],
                segment_length=shared_input["segment_length"],
                water_level_fairway_ref=shared_input["water_level_fairway_ref"],
                discharge_level_pars=single_parameters,
                bank_fairway_dist=shared_input["bank_fairway_dist"],
                water_depth_fairway=shared_input["water_depth_fairway"],
                erosion_inputs=erosion_inputs,
            )
        )

        assert np.allclose(
            eq_erosion_distance, np.array([9.30482459, 9.27206278, 9.26068715])
        )
        assert np.allclose(
            eq_erosion_volume, np.array([78.0826197, 328.12862225, 421.77232935])
        )

    def test_compute_bank_erosion_dynamics(
        self, single_calculation, single_parameters, erosion_inputs, shared_input
    ):
        single_calculation = ErosionCalculator.compute_bank_erosion_dynamics(
            single_calculation=single_calculation,
            bank_height=shared_input["bank_height"],
            segment_length=shared_input["segment_length"],
            bank_fairway_dist=shared_input["bank_fairway_dist"],
            water_level_fairway_ref=shared_input["water_level_fairway_ref"],
            discharge_level_pars=single_parameters,
            time_erosion=1.0,
            erosion_inputs=erosion_inputs,
        )

        assert np.allclose(
            single_calculation.erosion_volume_tot,
            np.array([0.0, 0.0, 0.0]),
        )
        assert np.allclose(
            single_calculation.erosion_distance_tot,
            np.array([0.0, 0.0, 0.0]),
        )
        assert np.allclose(
            single_calculation.erosion_distance_shipping, np.array([0.0, 0.0, 0.0])
        )
        assert np.allclose(
            single_calculation.erosion_distance_flow, np.array([0.0, 0.0, 0.0])
        )
        assert np.allclose(
            single_calculation.ship_wave_max,
            np.array([1.1096215, 1.10866935, 1.10853603]),
        )
        assert np.allclose(
            single_calculation.ship_wave_min,
            np.array([0.56151398, 0.5653226, 0.56585589]),
        )
