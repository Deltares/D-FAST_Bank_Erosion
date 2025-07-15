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
        """Fixture to provide single parameters for erosion calculations.

        Returns:
            SingleParameters:
                - ship_velocity: Velocity of the ship in m/s.
                - num_ship: Number of ships.
                - num_waves_per_ship: Number of waves per ship.
                - ship_draught: Draught of the ship in m.
                - ship_type: Type of the ship.
                - par_reed: Reed parameters (not used in this test).
                - par_slope: Slope parameters in degrees.
                - mu_reed: Reed friction coefficient (not used in this test).
                - mu_slope: Slope friction coefficient.
        """
        return SingleParameters(
            ship_velocity=np.array([5.0, 5.0, 5.0]),
            num_ship=np.array([15613, 15613, 15613]),
            num_waves_per_ship=np.array([5.0, 5.0, 5.0]),
            ship_draught=np.array([1.2, 1.2, 1.2]),
            ship_type=np.array([2.0, 2.0, 2.0]),
            par_reed=np.array([0.0, 0.0, 0.0]),
            par_slope=np.array([20.0, 20.0, 20.0]),
            mu_reed=np.array([0.0, 0.0, 0.0]),
            mu_slope=np.array([0.05, 0.05, 0.05]),
        )

    @pytest.fixture
    def single_calculation(self) -> SingleCalculation:
        """Fixture to provide a single calculation instance for erosion calculations.

        Returns:
            SingleCalculation:
                - bank_velocity: Velocity of the bank in m/s.
                - water_level: Water level in meters.
                - water_depth: Water depth in meters.
                - chezy: Chezy coefficient for flow resistance.
        """
        return SingleCalculation(
            bank_velocity=np.array([3.61503663e-02, 3.55049286e-02, 4.19792580e-02]),
            water_level=np.ma.array([11.10304451, 11.10307884, 11.10307884]),
            water_depth=np.ma.array([5.55804443, 5.47807884, 5.47807884]),
            chezy=np.ma.array([79.57007898, 79.57007898, 79.57007898]),
        )

    @pytest.fixture
    def erosion_inputs(self) -> SingleErosion:
        """Fixture to provide inputs for erosion calculations.

        Returns:
            SingleErosion:
                - wave_fairway_distance_0: Distance to the first wave in the fairway
                - wave_fairway_distance_1: Distance to the second wave in the fairway
                - bank_protection_level: Level of bank protection.
                - tauc: Critical shear stress for erosion.
        """
        return SingleErosion(
            wave_fairway_distance_0=np.array([150.0, 150.0, 150.0]),
            wave_fairway_distance_1=np.array([110.0, 110.0, 110.0]),
            bank_protection_level=np.array([-13.0, -13.0, -13.0]),
            tauc=np.array([1.0, 1.0, 1.0]),
        )

    @pytest.fixture
    def shared_input(self) -> dict:
        """Fixture to provide shared input data for the tests of the erosion calculator.

        Returns:
            dict: A dictionary containing shared input data for erosion calculations.
        """
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

    @pytest.mark.unit
    def test_comp_erosion_eq(
        self,
        single_parameters: SingleParameters,
        erosion_inputs: SingleErosion,
        shared_input: dict,
    ):
        """Test the computation of equilibrium erosion distance and volume.

        Args:
            single_parameters (SingleParameters):
                Parameters for the single calculation.
            erosion_inputs (SingleErosion):
                Inputs for erosion calculations.
            shared_input (dict):
                Shared input data for the test.

        Asserts:
            The computed equilibrium erosion distance matches expected values.
            The computed equilibrium erosion volume matches expected values.
        """
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

    @pytest.mark.unit
    def test_compute_bank_erosion_dynamics(
        self,
        single_calculation: SingleCalculation,
        single_parameters: SingleParameters,
        erosion_inputs: SingleErosion,
        shared_input: dict,
    ):
        """Test the computation of bank erosion dynamics.
        Args:
            single_calculation (SingleCalculation):
                The initial state of the single calculation.
            single_parameters (SingleParameters):
                Parameters for the single calculation.
            erosion_inputs (SingleErosion):
                Inputs for erosion calculations.
            shared_input (dict):
                Shared input data for the test.

        Asserts:
            The computed erosion volume totals match expected values.
            The computed erosion distance totals match expected values.
            The computed erosion distance for shipping matches expected values.
            The computed erosion distance for flow matches expected values.
            The computed maximum ship wave height matches expected values.
            The computed minimum ship wave height matches expected values.
        """
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
            np.array([11.19081363, 11.19093028, 11.1908225]),
        )
        assert np.allclose(
            single_calculation.ship_wave_min,
            np.array([10.75196801, 10.75167307, 10.7521042]),
        )

    @pytest.mark.unit
    def test_comp_hw_ship_at_bank(
        self,
        shared_input: dict,
        single_parameters: SingleParameters,
        erosion_inputs: SingleErosion,
    ):
        """Test the computation of ship wave height at the bank.

        Args:
            shared_input (dict):
                Shared input data for the test.
            single_parameters (SingleParameters):
                Parameters for the single calculation.
            erosion_inputs (SingleErosion):
                Inputs for erosion calculations.

        Asserts:
            The computed ship wave height at the bank matches expected values.
        """
        h0 = ErosionCalculator.comp_hw_ship_at_bank(
            shared_input["bank_fairway_dist"],
            erosion_inputs.wave_fairway_distance_0,
            erosion_inputs.wave_fairway_distance_1,
            shared_input["water_depth_fairway"],
            single_parameters.ship_type,
            single_parameters.ship_draught,
            single_parameters.ship_velocity,
        )

        assert np.allclose(h0, np.array([0.11631031, 0.11590078, 0.11575859]))
