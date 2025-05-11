"""Bank erosion calculator module."""

import sys
from typing import Tuple
import numpy as np
from dfastbe.bank_erosion.data_models.calculation import SingleCalculation, SingleParameters, SingleErosion

# Constants
EPS = sys.float_info.epsilon
WATER_DENSITY = 1000  # density of water [kg/m3]
g = 9.81  # gravitational acceleration [m/s2]


class ErosionCalculator:
    """Class for calculating bank erosion."""

    @staticmethod
    def comp_erosion_eq(
        bank_height: np.ndarray,
        segment_length: np.ndarray,
        water_level_fairway_ref: np.ndarray,
        discharge_level_pars: SingleParameters,
        bank_fairway_dist: np.ndarray,
        water_depth_fairway: np.ndarray,
        erosion_inputs: SingleErosion,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the equilibrium bank erosion.

        Args:
            bank_height : np.ndarray
                Array containing bank height [m]
            segment_length : np.ndarray
                Array containing length of the segment [m]
            water_level_fairway_ref : np.ndarray
                Array containing water level at fairway [m]
            discharge_level_pars (SingleParameters):
                Discharge level parameters object containing the following attributes.
                    ship_velocity : np.ndarray
                        Array containing ship velocity [m/s]
                    ship_type : np.ndarray
                        Array containing ship type [-]
                    ship_draught : np.ndarray
                        Array containing ship draught [m]
                    mu_slope : np.ndarray
                        Array containing slope [-]
            bank_fairway_dist : np.ndarray
                Array containing distance from bank to fairway [m]
            water_depth_fairway : np.ndarray
                Array containing water depth at the fairway [m]
            erosion_inputs (ErosionInputs):
                ErosionInputs object.

        Returns:
            dn_eq : np.ndarray
                 Equilibrium bank erosion distance [m]
            dv_eq : np.ndarray
                 Equilibrium bank erosion volume [m]
        """
        # ship induced wave height at the beginning of the foreshore
        h0 = ErosionCalculator.comp_hw_ship_at_bank(
            bank_fairway_dist,
            erosion_inputs.wave_fairway_distance_0,
            erosion_inputs.wave_fairway_distance_1,
            water_depth_fairway,
            discharge_level_pars.ship_type,
            discharge_level_pars.ship_draught,
            discharge_level_pars.ship_velocity,
        )
        h0 = np.maximum(h0, EPS)

        zup = np.minimum(bank_height, water_level_fairway_ref + 2 * h0)
        zdo = np.maximum(
            water_level_fairway_ref - 2 * h0, erosion_inputs.bank_protection_level
        )
        ht = np.maximum(zup - zdo, 0)
        hs = np.maximum(bank_height - water_level_fairway_ref + 2 * h0, 0)
        eq_erosion_distance = ht / discharge_level_pars.mu_slope
        eq_erosion_volume = (0.5 * ht + hs) * eq_erosion_distance * segment_length

        return eq_erosion_distance, eq_erosion_volume

    @staticmethod
    def compute_bank_erosion_dynamics(
        single_calculation: SingleCalculation,
        bank_height: np.ndarray,
        segment_length: np.ndarray,
        bank_fairway_dist: np.ndarray,
        water_level_fairway_ref: np.ndarray,
        discharge_level_pars: SingleParameters,
        time_erosion: float,
        erosion_inputs: SingleErosion,
    ) -> SingleCalculation:
        """
        Compute the bank erosion during a specific discharge level.

        Args:
            single_calculation (SingleCalculation):
                velocity : np.ndarray
                    Array containing flow velocity magnitude [m/s]
                water_level_fairway : np.ndarray
                    Array containing water levels at fairway [m]
                chezy : np.ndarray
                    Array containing Chezy values [m0.5/s]
            bank_height : np.ndarray
                Array containing bank height
            segment_length : np.ndarray
                Array containing length of line segment [m]
            water_level_fairway_ref : np.ndarray
                Array containing reference water levels at fairway [m]
            tauc : np.ndarray
                Array containing critical shear stress [N/m2]
            discharge_level_pars: SingleLevelParameters,
                num_ship : np.ndarray
                    Array containing number of ships [-]
                ship_velocity : np.ndarray
                    Array containing ship velocity [m/s]
                num_waves_per_ship : np.ndarray
                    Array containing number of waves per ship [-]
                ship_type : np.ndarray
                    Array containing ship type [-]
                ship_draught : np.ndarray
                    Array containing ship draught [m]
            time_erosion : float
                Erosion period [yr]
            bank_fairway_dist : np.ndarray
                Array containing distance from bank to fairway [m]
            fairway_wave_reduction_distance : np.ndarray
                Array containing distance from fairway at which wave reduction starts [m]
            fairway_wave_disappear_distance : np.ndarray
                Array containing distance from fairway at which all waves are gone [m]
            water_depth_fairway : np.ndarray
                Array containing water depth at fairway [m]
            dike_height : np.ndarray
                Array containing bank protection height [m]
            water_density : float
                Water density [kg/m3]

        Returns:
            parameters (CalculationParameters):
                erosion_distance : np.ndarray
                    Total bank erosion distance [m]
                erosion_volume : np.ndarray
                    Total bank erosion volume [m]
                erosion_distance_shipping : np.ndarray
                    Bank erosion distance due to shipping [m]
                erosion_distance_flow : np.ndarray
                    Bank erosion distance due to current [m]
                ship_wave_max : np.ndarray
                    Maximum bank level subject to ship waves [m]
                ship_wave_min : np.ndarray
                    Minimum bank level subject to ship waves [m]
        """
        sec_year = 3600 * 24 * 365

        # period of ship waves [s]
        ship_wave_period = 0.51 * discharge_level_pars.ship_velocity / g
        ts = (
                ship_wave_period
                * discharge_level_pars.num_ship
                * discharge_level_pars.num_waves_per_ship
        )
        vel = single_calculation.bank_velocity

        # the ship induced wave height at the beginning of the foreshore
        wave_height = ErosionCalculator.comp_hw_ship_at_bank(
            bank_fairway_dist,
            erosion_inputs.wave_fairway_distance_0,
            erosion_inputs.wave_fairway_distance_1,
            single_calculation.water_depth,
            discharge_level_pars.ship_type,
            discharge_level_pars.ship_draught,
            discharge_level_pars.ship_velocity,
        )
        wave_height = np.maximum(wave_height, EPS)

        # compute erosion parameters for each line part
        # erosion coefficient
        erosion_coef = 0.2 * np.sqrt(erosion_inputs.tauc) * 1e-6

        # critical velocity
        critical_velocity = np.sqrt(
            erosion_inputs.tauc / WATER_DENSITY * single_calculation.chezy ** 2 / g
        )

        # strength
        cE = 1.85e-4 / erosion_inputs.tauc

        # total wave damping coefficient
        # mu_tot = (mu_slope / H0) + mu_reed
        # water level along bank line
        ho_line_ship = np.minimum(
            single_calculation.water_level - erosion_inputs.bank_protection_level, 2 * wave_height
        )
        ho_line_flow = np.minimum(
            single_calculation.water_level - erosion_inputs.bank_protection_level,
            single_calculation.water_depth,
            )
        h_line_ship = np.maximum(bank_height - single_calculation.water_level + ho_line_ship, 0)
        h_line_flow = np.maximum(bank_height - single_calculation.water_level + ho_line_flow, 0)

        # compute displacement due to flow
        crit_ratio = np.ones(critical_velocity.shape)
        mask = (vel > critical_velocity) & (
                single_calculation.water_level > erosion_inputs.bank_protection_level
        )
        crit_ratio[mask] = (vel[mask] / critical_velocity[mask]) ** 2
        erosion_distance_flow = erosion_coef * (crit_ratio - 1) * time_erosion * sec_year

        # compute displacement due to ship waves
        ship_wave_max = single_calculation.water_level + 0.5 * wave_height
        ship_wave_min = single_calculation.water_level - 2 * wave_height
        mask = (ship_wave_min < water_level_fairway_ref) & (
                water_level_fairway_ref < ship_wave_max
        )
        # limit mu -> 0

        erosion_distance_shipping = cE * wave_height**2 * ts * time_erosion
        erosion_distance_shipping[~mask] = 0

        # compute erosion volume
        mask = (h_line_ship > 0) & (
                single_calculation.water_level > erosion_inputs.bank_protection_level
        )
        dv_ship = erosion_distance_shipping * segment_length * h_line_ship
        dv_ship[~mask] = 0.0
        erosion_distance_shipping[~mask] = 0.0

        mask = (h_line_flow > 0) & (
                single_calculation.water_level > erosion_inputs.bank_protection_level
        )
        dv_flow = erosion_distance_flow * segment_length * h_line_flow
        dv_flow[~mask] = 0.0
        erosion_distance_flow[~mask] = 0.0

        erosion_distance = erosion_distance_shipping + erosion_distance_flow
        erosion_volume = dv_ship + dv_flow
        single_calculation.erosion_volume_tot = erosion_volume
        single_calculation.erosion_distance_tot = erosion_distance
        single_calculation.erosion_distance_shipping = erosion_distance_shipping
        single_calculation.erosion_distance_flow = erosion_distance_flow
        single_calculation.ship_wave_max = ship_wave_max
        single_calculation.ship_wave_min = ship_wave_min
        return single_calculation

    @staticmethod
    def comp_hw_ship_at_bank(
        bank_fairway_dist: np.ndarray,
        fairway_wave_reduction_distance: np.ndarray,
        fairway_wave_disappear_distance: np.ndarray,
        water_depth_fairway: np.ndarray,
        ship_type: np.ndarray,
        ship_draught: np.ndarray,
        ship_velocity: np.ndarray,
    ) -> np.ndarray:
        """
        Compute wave heights at bank due to passing ships.

        Arguments
        ---------
        bank_fairway_dist : np.ndarray
            Array containing distance from bank to fairway [m]
        fairway_wave_reduction_distance : np.ndarray
            Array containing distance from fairway at which wave reduction starts [m]
        fairway_wave_disappear_distance : np.ndarray
            Array containing distance from fairway at which all waves are gone [m]
        water_depth_fairway : np.ndarray
            Array containing the water depth at the fairway [m]
        ship_type : np.ndarray
            Array containing the ship type [-]
        ship_draught : np.ndarray
            Array containing draught of the ships [m]
        ship_velocity : np.ndarray
            Array containing velocity of the ships [m/s]
        g : float
            Gravitational acceleration [m/s2]

        Returns
        -------
        h0 : np.ndarray
            Array containing wave height at the bank [m]
        """
        h = np.copy(water_depth_fairway)

        a1 = np.zeros(len(bank_fairway_dist))
        # multiple barge convoy set
        a1[ship_type == 1] = 0.5
        # RHK ship / motor ship
        a1[ship_type == 2] = 0.28 * ship_draught[ship_type == 2] ** 1.25
        # towboat
        a1[ship_type == 3] = 1

        froude = ship_velocity / np.sqrt(h * g)
        froude_limit = 0.8
        high_froude = froude > froude_limit
        h[high_froude] = ((ship_velocity[high_froude] / froude_limit) ** 2) / g
        froude[high_froude] = froude_limit

        A = 0.5 * (
                1
                + np.cos(
            (bank_fairway_dist - fairway_wave_disappear_distance)
            / (fairway_wave_reduction_distance - fairway_wave_disappear_distance)
            * np.pi
        )
        )
        A[bank_fairway_dist < fairway_wave_disappear_distance] = 1
        A[bank_fairway_dist > fairway_wave_reduction_distance] = 0

        h0 = a1 * h * (bank_fairway_dist / h) ** (-1 / 3) * froude**4 * A
        return h0