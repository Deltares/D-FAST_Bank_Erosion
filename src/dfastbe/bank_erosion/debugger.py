"""Bank Erosion Debugger."""

from dfastbe.bank_erosion.data_models import (
    DischargeCalculationParameters,
    FairwayData,
    ParametersPerBank,
    SingleBank,
    SingleErosion,
)
from dfastbe.io import write_csv, write_shp


class Debugger:
    """Class to handle debugging and output of bank erosion calculations."""

    def __init__(self, crs: str, output_dir: str):
        """Debugger constructor."""
        self.crs = crs
        self.output_dir = output_dir

    def last_discharge_level(
        self,
        bank_index: int,
        single_bank: SingleBank,
        fairway_data: FairwayData,
        erosion_inputs: SingleErosion,
        discharge_level_pars: ParametersPerBank,
        water_depth_fairway,
        dn_eq1,
        dv_eq1,
        bank_height,
    ):
        """Write the last discharge level to a shapefile and CSV file."""
        bank_coords_mind = single_bank.get_mid_points()
        params = {
            "chainage": single_bank.bank_chainage_midpoints,
            "x": bank_coords_mind[:, 0],
            "y": bank_coords_mind[:, 1],
            "iface_fw": single_bank.fairway_face_indices,
            "iface_bank": single_bank.bank_face_indices,
            "bank_height": bank_height[bank_index],
            "segment_length": single_bank.segment_length,
            "zw0": fairway_data.fairway_initial_water_levels[bank_index],
            "ship_velocity": discharge_level_pars.ship_velocity,
            "ship_type": discharge_level_pars.ship_type,
            "draught": discharge_level_pars.ship_draught,
            "mu_slp": discharge_level_pars.mu_slope,
            "bank_fairway_dist": single_bank.fairway_distances,
            "fairway_wave_reduction_distance": erosion_inputs.wave_fairway_distance_0,
            "fairway_wave_disappear_distance": erosion_inputs.wave_fairway_distance_1,
            "water_depth_fairway": water_depth_fairway,
            "dike_height": erosion_inputs.bank_protection_level,
            "erosion_distance": dn_eq1,
            "erosion_volume": dv_eq1,
        }

        bank_coords_geo = single_bank.get_mid_points(as_geo_series=True, crs=self.crs)
        write_shp(
            bank_coords_geo,
            params,
            f"{str(self.output_dir)}/debug.EQ.B{bank_index + 1}.shp",
        )
        write_csv(params, f"{str(self.output_dir)}/debug.EQ.B{bank_index + 1}.csv")

    def middle_levels(
        self,
        bank_ind: int,
        q_level: int,
        single_bank: SingleBank,
        fairway_data: FairwayData,
        erosion_inputs: SingleErosion,
        discharge_level_pars: ParametersPerBank,
        water_depth_fairway,
        velocity,
        bank_height,
        parameter: DischargeCalculationParameters,
    ):
        """Write the middle levels to a shapefile and CSV file."""
        bank_coords_mind = single_bank.get_mid_points()
        params = {
            "chainage": single_bank.bank_chainage_midpoints,
            "x": bank_coords_mind[:, 0],
            "y": bank_coords_mind[:, 1],
            "iface_fw": single_bank.fairway_face_indices,
            "iface_bank": single_bank.bank_face_indices,
            "velocity": velocity,
            "bank_height": bank_height[bank_ind],
            "segment_length": single_bank.segment_length,
            "zw": parameter.water_level,
            "zw0": fairway_data.fairway_initial_water_levels[bank_ind],
            "tauc": erosion_inputs.tauc,
            "num_ship": discharge_level_pars.num_ship,
            "ship_velocity": discharge_level_pars.ship_velocity,
            "num_waves_per_ship": discharge_level_pars.num_waves_per_ship,
            "ship_type": discharge_level_pars.ship_type,
            "draught": discharge_level_pars.ship_draught,
            "mu_slp": discharge_level_pars.mu_slope,
            "mu_reed": discharge_level_pars.mu_reed,
            "dist_fw": single_bank.fairway_distances,
            "fairway_wave_reduction_distance": erosion_inputs.wave_fairway_distance_0,
            "fairway_wave_disappear_distance": erosion_inputs.wave_fairway_distance_1,
            "water_depth_fairway": water_depth_fairway,
            "chez": parameter.chezy,
            "dike_height": erosion_inputs.bank_protection_level,
            "erosion_distance": parameter.erosion_distance_tot,
            "erosion_volume": parameter.erosion_volume_tot,
            "erosion_distance_shipping": parameter.erosion_distance_shipping,
            "erosion_distance_flow": parameter.erosion_distance_flow,
        }
        bank_coords_geo = single_bank.get_mid_points(as_geo_series=True, crs=self.crs)
        write_shp(
            bank_coords_geo,
            params,
            f"{str(self.output_dir)}/debug.Q{q_level + 1}.B{bank_ind + 1}.shp",
        )
        write_csv(
            params, f"{str(self.output_dir)}/debug.Q{q_level + 1}.B{bank_ind + 1}.csv"
        )
