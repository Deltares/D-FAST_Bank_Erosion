"""Bank Erosion Debugger."""

from typing import Dict

import numpy as np
from geopandas import GeoSeries
from geopandas.geodataframe import GeoDataFrame

from dfastbe.bank_erosion.data_models.calculation import (
    FairwayData,
    SingleBank,
    SingleCalculation,
    SingleErosion,
    SingleParameters,
)

__all__ = ["Debugger"]


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
        single_parameters: SingleParameters,
        single_calculation: SingleCalculation,
    ):
        """Write the last discharge level to a shapefile and CSV file."""
        bank_coords_mind = single_bank.get_mid_points()
        params = {
            "chainage": single_bank.bank_chainage_midpoints,
            "x": bank_coords_mind[:, 0],
            "y": bank_coords_mind[:, 1],
            "iface_fw": single_bank.fairway_face_indices,
            "iface_bank": single_bank.bank_face_indices,
            "bank_height": single_bank.height,
            "segment_length": single_bank.segment_length,
            "zw0": fairway_data.fairway_initial_water_levels[bank_index],
            "ship_velocity": single_parameters.ship_velocity,
            "ship_type": single_parameters.ship_type,
            "draught": single_parameters.ship_draught,
            "mu_slp": single_parameters.mu_slope,
            "bank_fairway_dist": single_bank.fairway_distances,
            "fairway_wave_reduction_distance": erosion_inputs.wave_fairway_distance_0,
            "fairway_wave_disappear_distance": erosion_inputs.wave_fairway_distance_1,
            "water_depth_fairway": single_calculation.water_depth,
            "dike_height": erosion_inputs.bank_protection_level,
            "erosion_distance": single_calculation.erosion_distance_eq,
            "erosion_volume": single_calculation.erosion_volume_eq,
        }

        path = f"{str(self.output_dir)}/debug.EQ.B{bank_index + 1}"
        bank_coords_geo = single_bank.get_mid_points(as_geo_series=True, crs=self.crs)
        self._write_data(bank_coords_geo, params, path)

    def middle_levels(
        self,
        bank_ind: int,
        q_level: int,
        single_bank: SingleBank,
        fairway_data: FairwayData,
        erosion_inputs: SingleErosion,
        single_parameters: SingleParameters,
        single_calculation: SingleCalculation,
    ):
        """Write the middle levels to a shapefile and CSV file."""
        bank_coords_mind = single_bank.get_mid_points()
        params = {
            "chainage": single_bank.bank_chainage_midpoints,
            "x": bank_coords_mind[:, 0],
            "y": bank_coords_mind[:, 1],
            "iface_fw": single_bank.fairway_face_indices,
            "iface_bank": single_bank.bank_face_indices,
            "velocity": single_calculation.bank_velocity,
            "bank_height": single_bank.height,
            "segment_length": single_bank.segment_length,
            "zw": single_calculation.water_level,
            "zw0": fairway_data.fairway_initial_water_levels[bank_ind],
            "tauc": erosion_inputs.tauc,
            "num_ship": single_parameters.num_ship,
            "ship_velocity": single_parameters.ship_velocity,
            "num_waves_per_ship": single_parameters.num_waves_per_ship,
            "ship_type": single_parameters.ship_type,
            "draught": single_parameters.ship_draught,
            "mu_slp": single_parameters.mu_slope,
            "mu_reed": single_parameters.mu_reed,
            "dist_fw": single_bank.fairway_distances,
            "fairway_wave_reduction_distance": erosion_inputs.wave_fairway_distance_0,
            "fairway_wave_disappear_distance": erosion_inputs.wave_fairway_distance_1,
            "water_depth_fairway": single_calculation.water_depth,
            "chez": single_calculation.chezy,
            "dike_height": erosion_inputs.bank_protection_level,
            "erosion_distance": single_calculation.erosion_distance_tot,
            "erosion_volume": single_calculation.erosion_volume_tot,
            "erosion_distance_shipping": single_calculation.erosion_distance_shipping,
            "erosion_distance_flow": single_calculation.erosion_distance_flow,
        }
        path = f"{str(self.output_dir)}/debug.Q{q_level + 1}.B{bank_ind + 1}"
        bank_coords_geo = single_bank.get_mid_points(as_geo_series=True, crs=self.crs)
        self._write_data(bank_coords_geo, params, path)

    @staticmethod
    def _write_data(coords: GeoSeries, data: Dict[str, np.ndarray], path: str):
        """Write the data to a shapefile and CSV file."""
        csv_path = f"{path}.csv"
        shp_path = f"{path}.shp"
        _write_shp(coords, data, shp_path)
        _write_csv(data, csv_path)


def _write_shp(geom: GeoSeries, data: Dict[str, np.ndarray], filename: str) -> None:
    """Write a shape file.

    Write a shape file for a given GeoSeries and dictionary of np arrays.
    The GeoSeries and all np should have equal length.

    Arguments
    ---------
    geom : geopandas.geoseries.GeoSeries
        geopandas GeoSeries containing k geometries.
    data : Dict[str, np.ndarray]
        Dictionary of quantities to be written, each np array should have length k.
    filename : str
        Name of the file to be written.

    Returns
    -------
    None
    """
    GeoDataFrame(data, geometry=geom).to_file(filename)


def _write_csv(data: Dict[str, np.ndarray], filename: str) -> None:
    """
    Write a data to csv file.

    Arguments
    ---------
    data : Dict[str, np.ndarray]
        Value(s) to be written.
    filename : str
        Name of the file to be written.

    Returns
    -------
    None
    """
    keys = [key for key in data.keys()]
    header = ""
    for i in range(len(keys)):
        if i < len(keys) - 1:
            header = header + '"' + keys[i] + '", '
        else:
            header = header + '"' + keys[i] + '"'

    data = np.column_stack([array for array in data.values()])
    np.savetxt(filename, data, delimiter=", ", header=header, comments="")