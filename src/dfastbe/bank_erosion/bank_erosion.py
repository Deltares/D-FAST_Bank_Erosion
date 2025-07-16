"""
Copyright (C) 2020 Stichting Deltares.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation version 2.1.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, see <http://www.gnu.org/licenses/>.

contact: delft3d.support@deltares.nl
Stichting Deltares
P.O. Box 177
2600 MH Delft, The Netherlands

All indications and logos of, and references to, "Delft3D" and "Deltares"
are registered trademarks of Stichting Deltares, and remain the property of
Stichting Deltares. All rights reserved.

INFORMATION
This file is part of D-FAST Bank Erosion: https://github.com/Deltares/D-FAST_Bank_Erosion
"""

import os
from typing import Any, Dict, List, Tuple

import numpy as np
from geopandas.geodataframe import GeoDataFrame
from geopandas.geoseries import GeoSeries
from shapely.geometry import LineString

from dfastbe import __version__
from dfastbe.bank_erosion.data_models.calculation import (
    BankData,
    DischargeLevels,
    ErosionInputs,
    ErosionResults,
    FairwayData,
    MeshData,
    SingleCalculation,
    SingleDischargeLevel,
    SingleErosion,
    SingleLevelParameters,
    SingleParameters,
    WaterLevelData,
)
from dfastbe.bank_erosion.data_models.inputs import (
    ErosionRiverData,
    ErosionSimulationData,
)
from dfastbe.bank_erosion.debugger import Debugger
from dfastbe.bank_erosion.erosion_calculator import ErosionCalculator
from dfastbe.bank_erosion.plotter import ErosionPlotter
from dfastbe.bank_erosion.utils import (
    BankLinesProcessor,
    calculate_alpha,
    get_km_bins,
    get_km_eroded_volume,
    intersect_line_mesh,
    move_line,
    write_km_eroded_volumes,
)
from dfastbe.io.config import ConfigFile
from dfastbe.io.data_models import LineGeometry
from dfastbe.io.logger import log_text, timed_logger

X_AXIS_TITLE = "x-coordinate [km]"
Y_AXIS_TITLE = "y-coordinate [km]"


class Erosion:
    """Class to handle the bank erosion calculations."""

    def __init__(self, config_file: ConfigFile, gui: bool = False):
        """Initialize the Erosion class."""
        self.root_dir = config_file.root_dir
        self._config_file = config_file
        self.gui = gui

        self.river_data = ErosionRiverData(config_file)
        self.river_center_line_arr = self.river_data.river_center_line.as_array()
        self.simulation_data = self.river_data.simulation_data()
        self.sim_files, self.p_discharge = self.river_data.get_erosion_sim_data(
            self.river_data.num_discharge_levels
        )
        self.bl_processor = BankLinesProcessor(self.river_data)
        self.debugger = Debugger(config_file.crs, self.river_data.output_dir)
        self.erosion_calculator = ErosionCalculator()
        self._results = None

    @property
    def config_file(self) -> ConfigFile:
        """Configuration file object."""
        return self._config_file

    @property
    def results(self) -> Dict[str, Any]:
        """dict: Results of the bank line detection analysis.

        Returns:
        """
        return self._results

    @results.setter
    def results(self, value: Dict[str, Any]):
        """Set the results of the bank erosion analysis."""
        self._results = value

    def get_ship_parameters(
        self, num_stations_per_bank: List[int]
    ) -> Dict[str, List[np.ndarray]]:
        """Get ship parameters from the configuration file."""
        ship_relative_velocity = self.config_file.get_parameter(
            "Erosion", "VShip", num_stations_per_bank, positive=True, onefile=True
        )
        num_ships_year = self.config_file.get_parameter(
            "Erosion", "NShip", num_stations_per_bank, positive=True, onefile=True
        )
        num_waves_p_ship = self.config_file.get_parameter(
            "Erosion",
            "NWave",
            num_stations_per_bank,
            default=5,
            positive=True,
            onefile=True,
        )
        ship_draught = self.config_file.get_parameter(
            "Erosion", "Draught", num_stations_per_bank, positive=True, onefile=True
        )
        ship_type = self.config_file.get_parameter(
            "Erosion", "ShipType", num_stations_per_bank, valid=[1, 2, 3], onefile=True
        )
        parslope0 = self.config_file.get_parameter(
            "Erosion",
            "Slope",
            num_stations_per_bank,
            default=20,
            positive=True,
            ext="slp",
        )
        reed_wave_damping_coeff = self.config_file.get_parameter(
            "Erosion",
            "Reed",
            num_stations_per_bank,
            default=0,
            positive=True,
            ext="rdd",
        )

        ship_data = {
            "vship0": ship_relative_velocity,
            "Nship0": num_ships_year,
            "nwave0": num_waves_p_ship,
            "Tship0": ship_draught,
            "ship0": ship_type,
            "parslope0": parslope0,
            "parreed0": reed_wave_damping_coeff,
        }
        return ship_data

    def _process_river_axis_by_center_line(self) -> LineGeometry:
        """Process the river axis by the center line.

        Intersect the river center line with the river axis to map the stations from the first to the latter
        then clip the river axis by the first and last station of the centerline.
        """
        river_axis = LineGeometry(self.river_data.river_axis, crs=self.config_file.crs)
        river_axis_numpy = river_axis.as_array()
        # optional sorting --> see 04_Waal_D3D example
        # check: sum all distances and determine maximum distance ...
        # if maximum > alpha * sum then perform sort
        # Waal OK: 0.0082 ratio max/sum, Waal NotOK: 0.13 - Waal: 2500 points,
        # so even when OK still some 21 times more than 1/2500 = 0.0004
        dist2 = (np.diff(river_axis_numpy, axis=0) ** 2).sum(axis=1)
        alpha = dist2.max() / dist2.sum()
        if alpha > 0.03:
            print("The river axis needs sorting!!")

        # map km to axis points, further using axis
        log_text("chainage_to_axis")
        river_axis_km = river_axis.intersect_with_line(self.river_center_line_arr)

        # clip river axis to reach of interest (get the closest point to the first and last station)
        i1 = np.argmin(
            ((self.river_center_line_arr[0, :2] - river_axis_numpy) ** 2).sum(axis=1)
        )
        i2 = np.argmin(
            ((self.river_center_line_arr[-1, :2] - river_axis_numpy) ** 2).sum(axis=1)
        )
        if i1 < i2:
            river_axis_km = river_axis_km[i1 : i2 + 1]
            river_axis_numpy = river_axis_numpy[i1 : i2 + 1]
        else:
            # reverse river axis
            river_axis_km = river_axis_km[i2 : i1 + 1][::-1]
            river_axis_numpy = river_axis_numpy[i2 : i1 + 1][::-1]

        river_axis = LineGeometry(river_axis_numpy, crs=self.config_file.crs)
        river_axis.add_data(data={"stations": river_axis_km})
        return river_axis

    def _get_fairway_data(
        self,
        river_axis: LineGeometry,
        mesh_data: MeshData,
    ):
        # map km to fairway points, further using axis
        log_text("chainage_to_fairway")
        # intersect fairway and mesh
        fairway_intersection_coords, fairway_face_indices = intersect_line_mesh(
            river_axis.as_array(), mesh_data
        )
        if self.river_data.debug:
            arr = (
                fairway_intersection_coords[:-1] + fairway_intersection_coords[1:]
            ) / 2
            line_geom = LineGeometry(arr, crs=self.config_file.crs)
            line_geom.to_file(
                file_name=f"{str(self.river_data.output_dir)}{os.sep}fairway_face_indices.shp",
                data={"iface": fairway_face_indices},
            )

        return FairwayData(fairway_face_indices, fairway_intersection_coords)

    def calculate_fairway_bank_line_distance(
        self,
        bank_data: BankData,
        fairway_data: FairwayData,
        simulation_data: ErosionSimulationData,
    ):
        """Map bank data to fairway data.

        Args:
            bank_data (BankData):
            fairway_data (FairwayData):
            simulation_data (ErosionSimulationData):

        Returns:
            FairwayData:
                The method updates the following attributes in the `bank_data` instance
                    - fairway_face_indices
                    - fairway_distances
            BankData:
                the following attributes in the `fairway_data` instance
                    - fairway_initial_water_levels
        """
        # distance fairway-bankline (bank-fairway)
        log_text("bank_distance_fairway")

        num_fairway_face_ind = len(fairway_data.fairway_face_indices)

        for bank_i, single_bank in enumerate(bank_data):
            bank_coords = single_bank.bank_line_coords
            coords_mid = (bank_coords[:-1] + bank_coords[1:]) / 2
            bank_fairway_dist = np.zeros(len(coords_mid))
            bp_fw_face_idx = np.zeros(len(coords_mid), dtype=int)

            for ind, coord_i in enumerate(coords_mid):
                # find closest fairway support node
                closest_ind = np.argmin(
                    ((coord_i - fairway_data.intersection_coords) ** 2).sum(axis=1)
                )
                fairway_coord = fairway_data.intersection_coords[closest_ind]
                fairway_bank_distance = ((coord_i - fairway_coord) ** 2).sum() ** 0.5
                # If fairway support node is also the closest projected fairway point, then it likely
                # that that point is one of the original support points (a corner) of the fairway path
                # and located inside a grid cell. The segments before and after that point will then
                # both be located inside that same grid cell, so let's pick the segment before the point.
                # If the point happens to coincide with a grid edge and the two segments are located
                # in different grid cells, then we could either simply choose one or add complexity to
                # average the values of the two grid cells. Let's go for the simplest approach ...
                iseg = max(closest_ind - 1, 0)
                if closest_ind > 0:
                    alpha = calculate_alpha(
                        fairway_data.intersection_coords,
                        closest_ind,
                        closest_ind - 1,
                        coord_i,
                    )
                    if 0 < alpha < 1:
                        fwp1 = fairway_data.intersection_coords[
                            closest_ind - 1
                        ] + alpha * (
                            fairway_data.intersection_coords[closest_ind]
                            - fairway_data.intersection_coords[closest_ind - 1]
                        )
                        d1 = ((coord_i - fwp1) ** 2).sum() ** 0.5
                        if d1 < fairway_bank_distance:
                            fairway_bank_distance = d1
                            # projected point located on segment before, which corresponds to initial choice: iseg = ifw - 1
                if closest_ind < num_fairway_face_ind:
                    alpha = calculate_alpha(
                        fairway_data.intersection_coords,
                        closest_ind + 1,
                        closest_ind,
                        coord_i,
                    )
                    if 0 < alpha < 1:
                        fwp1 = fairway_data.intersection_coords[closest_ind] + alpha * (
                            fairway_data.intersection_coords[closest_ind + 1]
                            - fairway_data.intersection_coords[closest_ind]
                        )
                        d1 = ((coord_i - fwp1) ** 2).sum() ** 0.5
                        if d1 < fairway_bank_distance:
                            fairway_bank_distance = d1
                            iseg = closest_ind

                bp_fw_face_idx[ind] = fairway_data.fairway_face_indices[iseg]
                bank_fairway_dist[ind] = fairway_bank_distance

            if self.river_data.debug:
                line_geom = LineGeometry(coords_mid, crs=self.config_file.crs)
                line_geom.to_file(
                    file_name=f"{self.river_data.output_dir}/bank_{bank_i + 1}_chainage_and_fairway_face_idx.shp",
                    data={
                        "chainage": single_bank.bank_chainage_midpoints,
                        "iface_fw": bp_fw_face_idx[bank_i],
                    },
                )

            single_bank.fairway_face_indices = bp_fw_face_idx
            single_bank.fairway_distances = bank_fairway_dist

        # water level at fairway
        water_level_fairway_ref = []
        for single_bank in bank_data:
            ii = single_bank.fairway_face_indices
            water_level_fairway_ref.append(simulation_data.water_level_face[ii])
        fairway_data.fairway_initial_water_levels = water_level_fairway_ref

    def _prepare_initial_conditions(
        self,
        config_file: ConfigFile,
        num_stations_per_bank: List[int],
        fairway_data: FairwayData,
    ) -> ErosionInputs:
        # wave reduction s0, s1
        wave_fairway_distance_0 = config_file.get_parameter(
            "Erosion",
            "Wave0",
            num_stations_per_bank,
            default=200,
            positive=True,
            onefile=True,
        )
        wave_fairway_distance_1 = config_file.get_parameter(
            "Erosion",
            "Wave1",
            num_stations_per_bank,
            default=150,
            positive=True,
            onefile=True,
        )

        # save 1_banklines
        # read vship, nship, nwave, draught (tship), shiptype ... independent of level number
        shipping_data = self.get_ship_parameters(num_stations_per_bank)

        # read classes flag (yes: banktype = taucp, no: banktype = tauc) and banktype (taucp: 0-4 ... or ... tauc = critical shear value)
        classes = config_file.get_bool("Erosion", "Classes")
        if classes:
            bank_type = config_file.get_parameter(
                "Erosion",
                "BankType",
                num_stations_per_bank,
                default=0,
                ext=".btp",
            )
            tauc = []
            for bank in bank_type:
                tauc.append(ErosionInputs.taucls[bank])
        else:
            tauc = config_file.get_parameter(
                "Erosion",
                "BankType",
                num_stations_per_bank,
                default=0,
                ext=".btp",
            )
            thr = (ErosionInputs.taucls[:-1] + ErosionInputs.taucls[1:]) / 2
            bank_type = [None] * len(thr)
            for ib, shear_stress in enumerate(tauc):
                bt = np.zeros(shear_stress.size)
                for thr_i in thr:
                    bt[shear_stress < thr_i] += 1
                bank_type[ib] = bt

        # read bank protection level dike_height
        zss_miss = -1000
        dike_height = config_file.get_parameter(
            "Erosion",
            "ProtectionLevel",
            num_stations_per_bank,
            default=zss_miss,
            ext=".bpl",
        )
        # if dike_height undefined, set dike_height equal to water_level_fairway_ref - 1
        for ib, one_zss in enumerate(dike_height):
            mask = one_zss == zss_miss
            one_zss[mask] = fairway_data.fairway_initial_water_levels[ib][mask] - 1

        data = {
            'wave_fairway_distance_0': wave_fairway_distance_0,
            'wave_fairway_distance_1': wave_fairway_distance_1,
            'bank_protection_level': dike_height,
            'tauc': tauc
        }
        return ErosionInputs.from_column_arrays(
            data, SingleErosion, shipping_data=shipping_data, bank_type=bank_type
        )

    def _calculate_bank_height(self, bank_data: BankData, simulation_data: ErosionSimulationData) -> BankData:
        # bank height = maximum bed elevation per cell
        for bank_i in bank_data:
            bank_i.height = simulation_data.calculate_bank_height(bank_i, self.river_data.zb_dx)

        return bank_data

    def _process_discharge_levels(
        self,
        km_mid,
        km_bin,
        config_file: ConfigFile,
        erosion_inputs: ErosionInputs,
        bank_data: BankData,
        fairway_data: FairwayData,
    ) -> Tuple[WaterLevelData, ErosionResults]:

        num_levels = self.river_data.num_discharge_levels
        num_km = len(km_mid)

        # initialize arrays for erosion loop over all discharges
        discharge_levels = []

        log_text("total_time", data={"t": self.river_data.erosion_time})

        for level_i in range(num_levels):
            log_text(
                "discharge_header",
                data={
                    "i": level_i + 1,
                    "p": self.p_discharge[level_i],
                    "t": self.p_discharge[level_i] * self.river_data.erosion_time,
                },
            )

            log_text("read_q_params", indent="  ")
            # 1) read level-specific parameters
            # read ship_velocity, num_ship, nwave, draught, ship_type, slope, reed, fairway_depth, ... (level specific values)
            level_parameters = self._read_discharge_parameters(
                level_i, erosion_inputs.shipping_data, bank_data.num_stations_per_bank
            )

            # 2) load FM result
            log_text("-", indent="  ")
            log_text(
                "read_simdata", data={"file": self.sim_files[level_i]}, indent="  "
            )
            simulation_data = ErosionSimulationData.read(
                self.sim_files[level_i], indent="  "
            )
            log_text("bank_erosion", indent="  ")

            if level_i == 0:
                bank_data = self._calculate_bank_height(bank_data, simulation_data)

            single_level, dvol_bank = self.compute_erosion_per_level(
                level_i,
                bank_data,
                simulation_data,
                fairway_data,
                level_parameters,
                erosion_inputs,
                km_bin,
                num_km,
            )

            discharge_levels.append(single_level)

            error_vol_file = config_file.get_str(
                "Erosion", f"EroVol{level_i + 1}", default=f"erovolQ{level_i + 1}.evo"
            )
            log_text("save_error_vol", data={"file": error_vol_file}, indent="  ")
            write_km_eroded_volumes(
                km_mid, dvol_bank, f"{self.river_data.output_dir}/{error_vol_file}"
            )

        # shape is (num_levels, 2, (num_stations_per_bank))
        # if num_levels = 13 and the num_stations_per_bank = [10, 15]
        # then shape = (13, 2, (10, 15)) list of 13 elements, each element is a list of 2 elements
        # first an array of 10 elements, and the second is array of 15 elements
        discharge_levels = DischargeLevels(discharge_levels)
        flow_erosion_dist = discharge_levels.accumulate("erosion_distance_flow")
        ship_erosion_dist = discharge_levels.accumulate("erosion_distance_shipping")
        total_erosion_dist = discharge_levels.accumulate("erosion_distance_tot")
        total_eroded_vol = discharge_levels.accumulate("erosion_volume_tot")

        erosion_results = ErosionResults(
            erosion_time=self.river_data.erosion_time,
            flow_erosion_dist=flow_erosion_dist,
            ship_erosion_dist=ship_erosion_dist,
            total_erosion_dist=total_erosion_dist,
            total_eroded_vol=total_eroded_vol,
            eq_erosion_dist=discharge_levels._get_attr_both_sides_level(
                "erosion_distance_eq", num_levels - 1
            ),
            eq_eroded_vol=discharge_levels._get_attr_both_sides_level(
                "erosion_volume_eq", num_levels - 1
            ),
        )
        water_level_data = discharge_levels.get_water_level_data()

        bank_data.left.bank_line_size, bank_data.right.bank_line_size = (
            bank_data.left.segment_length,
            bank_data.right.segment_length,
        )

        return water_level_data, erosion_results

    def _postprocess_erosion_results(
        self,
        km_bin: Tuple[float, float, float],
        km_mid,
        bank_data: BankData,
        erosion_results: ErosionResults,
    ) -> Tuple[List[LineString], List[LineString], List[LineString]]:
        """Postprocess the erosion results to get the new bank lines and volumes."""
        log_text("=")
        avg_erosion_rate = np.zeros(bank_data.n_bank_lines)
        dn_max = np.zeros(bank_data.n_bank_lines)
        d_nav_flow = np.zeros(bank_data.n_bank_lines)
        d_nav_ship = np.zeros(bank_data.n_bank_lines)
        d_nav_eq = np.zeros(bank_data.n_bank_lines)
        dn_max_eq = np.zeros(bank_data.n_bank_lines)
        eq_eroded_vol_per_km = np.zeros((len(km_mid), bank_data.n_bank_lines))
        total_eroded_vol_per_km = np.zeros((len(km_mid), bank_data.n_bank_lines))
        xy_line_new_list = []
        bankline_new_list = []
        xy_line_eq_list = []
        bankline_eq_list = []
        for ib, single_bank in enumerate(bank_data):
            bank_coords = single_bank.bank_line_coords
            avg_erosion_rate[ib] = (
                erosion_results.total_erosion_dist[ib] * single_bank.bank_line_size
            ).sum() / single_bank.bank_line_size.sum()
            dn_max[ib] = erosion_results.total_erosion_dist[ib].max()
            d_nav_flow[ib] = (
                erosion_results.flow_erosion_dist[ib] * single_bank.bank_line_size
            ).sum() / single_bank.bank_line_size.sum()
            d_nav_ship[ib] = (
                erosion_results.ship_erosion_dist[ib] * single_bank.bank_line_size
            ).sum() / single_bank.bank_line_size.sum()
            d_nav_eq[ib] = (
                erosion_results.eq_erosion_dist[ib] * single_bank.bank_line_size
            ).sum() / single_bank.bank_line_size.sum()
            dn_max_eq[ib] = erosion_results.eq_erosion_dist[ib].max()
            log_text("bank_dnav", data={"ib": ib + 1, "v": avg_erosion_rate[ib]})
            log_text("bank_dnavflow", data={"v": d_nav_flow[ib]})
            log_text("bank_dnavship", data={"v": d_nav_ship[ib]})
            log_text("bank_dnmax", data={"v": dn_max[ib]})
            log_text("bank_dnaveq", data={"v": d_nav_eq[ib]})
            log_text("bank_dnmaxeq", data={"v": dn_max_eq[ib]})

            xy_line_new = move_line(
                bank_coords,
                erosion_results.total_erosion_dist[ib],
                single_bank.is_right_bank,
            )
            xy_line_new_list.append(xy_line_new)
            bankline_new_list.append(LineString(xy_line_new))

            xy_line_eq = move_line(
                bank_coords,
                erosion_results.eq_erosion_dist[ib],
                single_bank.is_right_bank,
            )
            xy_line_eq_list.append(xy_line_eq)
            bankline_eq_list.append(LineString(xy_line_eq))

            dvol_eq = get_km_eroded_volume(
                single_bank.bank_chainage_midpoints,
                erosion_results.eq_eroded_vol[ib],
                km_bin,
            )
            eq_eroded_vol_per_km[:, ib] = dvol_eq
            dvol_tot = get_km_eroded_volume(
                single_bank.bank_chainage_midpoints,
                erosion_results.total_eroded_vol[ib],
                km_bin,
            )
            total_eroded_vol_per_km[:, ib] = dvol_tot
            if ib < bank_data.n_bank_lines - 1:
                log_text("-")

        erosion_results.avg_erosion_rate = avg_erosion_rate
        erosion_results.eq_eroded_vol_per_km = eq_eroded_vol_per_km
        erosion_results.total_eroded_vol_per_km = total_eroded_vol_per_km

        return bankline_new_list, bankline_eq_list, xy_line_eq_list

    def _get_param(
        self, name: str, default_val, iq_str, num_stations_per_bank, **kwargs
    ):
        return self.config_file.get_parameter(
            "Erosion",
            f"{name}{iq_str}",
            num_stations_per_bank,
            default=default_val,
            **kwargs,
        )

    def _read_discharge_parameters(
        self,
        level_i: int,
        shipping_data: Dict[str, List[np.ndarray]],
        num_stations_per_bank: List[int],
    ) -> SingleLevelParameters:
        """Read Discharge level parameters.

        Read all discharge-specific input arrays for level *iq*.
        Returns a dict with keys: vship, num_ship, n_wave, t_ship, ship_type,
        mu_slope, mu_reed, par_slope, par_reed.
        """
        iq_str = f"{level_i + 1}"

        ship_velocity = self._get_param(
            "VShip",
            shipping_data["vship0"],
            iq_str,
            num_stations_per_bank,
        )
        num_ship = self._get_param(
            "NShip",
            shipping_data["Nship0"],
            iq_str,
            num_stations_per_bank,
        )
        num_waves_per_ship = self._get_param(
            "NWave",
            shipping_data["nwave0"],
            iq_str,
            num_stations_per_bank,
        )
        ship_draught = self._get_param(
            "Draught",
            shipping_data["Tship0"],
            iq_str,
            num_stations_per_bank,
        )
        ship_type = self._get_param(
            "ShipType",
            shipping_data["ship0"],
            iq_str,
            num_stations_per_bank,
            valid=[1, 2, 3],
            onefile=True,
        )
        par_slope = self._get_param(
            "Slope",
            shipping_data["parslope0"],
            iq_str,
            num_stations_per_bank,
            positive=True,
            ext="slp",
        )
        par_reed = self._get_param(
            "Reed",
            shipping_data["parreed0"],
            iq_str,
            num_stations_per_bank,
            positive=True,
            ext="rdd",
        )

        mu_slope, mu_reed = [], []
        for ps, pr in zip(par_slope, par_reed):
            mus = ps.copy()
            mus[mus > 0] = 1.0 / mus[mus > 0]  # 1/slope for non-zero values
            mu_slope.append(mus)
            mu_reed.append(8.5e-4 * pr**0.8)  # empirical damping coefficient

        return SingleLevelParameters.from_column_arrays(
            {
                "id": level_i,
                "ship_velocity": ship_velocity,
                "num_ship": num_ship,
                "num_waves_per_ship": num_waves_per_ship,
                "ship_draught": ship_draught,
                "ship_type": ship_type,
                "par_slope": par_slope,
                "par_reed": par_reed,
                "mu_slope": mu_slope,
                "mu_reed": mu_reed,
            },
            SingleParameters,
        )

    def compute_erosion_per_level(
        self,
        level_i: int,
        bank_data: BankData,
        simulation_data: ErosionSimulationData,
        fairway_data: FairwayData,
        single_parameters: SingleLevelParameters,
        erosion_inputs: ErosionInputs,
        km_bin: Tuple[float, float, float],
        num_km: int,
    ) -> Tuple[SingleDischargeLevel, np.ndarray]:
        """Compute the bank erosion for a given level."""
        num_levels = self.river_data.num_discharge_levels
        dvol_bank = np.zeros((num_km, 2))
        hfw_max_level = 0
        par_list = []
        for ind, bank_i in enumerate(bank_data):

            single_calculation = SingleCalculation()
            # bank_i = 0: left bank, bank_i = 1: right bank
            # calculate velocity along banks ...
            single_calculation.bank_velocity = simulation_data.calculate_bank_velocity(
                bank_i, self.river_data.vel_dx
            )

            # get fairway face indices
            fairway_face_indices = bank_i.fairway_face_indices
            data = simulation_data.get_fairway_data(fairway_face_indices)
            single_calculation.water_level = data["water_level"]
            single_calculation.chezy = data["chezy"]
            single_calculation.water_depth = data["water_depth"]

            # get water depth along the fair-way
            hfw_max_level = max(hfw_max_level, single_calculation.water_depth.max())

            # last discharge level
            if level_i == num_levels - 1:
                erosion_distance_eq, erosion_volume_eq = self.erosion_calculator.comp_erosion_eq(
                    bank_i.height,
                    bank_i.segment_length,
                    fairway_data.fairway_initial_water_levels[ind],
                    single_parameters.get_bank(ind),
                    bank_i.fairway_distances,
                    single_calculation.water_depth,
                    erosion_inputs.get_bank(ind),
                )
                single_calculation.erosion_distance_eq = erosion_distance_eq
                single_calculation.erosion_volume_eq = erosion_volume_eq

            single_calculation = self.erosion_calculator.compute_bank_erosion_dynamics(
                single_calculation,
                bank_i.height,
                bank_i.segment_length,
                bank_i.fairway_distances,
                fairway_data.fairway_initial_water_levels[ind],
                single_parameters.get_bank(ind),
                self.river_data.erosion_time * self.p_discharge[level_i],
                erosion_inputs.get_bank(ind),
            )

            # accumulate eroded volumes per km
            volume_per_discharge = get_km_eroded_volume(
                bank_i.bank_chainage_midpoints, single_calculation.erosion_volume_tot, km_bin
            )
            single_calculation.volume_per_discharge = volume_per_discharge
            par_list.append(single_calculation)

            dvol_bank[:, ind] += volume_per_discharge

            if self.river_data.debug:
                self._debug_output(
                    level_i,
                    ind,
                    bank_data,
                    fairway_data,
                    erosion_inputs,
                    single_parameters,
                    num_levels,
                    single_calculation,
                )

        level_calculation = SingleDischargeLevel(
            left=par_list[0], right=par_list[1], hfw_max=hfw_max_level
        )

        return level_calculation, dvol_bank

    def _debug_output(
        self,
        level_i,
        ind,
        bank_data: BankData,
        fairway_data: FairwayData,
        erosion_inputs: ErosionInputs,
        single_level_parameters: SingleLevelParameters,
        num_levels: int,
        single_calculation: SingleCalculation,
    ):
        if level_i == num_levels - 1:
            # EQ debug
            self.debugger.last_discharge_level(
                ind,
                bank_data.get_bank(ind),
                fairway_data,
                erosion_inputs.get_bank(ind),
                single_level_parameters.get_bank(ind),
                single_calculation,
            )
        # Q-specific debug
        self.debugger.middle_levels(
            ind,
            level_i,
            bank_data.get_bank(ind),
            fairway_data,
            erosion_inputs.get_bank(ind),
            single_level_parameters.get_bank(ind),
            single_calculation,
        )

    def run(self) -> None:
        """Run the bank erosion analysis for a specified configuration."""
        timed_logger("-- start analysis --")
        log_text(
            "header_bankerosion",
            data={
                "version": __version__,
                "location": "https://github.com/Deltares/D-FAST_Bank_Erosion",
            },
        )
        log_text("-")
        config_file = self.config_file

        log_text("derive_topology")

        mesh_data = self.simulation_data.compute_mesh_topology()
        river_axis = self._process_river_axis_by_center_line()

        # map to the output interval
        km_bin = (
            river_axis.data["stations"].min(),
            river_axis.data["stations"].max(),
            self.river_data.output_intervals,
        )
        km_mid = get_km_bins(km_bin, station_type="mid")  # get mid-points

        fairway_data = self._get_fairway_data(river_axis, mesh_data)

        # map bank lines to mesh cells
        log_text("intersect_bank_mesh")
        bank_data = self.bl_processor.intersect_with_mesh(mesh_data)
        # map the bank data to the fairway data (the bank_data and fairway_data will be updated inside the `_map_bank_to_fairway` function)
        self.calculate_fairway_bank_line_distance(
            bank_data, fairway_data, self.simulation_data
        )

        erosion_inputs = self._prepare_initial_conditions(
            config_file, bank_data.num_stations_per_bank, fairway_data
        )

        # initialize arrays for erosion loop over all discharges
        water_level_data, erosion_results = self._process_discharge_levels(
            km_mid,
            km_bin,
            config_file,
            erosion_inputs,
            bank_data,
            fairway_data,
        )

        bankline_new_list, bankline_eq_list, xy_line_eq_list = (
            self._postprocess_erosion_results(
                km_bin,
                km_mid,
                bank_data,
                erosion_results,
            )
        )

        self.results = {
            "river_axis": river_axis,
            "bank_data": bank_data,
            "water_level_data": water_level_data,
            "erosion_results": erosion_results,
            "erosion_inputs": erosion_inputs,
            "bankline_new_list": bankline_new_list,
            "bankline_eq_list": bankline_eq_list,
            "xy_line_eq_list": xy_line_eq_list,
            "km_mid": km_mid,
        }

        log_text("end_bankerosion")
        timed_logger("-- end analysis --")

    def plot(self):
        # create various plots
        if self.river_data.plot_flags.plot_data:
            plotter = ErosionPlotter(
                self.gui,
                self.river_data.plot_flags,
                self.results["erosion_results"],
                self.results["bank_data"],
                self.results["water_level_data"],
                self.results["erosion_inputs"],
            )
            plotter.plot_all(
                self.results["river_axis"].data["stations"],
                self.results["xy_line_eq_list"],
                self.results["km_mid"],
                self.river_data.output_intervals,
                self.river_center_line_arr,
                self.simulation_data,
            )

    def save(self):
        self._write_bankline_shapefiles(
            self.results["bankline_new_list"],
            self.results["bankline_eq_list"],
            self.config_file
        )
        self._write_volume_outputs(
            self.results["erosion_results"],
            self.results["km_mid"]
        )
    def _write_bankline_shapefiles(
        self, bankline_new_list, bankline_eq_list, config_file: ConfigFile
    ):
        bankline_new_series = GeoSeries(bankline_new_list, crs=config_file.crs)
        bank_lines_new = GeoDataFrame(geometry=bankline_new_series)
        bank_name = self.config_file.get_str("General", "BankFile", "bankfile")

        bank_file = self.river_data.output_dir / f"{bank_name}_new.shp"
        log_text("save_banklines", data={"file": str(bank_file)})
        bank_lines_new.to_file(bank_file)

        bankline_eq_series = GeoSeries(bankline_eq_list, crs=config_file.crs)
        banklines_eq = GeoDataFrame(geometry=bankline_eq_series)

        bank_file = self.river_data.output_dir / f"{bank_name}_eq.shp"
        log_text("save_banklines", data={"file": str(bank_file)})
        banklines_eq.to_file(bank_file)

    def _write_volume_outputs(self, erosion_results: ErosionResults, km_mid):
        erosion_vol_file = self.config_file.get_str(
            "Erosion", "EroVol", default="erovol.evo"
        )
        log_text("save_tot_erovol", data={"file": erosion_vol_file})
        write_km_eroded_volumes(
            km_mid,
            erosion_results.total_eroded_vol_per_km,
            str(self.river_data.output_dir / erosion_vol_file),
        )

        # write eroded volumes per km (equilibrium)
        erosion_vol_file = self.config_file.get_str(
            "Erosion", "EroVolEqui", default="erovol_eq.evo"
        )
        log_text("save_eq_erovol", data={"file": erosion_vol_file})
        write_km_eroded_volumes(
            km_mid,
            erosion_results.eq_eroded_vol_per_km,
            str(self.river_data.output_dir / erosion_vol_file),
        )


def calculate_alpha(coords: np.ndarray, ind_1: int, ind_2: int, bp: Tuple[int, Any]):
    """Calculate the alpha value for the bank erosion model."""
    alpha = (
        (coords[ind_1, 0] - coords[ind_2, 0]) * (bp[0] - coords[ind_2, 0])
        + (coords[ind_1, 1] - coords[ind_2, 1]) * (bp[1] - coords[ind_2, 1])
    ) / (
        (coords[ind_1, 0] - coords[ind_2, 0]) ** 2
        + (coords[ind_1, 1] - coords[ind_2, 1]) ** 2
    )

    return alpha
