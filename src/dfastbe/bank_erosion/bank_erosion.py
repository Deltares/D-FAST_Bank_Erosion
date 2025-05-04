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

from typing import Tuple, List, Dict, Any
import os
from geopandas.geodataframe import GeoDataFrame
from geopandas.geoseries import GeoSeries
from numpy import ndarray
from shapely.geometry import LineString
import numpy as np
import matplotlib.pyplot as plt

from dfastbe import __version__
from dfastbe.kernel import get_km_bins, comp_erosion_eq, compute_bank_erosion_dynamics, get_km_eroded_volume, \
                            get_zoom_extends
from dfastbe.support import move_line
from dfastbe import plotting as df_plt
from dfastbe.bank_erosion.debugger import Debugger
from dfastbe.io import (
    LineGeometry, ConfigFile,
    log_text,
    write_km_eroded_volumes,
)
from dfastbe.bank_erosion.data_models import (
    ErosionRiverData,
    ErosionSimulationData,
    ErosionInputs,
    WaterLevelData,
    MeshData,
    BankData,
    FairwayData,
    ErosionResults,
)
from dfastbe.bank_erosion.utils import intersect_line_mesh, BankLinesProcessor
from dfastbe.utils import timed_logger


X_AXIS_TITLE = "x-coordinate [km]"
Y_AXIS_TITLE = "y-coordinate [km]"


class Erosion:
    def __init__(self, config_file: ConfigFile, gui: bool = False):
        self.root_dir = config_file.root_dir
        self._config_file = config_file
        self.gui = gui

        self.river_data = ErosionRiverData(config_file)
        self.river_center_line_arr = self.river_data.river_center_line.as_array()
        self.simulation_data = self.river_data.simulation_data()
        self.sim_files, self.p_discharge = self.river_data.get_erosion_sim_data(self.river_data.num_discharge_levels)
        self.bl_processor = BankLinesProcessor(self.river_data)
        self.debugger = Debugger(config_file, self.river_data)

    @property
    def config_file(self) -> ConfigFile:
        """Configuration file object."""
        return self._config_file

    def get_ship_parameters(self, num_stations_per_bank: List[int]) -> Dict[str, float]:

        ship_relative_velocity = self.config_file.get_parameter(
            "Erosion", "VShip", num_stations_per_bank, positive=True, onefile=True
        )
        num_ships_year = self.config_file.get_parameter(
            "Erosion", "NShip", num_stations_per_bank, positive=True, onefile=True
        )
        num_waves_p_ship = self.config_file.get_parameter(
            "Erosion", "NWave", num_stations_per_bank, default=5, positive=True, onefile=True
        )
        ship_draught = self.config_file.get_parameter(
            "Erosion", "Draught", num_stations_per_bank, positive=True, onefile=True
        )
        ship_type = self.config_file.get_parameter(
            "Erosion", "ShipType", num_stations_per_bank, valid=[1, 2, 3], onefile=True
        )
        parslope0 = self.config_file.get_parameter(
            "Erosion", "Slope", num_stations_per_bank, default=20, positive=True, ext="slp"
        )
        reed_wave_damping_coeff = self.config_file.get_parameter(
            "Erosion", "Reed", num_stations_per_bank, default=0, positive=True, ext="rdd"
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
        """
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

        # clip river axis to reach of interest (get closest point to the first and last station)
        i1 = np.argmin(((self.river_center_line_arr[0, :2] - river_axis_numpy) ** 2).sum(axis=1))
        i2 = np.argmin(((self.river_center_line_arr[-1, :2] - river_axis_numpy) ** 2).sum(axis=1))
        if i1 < i2:
            river_axis_km = river_axis_km[i1 : i2 + 1]
            river_axis_numpy = river_axis_numpy[i1 : i2 + 1]
        else:
            # reverse river axis
            river_axis_km = river_axis_km[i2 : i1 + 1][::-1]
            river_axis_numpy = river_axis_numpy[i2 : i1 + 1][::-1]

        # river_axis = LineString(river_axis_numpy)
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
        # log_text("intersect_fairway_mesh", data={"n": len(fairway_numpy)})
        fairway_intersection_coords, fairway_face_indices = intersect_line_mesh(
            river_axis.as_array(), mesh_data
        )
        if self.river_data.debug:
            arr = (fairway_intersection_coords[:-1] + fairway_intersection_coords[1:])/ 2
            line_geom = LineGeometry(arr, crs=self.config_file.crs)
            line_geom.to_file(
                file_name=f"{str(self.river_data.output_dir)}{os.sep}fairway_face_indices.shp",
                data={"iface": fairway_face_indices},
            )

        return FairwayData(fairway_face_indices, fairway_intersection_coords)

    def _map_bank_to_fairway(
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
            The method updates the following attributes in the `bank_data` instance
                - fairway_face_indices
                - fairway_distances
            and the following attributes in the `fairway_data` instance
                - fairway_initial_water_levels
        """
        # distance fairway-bankline (bank-fairway)
        log_text("bank_distance_fairway")
        bank_fairway_dist = []
        bp_fw_face_idx = []
        num_fairway_face_ind = len(fairway_data.fairway_face_indices)
        for bank_i, bank_coords in enumerate(bank_data.bank_line_coords):
            coords_mid = (bank_coords[:-1] + bank_coords[1:]) / 2
            bank_fairway_dist.append(np.zeros(len(coords_mid)))
            bp_fw_face_idx.append(np.zeros(len(coords_mid), dtype=int))
            for ip, bp in enumerate(coords_mid):
                # find closest fairway support node
                ifw = np.argmin(
                    ((bp - fairway_data.intersection_coords) ** 2).sum(axis=1)
                )
                fwp = fairway_data.intersection_coords[ifw]
                dbfw = ((bp - fwp) ** 2).sum() ** 0.5
                # If fairway support node is also the closest projected fairway point, then it likely
                # that that point is one of the original support points (a corner) of the fairway path
                # and located inside a grid cell. The segments before and after that point will then
                # both be located inside that same grid cell, so let's pick the segment before the point.
                # If the point happens to coincide with a grid edge and the two segments are located
                # in different grid cells, then we could either simply choose one or add complexity to
                # average the values of the two grid cells. Let's go for the simplest approach ...
                iseg = max(ifw - 1, 0)
                if ifw > 0:
                    alpha = (
                        (
                            fairway_data.intersection_coords[ifw, 0]
                            - fairway_data.intersection_coords[ifw - 1, 0]
                        )
                        * (bp[0] - fairway_data.intersection_coords[ifw - 1, 0])
                        + (
                            fairway_data.intersection_coords[ifw, 1]
                            - fairway_data.intersection_coords[ifw - 1, 1]
                        )
                        * (bp[1] - fairway_data.intersection_coords[ifw - 1, 1])
                    ) / (
                        (
                            fairway_data.intersection_coords[ifw, 0]
                            - fairway_data.intersection_coords[ifw - 1, 0]
                        )
                        ** 2
                        + (
                            fairway_data.intersection_coords[ifw, 1]
                            - fairway_data.intersection_coords[ifw - 1, 1]
                        )
                        ** 2
                    )
                    if 0 < alpha < 1:
                        fwp1 = fairway_data.intersection_coords[ifw - 1] + alpha * (
                            fairway_data.intersection_coords[ifw]
                            - fairway_data.intersection_coords[ifw - 1]
                        )
                        d1 = ((bp - fwp1) ** 2).sum() ** 0.5
                        if d1 < dbfw:
                            dbfw = d1
                            # projected point located on segment before, which corresponds to initial choice: iseg = ifw - 1
                if ifw < num_fairway_face_ind:
                    alpha = (
                        (
                            fairway_data.intersection_coords[ifw + 1, 0]
                            - fairway_data.intersection_coords[ifw, 0]
                        )
                        * (bp[0] - fairway_data.intersection_coords[ifw, 0])
                        + (
                            fairway_data.intersection_coords[ifw + 1, 1]
                            - fairway_data.intersection_coords[ifw, 1]
                        )
                        * (bp[1] - fairway_data.intersection_coords[ifw, 1])
                    ) / (
                        (
                            fairway_data.intersection_coords[ifw + 1, 0]
                            - fairway_data.intersection_coords[ifw, 0]
                        )
                        ** 2
                        + (
                            fairway_data.intersection_coords[ifw + 1, 1]
                            - fairway_data.intersection_coords[ifw, 1]
                        )
                        ** 2
                    )
                    if 0 < alpha < 1:
                        fwp1 = fairway_data.intersection_coords[ifw] + alpha * (
                            fairway_data.intersection_coords[ifw + 1]
                            - fairway_data.intersection_coords[ifw]
                        )
                        d1 = ((bp - fwp1) ** 2).sum() ** 0.5
                        if d1 < dbfw:
                            dbfw = d1
                            iseg = ifw

                bp_fw_face_idx[bank_i][ip] = fairway_data.fairway_face_indices[iseg]
                bank_fairway_dist[bank_i][ip] = dbfw

            if self.river_data.debug:
                line_geom = LineGeometry(coords_mid, crs=self.config_file.crs)
                line_geom.to_file(
                    file_name=f"{self.river_data.output_dir}/bank_{bank_i + 1}_chainage_and_fairway_face_idx.shp",
                    data={
                        "chainage": bank_data.bank_chainage_midpoints[bank_i],
                        "iface_fw": bp_fw_face_idx[bank_i],
                    },
                )

        bank_data.fairway_face_indices = bp_fw_face_idx
        bank_data.fairway_distances = bank_fairway_dist

        # water level at fairway
        water_level_fairway_ref = []
        for bank_i in range(bank_data.n_bank_lines):
            ii = bank_data.fairway_face_indices[bank_i]
            water_level_fairway_ref.append(simulation_data.water_level_face[ii])
        fairway_data.fairway_initial_water_levels = water_level_fairway_ref

    def _prepare_initial_conditions(
        self, config_file: ConfigFile, num_stations_per_bank: List[int], fairway_data: FairwayData
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

        return ErosionInputs(
            shipping_data=shipping_data,
            wave_fairway_distance_0=wave_fairway_distance_0,
            wave_fairway_distance_1=wave_fairway_distance_1,
            bank_protection_level=dike_height,
            tauc=tauc,
            bank_type=bank_type,
        )

    def _process_discharge_levels(
        self,
        km_mid,
        km_bin,
        config_file: ConfigFile,
        erosion_inputs: ErosionInputs,
        bank_data: BankData,
        fairway_data: FairwayData,
    ) -> Tuple[WaterLevelData, ErosionResults]:
        num_stations_per_bank = [len(bank_i) for bank_i in bank_data.bank_chainage_midpoints]
        num_levels = self.river_data.num_discharge_levels
        num_km = len(km_mid)
        num_bank = bank_data.n_bank_lines
        segment_length = bank_data.segment_length
        # initialize arrays for erosion loop over all discharges
        # shape is (num_levels, 2, (num_stations_per_bank))
        # if num_levels = 13 and the num_stations_per_bank = [10, 15]
        # then shape = (13, 2, (10, 15)) list of 13 elements, each element is a list of 2 elements
        # first an array of 10 elements, and the second is array of 15 elements
        velocity_all: List[List[np.ndarray]] = []
        water_level_all: List[List[np.ndarray]] = []
        chezy_all: List[List[np.ndarray]] = []
        vol_per_discharge_all: List[List[np.ndarray]] = []
        ship_wave_max_all: List[List[np.ndarray]] = []
        ship_wave_min_all: List[List[np.ndarray]] = []

        bank_height = []
        flow_erosion_dist = []
        ship_erosion_dist = []
        total_erosion_dist = []
        total_eroded_vol = []

        eq_erosion_dist = []
        eq_eroded_vol = []

        log_text("total_time", data={"t": self.river_data.erosion_time})


        for level_i in range(num_levels):
            log_text(
                "discharge_header",
                data={"i": level_i + 1, "p": self.p_discharge[level_i], "t": self.p_discharge[level_i] * self.river_data.erosion_time, },
            )

            log_text("read_q_params", indent="  ")
            # 1) read level-specific parameters
            # read v_ship, n_ship, nwave, draught, ship_type, slope, reed, fairway_depth, ... (level specific values)
            pars = self._read_discharge_parameters(level_i, erosion_inputs, num_stations_per_bank)

            # 2) load FM result
            log_text("-", indent="  ")
            log_text("read_simdata", data={"file": self.sim_files[level_i]}, indent="  ")
            log_text("-", indent="  ")
            simulation_data = ErosionSimulationData.read(self.sim_files[level_i], indent="  ")
            log_text("-", indent="  ")

            log_text("bank_erosion", indent="  ")

            velocity_all.append([])
            water_level_all.append([])
            chezy_all.append([])

            ship_wave_max_all.append([])
            ship_wave_min_all.append([])
            vol_per_discharge_all.append([])

            dvol_bank = np.zeros((num_km, num_bank))

            hfw_max_level = 0

            for bank_i, bank_i_coords in enumerate(bank_data.bank_line_coords):
                # bank_i = 0: left bank, bank_i = 1: right bank
                # calculate velocity along banks ...
                vel_bank = simulation_data.calculate_bank_velocity(bank_data, bank_i, self.river_data.vel_dx)
                velocity_all[level_i].append(vel_bank)

                if level_i == 0:
                    # determine velocity and bank height along banks ...
                    # bank height = maximum bed elevation per cell
                    zb_bank = simulation_data.calculate_bank_height(bank_i, bank_data, self.river_data.zb_dx)
                    bank_height.append(zb_bank)

                # get water depth along the fair-way
                ii_face = bank_data.fairway_face_indices[bank_i]
                water_depth_fairway = simulation_data.water_depth_face[ii_face]
                hfw_max_level = max(hfw_max_level, water_depth_fairway.max())

                water_level_all[level_i].append(simulation_data.water_level_face[ii_face])
                chez_face = simulation_data.chezy_face[ii_face]
                chezy_all[level_i].append(0 * chez_face + chez_face.mean())

                # last discharge level
                if level_i == num_levels - 1:
                    dn_eq1, dv_eq1 = comp_erosion_eq(
                        bank_height[bank_i],
                        segment_length[bank_i],
                        fairway_data.fairway_initial_water_levels[bank_i],
                        pars["v_ship"][bank_i],
                        pars["ship_type"][bank_i],
                        pars["t_ship"][bank_i],
                        pars["mu_slope"][bank_i],
                        bank_data.fairway_distances[bank_i],
                        water_depth_fairway,
                        erosion_inputs,
                        bank_i,
                    )
                    eq_erosion_dist.append(dn_eq1)
                    eq_eroded_vol.append(dv_eq1)


                dn_tot, dv_tot, erosion_distance_shipping, erosion_distance_flow, ship_w_max, ship_w_min = (
                    compute_bank_erosion_dynamics(
                        velocity_all[level_i][bank_i],
                        bank_height[bank_i],
                        segment_length[bank_i],
                        water_level_all[level_i][bank_i],
                        fairway_data.fairway_initial_water_levels[bank_i],
                        pars["n_ship"][bank_i],
                        pars["v_ship"][bank_i],
                        pars["n_wave"][bank_i],
                        pars["ship_type"][bank_i],
                        pars["t_ship"][bank_i],
                        self.river_data.erosion_time * self.p_discharge[level_i],
                        bank_data.fairway_distances[bank_i],
                        water_depth_fairway,
                        chezy_all[level_i][bank_i],
                        erosion_inputs,
                        bank_i,
                    )
                )
                ship_wave_max_all[level_i].append(ship_w_max)
                ship_wave_min_all[level_i].append(ship_w_min)

                if self.river_data.debug:
                    if level_i == num_levels - 1:
                        # EQ debug
                        self.debugger.debug_process_discharge_levels_1(
                            bank_i, bank_data, fairway_data, erosion_inputs, pars, water_depth_fairway, dn_eq1, dv_eq1, bank_i_coords,
                            bank_height, segment_length
                        )
                    # Q-specific debug
                    self.debugger.debug_process_discharge_levels_2(
                        bank_i, level_i, bank_data, fairway_data, erosion_inputs, pars, water_depth_fairway, bank_i_coords, velocity_all, bank_height, segment_length,
                        water_level_all, chezy_all, dn_tot, dv_tot, erosion_distance_shipping, erosion_distance_flow
                    )

                # shift bank lines
                if len(total_erosion_dist) == bank_i:
                    flow_erosion_dist.append(erosion_distance_flow.copy())
                    ship_erosion_dist.append(erosion_distance_shipping.copy())
                    total_erosion_dist.append(dn_tot.copy())
                    total_eroded_vol.append(dv_tot.copy())
                else:
                    flow_erosion_dist[bank_i] += erosion_distance_flow
                    ship_erosion_dist[bank_i] += erosion_distance_shipping
                    total_erosion_dist[bank_i] += dn_tot
                    total_eroded_vol[bank_i] += dv_tot

                # accumulate eroded volumes per km
                dvol = get_km_eroded_volume(
                    bank_data.bank_chainage_midpoints[bank_i], dv_tot, km_bin
                )
                vol_per_discharge_all[level_i].append(dvol)
                dvol_bank[:, bank_i] += dvol

            error_vol_file = config_file.get_str("Erosion", f"EroVol{level_i + 1}", default=f"erovolQ{level_i + 1}.evo")
            log_text("save_error_vol", data={"file": error_vol_file}, indent="  ")
            write_km_eroded_volumes(km_mid, dvol_bank, f"{self.river_data.output_dir}/{error_vol_file}")

        erosion_results = ErosionResults(
            eq_erosion_dist=eq_erosion_dist,
            total_erosion_dist=total_erosion_dist,
            flow_erosion_dist=flow_erosion_dist,
            ship_erosion_dist=ship_erosion_dist,
            vol_per_discharge=vol_per_discharge_all,
            eq_eroded_vol=eq_eroded_vol,
            total_eroded_vol=total_eroded_vol,
            erosion_time=self.river_data.erosion_time,
        )

        water_level_data = WaterLevelData(
            hfw_max=hfw_max_level,
            water_level=water_level_all,
            ship_wave_max=ship_wave_max_all,
            ship_wave_min=ship_wave_min_all,
            velocity=velocity_all,
            bank_height=bank_height,
            chezy=chezy_all,
        )
        bank_data.bank_line_size = segment_length

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
        for ib, bank_coords in enumerate(bank_data.bank_line_coords):
            avg_erosion_rate[ib] = (
                erosion_results.total_erosion_dist[ib] * bank_data.bank_line_size[ib]
            ).sum() / bank_data.bank_line_size[ib].sum()
            dn_max[ib] = erosion_results.total_erosion_dist[ib].max()
            d_nav_flow[ib] = (
                erosion_results.flow_erosion_dist[ib] * bank_data.bank_line_size[ib]
            ).sum() / bank_data.bank_line_size[ib].sum()
            d_nav_ship[ib] = (
                erosion_results.ship_erosion_dist[ib] * bank_data.bank_line_size[ib]
            ).sum() / bank_data.bank_line_size[ib].sum()
            d_nav_eq[ib] = (
                erosion_results.eq_erosion_dist[ib] * bank_data.bank_line_size[ib]
            ).sum() / bank_data.bank_line_size[ib].sum()
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
                bank_data.is_right_bank[ib],
            )
            xy_line_new_list.append(xy_line_new)
            bankline_new_list.append(LineString(xy_line_new))

            xy_line_eq = move_line(
                bank_coords,
                erosion_results.eq_erosion_dist[ib],
                bank_data.is_right_bank[ib],
            )
            xy_line_eq_list.append(xy_line_eq)
            bankline_eq_list.append(LineString(xy_line_eq))

            dvol_eq = get_km_eroded_volume(
                bank_data.bank_chainage_midpoints[ib],
                erosion_results.eq_eroded_vol[ib],
                km_bin,
            )
            eq_eroded_vol_per_km[:, ib] = dvol_eq
            dvol_tot = get_km_eroded_volume(
                bank_data.bank_chainage_midpoints[ib],
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

    def _get_param(self, name: str, default_val, iq_str, num_stations_per_bank, **kwargs):
        return self.config_file.get_parameter(
            "Erosion", f"{name}{iq_str}", num_stations_per_bank,
            default=default_val, **kwargs
        )

    def _read_discharge_parameters(
        self,
        iq: int,
        erosion_inputs: ErosionInputs,
        num_stations_per_bank: List[int],
    ) -> dict[str, list[ndarray] | list[Any]]:
        """
        Read all discharge-specific input arrays for level *iq*.
        Returns a dict with keys: vship, n_ship, n_wave, t_ship, ship_type,
        mu_slope, mu_reed, par_slope, par_reed.
        """
        iq_str = f"{iq + 1}"

        ship_velocity = self._get_param("VShip", erosion_inputs.shipping_data["vship0"], iq_str, num_stations_per_bank)
        num_ship = self._get_param("NShip", erosion_inputs.shipping_data["Nship0"], iq_str, num_stations_per_bank)
        n_wave = self._get_param("NWave", erosion_inputs.shipping_data["nwave0"], iq_str, num_stations_per_bank)
        t_ship = self._get_param("Draught", erosion_inputs.shipping_data["Tship0"], iq_str, num_stations_per_bank)
        ship_type = self._get_param("ShipType", erosion_inputs.shipping_data["ship0"], iq_str, num_stations_per_bank, valid=[1, 2, 3], onefile=True)
        par_slope = self._get_param("Slope", erosion_inputs.shipping_data["parslope0"], iq_str, num_stations_per_bank, positive=True, ext="slp")
        par_reed = self._get_param("Reed", erosion_inputs.shipping_data["parreed0"], iq_str, num_stations_per_bank, positive=True, ext="rdd")

        mu_slope, mu_reed = [], []
        for ps, pr in zip(par_slope, par_reed):
            mus = ps.copy()
            mus[mus > 0] = 1.0 / mus[mus > 0]   # 1/slope for non-zero values
            mu_slope.append(mus)
            mu_reed.append(8.5e-4 * pr ** 0.8)  # empirical damping coefficient

        return {
            "v_ship": ship_velocity, "n_ship": num_ship, "n_wave": n_wave, "t_ship": t_ship,
            "ship_type": ship_type, "par_slope": par_slope, "par_reed": par_reed,
            "mu_slope": mu_slope, "mu_reed": mu_reed,
        }

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
        km_bin = (river_axis.data["stations"].min(), river_axis.data["stations"].max(), self.river_data.output_intervals)
        km_mid = get_km_bins(km_bin, type=3)  # get mid-points

        fairway_data = self._get_fairway_data(river_axis, mesh_data)

        # map bank lines to mesh cells
        log_text("intersect_bank_mesh")
        bank_data = self.bl_processor.intersect_with_mesh(mesh_data)
        # map the bank data to the fairway data (the bank_data and fairway_data will be updated inside the `_map_bank_to_fairway` function)
        self._map_bank_to_fairway(bank_data, fairway_data, self.simulation_data)

        num_stations_per_bank = [len(bank_i) for bank_i in bank_data.bank_chainage_midpoints]
        erosion_inputs = self._prepare_initial_conditions(
            config_file, num_stations_per_bank, fairway_data
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

        self._write_bankline_shapefiles(bankline_new_list, bankline_eq_list, config_file)
        self._write_volume_outputs(erosion_results, km_mid)

        # create various plots
        self._generate_plots(
            river_axis.data["stations"],
            self.simulation_data,
            xy_line_eq_list,
            km_mid,
            self.river_data.output_intervals,
            erosion_inputs,
            water_level_data,
            mesh_data,
            bank_data,
            erosion_results,
        )
        log_text("end_bankerosion")
        timed_logger("-- end analysis --")

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
        erosion_vol_file = self.config_file.get_str("Erosion", "EroVol", default="erovol.evo")
        log_text("save_tot_erovol", data={"file": erosion_vol_file})
        write_km_eroded_volumes(
            km_mid,
            erosion_results.total_eroded_vol_per_km,
            str(self.river_data.output_dir / erosion_vol_file),
        )

        # write eroded volumes per km (equilibrium)
        erosion_vol_file = self.config_file.get_str("Erosion", "EroVolEqui", default="erovol_eq.evo")
        log_text("save_eq_erovol", data={"file": erosion_vol_file})
        write_km_eroded_volumes(
            km_mid,
            erosion_results.eq_eroded_vol_per_km,
            str(self.river_data.output_dir / erosion_vol_file),
        )

    def _generate_plots(
        self,
        river_axis_km,
        simulation_data: ErosionSimulationData,
        xy_line_eq_list,
        km_mid,
        km_step,
        erosion_inputs: ErosionInputs,
        water_level_data: WaterLevelData,
        mesh_data: MeshData,
        bank_data: BankData,
        erosion_results: ErosionResults,
    ):
        # create various plots
        if self.river_data.plot_flags["plot_data"]:
            log_text("=")
            log_text("create_figures")
            fig_i = 0
            bbox = self.river_data.get_bbox(self.river_center_line_arr)

            if self.river_data.plot_flags["save_plot_zoomed"]:
                bank_coords_mid = []
                for ib in range(bank_data.n_bank_lines):
                    bank_coords_mid.append(
                        (
                            bank_data.bank_line_coords[ib][:-1, :]
                            + bank_data.bank_line_coords[ib][1:, :]
                        )
                        / 2
                    )
                km_zoom, xy_zoom = get_zoom_extends(
                    river_axis_km.min(),
                    river_axis_km.max(),
                    self.river_data.plot_flags["zoom_km_step"],
                    bank_coords_mid,
                    bank_data.bank_chainage_midpoints,
                )

            fig, ax = df_plt.plot1_waterdepth_and_banklines(
                bbox,
                self.river_center_line_arr,
                bank_data.bank_lines,
                simulation_data.face_node,
                simulation_data.n_nodes,
                simulation_data.x_node,
                simulation_data.y_node,
                simulation_data.water_depth_face,
                1.1 * water_level_data.hfw_max,
                X_AXIS_TITLE,
                Y_AXIS_TITLE,
                "water depth and initial bank lines",
                "water depth [m]",
            )
            if self.river_data.plot_flags["save_plot"]:
                fig_i = fig_i + 1
                fig_base = f"{self.river_data.plot_flags['fig_dir']}{os.sep}{fig_i}_banklines"

                if self.river_data.plot_flags["save_plot_zoomed"]:
                    df_plt.zoom_xy_and_save(fig, ax, fig_base, self.river_data.plot_flags["plot_ext"], xy_zoom)

                fig_path = fig_base + self.river_data.plot_flags["plot_ext"]
                df_plt.savefig(fig, fig_path)

            fig, ax = df_plt.plot2_eroded_distance_and_equilibrium(
                bbox,
                self.river_center_line_arr,
                bank_data.bank_line_coords,
                erosion_results.total_erosion_dist,
                bank_data.is_right_bank,
                erosion_results.avg_erosion_rate,
                xy_line_eq_list,
                mesh_data.x_edge_coords,
                mesh_data.y_edge_coords,
                X_AXIS_TITLE,
                Y_AXIS_TITLE,
                "eroded distance and equilibrium bank location",
                f"eroded during {erosion_results.erosion_time} year",
                "eroded distance [m]",
                "equilibrium location",
            )
            if self.river_data.plot_flags["save_plot"]:
                fig_i = fig_i + 1
                fig_base = f"{self.river_data.plot_flags['fig_dir']}{os.sep}{fig_i}_erosion_sensitivity"

                if self.river_data.plot_flags["save_plot_zoomed"]:
                    df_plt.zoom_xy_and_save(fig, ax, fig_base, self.river_data.plot_flags["plot_ext"], xy_zoom)

                fig_path = fig_base + self.river_data.plot_flags["plot_ext"]
                df_plt.savefig(fig, fig_path)

            fig, ax = df_plt.plot3_eroded_volume(
                km_mid,
                km_step,
                "river chainage [km]",
                erosion_results.vol_per_discharge,
                "eroded volume [m^3]",
                f"eroded volume per {km_step} chainage km ({erosion_results.erosion_time} years)",
                "Q{iq}",
                "Bank {ib}",
            )
            if self.river_data.plot_flags["save_plot"]:
                fig_i = fig_i + 1
                fig_base = f"{self.river_data.plot_flags['fig_dir']}{os.sep}{fig_i}_eroded_volume"

                if self.river_data.plot_flags["save_plot_zoomed"]:
                    df_plt.zoom_x_and_save(fig, ax, fig_base, self.river_data.plot_flags["plot_ext"], km_zoom)

                fig_path = fig_base + self.river_data.plot_flags["plot_ext"]
                df_plt.savefig(fig, fig_path)

            fig, ax = df_plt.plot3_eroded_volume_subdivided_1(
                km_mid,
                km_step,
                "river chainage [km]",
                erosion_results.vol_per_discharge,
                "eroded volume [m^3]",
                f"eroded volume per {km_step} chainage km ({erosion_results.erosion_time} years)",
                "Q{iq}",
            )
            if self.river_data.plot_flags["save_plot"]:
                fig_i = fig_i + 1
                fig_base = self.river_data.plot_flags["fig_dir"] + os.sep + str(fig_i) + "_eroded_volume_per_discharge"
                if self.river_data.plot_flags["save_plot_zoomed"]:
                    df_plt.zoom_x_and_save(fig, ax, fig_base, self.river_data.plot_flags["plot_ext"], km_zoom)
                fig_path = fig_base + self.river_data.plot_flags["plot_ext"]
                df_plt.savefig(fig, fig_path)

            fig, ax = df_plt.plot3_eroded_volume_subdivided_2(
                km_mid,
                km_step,
                "river chainage [km]",
                erosion_results.vol_per_discharge,
                "eroded volume [m^3]",
                f"eroded volume per {km_step} chainage km ({erosion_results.erosion_time} years)",
                "Bank {ib}",
            )
            if self.river_data.plot_flags["save_plot"]:
                fig_i = fig_i + 1
                fig_base = self.river_data.plot_flags["fig_dir"] + os.sep + str(fig_i) + "_eroded_volume_per_bank"
                if self.river_data.plot_flags["save_plot_zoomed"]:
                    df_plt.zoom_x_and_save(fig, ax, fig_base, self.river_data.plot_flags["plot_ext"], km_zoom)
                fig_path = fig_base + self.river_data.plot_flags["plot_ext"]
                df_plt.savefig(fig, fig_path)

            fig, ax = df_plt.plot4_eroded_volume_eq(
                km_mid,
                km_step,
                "river chainage [km]",
                erosion_results.eq_eroded_vol_per_km,
                "eroded volume [m^3]",
                f"eroded volume per {km_step} chainage km (equilibrium)",
            )
            if self.river_data.plot_flags["save_plot"]:
                fig_i = fig_i + 1
                fig_base = self.river_data.plot_flags["fig_dir"] + os.sep + str(fig_i) + "_eroded_volume_eq"
                if self.river_data.plot_flags["save_plot_zoomed"]:
                    df_plt.zoom_x_and_save(fig, ax, fig_base, self.river_data.plot_flags["plot_ext"], km_zoom)
                fig_path = fig_base + self.river_data.plot_flags["plot_ext"]
                df_plt.savefig(fig, fig_path)

            figlist, axlist = df_plt.plot5series_waterlevels_per_bank(
                bank_data.bank_chainage_midpoints,
                "river chainage [km]",
                water_level_data.water_level,
                water_level_data.ship_wave_max,
                water_level_data.ship_wave_min,
                "water level at Q{iq}",
                "average water level",
                "wave influenced range",
                water_level_data.bank_height,
                "level of bank",
                erosion_inputs.bank_protection_level,
                "bank protection level",
                "elevation",
                "(water)levels along bank line {ib}",
                "[m NAP]",
            )
            if self.river_data.plot_flags["save_plot"]:
                for ib, fig in enumerate(figlist):
                    fig_i = fig_i + 1
                    fig_base = f"{self.river_data.plot_flags['fig_dir']}/{fig_i}_levels_bank_{ib + 1}"

                    if self.river_data.plot_flags["save_plot_zoomed"]:
                        df_plt.zoom_x_and_save(fig, axlist[ib], fig_base, self.river_data.plot_flags["plot_ext"], km_zoom)
                    fig_file = f"{fig_base}{self.river_data.plot_flags['plot_ext']}"
                    df_plt.savefig(fig, fig_file)

            figlist, axlist = df_plt.plot6series_velocity_per_bank(
                bank_data.bank_chainage_midpoints,
                "river chainage [km]",
                water_level_data.velocity,
                "velocity at Q{iq}",
                erosion_inputs.tauc,
                water_level_data.chezy[0],
                "critical velocity",
                "velocity",
                "velocity along bank line {ib}",
                "[m/s]",
            )
            if self.river_data.plot_flags["save_plot"]:
                for ib, fig in enumerate(figlist):
                    fig_i = fig_i + 1
                    fig_base = f"{self.river_data.plot_flags['fig_dir']}{os.sep}{fig_i}_velocity_bank_{ib + 1}"

                    if self.river_data.plot_flags["save_plot_zoomed"]:
                        df_plt.zoom_x_and_save(fig, axlist[ib], fig_base, self.river_data.plot_flags["plot_ext"], km_zoom)

                    fig_file = fig_base + self.river_data.plot_flags["plot_ext"]
                    df_plt.savefig(fig, fig_file)

            fig, ax = df_plt.plot7_banktype(
                bbox,
                self.river_center_line_arr,
                bank_data.bank_line_coords,
                erosion_inputs.bank_type,
                erosion_inputs.taucls_str,
                X_AXIS_TITLE,
                Y_AXIS_TITLE,
                "bank type",
            )
            if self.river_data.plot_flags["save_plot"]:
                fig_i = fig_i + 1
                fig_base = self.river_data.plot_flags["fig_dir"] + os.sep + str(fig_i) + "_banktype"
                if self.river_data.plot_flags["save_plot_zoomed"]:
                    df_plt.zoom_xy_and_save(fig, ax, fig_base, self.river_data.plot_flags["plot_ext"], xy_zoom)
                fig_file = fig_base + self.river_data.plot_flags["plot_ext"]
                df_plt.savefig(fig, fig_file)

            fig, ax = df_plt.plot8_eroded_distance(
                bank_data.bank_chainage_midpoints,
                "river chainage [km]",
                erosion_results.total_erosion_dist,
                "Bank {ib}",
                erosion_results.eq_erosion_dist,
                "Bank {ib} (eq)",
                "eroded distance",
                "[m]",
            )
            if self.river_data.plot_flags["save_plot"]:
                fig_i = fig_i + 1
                fig_base = self.river_data.plot_flags["fig_dir"] + os.sep + str(fig_i) + "_erodis"
                if self.river_data.plot_flags["save_plot_zoomed"]:
                    df_plt.zoom_x_and_save(fig, ax, fig_base, self.river_data.plot_flags["plot_ext"], km_zoom)
                fig_file = fig_base + self.river_data.plot_flags["plot_ext"]
                df_plt.savefig(fig, fig_file)

            if self.river_data.plot_flags["close_plot"]:
                plt.close("all")
            else:
                plt.show(block=not self.gui)
