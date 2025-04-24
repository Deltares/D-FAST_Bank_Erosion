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

from typing import Tuple, List, Dict
import os
from geopandas.geodataframe import GeoDataFrame
from geopandas.geoseries import GeoSeries
from shapely.geometry import LineString, Point
import numpy as np
import matplotlib.pyplot as plt

from dfastbe import __version__
from dfastbe.kernel import get_km_bins, moving_avg, comp_erosion_eq, comp_erosion, get_km_eroded_volume, \
                            get_zoom_extends
from dfastbe.support import on_right_side, project_km_on_line, intersect_line_mesh, move_line
from dfastbe import plotting as df_plt
from dfastbe.io import (
    ConfigFile,
    log_text,
    write_shp_pnt,
    write_km_eroded_volumes,
    write_shp,
    write_csv,
)
from dfastbe.erosion.data_models import (
    ErosionRiverData,
    ErosionSimulationData,
    ErosionInputs,
    WaterLevelData,
    MeshData,
    BankData,
    FairwayData,
    ErosionResults,
)
from dfastbe.utils import timed_logger


RHO = 1000  # density of water [kg/m3]
g = 9.81  # gravitational acceleration [m/s2]

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

    @property
    def config_file(self) -> ConfigFile:
        """Configuration file object."""
        return self._config_file

    def get_ship_parameters(self, bank_km_mid) -> Dict[str, float]:

        ship_relative_velocity = self.config_file.get_parameter(
            "Erosion", "VShip", bank_km_mid, positive=True, onefile=True
        )
        num_ships_year = self.config_file.get_parameter(
            "Erosion", "NShip", bank_km_mid, positive=True, onefile=True
        )
        num_waves_p_ship = self.config_file.get_parameter(
            "Erosion", "NWave", bank_km_mid, default=5, positive=True, onefile=True
        )
        ship_draught = self.config_file.get_parameter(
            "Erosion", "Draught", bank_km_mid, positive=True, onefile=True
        )
        ship_type = self.config_file.get_parameter(
            "Erosion", "ShipType", bank_km_mid, valid=[1, 2, 3], onefile=True
        )
        parslope0 = self.config_file.get_parameter(
            "Erosion", "Slope", bank_km_mid, default=20, positive=True, ext="slp"
        )
        reed_wave_damping_coeff = self.config_file.get_parameter(
            "Erosion", "Reed", bank_km_mid, default=0, positive=True, ext="rdd"
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


    def intersect_bank_lines_with_mesh(
        self,
        stations_coords: np.ndarray,
        mesh_data: MeshData,
    ) -> BankData:
        n_bank_lines = len(self.river_data.bank_lines)

        bank_line_coords = []
        bank_face_indices = []
        for bank_index in range(n_bank_lines):
            line_coords = np.array(self.river_data.bank_lines.geometry[bank_index].coords)
            log_text("bank_nodes", data={"ib": bank_index + 1, "n": len(line_coords)})

            coords_along_bank, face_indices = intersect_line_mesh(
                line_coords, mesh_data
            )
            bank_line_coords.append(coords_along_bank)
            bank_face_indices.append(face_indices)

        # linking bank lines to chainage
        log_text("chainage_to_banks")
        bank_chainage_midpoints = [None] * n_bank_lines
        is_right_bank = [True] * n_bank_lines
        for bank_index, coords in enumerate(bank_line_coords):
            segment_mid_points = (coords[:-1, :] + coords[1:, :]) / 2
            chainage_mid_points = project_km_on_line(segment_mid_points,
                                                     self.river_center_line_arr)

            # check if the bank line is defined from low chainage to high chainage
            if chainage_mid_points[0] > chainage_mid_points[-1]:
                # if not, flip the bank line and all associated data
                chainage_mid_points = chainage_mid_points[::-1]
                bank_line_coords[bank_index] = bank_line_coords[bank_index][::-1, :]
                bank_face_indices[bank_index] = bank_face_indices[bank_index][::-1]

            bank_chainage_midpoints[bank_index] = chainage_mid_points

            # check if the bank line is left or right bank
            # when looking from low to high chainage
            is_right_bank[bank_index] = on_right_side(coords, stations_coords)
            if is_right_bank[bank_index]:
                log_text("right_side_bank", data={"ib": bank_index + 1})
            else:
                log_text("left_side_bank", data={"ib": bank_index + 1})

        return BankData(
            bank_line_coords=bank_line_coords,
            bank_face_indices=bank_face_indices,
            bank_chainage_midpoints=bank_chainage_midpoints,
            is_right_bank=is_right_bank,
            bank_lines=self.river_data.bank_lines,
            n_bank_lines=n_bank_lines,
        )

    def _prepare_river_axis(
        self, stations_coords: np.ndarray, config_file: ConfigFile
    ) -> Tuple[np.ndarray, np.ndarray, LineString]:
        # read river axis file
        river_axis = self.river_data.read_river_axis()
        river_axis_numpy = np.array(river_axis.coords)
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
        river_axis_km = project_km_on_line(river_axis_numpy, self.river_center_line_arr)
        write_shp_pnt(
            river_axis_numpy,
            {"chainage": river_axis_km},
            f"{str(self.river_data.output_dir)}{os.sep}river_axis_chainage.shp",
            config_file,
        )

        # clip river axis to reach of interest
        i1 = np.argmin(((stations_coords[0] - river_axis_numpy) ** 2).sum(axis=1))
        i2 = np.argmin(((stations_coords[-1] - river_axis_numpy) ** 2).sum(axis=1))
        if i1 < i2:
            river_axis_km = river_axis_km[i1 : i2 + 1]
            river_axis_numpy = river_axis_numpy[i1 : i2 + 1]
        else:
            # reverse river axis
            river_axis_km = river_axis_km[i2 : i1 + 1][::-1]
            river_axis_numpy = river_axis_numpy[i2 : i1 + 1][::-1]
        river_axis = LineString(river_axis_numpy)

        return river_axis_km, river_axis_numpy, river_axis

    def _prepare_fairway(
        self,
        river_axis: LineString,
        stations_coords: np.ndarray,
        mesh_data: MeshData,
        config_file: ConfigFile,
    ):
        # read fairway file
        fairway_file = self.config_file.get_str("Erosion", "Fairway")
        log_text("read_fairway", data={"file": fairway_file})

        # map km to fairway points, further using axis
        log_text("chainage_to_fairway")
        fairway_numpy = np.array(river_axis.coords)
        fairway_km = project_km_on_line(fairway_numpy, self.river_center_line_arr)
        write_shp_pnt(
            fairway_numpy,
            {"chainage": fairway_km},
            str(self.river_data.output_dir) + os.sep + "fairway_chainage.shp",
            config_file,
        )

        # clip fairway to reach of interest
        i1 = np.argmin(((stations_coords[0] - fairway_numpy) ** 2).sum(axis=1))
        i2 = np.argmin(((stations_coords[-1] - fairway_numpy) ** 2).sum(axis=1))
        if i1 < i2:
            fairway_numpy = fairway_numpy[i1 : i2 + 1]
        else:
            # reverse fairway
            fairway_numpy = fairway_numpy[i2 : i1 + 1][::-1]

        # intersect fairway and mesh
        log_text("intersect_fairway_mesh", data={"n": len(fairway_numpy)})
        fairway_intersection_coords, fairway_face_indices = intersect_line_mesh(
            fairway_numpy, mesh_data
        )
        if self.river_data.debug:
            write_shp_pnt(
                (fairway_intersection_coords[:-1] + fairway_intersection_coords[1:])
                / 2,
                {"iface": fairway_face_indices},
                f"{str(self.river_data.output_dir)}{os.sep}fairway_face_indices.shp",
                config_file,
            )

        return FairwayData(fairway_face_indices, fairway_intersection_coords)

    def _map_bank_to_fairway(
        self,
        bank_data: BankData,
        fairway_data: FairwayData,
        simulation_data: ErosionSimulationData,
        config_file: ConfigFile,
    ):
        # distance fairway-bankline (bankfairway)
        log_text("bank_distance_fairway")
        distance_fw = []
        bp_fw_face_idx = []
        nfw = len(fairway_data.fairway_face_indices)
        for ib, bcrds in enumerate(bank_data.bank_line_coords):
            bcrds_mid = (bcrds[:-1] + bcrds[1:]) / 2
            distance_fw.append(np.zeros(len(bcrds_mid)))
            bp_fw_face_idx.append(np.zeros(len(bcrds_mid), dtype=int))
            for ip, bp in enumerate(bcrds_mid):
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
                    if alpha > 0 and alpha < 1:
                        fwp1 = fairway_data.intersection_coords[ifw - 1] + alpha * (
                            fairway_data.intersection_coords[ifw]
                            - fairway_data.intersection_coords[ifw - 1]
                        )
                        d1 = ((bp - fwp1) ** 2).sum() ** 0.5
                        if d1 < dbfw:
                            dbfw = d1
                            # projected point located on segment before, which corresponds to initial choice: iseg = ifw - 1
                if ifw < nfw:
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
                    if alpha > 0 and alpha < 1:
                        fwp1 = fairway_data.intersection_coords[ifw] + alpha * (
                            fairway_data.intersection_coords[ifw + 1]
                            - fairway_data.intersection_coords[ifw]
                        )
                        d1 = ((bp - fwp1) ** 2).sum() ** 0.5
                        if d1 < dbfw:
                            dbfw = d1
                            iseg = ifw

                bp_fw_face_idx[ib][ip] = fairway_data.fairway_face_indices[iseg]
                distance_fw[ib][ip] = dbfw

            if self.river_data.debug:
                write_shp_pnt(
                    bcrds_mid,
                    {
                        "chainage": bank_data.bank_chainage_midpoints[ib],
                        "iface_fw": bp_fw_face_idx[ib],
                    },
                    f"{self.river_data.output_dir}/bank_{ib + 1}_chainage_and_fairway_face_idx.shp",
                    config_file,
                )

        bank_data.fairway_face_indices = bp_fw_face_idx
        bank_data.fairway_distances = distance_fw

        # water level at fairway
        zfw_ini = []
        for ib in range(bank_data.n_bank_lines):
            ii = bank_data.fairway_face_indices[ib]
            zfw_ini.append(simulation_data.water_level_face[ii])
        fairway_data.fairway_initial_water_levels = zfw_ini

    def _prepare_initial_conditions(
        self, config_file: ConfigFile, bank_data: BankData, fairway_data: FairwayData
    ) -> ErosionInputs:
        # wave reduction s0, s1
        wave_fairway_distance_0 = config_file.get_parameter(
            "Erosion",
            "Wave0",
            bank_data.bank_chainage_midpoints,
            default=200,
            positive=True,
            onefile=True,
        )
        wave_fairway_distance_1 = config_file.get_parameter(
            "Erosion",
            "Wave1",
            bank_data.bank_chainage_midpoints,
            default=150,
            positive=True,
            onefile=True,
        )

        # save 1_banklines
        # read vship, nship, nwave, draught (tship), shiptype ... independent of level number
        shipping_data = self.get_ship_parameters(bank_data.bank_chainage_midpoints)

        # read classes flag (yes: banktype = taucp, no: banktype = tauc) and banktype (taucp: 0-4 ... or ... tauc = critical shear value)
        classes = config_file.get_bool("Erosion", "Classes")
        if classes:
            bank_type = config_file.get_parameter(
                "Erosion",
                "BankType",
                bank_data.bank_chainage_midpoints,
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
                bank_data.bank_chainage_midpoints,
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

        # read bank protection level zss
        zss_miss = -1000
        zss = config_file.get_parameter(
            "Erosion",
            "ProtectionLevel",
            bank_data.bank_chainage_midpoints,
            default=zss_miss,
            ext=".bpl",
        )
        # if zss undefined, set zss equal to zfw_ini - 1
        for ib, one_zss in enumerate(zss):
            mask = one_zss == zss_miss
            one_zss[mask] = fairway_data.fairway_initial_water_levels[ib][mask] - 1

        return ErosionInputs(
            shipping_data=shipping_data,
            wave_fairway_distance_0=wave_fairway_distance_0,
            wave_fairway_distance_1=wave_fairway_distance_1,
            bank_protection_level=zss,
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
        # initialize arrays for erosion loop over all discharges
        velocity: List[List[np.ndarray]] = []
        bank_height: List[np.ndarray] = []
        water_level: List[List[np.ndarray]] = []
        chezy: List[List[np.ndarray]] = []
        vol_per_discharge: List[List[np.ndarray]] = []
        ship_wave_max: List[List[np.ndarray]] = []
        ship_wave_min: List[List[np.ndarray]] = []

        line_size: List[np.ndarray] = []
        flow_erosion_dist: List[np.ndarray] = []
        ship_erosion_dist: List[np.ndarray] = []
        total_erosion_dist: List[np.ndarray] = []
        total_eroded_vol: List[np.ndarray] = []
        eq_erosion_dist: List[np.ndarray] = []
        eq_eroded_vol: List[np.ndarray] = []

        erosion_time = config_file.get_int("Erosion", "TErosion", positive=True)
        log_text("total_time", data={"t": erosion_time})

        for iq in range(self.river_data.num_discharge_levels):
            log_text(
                "discharge_header",
                data={
                    "i": iq + 1,
                    "p": self.p_discharge[iq],
                    "t": self.p_discharge[iq] * erosion_time,
                },
            )

            iq_str = "{}".format(iq + 1)

            log_text("read_q_params", indent="  ")
            # read vship, nship, nwave, draught, shiptype, slope, reed, fairwaydepth, ... (level specific values)
            vship = config_file.get_parameter(
                "Erosion",
                f"VShip{iq_str}",
                bank_data.bank_chainage_midpoints,
                default=erosion_inputs.shipping_data["vship0"],
                positive=True,
                onefile=True,
            )
            Nship = config_file.get_parameter(
                "Erosion",
                f"NShip{iq_str}",
                bank_data.bank_chainage_midpoints,
                default=erosion_inputs.shipping_data["Nship0"],
                positive=True,
                onefile=True,
            )
            nwave = config_file.get_parameter(
                "Erosion",
                f"NWave{iq_str}",
                bank_data.bank_chainage_midpoints,
                default=erosion_inputs.shipping_data["nwave0"],
                positive=True,
                onefile=True,
            )
            Tship = config_file.get_parameter(
                "Erosion",
                f"Draught{iq_str}",
                bank_data.bank_chainage_midpoints,
                default=erosion_inputs.shipping_data["Tship0"],
                positive=True,
                onefile=True,
            )
            ship_type = config_file.get_parameter(
                "Erosion",
                f"ShipType{iq_str}",
                bank_data.bank_chainage_midpoints,
                default=erosion_inputs.shipping_data["ship0"],
                valid=[1, 2, 3],
                onefile=True,
            )

            parslope = config_file.get_parameter(
                "Erosion",
                f"Slope{iq_str}",
                bank_data.bank_chainage_midpoints,
                default=erosion_inputs.shipping_data["parslope0"],
                positive=True,
                ext="slp",
            )
            parreed = config_file.get_parameter(
                "Erosion",
                f"Reed{iq_str}",
                bank_data.bank_chainage_midpoints,
                default=erosion_inputs.shipping_data["parreed0"],
                positive=True,
                ext="rdd",
            )
            mu_slope = [None] * bank_data.n_bank_lines
            mu_reed = [None] * bank_data.n_bank_lines
            for ib in range(bank_data.n_bank_lines):
                mus = parslope[ib].copy()
                mus[mus > 0] = 1 / mus[mus > 0]
                mu_slope[ib] = mus
                mu_reed[ib] = 8.5e-4 * parreed[ib] ** 0.8

            log_text("-", indent="  ")
            log_text("read_simdata", data={"file": self.sim_files[iq]}, indent="  ")
            log_text("-", indent="  ")
            simulation_data = ErosionSimulationData.read(self.sim_files[iq], indent="  ")
            log_text("-", indent="  ")

            log_text("bank_erosion", indent="  ")
            velocity.append([])
            water_level.append([])
            chezy.append([])
            vol_per_discharge.append([])
            ship_wave_max.append([])
            ship_wave_min.append([])

            dvol_bank = np.zeros((len(km_mid), bank_data.n_bank_lines))
            hfw_max = 0
            for ib, bcrds in enumerate(bank_data.bank_line_coords):
                # determine velocity along banks ...
                dx = np.diff(bcrds[:, 0])
                dy = np.diff(bcrds[:, 1])
                if iq == 0:
                    line_size.append(np.sqrt(dx ** 2 + dy ** 2))

                bank_index = bank_data.bank_face_indices[ib]
                vel_bank = (
                    np.absolute(
                        simulation_data.velocity_x_face[bank_index] * dx
                        + simulation_data.velocity_y_face[bank_index] * dy
                    )
                    / line_size[ib]
                )
                if self.river_data.vel_dx > 0.0:
                    if ib == 0:
                        log_text(
                            "apply_velocity_filter", indent="  ", data={"dx": self.river_data.vel_dx}
                        )
                    vel_bank = moving_avg(
                        bank_data.bank_chainage_midpoints[ib], vel_bank, self.river_data.vel_dx
                    )
                velocity[iq].append(vel_bank)
                #
                if iq == 0:
                    # determine velocity and bankheight along banks ...
                    # bankheight = maximum bed elevation per cell
                    if simulation_data.bed_elevation_location == "node":
                        zb = simulation_data.bed_elevation_values
                        zb_all_nodes = ErosionSimulationData.apply_masked_indexing(
                            zb, simulation_data.face_node[bank_index, :]
                        )
                        zb_bank = zb_all_nodes.max(axis=1)
                        if self.river_data.zb_dx > 0.0:
                            if ib == 0:
                                log_text(
                                    "apply_banklevel_filter",
                                    indent="  ",
                                    data={"dx": self.river_data.zb_dx},
                                )
                            zb_bank = moving_avg(
                                bank_data.bank_chainage_midpoints[ib],
                                zb_bank,
                                self.river_data.zb_dx,
                            )
                        bank_height.append(zb_bank)
                    else:
                        # don't know ... need to check neighbouring cells ...
                        bank_height.append(None)

                # get water depth along fairway
                ii = bank_data.fairway_face_indices[ib]
                hfw = simulation_data.water_depth_face[ii]
                hfw_max = max(hfw_max, hfw.max())
                water_level[iq].append(simulation_data.water_level_face[ii])
                chez = simulation_data.chezy_face[ii]
                chezy[iq].append(0 * chez + chez.mean())

                if iq == self.river_data.num_discharge_levels - 1:  # ref_level:
                    dn_eq1, dv_eq1 = comp_erosion_eq(
                        bank_height[ib],
                        line_size[ib],
                        fairway_data.fairway_initial_water_levels[ib],
                        vship[ib],
                        ship_type[ib],
                        Tship[ib],
                        mu_slope[ib],
                        bank_data.fairway_distances[ib],
                        hfw,
                        erosion_inputs,
                        ib,
                        g,
                    )
                    eq_erosion_dist.append(dn_eq1)
                    eq_eroded_vol.append(dv_eq1)

                    if self.river_data.debug:
                        bcrds_mid = (bcrds[:-1] + bcrds[1:]) / 2
                        bank_coords_points = [Point(xy1) for xy1 in bcrds_mid]
                        bank_coords_geo = GeoSeries(
                            bank_coords_points, crs=config_file.crs
                        )
                        params = {
                            "chainage": bank_data.bank_chainage_midpoints[ib],
                            "x": bcrds_mid[:, 0],
                            "y": bcrds_mid[:, 1],
                            "iface_fw": bank_data.fairway_face_indices[ib],
                            "iface_bank": bank_data.bank_face_indices[ib],  # bank_index
                            "zb": bank_height[ib],
                            "len": line_size[ib],
                            "zw0": fairway_data.fairway_initial_water_levels[ib],
                            "vship": vship[ib],
                            "shiptype": ship_type[ib],
                            "draught": Tship[ib],
                            "mu_slp": mu_slope[ib],
                            "dist_fw": bank_data.fairway_distances[ib],
                            "dfw0": erosion_inputs.wave_fairway_distance_0[ib],
                            "dfw1": erosion_inputs.wave_fairway_distance_1[ib],
                            "hfw": hfw,
                            "zss": erosion_inputs.bank_protection_level[ib],
                            "dn": dn_eq1,
                            "dv": dv_eq1,
                        }

                        write_shp(bank_coords_geo, params, f"{str(self.river_data.output_dir)}{os.sep}debug.EQ.B{ib + 1}.shp")
                        write_csv(params, f"{str(self.river_data.output_dir)}{os.sep}debug.EQ.B{ib + 1}.csv")

                dniqib, dviqib, dn_ship, dn_flow, ship_wave_max_ib, ship_wave_min_ib = (
                    comp_erosion(
                        velocity[iq][ib],
                        bank_height[ib],
                        line_size[ib],
                        water_level[iq][ib],
                        fairway_data.fairway_initial_water_levels[ib],
                        Nship[ib],
                        vship[ib],
                        nwave[ib],
                        ship_type[ib],
                        Tship[ib],
                        erosion_time * self.p_discharge[iq],
                        mu_slope[ib],
                        mu_reed[ib],
                        bank_data.fairway_distances[ib],
                        hfw,
                        chezy[iq][ib],
                        erosion_inputs,
                        RHO,
                        g,
                        ib,
                    )
                )
                ship_wave_max[iq].append(ship_wave_max_ib)
                ship_wave_min[iq].append(ship_wave_min_ib)

                if self.river_data.debug:
                    bcrds_mid = (bcrds[:-1] + bcrds[1:]) / 2

                    bank_coords_points = [Point(xy1) for xy1 in bcrds_mid]
                    bank_coords_geo = GeoSeries(bank_coords_points, crs=config_file.crs)
                    params = {
                        "chainage": bank_data.bank_chainage_midpoints[ib],
                        "x": bcrds_mid[:, 0],
                        "y": bcrds_mid[:, 1],
                        "iface_fw": bank_data.fairway_face_indices[ib],
                        "iface_bank": bank_data.bank_face_indices[ib],  # bank_index
                        "u": velocity[iq][ib],
                        "zb": bank_height[ib],
                        "len": line_size[ib],
                        "zw": water_level[iq][ib],
                        "zw0": fairway_data.fairway_initial_water_levels[ib],
                        "tauc": erosion_inputs.tauc[ib],
                        "nship": Nship[ib],
                        "vship": vship[ib],
                        "nwave": nwave[ib],
                        "shiptype": ship_type[ib],
                        "draught": Tship[ib],
                        "mu_slp": mu_slope[ib],
                        "mu_reed": mu_reed[ib],
                        "dist_fw": bank_data.fairway_distances[ib],
                        "dfw0": erosion_inputs.wave_fairway_distance_0[ib],
                        "dfw1": erosion_inputs.wave_fairway_distance_1[ib],
                        "hfw": hfw,
                        "chez": chezy[iq][ib],
                        "zss": erosion_inputs.bank_protection_level[ib],
                        "dn": dniqib,
                        "dv": dviqib,
                        "dnship": dn_ship,
                        "dnflow": dn_flow,
                    }
                    write_shp(bank_coords_geo, params, f"{str(self.river_data.output_dir)}{os.sep}debug.Q{iq + 1}.B{ib + 1}.shp")
                    write_csv(params, f"{str(self.river_data.output_dir)}{os.sep}debug.Q{iq + 1}.B{ib + 1}.csv")

                # shift bank lines
                if len(total_erosion_dist) == ib:
                    flow_erosion_dist.append(dn_flow.copy())
                    ship_erosion_dist.append(dn_ship.copy())
                    total_erosion_dist.append(dniqib.copy())
                    total_eroded_vol.append(dviqib.copy())
                else:
                    flow_erosion_dist[ib] += dn_flow
                    ship_erosion_dist[ib] += dn_ship
                    total_erosion_dist[ib] += dniqib
                    total_eroded_vol[ib] += dviqib

                # accumulate eroded volumes per km
                dvol = get_km_eroded_volume(
                    bank_data.bank_chainage_midpoints[ib], dviqib, km_bin
                )
                vol_per_discharge[iq].append(dvol)
                dvol_bank[:, ib] += dvol

            erovol_file = config_file.get_str("Erosion", f"EroVol{iq_str}", default=f"erovolQ{iq_str}.evo")
            log_text("save_erovol", data={"file": erovol_file}, indent="  ")
            write_km_eroded_volumes(
                km_mid, dvol_bank, str(self.river_data.output_dir) + os.sep + erovol_file
            )

        erosion_results = ErosionResults(
            eq_erosion_dist=eq_erosion_dist,
            total_erosion_dist=total_erosion_dist,
            flow_erosion_dist=flow_erosion_dist,
            ship_erosion_dist=ship_erosion_dist,
            vol_per_discharge=vol_per_discharge,
            eq_eroded_vol=eq_eroded_vol,
            total_eroded_vol=total_eroded_vol,
            erosion_time=erosion_time,
        )

        water_level_data = WaterLevelData(
            hfw_max=hfw_max,
            water_level=water_level,
            ship_wave_max=ship_wave_max,
            ship_wave_min=ship_wave_min,
            velocity=velocity,
            bank_height=bank_height,
            chezy=chezy,
        )
        bank_data.bank_line_size = line_size

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
        river_center_line_coords = self.river_center_line_arr[:, :2]

        # map bank lines to mesh cells
        log_text("intersect_bank_mesh")

        bank_data = self.intersect_bank_lines_with_mesh(
            config_file, river_center_line_coords, mesh_data
        )

        river_axis_km, _, river_axis = self._prepare_river_axis(
            river_center_line_coords, config_file
        )
        # map to output interval
        km_bin = (river_axis_km.min(), river_axis_km.max(), self.river_data.output_intervals)
        km_mid = get_km_bins(km_bin, type=3)  # get mid-points

        fairway_data = self._prepare_fairway(river_axis, river_center_line_coords, mesh_data, config_file)

        self._map_bank_to_fairway(bank_data, fairway_data, self.simulation_data, config_file)

        erosion_inputs = self._prepare_initial_conditions(
            config_file, bank_data, fairway_data
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
            river_axis_km,
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
                RHO,
                g,
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

