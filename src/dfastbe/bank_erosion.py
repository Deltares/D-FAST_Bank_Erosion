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
from pathlib import Path

import os
from geopandas.geodataframe import GeoDataFrame
from geopandas.geoseries import GeoSeries
from shapely.geometry import LineString, Point
import numpy as np
import matplotlib.pyplot as plt

from dfastbe import __version__
from dfastbe.kernel import get_km_bins, moving_avg, comp_erosion_eq, comp_erosion, get_km_eroded_volume, \
                            get_zoom_extends, get_bbox
from dfastbe.support import on_right_side, project_km_on_line, intersect_line_mesh, move_line
from dfastbe import plotting as df_plt
from dfastbe.io import ConfigFile, log_text, read_simulation_data, \
    write_shp_pnt, write_km_eroded_volumes, write_shp, write_csv, RiverData, SimulationObject
from dfastbe.structures import (
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
        self.bank_dir = self._get_bank_line_dir()
        self.output_dir = config_file.get_output_dir("erosion")
        # check if additional debug output is requested
        self.debug = config_file.get_bool("General", "DebugOutput", False)
        # set plotting flags
        self.plot_flags = config_file.get_plotting_flags(self.root_dir)
        self.river_data = RiverData(config_file)

        # get filter settings for bank levels and flow velocities along banks
        self.zb_dx = config_file.get_float("Erosion", "BedFilterDist", 0.0, positive=True)
        self.vel_dx = config_file.get_float("Erosion", "VelFilterDist", 0.0, positive=True)
        log_text("get_levels")
        self.num_levels = config_file.get_int("Erosion", "NLevel")
        self.ref_level = config_file.get_int("Erosion", "RefLevel") - 1
        self.sim_files, self.p_discharge = self.get_sim_data()

    @property
    def config_file(self) -> ConfigFile:
        """Configuration file object."""
        return self._config_file

    def _get_bank_line_dir(self) -> Path:
        bank_dir = self.config_file.get_str("General", "BankDir")
        log_text("bankdir_in", data={"dir": bank_dir})
        bank_dir = Path(bank_dir)
        if not bank_dir.exists():
            log_text("missing_dir", data={"dir": bank_dir})
            raise BankLinesResultsError(
                f"Required bank line directory:{bank_dir} does not exist. please use the banklines command to run the "
                "bankline detection tool first it."
            )
        else:
            return bank_dir

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

    def get_sim_data(self):
        # get pdischarges
        sim_files = []
        p_discharge = []
        for iq in range(self.num_levels):
            iq_str = str(iq + 1)
            sim_files.append(self.config_file.get_sim_file("Erosion", iq_str))
            p_discharge.append(
                self.config_file.get_float("Erosion", f"PDischarge{iq_str}")
            )
        return sim_files, p_discharge

    def intersect_bank_lines_with_mesh(
        self,
        config_file: ConfigFile,
        stations_coords: np.ndarray,
        mesh_data: MeshData,
    ) -> BankData:
        bank_lines = config_file.get_bank_lines(str(self.bank_dir))
        n_bank_lines = len(bank_lines)

        bank_line_coords = []
        bank_face_indices = []
        for bank_index in range(n_bank_lines):
            line_coords = np.array(bank_lines.geometry[bank_index])
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
            chainage_mid_points = project_km_on_line(segment_mid_points, self.river_data.masked_profile_arr)

            # check if bank line is defined from low chainage to high chainage
            if chainage_mid_points[0] > chainage_mid_points[-1]:
                # if not, flip the bank line and all associated data
                chainage_mid_points = chainage_mid_points[::-1]
                bank_line_coords[bank_index] = bank_line_coords[bank_index][::-1, :]
                bank_face_indices[bank_index] = bank_face_indices[bank_index][::-1]

            bank_chainage_midpoints[bank_index] = chainage_mid_points

            # check if bank line is left or right bank
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
            bank_lines=bank_lines,
            n_bank_lines=n_bank_lines,
        )

    def _prepare_river_axis(self, stations_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, LineString]:
        # read river axis file
        river_axis = self.river_data.read_river_axis()
        river_axis_numpy = np.array(river_axis)
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
        river_axis_km = project_km_on_line(river_axis_numpy, self.river_data.masked_profile_arr)
        write_shp_pnt(
            river_axis_numpy, {"chainage": river_axis_km}, f"{str(self.output_dir)}{os.sep}river_axis_chainage.shp"
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
    ):
        # read fairway file
        fairway_file = self.config_file.get_str("Erosion", "Fairway")
        log_text("read_fairway", data={"file": fairway_file})

        # map km to fairway points, further using axis
        log_text("chainage_to_fairway")
        fairway_numpy = np.array(river_axis.coords)
        fairway_km = project_km_on_line(fairway_numpy, self.river_data.masked_profile_arr)
        write_shp_pnt(fairway_numpy, {"chainage": fairway_km}, str(self.output_dir) + os.sep + "fairway_chainage.shp")

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
        if self.debug:
            write_shp_pnt(
                (fairway_intersection_coords[:-1] + fairway_intersection_coords[1:])
                / 2,
                {"iface": fairway_face_indices},
                f"{str(self.output_dir)}{os.sep}fairway_face_indices.shp",
            )

        return FairwayData(fairway_face_indices, fairway_intersection_coords)

    def _map_bank_to_fairway(
        self, bank_data: BankData, fairway_data: FairwayData, sim: SimulationObject
    ):
        # distance fairway-bankline (bankfairway)
        log_text("bank_distance_fairway")
        distance_fw = []
        bp_fw_face_idx = []
        nfw = len(fairway_data.fairway_face_indices)
        for ib, bcrds in enumerate(bank_data.bank_line_coords):
            bcrds_mid = (bcrds[:-1] + bcrds[1:]) / 2
            distance_fw.append(np.zeros(len(bcrds_mid)))
            bp_fw_face_idx.append(np.zeros(len(bcrds_mid), dtype=np.int64))
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

            if self.debug:
                write_shp_pnt(
                    bcrds_mid,
                    {
                        "chainage": bank_data.bank_chainage_midpoints[ib],
                        "iface_fw": bp_fw_face_idx[ib],
                    },
                    str(self.output_dir)
                    + f"/bank_{ib + 1}_chainage_and_fairway_face_idx.shp",
                )

        bank_data.fairway_face_indices = bp_fw_face_idx
        bank_data.fairway_distances = distance_fw

        # water level at fairway
        zfw_ini = []
        for ib in range(bank_data.n_bank_lines):
            ii = bank_data.fairway_face_indices[ib]
            zfw_ini.append(sim["zw_face"][ii])
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
    ) -> Tuple[
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[List[np.ndarray]],
        List[np.ndarray],
        List[np.ndarray],
        WaterLevelData,
    ]:
        # initialize arrays for erosion loop over all discharges
        velocity: List[List[np.ndarray]] = []
        bank_height: List[np.ndarray] = []
        water_level: List[List[np.ndarray]] = []
        chezy: List[List[np.ndarray]] = []
        dv: List[List[np.ndarray]] = []
        ship_wave_max: List[List[np.ndarray]] = []
        ship_wave_min: List[List[np.ndarray]] = []

        line_size: List[np.ndarray] = []
        dn_flow_tot: List[np.ndarray] = []
        dn_ship_tot: List[np.ndarray] = []
        dn_tot: List[np.ndarray] = []
        dv_tot: List[np.ndarray] = []
        dn_eq: List[np.ndarray] = []
        dv_eq: List[np.ndarray] = []

        t_erosion = config_file.get_int("Erosion", "TErosion", positive=True)
        log_text("total_time", data={"t": t_erosion})

        for iq in range(self.num_levels):
            log_text(
                "discharge_header",
                data={"i": iq + 1, "p": self.p_discharge[iq], "t": self.p_discharge[iq] * t_erosion},
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
            sim, _ = read_simulation_data(self.sim_files[iq], indent="  ")
            log_text("-", indent="  ")
            fnc = sim["facenode"]

            log_text("bank_erosion", indent="  ")
            velocity.append([])
            water_level.append([])
            chezy.append([])
            dv.append([])
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
                            sim["ucx_face"][bank_index] * dx + sim["ucy_face"][bank_index] * dy
                        )
                        / line_size[ib]
                )
                if self.vel_dx > 0.0:
                    if ib == 0:
                        log_text(
                            "apply_velocity_filter", indent="  ", data={"dx": self.vel_dx}
                        )
                    vel_bank = moving_avg(
                        bank_data.bank_chainage_midpoints[ib], vel_bank, self.vel_dx
                    )
                velocity[iq].append(vel_bank)
                #
                if iq == 0:
                    # determine velocity and bankheight along banks ...
                    # bankheight = maximum bed elevation per cell
                    if sim["zb_location"] == "node":
                        zb = sim["zb_val"]
                        zb_all_nodes = _apply_masked_indexing(zb, fnc[bank_index, :])
                        zb_bank = zb_all_nodes.max(axis=1)
                        if self.zb_dx > 0.0:
                            if ib == 0:
                                log_text(
                                    "apply_banklevel_filter",
                                    indent="  ",
                                    data={"dx": self.zb_dx},
                                )
                            zb_bank = moving_avg(
                                bank_data.bank_chainage_midpoints[ib],
                                zb_bank,
                                self.zb_dx,
                            )
                        bank_height.append(zb_bank)
                    else:
                        # don't know ... need to check neighbouring cells ...
                        bank_height.append(None)

                # get water depth along fairway
                ii = bank_data.fairway_face_indices[ib]
                hfw = sim["h_face"][ii]
                hfw_max = max(hfw_max, hfw.max())
                water_level[iq].append(sim["zw_face"][ii])
                chez = sim["chz_face"][ii]
                chezy[iq].append(0 * chez + chez.mean())

                if iq == self.num_levels - 1:  # ref_level:
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
                    dn_eq.append(dn_eq1)
                    dv_eq.append(dv_eq1)

                    if self.debug:
                        bcrds_mid = (bcrds[:-1] + bcrds[1:]) / 2
                        bank_coords_points = [Point(xy1) for xy1 in bcrds_mid]
                        bank_coords_geo = GeoSeries(bank_coords_points)
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

                        write_shp(bank_coords_geo, params, f"{str(self.output_dir)}{os.sep}debug.EQ.B{ib + 1}.shp")
                        write_csv(params, f"{str(self.output_dir)}{os.sep}debug.EQ.B{ib + 1}.csv")

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
                        t_erosion * self.p_discharge[iq],
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

                if self.debug:
                    bcrds_mid = (bcrds[:-1] + bcrds[1:]) / 2

                    bank_coords_points = [Point(xy1) for xy1 in bcrds_mid]
                    bank_coords_geo = GeoSeries(bank_coords_points)
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
                    write_shp(bank_coords_geo, params, f"{str(self.output_dir)}{os.sep}debug.Q{iq + 1}.B{ib + 1}.shp")
                    write_csv(params, f"{str(self.output_dir)}{os.sep}debug.Q{iq + 1}.B{ib + 1}.csv")

                # shift bank lines
                if len(dn_tot) == ib:
                    dn_flow_tot.append(dn_flow.copy())
                    dn_ship_tot.append(dn_ship.copy())
                    dn_tot.append(dniqib.copy())
                    dv_tot.append(dviqib.copy())
                else:
                    dn_flow_tot[ib] += dn_flow
                    dn_ship_tot[ib] += dn_ship
                    dn_tot[ib] += dniqib
                    dv_tot[ib] += dviqib

                # accumulate eroded volumes per km
                dvol = get_km_eroded_volume(
                    bank_data.bank_chainage_midpoints[ib], dviqib, km_bin
                )
                dv[iq].append(dvol)
                dvol_bank[:, ib] += dvol

            erovol_file = config_file.get_str("Erosion", f"EroVol{iq_str}", default=f"erovolQ{iq_str}.evo")
            log_text("save_erovol", data={"file": erovol_file}, indent="  ")
            write_km_eroded_volumes(
                km_mid, dvol_bank, str(self.output_dir) + os.sep + erovol_file
            )

        erosion_results = ErosionResults(
            dn_eq=dn_eq,
            dn_tot=dn_tot,
            dn_flow_tot=dn_flow_tot,
            dn_ship_tot=dn_ship_tot,
            dv=dv,
            dv_eq=dv_eq,
            dv_tot=dv_tot,
            t_erosion=t_erosion,
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

        return (
            erosion_results,
            water_level_data,
        )

    def _postprocess_erosion_results(
        self,
        km_bin: Tuple[float, float, float],
        km_mid,
        bank_data: BankData,
        erosion_results: ErosionResults,
    ):
        log_text("=")
        d_nav = np.zeros(bank_data.n_bank_lines)
        dn_max = np.zeros(bank_data.n_bank_lines)
        d_nav_flow = np.zeros(bank_data.n_bank_lines)
        d_nav_ship = np.zeros(bank_data.n_bank_lines)
        d_nav_eq = np.zeros(bank_data.n_bank_lines)
        dn_max_eq = np.zeros(bank_data.n_bank_lines)
        vol_eq = np.zeros((len(km_mid), bank_data.n_bank_lines))
        vol_tot = np.zeros((len(km_mid), bank_data.n_bank_lines))
        xy_line_new_list = []
        bankline_new_list = []
        xy_line_eq_list = []
        bankline_eq_list = []
        for ib, bank_coords in enumerate(bank_data.bank_line_coords):
            d_nav[ib] = (
                erosion_results.dn_tot[ib] * bank_data.bank_line_size[ib]
            ).sum() / bank_data.bank_line_size[ib].sum()
            dn_max[ib] = erosion_results.dn_tot[ib].max()
            d_nav_flow[ib] = (
                erosion_results.dn_flow_tot[ib] * bank_data.bank_line_size[ib]
            ).sum() / bank_data.bank_line_size[ib].sum()
            d_nav_ship[ib] = (
                erosion_results.dn_ship_tot[ib] * bank_data.bank_line_size[ib]
            ).sum() / bank_data.bank_line_size[ib].sum()
            d_nav_eq[ib] = (
                erosion_results.dn_eq[ib] * bank_data.bank_line_size[ib]
            ).sum() / bank_data.bank_line_size[ib].sum()
            dn_max_eq[ib] = erosion_results.dn_eq[ib].max()
            log_text("bank_dnav", data={"ib": ib + 1, "v": d_nav[ib]})
            log_text("bank_dnavflow", data={"v": d_nav_flow[ib]})
            log_text("bank_dnavship", data={"v": d_nav_ship[ib]})
            log_text("bank_dnmax", data={"v": dn_max[ib]})
            log_text("bank_dnaveq", data={"v": d_nav_eq[ib]})
            log_text("bank_dnmaxeq", data={"v": dn_max_eq[ib]})

            xy_line_new = move_line(
                bank_coords, erosion_results.dn_tot[ib], bank_data.is_right_bank[ib]
            )
            xy_line_new_list.append(xy_line_new)
            bankline_new_list.append(LineString(xy_line_new))

            xy_line_eq = move_line(
                bank_coords, erosion_results.dn_eq[ib], bank_data.is_right_bank[ib]
            )
            xy_line_eq_list.append(xy_line_eq)
            bankline_eq_list.append(LineString(xy_line_eq))

            dvol_eq = get_km_eroded_volume(
                bank_data.bank_chainage_midpoints[ib], erosion_results.dv_eq[ib], km_bin
            )
            vol_eq[:, ib] = dvol_eq
            dvol_tot = get_km_eroded_volume(
                bank_data.bank_chainage_midpoints[ib],
                erosion_results.dv_tot[ib],
                km_bin,
            )
            vol_tot[:, ib] = dvol_tot
            if ib < bank_data.n_bank_lines - 1:
                log_text("-")

        erosion_results.d_nav = d_nav
        erosion_results.vol_eq = vol_eq
        erosion_results.vol_tot = vol_tot

        return bankline_new_list, bankline_eq_list, xy_line_eq_list

    def bankerosion_core(self) -> None:
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

        # read simulation data (get_sim_data)
        sim_file = config_file.get_sim_file("Erosion", str(self.ref_level + 1))
        log_text("-")
        log_text("read_simdata", data={"file": sim_file})
        log_text("-")
        sim, _ = read_simulation_data(sim_file)
        log_text("-")

        log_text("derive_topology")

        mesh_data = _compute_mesh_topology(sim)

        # clip the chainage path to the range of chainages of interest
        km_bounds = self.river_data.station_bounds
        log_text("clip_chainage", data={"low": km_bounds[0], "high": km_bounds[1]})

        stations_coords = self.river_data.masked_profile_arr[:, :2]

        # map bank lines to mesh cells
        log_text("intersect_bank_mesh")

        bank_data = self.intersect_bank_lines_with_mesh(
            config_file, stations_coords, mesh_data
        )

        river_axis_km, _, river_axis = self._prepare_river_axis(stations_coords)

        # get output interval
        km_step = config_file.get_float("Erosion", "OutputInterval", 1.0)
        # map to output interval
        km_bin = (river_axis_km.min(), river_axis_km.max(), km_step)
        km_mid = get_km_bins(km_bin, type=3)  # get mid-points

        fairway_data = self._prepare_fairway(river_axis, stations_coords, mesh_data)

        self._map_bank_to_fairway(bank_data, fairway_data, sim)

        erosion_inputs = self._prepare_initial_conditions(
            config_file, bank_data, fairway_data
        )

        # initialize arrays for erosion loop over all discharges
        (water_level_data, erosion_results) = self._process_discharge_levels(
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

        self._write_bankline_shapefiles(bankline_new_list, bankline_eq_list)
        self._write_volume_outputs(vol_tot, vol_eq, km_mid)

        # create various plots
        self._generate_plots(
            river_axis_km,
            sim,
            dn_tot,
            xy_line_eq_list,
            d_nav,
            t_erosion,
            km_mid,
            km_step,
            dv,
            vol_eq,
            dn_eq,
            erosion_inputs,
            water_level_data,
            mesh_data,
            bank_data,
        )
        log_text("end_bankerosion")
        timed_logger("-- end analysis --")

    def _write_bankline_shapefiles(self, bankline_new_list, bankline_eq_list):
        bankline_new_series = GeoSeries(bankline_new_list)
        bank_lines_new = GeoDataFrame.from_features(bankline_new_series)
        bank_name = self.config_file.get_str("General", "BankFile", "bankfile")

        bank_file = self.output_dir / f"{bank_name}_new.shp"
        log_text("save_banklines", data={"file": str(bank_file)})
        bank_lines_new.to_file(bank_file)

        bankline_eq_series = GeoSeries(bankline_eq_list)
        banklines_eq = GeoDataFrame.from_features(bankline_eq_series)

        bank_file = self.output_dir / f"{bank_name}_eq.shp"
        log_text("save_banklines", data={"file": str(bank_file)})
        banklines_eq.to_file(bank_file)

    def _write_volume_outputs(self, vol_tot, vol_eq, km_mid):
        erosion_vol_file = self.config_file.get_str("Erosion", "EroVol", default="erovol.evo")
        log_text("save_tot_erovol", data={"file": erosion_vol_file})
        write_km_eroded_volumes(km_mid, vol_tot, str(self.output_dir) + os.sep + erosion_vol_file)

        # write eroded volumes per km (equilibrium)
        erosion_vol_file = self.config_file.get_str("Erosion", "EroVolEqui", default="erovol_eq.evo")
        log_text("save_eq_erovol", data={"file": erosion_vol_file})
        write_km_eroded_volumes(km_mid, vol_eq, str(self.output_dir) + os.sep + erosion_vol_file)

    def _generate_plots(
        self,
        river_axis_km,
        sim,
        dn_tot,
        xy_line_eq_list,
        d_nav,
        t_erosion,
        km_mid,
        km_step,
        dv,
        vol_eq,
        dn_eq,
        erosion_inputs: ErosionInputs,
        water_level_data: WaterLevelData,
        mesh_data: MeshData,
        bank_data: BankData,
    ):
        # create various plots
        if self.plot_flags["plot_data"]:
            log_text("=")
            log_text("create_figures")
            fig_i = 0
            bbox = get_bbox(self.river_data.masked_profile_arr)

            if self.plot_flags["save_plot_zoomed"]:
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
                    self.plot_flags["zoom_km_step"],
                    bank_coords_mid,
                    bank_data.bank_chainage_midpoints,
                )

            fig, ax = df_plt.plot1_waterdepth_and_banklines(
                bbox,
                self.river_data.masked_profile_arr,
                bank_data.bank_lines,
                mesh_data.face_node,
                sim["nnodes"],
                sim["x_node"],
                sim["y_node"],
                sim["h_face"],
                1.1 * water_level_data.hfw_max,
                X_AXIS_TITLE,
                Y_AXIS_TITLE,
                "water depth and initial bank lines",
                "water depth [m]",
            )
            if self.plot_flags["save_plot"]:
                fig_i = fig_i + 1
                fig_base = f"{self.plot_flags['fig_dir']}{os.sep}{fig_i}_banklines"

                if self.plot_flags["save_plot_zoomed"]:
                    df_plt.zoom_xy_and_save(fig, ax, fig_base, self.plot_flags["plot_ext"], xy_zoom)

                fig_path = fig_base + self.plot_flags["plot_ext"]
                df_plt.savefig(fig, fig_path)

            fig, ax = df_plt.plot2_eroded_distance_and_equilibrium(
                bbox,
                self.river_data.masked_profile_arr,
                bank_data.bank_line_coords,
                dn_tot,
                bank_data.is_right_bank,
                d_nav,
                xy_line_eq_list,
                mesh_data.x_edge_coords,
                mesh_data.y_edge_coords,
                X_AXIS_TITLE,
                Y_AXIS_TITLE,
                "eroded distance and equilibrium bank location",
                f"eroded during {t_erosion} year",
                "eroded distance [m]",
                "equilibrium location",
            )
            if self.plot_flags["save_plot"]:
                fig_i = fig_i + 1
                fig_base = f"{self.plot_flags['fig_dir']}{os.sep}{fig_i}_erosion_sensitivity"

                if self.plot_flags["save_plot_zoomed"]:
                    df_plt.zoom_xy_and_save(fig, ax, fig_base, self.plot_flags["plot_ext"], xy_zoom)

                fig_path = fig_base + self.plot_flags["plot_ext"]
                df_plt.savefig(fig, fig_path)

            fig, ax = df_plt.plot3_eroded_volume(
                km_mid,
                km_step,
                "river chainage [km]",
                dv,
                "eroded volume [m^3]",
                f"eroded volume per {km_step} chainage km ({t_erosion} years)",
                "Q{iq}",
                "Bank {ib}",
            )
            if self.plot_flags["save_plot"]:
                fig_i = fig_i + 1
                fig_base = f"{self.plot_flags['fig_dir']}{os.sep}{fig_i}_eroded_volume"

                if self.plot_flags["save_plot_zoomed"]:
                    df_plt.zoom_x_and_save(fig, ax, fig_base, self.plot_flags["plot_ext"], km_zoom)

                fig_path = fig_base + self.plot_flags["plot_ext"]
                df_plt.savefig(fig, fig_path)

            fig, ax = df_plt.plot3_eroded_volume_subdivided_1(
                km_mid,
                km_step,
                "river chainage [km]",
                dv,
                "eroded volume [m^3]",
                f"eroded volume per {km_step} chainage km ({t_erosion} years)",
                "Q{iq}",
            )
            if self.plot_flags["save_plot"]:
                fig_i = fig_i + 1
                fig_base = self.plot_flags["fig_dir"] + os.sep + str(fig_i) + "_eroded_volume_per_discharge"
                if self.plot_flags["save_plot_zoomed"]:
                    df_plt.zoom_x_and_save(fig, ax, fig_base, self.plot_flags["plot_ext"], km_zoom)
                fig_path = fig_base + self.plot_flags["plot_ext"]
                df_plt.savefig(fig, fig_path)

            fig, ax = df_plt.plot3_eroded_volume_subdivided_2(
                km_mid,
                km_step,
                "river chainage [km]",
                dv,
                "eroded volume [m^3]",
                f"eroded volume per {km_step} chainage km ({t_erosion} years)",
                "Bank {ib}",
            )
            if self.plot_flags["save_plot"]:
                fig_i = fig_i + 1
                fig_base = self.plot_flags["fig_dir"] + os.sep + str(fig_i) + "_eroded_volume_per_bank"
                if self.plot_flags["save_plot_zoomed"]:
                    df_plt.zoom_x_and_save(fig, ax, fig_base, self.plot_flags["plot_ext"], km_zoom)
                fig_path = fig_base + self.plot_flags["plot_ext"]
                df_plt.savefig(fig, fig_path)

            fig, ax = df_plt.plot4_eroded_volume_eq(
                km_mid,
                km_step,
                "river chainage [km]",
                vol_eq,
                "eroded volume [m^3]",
                f"eroded volume per {km_step} chainage km (equilibrium)",
            )
            if self.plot_flags["save_plot"]:
                fig_i = fig_i + 1
                fig_base = self.plot_flags["fig_dir"] + os.sep + str(fig_i) + "_eroded_volume_eq"
                if self.plot_flags["save_plot_zoomed"]:
                    df_plt.zoom_x_and_save(fig, ax, fig_base, self.plot_flags["plot_ext"], km_zoom)
                fig_path = fig_base + self.plot_flags["plot_ext"]
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
            if self.plot_flags["save_plot"]:
                for ib, fig in enumerate(figlist):
                    fig_i = fig_i + 1
                    fig_base = f"{self.plot_flags['fig_dir']}/{fig_i}_levels_bank_{ib + 1}"

                    if self.plot_flags["save_plot_zoomed"]:
                        df_plt.zoom_x_and_save(fig, axlist[ib], fig_base, self.plot_flags["plot_ext"], km_zoom)
                    fig_file = f"{fig_base}{self.plot_flags['plot_ext']}"
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
            if self.plot_flags["save_plot"]:
                for ib, fig in enumerate(figlist):
                    fig_i = fig_i + 1
                    fig_base = f"{self.plot_flags['fig_dir']}{os.sep}{fig_i}_velocity_bank_{ib + 1}"

                    if self.plot_flags["save_plot_zoomed"]:
                        df_plt.zoom_x_and_save(fig, axlist[ib], fig_base, self.plot_flags["plot_ext"], km_zoom)

                    fig_file = fig_base + self.plot_flags["plot_ext"]
                    df_plt.savefig(fig, fig_file)

            fig, ax = df_plt.plot7_banktype(
                bbox,
                self.river_data.masked_profile_arr,
                bank_data.bank_line_coords,
                erosion_inputs.bank_type,
                erosion_inputs.taucls_str,
                X_AXIS_TITLE,
                Y_AXIS_TITLE,
                "bank type",
            )
            if self.plot_flags["save_plot"]:
                fig_i = fig_i + 1
                fig_base = self.plot_flags["fig_dir"] + os.sep + str(fig_i) + "_banktype"
                if self.plot_flags["save_plot_zoomed"]:
                    df_plt.zoom_xy_and_save(fig, ax, fig_base, self.plot_flags["plot_ext"], xy_zoom)
                fig_file = fig_base + self.plot_flags["plot_ext"]
                df_plt.savefig(fig, fig_file)

            fig, ax = df_plt.plot8_eroded_distance(
                bank_data.bank_chainage_midpoints,
                "river chainage [km]",
                dn_tot,
                "Bank {ib}",
                dn_eq,
                "Bank {ib} (eq)",
                "eroded distance",
                "[m]",
            )
            if self.plot_flags["save_plot"]:
                fig_i = fig_i + 1
                fig_base = self.plot_flags["fig_dir"] + os.sep + str(fig_i) + "_erodis"
                if self.plot_flags["save_plot_zoomed"]:
                    df_plt.zoom_x_and_save(fig, ax, fig_base, self.plot_flags["plot_ext"], km_zoom)
                fig_file = fig_base + self.plot_flags["plot_ext"]
                df_plt.savefig(fig, fig_file)

            if self.plot_flags["close_plot"]:
                plt.close("all")
            else:
                plt.show(block=not self.gui)


def _apply_masked_indexing(x0: np.array, idx: np.ma.masked_array) -> np.ma.masked_array:
    """
    Index one array by another transferring the mask.

    Args:
        x0 : np.ndarray
            A linear array.
        idx : np.ma.masked_array
            An index array with possibly masked indices.

    returns:
        x1: np.ma.masked_array
            An array with same shape as idx, with mask.
    """
    idx_safe = idx.copy()
    idx_safe.data[np.ma.getmask(idx)] = 0
    x1 = np.ma.masked_where(np.ma.getmask(idx), x0[idx_safe])
    return x1


def _compute_mesh_topology(
    sim: SimulationObject,
) -> MeshData:
    """Derive secondary topology arrays from the face-node connectivity of the mesh.

    This function computes the edge-node, edge-face, and face-edge connectivity arrays,
    as well as the boundary edges of the mesh, based on the face-node connectivity provided
    in the simulation data.

    Args:
        sim (SimulationObject):
            A simulation object containing mesh-related data, including face-node connectivity
            (`facenode`), the number of nodes per face (`nnodes`), and node coordinates (`x_node`, `y_node`).

    Returns:
        MeshData: a dataclass containing the following attributes:
            - `x_face_coords`: x-coordinates of face nodes
            - `y_face_coords`: y-coordinates of face nodes
            - `x_edge_coords`: x-coordinates of edge nodes
            - `y_edge_coords`: y-coordinates of edge nodes
            - `face_node`: the node indices for each of the mesh faces.
            - `n_nodes`: number of nodes per face
            - `edge_node`: the node indices for each of the mesh edges.
            - `edge_face_connectivity`: the face indices for each of the mesh edge
            - `face_edge_connectivity`: the edge indices for each of the mesh face
            - `boundary_edge_nrs`: indices of boundary edges

    Raises:
        KeyError:
            If required keys (e.g., `facenode`, `nnodes`, `x_node`, `y_node`) are missing from the `sim` object.

    Notes:
        - The function identifies unique edges by sorting and comparing node indices.
        - Boundary edges are identified as edges that belong to only one face.
        - The function assumes that the mesh is well-formed, with consistent face-node connectivity.
    """

    # get a sorted list of edge node connections (shared edges occur twice)
    # face_nr contains the face index to which the edge belongs
    face_node = sim["facenode"]
    n_nodes = sim["nnodes"]
    n_faces = face_node.shape[0]
    n_edges = sum(n_nodes)
    edge_node = np.zeros((n_edges, 2), dtype=np.int64)
    face_nr = np.zeros((n_edges,), dtype=np.int64)
    i = 0
    for face_i in range(n_faces):
        num_edges = n_nodes[face_i]  # note: nEdges = nNodes
        for edge_i in range(num_edges):
            if edge_i == 0:
                edge_node[i, 1] = face_node[face_i, num_edges - 1]
            else:
                edge_node[i, 1] = face_node[face_i, edge_i - 1]
            edge_node[i, 0] = face_node[face_i, edge_i]
            face_nr[i] = face_i
            i = i + 1
    edge_node.sort(axis=1)
    i2 = np.argsort(edge_node[:, 1], kind="stable")
    i1 = np.argsort(edge_node[i2, 0], kind="stable")
    i12 = i2[i1]
    edge_node = edge_node[i12, :]
    face_nr = face_nr[i12]

    # detect which edges are equal to the previous edge, and get a list of all unique edges
    numpy_true = np.array([True])
    equal_to_previous = np.concatenate(
        (~numpy_true, (np.diff(edge_node, axis=0) == 0).all(axis=1))
    )
    unique_edge = ~equal_to_previous
    n_unique_edges = np.sum(unique_edge)
    # reduce the edge node connections to only the unique edges
    edge_node = edge_node[unique_edge, :]

    # number the edges
    edge_nr = np.zeros(n_edges, dtype=np.int64)
    edge_nr[unique_edge] = np.arange(n_unique_edges, dtype=np.int64)
    edge_nr[equal_to_previous] = edge_nr[
        np.concatenate((equal_to_previous[1:], equal_to_previous[:1]))
    ]

    # if two consecutive edges are unique, the first one occurs only once and represents a boundary edge
    is_boundary_edge = unique_edge & np.concatenate((unique_edge[1:], numpy_true))
    boundary_edge_nrs = edge_nr[is_boundary_edge]

    # go back to the original face order
    edge_nr_in_face_order = np.zeros(n_edges, dtype=np.int64)
    edge_nr_in_face_order[i12] = edge_nr
    # create the face edge connectivity array
    face_edge_connectivity = np.zeros(face_node.shape, dtype=np.int64)

    i = 0
    for face_i in range(n_faces):
        num_edges = n_nodes[face_i]  # note: num_edges = n_nodes
        for edge_i in range(num_edges):
            face_edge_connectivity[face_i, edge_i] = edge_nr_in_face_order[i]
            i = i + 1

    # determine the edge face connectivity
    edge_face = -np.ones((n_unique_edges, 2), dtype=np.int64)
    edge_face[edge_nr[unique_edge], 0] = face_nr[unique_edge]
    edge_face[edge_nr[equal_to_previous], 1] = face_nr[equal_to_previous]

    x_face_coords = _apply_masked_indexing(sim["x_node"], face_node)
    y_face_coords = _apply_masked_indexing(sim["y_node"], face_node)
    x_edge_coords = sim["x_node"][edge_node]
    y_edge_coords = sim["y_node"][edge_node]

    return MeshData(
        x_face_coords=x_face_coords,
        y_face_coords=y_face_coords,
        x_edge_coords=x_edge_coords,
        y_edge_coords=y_edge_coords,
        face_node=face_node,
        n_nodes=n_nodes,
        edge_node=edge_node,
        edge_face_connectivity=edge_face,
        face_edge_connectivity=face_edge_connectivity,
        boundary_edge_nrs=boundary_edge_nrs,
    )


class BankLinesResultsError(Exception):
    """Custom exception for BankLine results errors."""

    pass
