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

import configparser
import os
import time
from typing import List, Tuple

import geopandas as gpd
import numpy as np
import shapely
from matplotlib import pyplot as plt

import dfastbe.io as df_io
from dfastbe import __version__, kernel, plotting, support

FIRST_TIME: float
LAST_TIME: float


def banklines(filename: str = "dfastbe.cfg") -> None:
    """
    Run the bank line detection analysis using a configuration specified by file name.

    Arguments
    ---------
    filename : str
        Name of the configuration file.
    """
    # read configuration file
    timedlogger("reading configuration file ...")
    config = df_io.read_config(filename)
    rootdir, config = adjust_filenames(filename, config)
    banklines_core(config, rootdir, False)


def adjust_filenames(
    filename: str, config: configparser.ConfigParser
) -> Tuple[str, configparser.ConfigParser]:
    """
    Convert all paths to relative to current working directory.

    Arguments
    ---------
    filename : str
        Name of the configuration file.
    config : configparser.ConfigParser
        Analysis configuration settings.

    Returns
    -------
    rootdir : str
        Location of configuration file relative to current working directory.
    config : configparser.ConfigParser
        Analysis configuration settings using paths relative to current working directory.
    """
    rootdir = os.path.dirname(filename)
    cwd = os.getcwd()
    config = config_to_absolute_paths(rootdir, config)
    config = config_to_relative_paths(cwd, config)
    rootdir = df_io.relative_path(cwd, rootdir)

    return rootdir, config


def banklines_core(config: configparser.ConfigParser, rootdir: str, gui: bool) -> None:
    """
    Run the bank line detection analysis for a specified configuration.

    Arguments
    ---------
    config : configparser.ConfigParser
        Analysis configuration settings.
    rootdir : str
        Root folder for the analysis (may be relative to current work directory).
    gui : bool
        Flag indicating whether this routine is called from the GUI.
    """
    timedlogger("-- start analysis --")

    df_io.log_text(
        "header_banklines",
        dict={
            "version": __version__,
            "location": "https://github.com/Deltares/D-FAST_Bank_Erosion",
        },
    )
    df_io.log_text("-")

    # check output dir for bank lines
    bankdir = df_io.config_get_str(config, "General", "BankDir")
    df_io.log_text("bankdir_out", dict={"dir": bankdir})
    if os.path.exists(bankdir):
        df_io.log_text("overwrite_dir", dict={"dir": bankdir})
    else:
        os.makedirs(bankdir)

    # set plotting flags
    plot_bool = df_io.config_get_bool(config, "General", "Plotting", True)
    if plot_bool:
        saveplot = df_io.config_get_bool(config, "General", "SavePlots", True)
        saveplot_zoomed = df_io.config_get_bool(
            config, "General", "SaveZoomPlots", True
        )
        zoom_km_step = df_io.config_get_float(config, "General", "ZoomStepKM", 1.0)
        if zoom_km_step < 0.01:
            saveplot_zoomed = False
        closeplot = df_io.config_get_bool(config, "General", "ClosePlots", False)
    else:
        saveplot = False
        saveplot_zoomed = False
        closeplot = False

    # as appropriate check output dir for figures and file format
    if saveplot:
        figdir = df_io.config_get_str(
            config, "General", "FigureDir", rootdir + os.sep + "figure"
        )
        df_io.log_text("figure_dir", dict={"dir": figdir})
        if os.path.exists(figdir):
            df_io.log_text("overwrite_dir", dict={"dir": figdir})
        else:
            os.makedirs(figdir)
        plot_ext = df_io.config_get_str(config, "General", "FigureExt", ".png")

    # read chainage path
    xykm = df_io.config_get_xykm(config)

    # clip the chainage path to the range of chainages of interest
    kmbounds = df_io.config_get_kmbounds(config)
    df_io.log_text("clip_chainage", dict={"low": kmbounds[0], "high": kmbounds[1]})
    xykm = df_io.clip_path_to_kmbounds(xykm, kmbounds)
    xykm_numpy = np.array(xykm)
    xy_numpy = xykm_numpy[:, :2]

    # read bank search lines
    max_river_width = 1000
    search_lines = df_io.config_get_search_lines(config)
    search_lines, maxmaxd = support.clip_search_lines(
        search_lines, xykm, max_river_width
    )
    n_searchlines = len(search_lines)

    # convert search lines to bank polygons
    dlines = df_io.config_get_bank_search_distances(config, n_searchlines)
    bankareas = support.convert_search_lines_to_bank_polygons(search_lines, dlines)

    # determine whether search lines are located on left or right
    to_right = [True] * n_searchlines
    for ib in range(n_searchlines):
        to_right[ib] = support.on_right_side(np.array(search_lines[ib]), xy_numpy)

    # get simulation file name
    simfile = df_io.config_get_simfile(config, "Detect", "")

    # get critical water depth used for defining bank line (default = 0.0 m)
    h0 = df_io.config_get_float(config, "Detect", "WaterDepth", default=0)

    # read simulation data and drying flooding threshold dh0
    df_io.log_text("-")
    df_io.log_text("read_simdata", dict={"file": simfile})
    df_io.log_text("-")
    sim, dh0 = df_io.read_simdata(simfile)
    df_io.log_text("-")

    # increase critical water depth h0 by flooding threshold dh0
    h0 = h0 + dh0

    # clip simulation data to boundaries ...
    df_io.log_text("clip_data")
    sim = support.clip_simdata(sim, xykm, maxmaxd)

    # derive bank lines (getbanklines)
    df_io.log_text("identify_banklines")
    banklines = support.get_banklines(sim, h0)
    banklines.to_file(bankdir + os.sep + "raw_detected_bankline_fragments.shp")
    gpd.GeoSeries(bankareas).to_file(bankdir + os.sep + "bank_areas.shp")

    # clip the set of detected bank lines to the bank areas
    df_io.log_text("simplify_banklines")
    bank = [None] * n_searchlines
    clipped_banklines = [None] * n_searchlines
    for ib, bankarea in enumerate(bankareas):
        df_io.log_text("bank_lines", dict={"ib": ib + 1})
        clipped_banklines[ib] = support.clip_bank_lines(banklines, bankarea)
        bank[ib] = support.sort_connect_bank_lines(
            clipped_banklines[ib], xykm, to_right[ib]
        )
    gpd.GeoSeries(clipped_banklines).to_file(
        bankdir + os.sep + "bankline_fragments_per_bank_area.shp"
    )
    df_io.log_text("-")

    # save bankfile
    bankname = df_io.config_get_str(config, "General", "BankFile", "bankfile")
    bankfile = bankdir + os.sep + bankname + ".shp"
    df_io.log_text("save_banklines", dict={"file": bankfile})
    gpd.GeoSeries(bank).to_file(bankfile)

    if plotting:
        df_io.log_text("=")
        df_io.log_text("create_figures")
        ifig = 0
        bbox = get_bbox(xykm_numpy)

        if saveplot_zoomed:
            bank_crds: List[np.ndarray] = []
            bank_km: List[np.ndarray] = []
            for ib in range(n_searchlines):
                bcrds_numpy = np.array(bank[ib])
                km_numpy = support.project_km_on_line(bcrds_numpy, xykm_numpy)
                bank_crds.append(bcrds_numpy)
                bank_km.append(km_numpy)
            _, xyzoom = get_zoom_extends(
                kmbounds[0], kmbounds[1], zoom_km_step, bank_crds, bank_km
            )

        fig, ax = plotting.plot_detect1(
            bbox,
            xykm_numpy,
            bankareas,
            bank,
            sim["facenode"],
            sim["nnodes"],
            sim["x_node"],
            sim["y_node"],
            sim["h_face"],
            1.1 * sim["h_face"].max(),
            "x-coordinate [m]",
            "y-coordinate [m]",
            "water depth and detected bank lines",
            "water depth [m]",
            "bank search area",
            "detected bank line",
        )
        if saveplot:
            ifig = ifig + 1
            figbase = figdir + os.sep + str(ifig) + "_banklinedetection"
            if saveplot_zoomed:
                plotting.zoom_xy_and_save(fig, ax, figbase, plot_ext, xyzoom, scale=1)
            figfile = figbase + plot_ext
            plotting.savefig(fig, figfile)

    if plotting:
        if closeplot:
            plt.close("all")
        else:
            plt.show(block=not gui)

    df_io.log_text("end_banklines")
    timedlogger("-- stop analysis --")


def bankerosion(filename="dfastbe.cfg") -> None:
    """
    Run the bank erosion analysis using a configuration specified by file name.

    Arguments
    ---------
    filename : str
        Name of the configuration file.
    """
    # read configuration file
    timedlogger("reading configuration file ...")
    config = df_io.read_config(filename)
    rootdir, config = adjust_filenames(filename, config)
    bankerosion_core(config, rootdir, False)


def bankerosion_core(
    config: configparser.ConfigParser, rootdir: str, gui: bool
) -> None:
    """
    Run the bank erosion analysis for a specified configuration.

    Arguments
    ---------
    config : configparser.ConfigParser
        Analysis configuration settings.
    rootdir : str
        Root folder for the analysis (may be relative to current work directory).
    gui : bool
        Flag indicating whether this routine is called from the GUI.
    """
    banklines: gpd.GeoSeries
    timedlogger("-- start analysis --")

    rho = 1000  # density of water [kg/m3]
    g = 9.81  # gravititional acceleration [m/s2]
    df_io.log_text(
        "header_bankerosion",
        dict={
            "version": __version__,
            "location": "https://github.com/Deltares/D-FAST_Bank_Erosion",
        },
    )
    df_io.log_text("-")

    # check if additional debug output is requested
    debug = df_io.config_get_bool(config, "General", "DebugOutput", False)

    # check bankdir for input
    bankdir = df_io.config_get_str(config, "General", "BankDir")
    df_io.log_text("bankdir_in", dict={"dir": bankdir})
    if not os.path.exists(bankdir):
        df_io.log_text("missing_dir", dict={"dir": bankdir})
        return

    # check outputdir
    outputdir = df_io.config_get_str(config, "Erosion", "OutputDir")
    df_io.log_text("output_dir", dict={"dir": outputdir})
    if os.path.exists(outputdir):
        df_io.log_text("overwrite_dir", dict={"dir": outputdir})
    else:
        os.makedirs(outputdir)

    # set plotting flags
    plotting = df_io.config_get_bool(config, "General", "Plotting", True)
    if plotting:
        saveplot = df_io.config_get_bool(config, "General", "SavePlots", True)
        saveplot_zoomed = df_io.config_get_bool(
            config, "General", "SaveZoomPlots", True
        )
        zoom_km_step = df_io.config_get_float(config, "General", "ZoomStepKM", 1.0)
        if zoom_km_step < 0.01:
            saveplot_zoomed = False
        closeplot = df_io.config_get_bool(config, "General", "ClosePlots", False)
    else:
        saveplot = False
        saveplot_zoomed = False
        closeplot = False

    # as appropriate check output dir for figures and file format
    if saveplot:
        figdir = df_io.config_get_str(
            config, "General", "FigureDir", rootdir + os.sep + "figure"
        )
        df_io.log_text("figure_dir", dict={"dir": figdir})
        if os.path.exists(figdir):
            df_io.log_text("overwrite_dir", dict={"dir": figdir})
        else:
            os.makedirs(figdir)
        plot_ext = df_io.config_get_str(config, "General", "FigureExt", ".png")

    # get simulation time terosion
    teros = df_io.config_get_int(config, "Erosion", "TErosion", positive=True)
    df_io.log_text("total_time", dict={"t": teros})

    # get filter settings for bank levels and flow velocities along banks
    zb_dx = df_io.config_get_float(
        config, "Erosion", "BedFilterDist", 0.0, positive=True
    )
    vel_dx = df_io.config_get_float(
        config, "Erosion", "VelFilterDist", 0.0, positive=True
    )

    # get pdischarges
    df_io.log_text("get_levels")
    num_levels = df_io.config_get_int(config, "Erosion", "NLevel")
    ref_level = df_io.config_get_int(config, "Erosion", "RefLevel") - 1
    simfiles = []
    pdischarge = []
    for iq in range(num_levels):
        iq_str = str(iq + 1)
        simfiles.append(df_io.config_get_simfile(config, "Erosion", iq_str))
        pdischarge.append(
            df_io.config_get_float(config, "Erosion", "PDischarge" + iq_str)
        )

    # read simulation data (getsimdata)
    simfile = df_io.config_get_simfile(config, "Erosion", str(ref_level + 1))
    df_io.log_text("-")
    df_io.log_text("read_simdata", dict={"file": simfile})
    df_io.log_text("-")
    sim, _ = df_io.read_simdata(simfile)
    df_io.log_text("-")

    df_io.log_text("derive_topology")
    fn = sim["facenode"]
    nnodes = sim["nnodes"]
    en, ef, fe, boundary_edge_nrs = derive_topology_arrays(fn, nnodes)

    # read chainage path
    xykm = df_io.config_get_xykm(config)

    # clip the chainage path to the range of chainages of interest
    kmbounds = df_io.config_get_kmbounds(config)
    df_io.log_text("clip_chainage", dict={"low": kmbounds[0], "high": kmbounds[1]})

    xykm = df_io.clip_path_to_kmbounds(xykm, kmbounds)
    xykm_numpy = np.array(xykm)
    xy_numpy = xykm_numpy[:, :2]

    # read bank lines
    banklines = df_io.config_get_bank_lines(config, bankdir)
    n_banklines = len(banklines)

    # map bank lines to mesh cells
    df_io.log_text("intersect_bank_mesh")

    xf = masked_index(sim["x_node"], fn)
    yf = masked_index(sim["y_node"], fn)
    xe = sim["x_node"][en]
    ye = sim["y_node"][en]
    bank_crds = []
    bank_idx = []
    for ib in range(n_banklines):
        bp = np.array(banklines.geometry[ib])
        df_io.log_text("bank_nodes", dict={"ib": ib + 1, "n": len(bp)})

        crds, idx = support.intersect_line_mesh(
            bp, xf, yf, xe, ye, fe, ef, fn, en, nnodes, boundary_edge_nrs
        )
        bank_crds.append(crds)
        bank_idx.append(idx)

    # linking bank lines to chainage
    df_io.log_text("chainage_to_banks")
    bank_km_mid = [None] * n_banklines
    to_right = [True] * n_banklines
    for ib, bcrds in enumerate(bank_crds):
        bcrds_mid = (bcrds[:-1, :] + bcrds[1:, :]) / 2
        km_mid = support.project_km_on_line(bcrds_mid, xykm_numpy)

        # check if bank line is defined from low chainage to high chainage
        if km_mid[0] > km_mid[-1]:
            # if not, flip the bank line and all associated data
            km_mid = km_mid[::-1]
            bank_crds[ib] = bank_crds[ib][::-1, :]
            bank_idx[ib] = bank_idx[ib][::-1]

        bank_km_mid[ib] = km_mid

        # check if bank line is left or right bank
        # when looking from low to high chainage
        to_right[ib] = support.on_right_side(bcrds, xy_numpy)
        if to_right[ib]:
            df_io.log_text("right_side_bank", dict={"ib": ib + 1})
        else:
            df_io.log_text("left_side_bank", dict={"ib": ib + 1})

    # read river axis file
    river_axis_file = df_io.config_get_str(config, "Erosion", "RiverAxis")
    df_io.log_text("read_river_axis", dict={"file": river_axis_file})
    river_axis = df_io.read_xyc(river_axis_file)
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
    df_io.log_text("chainage_to_axis")
    river_axis_km = support.project_km_on_line(river_axis_numpy, xykm_numpy)
    df_io.write_shp_pnt(
        river_axis_numpy,
        {"chainage": river_axis_km},
        outputdir + os.sep + "river_axis_chainage.shp",
    )

    # clip river axis to reach of interest
    i1 = np.argmin(((xy_numpy[0] - river_axis_numpy) ** 2).sum(axis=1))
    i2 = np.argmin(((xy_numpy[-1] - river_axis_numpy) ** 2).sum(axis=1))
    if i1 < i2:
        river_axis_km = river_axis_km[i1 : i2 + 1]
        river_axis_numpy = river_axis_numpy[i1 : i2 + 1]
    else:
        # reverse river axis
        river_axis_km = river_axis_km[i2 : i1 + 1][::-1]
        river_axis_numpy = river_axis_numpy[i2 : i1 + 1][::-1]
    river_axis = shapely.geometry.LineString(river_axis_numpy)

    # get output interval
    km_step = df_io.config_get_float(config, "Erosion", "OutputInterval", 1.0)
    # map to output interval
    km_bin = (river_axis_km.min(), river_axis_km.max(), km_step)
    km_mid = kernel.get_km_bins(km_bin, type=3)  # get mid points

    # read fairway file
    fairway_file = df_io.config_get_str(config, "Erosion", "Fairway")
    df_io.log_text("read_fairway", dict={"file": fairway_file})
    # map km to fairway points, further using axis
    df_io.log_text("chainage_to_fairway")
    fairway_numpy = np.array(river_axis.coords)
    fairway_km = support.project_km_on_line(fairway_numpy, xykm_numpy)
    df_io.write_shp_pnt(
        fairway_numpy,
        {"chainage": fairway_km},
        f"{outputdir}{os.sep}fairway_chainage.shp",
    )

    # clip fairway to reach of interest
    i1 = np.argmin(((xy_numpy[0] - fairway_numpy) ** 2).sum(axis=1))
    i2 = np.argmin(((xy_numpy[-1] - fairway_numpy) ** 2).sum(axis=1))
    if i1 < i2:
        fairway_numpy = fairway_numpy[i1 : i2 + 1]
    else:
        # reverse fairway
        # fairway_km = fairway_km[i2 : i1 + 1][::-1]
        fairway_numpy = fairway_numpy[i2 : i1 + 1][::-1]

    # intersect fairway and mesh
    df_io.log_text("intersect_fairway_mesh", dict={"n": len(fairway_numpy)})
    ifw_numpy, ifw_face_idx = support.intersect_line_mesh(
        fairway_numpy, xf, yf, xe, ye, fe, ef, fn, en, nnodes, boundary_edge_nrs
    )
    if debug:
        df_io.write_shp_pnt(
            (ifw_numpy[:-1] + ifw_numpy[1:]) / 2,
            {"iface": ifw_face_idx},
            outputdir + os.sep + "fairway_face_indices.shp",
        )

    # distance fairway-bankline (bankfairway)
    df_io.log_text("bank_distance_fairway")
    distance_fw = []
    bp_fw_face_idx = []
    nfw = len(ifw_face_idx)
    for ib, bcrds in enumerate(bank_crds):
        bcrds_mid = (bcrds[:-1] + bcrds[1:]) / 2
        distance_fw.append(np.zeros(len(bcrds_mid)))
        bp_fw_face_idx.append(np.zeros(len(bcrds_mid), dtype=np.int64))
        for ip, bp in enumerate(bcrds_mid):
            # find closest fairway support node
            ifw = np.argmin(((bp - ifw_numpy) ** 2).sum(axis=1))
            fwp = ifw_numpy[ifw]
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
                    (ifw_numpy[ifw, 0] - ifw_numpy[ifw - 1, 0])
                    * (bp[0] - ifw_numpy[ifw - 1, 0])
                    + (ifw_numpy[ifw, 1] - ifw_numpy[ifw - 1, 1])
                    * (bp[1] - ifw_numpy[ifw - 1, 1])
                ) / (
                    (ifw_numpy[ifw, 0] - ifw_numpy[ifw - 1, 0]) ** 2
                    + (ifw_numpy[ifw, 1] - ifw_numpy[ifw - 1, 1]) ** 2
                )
                if 0 < alpha < 1:
                    fwp1 = ifw_numpy[ifw - 1] + alpha * (
                        ifw_numpy[ifw] - ifw_numpy[ifw - 1]
                    )
                    d1 = ((bp - fwp1) ** 2).sum() ** 0.5
                    if d1 < dbfw:
                        dbfw = d1
                        # projected point located on segment before, which corresponds to initial choice: iseg = ifw - 1
            if ifw < nfw:
                alpha = (
                    (ifw_numpy[ifw + 1, 0] - ifw_numpy[ifw, 0])
                    * (bp[0] - ifw_numpy[ifw, 0])
                    + (ifw_numpy[ifw + 1, 1] - ifw_numpy[ifw, 1])
                    * (bp[1] - ifw_numpy[ifw, 1])
                ) / (
                    (ifw_numpy[ifw + 1, 0] - ifw_numpy[ifw, 0]) ** 2
                    + (ifw_numpy[ifw + 1, 1] - ifw_numpy[ifw, 1]) ** 2
                )
                if 0 < alpha < 1:
                    fwp1 = ifw_numpy[ifw] + alpha * (
                        ifw_numpy[ifw + 1] - ifw_numpy[ifw]
                    )
                    d1 = ((bp - fwp1) ** 2).sum() ** 0.5
                    if d1 < dbfw:
                        dbfw = d1
                        iseg = ifw

            bp_fw_face_idx[ib][ip] = ifw_face_idx[iseg]
            distance_fw[ib][ip] = dbfw

        if debug:
            df_io.write_shp_pnt(
                bcrds_mid,
                {"chainage": bank_km_mid[ib], "iface_fw": bp_fw_face_idx[ib]},
                f"{outputdir}{os.sep}bank_{ib + 1}_chainage_and_fairway_face_idx.shp",
            )

    # water level at fairway
    # s1 = sim["zw_face"]
    zfw_ini = []
    for ib in range(n_banklines):
        ii = bp_fw_face_idx[ib]
        zfw_ini.append(sim["zw_face"][ii])

    # wave reduction s0, s1
    dfw0 = df_io.config_get_parameter(
        config,
        "Erosion",
        "Wave0",
        bank_km_mid,
        default=200,
        positive=True,
        onefile=True,
    )
    dfw1 = df_io.config_get_parameter(
        config,
        "Erosion",
        "Wave1",
        bank_km_mid,
        default=150,
        positive=True,
        onefile=True,
    )

    # save 1_banklines

    # read vship, nship, nwave, draught (tship), shiptype ... independent of level number
    vship0 = df_io.config_get_parameter(
        config, "Erosion", "VShip", bank_km_mid, positive=True, onefile=True
    )
    Nship0 = df_io.config_get_parameter(
        config, "Erosion", "NShip", bank_km_mid, positive=True, onefile=True
    )
    nwave0 = df_io.config_get_parameter(
        config, "Erosion", "NWave", bank_km_mid, default=5, positive=True, onefile=True
    )
    tship0 = df_io.config_get_parameter(
        config, "Erosion", "Draught", bank_km_mid, positive=True, onefile=True
    )
    ship0 = df_io.config_get_parameter(
        config, "Erosion", "ShipType", bank_km_mid, valid=[1, 2, 3], onefile=True
    )
    parslope0 = df_io.config_get_parameter(
        config, "Erosion", "Slope", bank_km_mid, default=20, positive=True, ext="slp"
    )
    parreed0 = df_io.config_get_parameter(
        config, "Erosion", "Reed", bank_km_mid, default=0, positive=True, ext="rdd"
    )

    # read classes flag (yes: banktype = taucp, no: banktype = tauc) and banktype (taucp: 0-4 ... or ... tauc = critical shear value)
    classes = df_io.config_get_bool(config, "Erosion", "Classes")
    taucls = np.array([1e20, 95, 3.0, 0.95, 0.15])
    taucls_str = ["protected", "vegetation", "good clay", "moderate/bad clay", "sand"]
    if classes:
        banktype = df_io.config_get_parameter(
            config, "Erosion", "BankType", bank_km_mid, default=0, ext=".btp"
        )
        tauc = []
        for ib in range(len(banktype)):
            tauc.append(taucls[banktype[ib]])
    else:
        tauc = df_io.config_get_parameter(
            config, "Erosion", "BankType", bank_km_mid, default=0, ext=".btp"
        )
        thr = (taucls[:-1] + taucls[1:]) / 2
        banktype = [None] * len(thr)
        for ib in range(len(tauc)):
            bt = np.zeros(tauc[ib].size)
            for thr_i in thr:
                bt[tauc[ib] < thr_i] += 1
            banktype[ib] = bt

    # read bank protection level zss
    zss_miss = -1000
    zss = df_io.config_get_parameter(
        config, "Erosion", "ProtectionLevel", bank_km_mid, default=zss_miss, ext=".bpl"
    )
    # if zss undefined, set zss equal to zfw_ini - 1
    for ib in range(len(zss)):
        mask = zss[ib] == zss_miss
        zss[ib][mask] = zfw_ini[ib][mask] - 1

    # initialize arrays for erosion loop over all discharges
    velocity: List[List[np.ndarray]] = []
    bankheight: List[np.ndarray] = []
    waterlevel: List[List[np.ndarray]] = []
    chezy: List[List[np.ndarray]] = []
    dv: List[List[np.ndarray]] = []
    shipwavemax: List[List[np.ndarray]] = []
    shipwavemin: List[List[np.ndarray]] = []

    linesize: List[np.ndarray] = []
    dn_flow_tot: List[np.ndarray] = []
    dn_ship_tot: List[np.ndarray] = []
    dn_tot: List[np.ndarray] = []
    dv_tot: List[np.ndarray] = []
    dn_eq: List[np.ndarray] = []
    dv_eq: List[np.ndarray] = []
    for iq in range(num_levels):
        df_io.log_text(
            "discharge_header",
            dict={"i": iq + 1, "p": pdischarge[iq], "t": pdischarge[iq] * teros},
        )

        iq_str = "{}".format(iq + 1)

        df_io.log_text("read_q_params", indent="  ")
        # read vship, nship, nwave, draught, shiptype, slope, reed, fairwaydepth, ... (level specific values)
        vship = df_io.config_get_parameter(
            config,
            "Erosion",
            "VShip" + iq_str,
            bank_km_mid,
            default=vship0,
            positive=True,
            onefile=True,
        )
        nship = df_io.config_get_parameter(
            config,
            "Erosion",
            "NShip" + iq_str,
            bank_km_mid,
            default=Nship0,
            positive=True,
            onefile=True,
        )
        nwave = df_io.config_get_parameter(
            config,
            "Erosion",
            "NWave" + iq_str,
            bank_km_mid,
            default=nwave0,
            positive=True,
            onefile=True,
        )
        tship = df_io.config_get_parameter(
            config,
            "Erosion",
            "Draught" + iq_str,
            bank_km_mid,
            default=tship0,
            positive=True,
            onefile=True,
        )
        ship_type = df_io.config_get_parameter(
            config,
            "Erosion",
            "ShipType" + iq_str,
            bank_km_mid,
            default=ship0,
            valid=[1, 2, 3],
            onefile=True,
        )

        parslope = df_io.config_get_parameter(
            config,
            "Erosion",
            "Slope" + iq_str,
            bank_km_mid,
            default=parslope0,
            positive=True,
            ext="slp",
        )
        parreed = df_io.config_get_parameter(
            config,
            "Erosion",
            "Reed" + iq_str,
            bank_km_mid,
            default=parreed0,
            positive=True,
            ext="rdd",
        )
        mu_slope = [None] * n_banklines
        mu_reed = [None] * n_banklines
        for ib in range(n_banklines):
            mus = parslope[ib].copy()
            mus[mus > 0] = 1 / mus[mus > 0]
            mu_slope[ib] = mus
            mu_reed[ib] = 8.5e-4 * parreed[ib] ** 0.8

        df_io.log_text("-", indent="  ")
        df_io.log_text("read_simdata", dict={"file": simfiles[iq]}, indent="  ")
        df_io.log_text("-", indent="  ")
        sim, dh0 = df_io.read_simdata(simfiles[iq], indent="  ")
        df_io.log_text("-", indent="  ")
        fnc = sim["facenode"]

        df_io.log_text("bank_erosion", indent="  ")
        velocity.append([])
        waterlevel.append([])
        chezy.append([])
        dv.append([])
        shipwavemax.append([])
        shipwavemin.append([])

        dvol_bank = np.zeros((len(km_mid), n_banklines))
        hfw_max = 0
        for ib, bcrds in enumerate(bank_crds):
            # determine velocity along banks ...
            dx = np.diff(bcrds[:, 0])
            dy = np.diff(bcrds[:, 1])
            if iq == 0:
                linesize.append(np.sqrt(dx**2 + dy**2))

            bank_index = bank_idx[ib]
            vel_bank = (
                np.absolute(
                    sim["ucx_face"][bank_index] * dx + sim["ucy_face"][bank_index] * dy
                )
                / linesize[ib]
            )

            if vel_dx > 0.0:
                if ib == 0:
                    df_io.log_text(
                        "apply_velocity_filter", indent="  ", dict={"dx": vel_dx}
                    )
                vel_bank = kernel.moving_avg(bank_km_mid[ib], vel_bank, vel_dx)
            velocity[iq].append(vel_bank)
            #
            if iq == 0:
                # determine velocity and bankheight along banks ...
                # bankheight = maximum bed elevation per cell
                if sim["zb_location"] == "node":
                    zb = sim["zb_val"]
                    zb_all_nodes = masked_index(zb, fnc[bank_index, :])
                    zb_bank = zb_all_nodes.max(axis=1)
                    if zb_dx > 0.0:
                        if ib == 0:
                            df_io.log_text(
                                "apply_banklevel_filter",
                                indent="  ",
                                dict={"dx": zb_dx},
                            )
                        zb_bank = kernel.moving_avg(bank_km_mid[ib], zb_bank, zb_dx)
                    bankheight.append(zb_bank)
                else:
                    # don't know ... need to check neighbouring cells ...
                    bankheight.append(None)

            # get water depth along fairway
            ii = bp_fw_face_idx[ib]
            hfw = sim["h_face"][ii]
            hfw_max = max(hfw_max, hfw.max())
            waterlevel[iq].append(sim["zw_face"][ii])
            chez = sim["chz_face"][ii]
            chezy[iq].append(0 * chez + chez.mean())

            if iq == num_levels - 1:  # ref_level:
                dn_eq1, dv_eq1 = kernel.comp_erosion_eq(
                    bankheight[ib],
                    linesize[ib],
                    zfw_ini[ib],
                    vship[ib],
                    ship_type[ib],
                    tship[ib],
                    mu_slope[ib],
                    distance_fw[ib],
                    dfw0[ib],
                    dfw1[ib],
                    hfw,
                    zss[ib],
                    g,
                )
                dn_eq.append(dn_eq1)
                dv_eq.append(dv_eq1)

                if debug:
                    bcrds_mid = (bcrds[:-1] + bcrds[1:]) / 2
                    bcrds_pnt = [shapely.geometry.Point(xy1) for xy1 in bcrds_mid]
                    bcrds_geo = gpd.geoseries.GeoSeries(bcrds_pnt)
                    params = {
                        "chainage": bank_km_mid[ib],
                        "x": bcrds_mid[:, 0],
                        "y": bcrds_mid[:, 1],
                        "iface_fw": bp_fw_face_idx[ib],  # ii
                        "iface_bank": bank_idx[ib],  # bank_index
                        "zb": bankheight[ib],
                        "len": linesize[ib],
                        "zw0": zfw_ini[ib],
                        "vship": vship[ib],
                        "shiptype": ship_type[ib],
                        "draught": tship[ib],
                        "mu_slp": mu_slope[ib],
                        "dist_fw": distance_fw[ib],
                        "dfw0": dfw0[ib],
                        "dfw1": dfw1[ib],
                        "hfw": hfw,
                        "zss": zss[ib],
                        "dn": dn_eq1,
                        "dv": dv_eq1,
                    }

                    df_io.write_shp(
                        bcrds_geo,
                        params,
                        outputdir + os.sep + "debug.EQ.B{}.shp".format(ib + 1),
                    )
                    df_io.write_csv(
                        params,
                        outputdir + os.sep + "debug.EQ.B{}.csv".format(ib + 1),
                    )

            dniqib, dviqib, dnship, dnflow, shipwavemax_ib, shipwavemin_ib = (
                kernel.comp_erosion(
                    velocity[iq][ib],
                    bankheight[ib],
                    linesize[ib],
                    waterlevel[iq][ib],
                    zfw_ini[ib],
                    tauc[ib],
                    nship[ib],
                    vship[ib],
                    nwave[ib],
                    ship_type[ib],
                    tship[ib],
                    teros * pdischarge[iq],
                    mu_slope[ib],
                    mu_reed[ib],
                    distance_fw[ib],
                    dfw0[ib],
                    dfw1[ib],
                    hfw,
                    chezy[iq][ib],
                    zss[ib],
                    rho,
                    g,
                )
            )
            shipwavemax[iq].append(shipwavemax_ib)
            shipwavemin[iq].append(shipwavemin_ib)

            if debug:
                bcrds_mid = (bcrds[:-1] + bcrds[1:]) / 2

                bcrds_pnt = [shapely.geometry.Point(xy1) for xy1 in bcrds_mid]
                bcrds_geo = gpd.geoseries.GeoSeries(bcrds_pnt)
                params = {
                    "chainage": bank_km_mid[ib],
                    "x": bcrds_mid[:, 0],
                    "y": bcrds_mid[:, 1],
                    "iface_fw": bp_fw_face_idx[ib],  # ii
                    "iface_bank": bank_idx[ib],  # bank_index
                    "u": velocity[iq][ib],
                    "zb": bankheight[ib],
                    "len": linesize[ib],
                    "zw": waterlevel[iq][ib],
                    "zw0": zfw_ini[ib],
                    "tauc": tauc[ib],
                    "nship": nship[ib],
                    "vship": vship[ib],
                    "nwave": nwave[ib],
                    "shiptype": ship_type[ib],
                    "draught": tship[ib],
                    "mu_slp": mu_slope[ib],
                    "mu_reed": mu_reed[ib],
                    "dist_fw": distance_fw[ib],
                    "dfw0": dfw0[ib],
                    "dfw1": dfw1[ib],
                    "hfw": hfw,
                    "chez": chezy[iq][ib],
                    "zss": zss[ib],
                    "dn": dniqib,
                    "dv": dviqib,
                    "dnship": dnship,
                    "dnflow": dnflow,
                }
                df_io.write_shp(
                    bcrds_geo,
                    params,
                    outputdir + os.sep + "debug.Q{}.B{}.shp".format(iq + 1, ib + 1),
                )
                df_io.write_csv(
                    params,
                    outputdir + os.sep + "debug.Q{}.B{}.csv".format(iq + 1, ib + 1),
                )

            # shift bank lines
            # xylines_new = support.move_line(bcrds, dniqib, to_right[ib])

            if len(dn_tot) == ib:
                dn_flow_tot.append(dnflow.copy())
                dn_ship_tot.append(dnship.copy())
                dn_tot.append(dniqib.copy())
                dv_tot.append(dviqib.copy())
            else:
                dn_flow_tot[ib] += dnflow
                dn_ship_tot[ib] += dnship
                dn_tot[ib] += dniqib
                dv_tot[ib] += dviqib

            # accumulate eroded volumes per km
            dvol = kernel.get_km_eroded_volume(bank_km_mid[ib], dviqib, km_bin)
            dv[iq].append(dvol)
            dvol_bank[:, ib] += dvol

        erovol_file = df_io.config_get_str(
            config, "Erosion", "EroVol" + iq_str, default="erovolQ" + iq_str + ".evo"
        )
        df_io.log_text("save_erovol", dict={"file": erovol_file}, indent="  ")
        df_io.write_km_eroded_volumes(
            km_mid, dvol_bank, outputdir + os.sep + erovol_file
        )

    df_io.log_text("=")
    dnav = np.zeros(n_banklines)
    dnmax = np.zeros(n_banklines)
    dnavflow = np.zeros(n_banklines)
    dnavship = np.zeros(n_banklines)
    dnaveq = np.zeros(n_banklines)
    dnmaxeq = np.zeros(n_banklines)
    vol_eq = np.zeros((len(km_mid), n_banklines))
    vol_tot = np.zeros((len(km_mid), n_banklines))
    xyline_new_list = []
    bankline_new_list = []
    xyline_eq_list = []
    bankline_eq_list = []
    for ib, bcrds in enumerate(bank_crds):
        dnav[ib] = (dn_tot[ib] * linesize[ib]).sum() / linesize[ib].sum()
        dnmax[ib] = dn_tot[ib].max()
        dnavflow[ib] = (dn_flow_tot[ib] * linesize[ib]).sum() / linesize[ib].sum()
        dnavship[ib] = (dn_ship_tot[ib] * linesize[ib]).sum() / linesize[ib].sum()
        dnaveq[ib] = (dn_eq[ib] * linesize[ib]).sum() / linesize[ib].sum()
        dnmaxeq[ib] = dn_eq[ib].max()
        df_io.log_text("bank_dnav", dict={"ib": ib + 1, "v": dnav[ib]})
        df_io.log_text("bank_dnavflow", dict={"v": dnavflow[ib]})
        df_io.log_text("bank_dnavship", dict={"v": dnavship[ib]})
        df_io.log_text("bank_dnmax", dict={"v": dnmax[ib]})
        df_io.log_text("bank_dnaveq", dict={"v": dnaveq[ib]})
        df_io.log_text("bank_dnmaxeq", dict={"v": dnmaxeq[ib]})

        xyline_new = support.move_line(bcrds, dn_tot[ib], to_right[ib])
        xyline_new_list.append(xyline_new)
        bankline_new_list.append(shapely.geometry.LineString(xyline_new))

        xyline_eq = support.move_line(bcrds, dn_eq[ib], to_right[ib])
        xyline_eq_list.append(xyline_eq)
        bankline_eq_list.append(shapely.geometry.LineString(xyline_eq))

        dvol_eq = kernel.get_km_eroded_volume(bank_km_mid[ib], dv_eq[ib], km_bin)
        vol_eq[:, ib] = dvol_eq
        dvol_tot = kernel.get_km_eroded_volume(bank_km_mid[ib], dv_tot[ib], km_bin)
        vol_tot[:, ib] = dvol_tot
        if ib < n_banklines - 1:
            df_io.log_text("-")

    # write bank line files
    bankline_new_series = gpd.geoseries.GeoSeries(bankline_new_list)
    banklines_new = gpd.geodataframe.GeoDataFrame.from_features(bankline_new_series)
    bankname = df_io.config_get_str(config, "General", "BankFile", "bankfile")
    bankfile = outputdir + os.sep + bankname + "_new.shp"
    df_io.log_text("save_banklines", dict={"file": bankfile})
    banklines_new.to_file(bankfile)

    bankline_eq_series = gpd.geoseries.GeoSeries(bankline_eq_list)
    banklines_eq = gpd.geodataframe.GeoDataFrame.from_features(bankline_eq_series)
    bankfile = outputdir + os.sep + bankname + "_eq.shp"
    df_io.log_text("save_banklines", dict={"file": bankfile})
    banklines_eq.to_file(bankfile)

    # write eroded volumes per km (total)
    erovol_file = df_io.config_get_str(
        config, "Erosion", "EroVol", default="erovol.evo"
    )
    df_io.log_text("save_tot_erovol", dict={"file": erovol_file})
    df_io.write_km_eroded_volumes(km_mid, vol_tot, outputdir + os.sep + erovol_file)

    # write eroded volumes per km (equilibrium)
    erovol_file = df_io.config_get_str(
        config, "Erosion", "EroVolEqui", default="erovol_eq.evo"
    )
    df_io.log_text("save_eq_erovol", dict={"file": erovol_file})
    df_io.write_km_eroded_volumes(km_mid, vol_eq, outputdir + os.sep + erovol_file)

    # create various plots
    if plotting:
        df_io.log_text("=")
        df_io.log_text("create_figures")
        ifig = 0
        bbox = get_bbox(xykm_numpy)

        if saveplot_zoomed:
            bank_crds_mid = []
            for ib in range(n_banklines):
                bank_crds_mid.append((bank_crds[ib][:-1, :] + bank_crds[ib][1:, :]) / 2)
            kmzoom, xyzoom = get_zoom_extends(
                river_axis_km.min(),
                river_axis_km.max(),
                zoom_km_step,
                bank_crds_mid,
                bank_km_mid,
            )

        fig, ax = plotting.plot1_waterdepth_and_banklines(
            bbox,
            xykm_numpy,
            banklines,
            fn,
            sim["nnodes"],
            sim["x_node"],
            sim["y_node"],
            sim["h_face"],
            1.1 * hfw_max,
            "x-coordinate [km]",
            "y-coordinate [km]",
            "water depth and initial bank lines",
            "water depth [m]",
        )
        if saveplot:
            ifig = ifig + 1
            figbase = figdir + os.sep + str(ifig) + "_banklines"
            if saveplot_zoomed:
                plotting.zoom_xy_and_save(fig, ax, figbase, plot_ext, xyzoom)
            figfile = figbase + plot_ext
            plotting.savefig(fig, figfile)

        fig, ax = plotting.plot2_eroded_distance_and_equilibrium(
            bbox,
            xykm_numpy,
            bank_crds,
            dn_tot,
            to_right,
            dnav,
            xyline_eq_list,
            xe,
            ye,
            "x-coordinate [km]",
            "y-coordinate [km]",
            "eroded distance and equilibrium bank location",
            "eroded during {t} year".format(t=teros),
            "eroded distance [m]",
            "equilibrium location",
        )
        if saveplot:
            ifig = ifig + 1
            figbase = figdir + os.sep + str(ifig) + "_erosion_sensitivity"
            if saveplot_zoomed:
                plotting.zoom_xy_and_save(fig, ax, figbase, plot_ext, xyzoom)
            figfile = figbase + plot_ext
            plotting.savefig(fig, figfile)

        fig, ax = plotting.plot3_eroded_volume(
            km_mid,
            km_step,
            "river chainage [km]",
            dv,
            "eroded volume [m^3]",
            "eroded volume per {ds} chainage km ({t} years)".format(
                ds=km_step, t=teros
            ),
            "Q{iq}",
            "Bank {ib}",
        )
        if saveplot:
            ifig = ifig + 1
            figbase = figdir + os.sep + str(ifig) + "_eroded_volume"
            if saveplot_zoomed:
                plotting.zoom_x_and_save(fig, ax, figbase, plot_ext, kmzoom)
            figfile = figbase + plot_ext
            plotting.savefig(fig, figfile)

        fig, ax = plotting.plot3_eroded_volume_subdivided_1(
            km_mid,
            km_step,
            "river chainage [km]",
            dv,
            "eroded volume [m^3]",
            "eroded volume per {ds} chainage km ({t} years)".format(
                ds=km_step, t=teros
            ),
            "Q{iq}",
        )
        if saveplot:
            ifig = ifig + 1
            figbase = figdir + os.sep + str(ifig) + "_eroded_volume_per_discharge"
            if saveplot_zoomed:
                plotting.zoom_x_and_save(fig, ax, figbase, plot_ext, kmzoom)
            figfile = figbase + plot_ext
            plotting.savefig(fig, figfile)

        fig, ax = plotting.plot3_eroded_volume_subdivided_2(
            km_mid,
            km_step,
            "river chainage [km]",
            dv,
            "eroded volume [m^3]",
            "eroded volume per {ds} chainage km ({t} years)".format(
                ds=km_step, t=teros
            ),
            "Bank {ib}",
        )
        if saveplot:
            ifig = ifig + 1
            figbase = figdir + os.sep + str(ifig) + "_eroded_volume_per_bank"
            if saveplot_zoomed:
                plotting.zoom_x_and_save(fig, ax, figbase, plot_ext, kmzoom)
            figfile = figbase + plot_ext
            plotting.savefig(fig, figfile)

        fig, ax = plotting.plot4_eroded_volume_eq(
            km_mid,
            km_step,
            "river chainage [km]",
            vol_eq,
            "eroded volume [m^3]",
            "eroded volume per {ds} chainage km (equilibrium)".format(ds=km_step),
        )
        if saveplot:
            ifig = ifig + 1
            figbase = figdir + os.sep + str(ifig) + "_eroded_volume_eq"
            if saveplot_zoomed:
                plotting.zoom_x_and_save(fig, ax, figbase, plot_ext, kmzoom)
            figfile = figbase + plot_ext
            plotting.savefig(fig, figfile)

        figlist, axlist = plotting.plot5series_waterlevels_per_bank(
            bank_km_mid,
            "river chainage [km]",
            waterlevel,
            shipwavemax,
            shipwavemin,
            "water level at Q{iq}",
            "average water level",
            "wave influenced range",
            bankheight,
            "level of bank",
            zss,
            "bank protection level",
            "elevation",
            "(water)levels along bank line {ib}",
            "[m NAP]",
        )
        if saveplot:
            for ib, fig in enumerate(figlist):
                ifig = ifig + 1
                figbase = f"{figdir}{os.sep}{ifig}_levels_bank_{ib + 1}"
                if saveplot_zoomed:
                    plotting.zoom_x_and_save(fig, axlist[ib], figbase, plot_ext, kmzoom)
                figfile = figbase + plot_ext
                plotting.savefig(fig, figfile)

        figlist, axlist = plotting.plot6series_velocity_per_bank(
            bank_km_mid,
            "river chainage [km]",
            velocity,
            "velocity at Q{iq}",
            tauc,
            chezy[0],
            rho,
            g,
            "critical velocity",
            "velocity",
            "velocity along bank line {ib}",
            "[m/s]",
        )
        if saveplot:
            for ib, fig in enumerate(figlist):
                ifig = ifig + 1
                figbase = figdir + os.sep + str(ifig) + "_velocity_bank_" + str(ib + 1)

                if saveplot_zoomed:
                    plotting.zoom_x_and_save(fig, axlist[ib], figbase, plot_ext, kmzoom)
                figfile = figbase + plot_ext
                plotting.savefig(fig, figfile)

        fig, ax = plotting.plot7_banktype(
            bbox,
            xykm_numpy,
            bank_crds,
            banktype,
            taucls_str,
            "x-coordinate [km]",
            "y-coordinate [km]",
            "bank type",
        )
        if saveplot:
            ifig = ifig + 1
            figbase = figdir + os.sep + str(ifig) + "_banktype"
            if saveplot_zoomed:
                plotting.zoom_xy_and_save(fig, ax, figbase, plot_ext, xyzoom)
            figfile = figbase + plot_ext
            plotting.savefig(fig, figfile)

        fig, ax = plotting.plot8_eroded_distance(
            bank_km_mid,
            "river chainage [km]",
            dn_tot,
            "Bank {ib}",
            dn_eq,
            "Bank {ib} (eq)",
            "eroded distance",
            "[m]",
        )
        if saveplot:
            ifig = ifig + 1
            figbase = figdir + os.sep + str(ifig) + "_erodis"
            if saveplot_zoomed:
                plotting.zoom_x_and_save(fig, ax, figbase, plot_ext, kmzoom)
            figfile = figbase + plot_ext
            plotting.savefig(fig, figfile)

        if closeplot:
            plt.close("all")
        else:
            plt.show(block=not gui)

    df_io.log_text("end_bankerosion")
    timedlogger("-- end analysis --")


def masked_index(x0: np.array, idx: np.ma.masked_array) -> np.ma.masked_array:
    """
    Index one array by another transferring the mask.

    Arguments
    ---------
    x0 : np.ndarray
        A linear array.
    idx : np.ma.masked_array
        An index array with possibly masked indices.

    Results
    -------
    x1: np.ma.masked_array
        An array with same shape as idx, with mask.
    """
    idx_safe = idx.copy()
    idx_safe.data[np.ma.getmask(idx)] = 0
    x1 = np.ma.masked_where(np.ma.getmask(idx), x0[idx_safe])
    return x1


def get_bbox(
    xykm: np.ndarray, buffer: float = 0.1
) -> Tuple[float, float, float, float]:
    """
    Derive the bounding box from a line.

    Arguments
    ---------
    xybm : np.ndarray
        An N x M array containing x- and y-coordinates as first two M entries
    buffer : float
        Buffer fraction surrounding the tight bounding box

    Results
    -------
    bbox : Tuple[float, float, float, float]
        Tuple bounding box consisting of [min x, min y, max x, max y)
    """
    x = xykm[:, 0]
    y = xykm[:, 1]
    xmin = x.min()
    ymin = y.min()
    xmax = x.max()
    ymax = y.max()
    d = buffer * max(xmax - xmin, ymax - ymin)
    bbox = (xmin - d, ymin - d, xmax + d, ymax + d)
    return bbox


def derive_topology_arrays(
    fn: np.ndarray, n_nodes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive the secondary topology arrays from the face_node_connectivity.

    Arguments
    ---------
    fn : np.ndarray
        An N x M array containing the node indices (max M) for each of the N mesh faces.
    n_nodes: np.ndarray
        A array of length N containing the number of nodes for each one of the N mesh faces.

    Results
    -------
    en : np.ndarray
        An L x 2 array containing the node indices (2) for each of the L mesh edges.
    ef : np.ndarray
        An L x 2 array containing the face indices (max 2) for each of the L mesh edges.
    fe : np.ndarray
        An N x M array containing the edge indices (max M) for each of the N mesh faces.
    boundary_edge_nrs : np.ndarray
        A array of length K containing the edge indices making the mesh outer (or inner) boundary.
    """

    # get a sorted list of edge node connections (shared edges occur twice)
    # face_nr contains the face index to which the edge belongs
    n_faces = fn.shape[0]

    n_edges = sum(n_nodes)
    en = np.zeros((n_edges, 2), dtype=np.int64)
    face_nr = np.zeros((n_edges,), dtype=np.int64)
    i = 0
    for i_face in range(n_faces):
        n_edges = n_nodes[i_face]  # note: nEdges = nNodes
        for i_edge in range(n_edges):
            if i_edge == 0:
                en[i, 1] = fn[i_face, n_edges - 1]
            else:
                en[i, 1] = fn[i_face, i_edge - 1]
            en[i, 0] = fn[i_face, i_edge]
            face_nr[i] = i_face
            i = i + 1
    en.sort(axis=1)
    i2 = np.argsort(en[:, 1], kind="stable")
    i1 = np.argsort(en[i2, 0], kind="stable")
    i12 = i2[i1]
    en = en[i12, :]
    face_nr = face_nr[i12]

    # detect which edges are equal to the previous edge, and get a list of all unique edges
    numpy_true = np.array([True])
    equal_to_previous = np.concatenate(
        (~numpy_true, (np.diff(en, axis=0) == 0).all(axis=1))
    )
    unique_edge = ~equal_to_previous
    n_unique_edges = np.sum(unique_edge)
    # reduce the edge node connections to only the unique edges
    en = en[unique_edge, :]

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
    fe = np.zeros(fn.shape, dtype=np.int64)
    i = 0
    for i_face in range(n_faces):
        n_edges = n_nodes[i_face]  # note: nEdges = nNodes
        for i_edge in range(n_edges):
            fe[i_face, i_edge] = edge_nr_in_face_order[i]
            i = i + 1

    # determine the edge face connectivity
    ef = -np.ones((n_unique_edges, 2), dtype=np.int64)
    ef[edge_nr[unique_edge], 0] = face_nr[unique_edge]
    ef[edge_nr[equal_to_previous], 1] = face_nr[equal_to_previous]

    return en, ef, fe, boundary_edge_nrs


def timedlogger(label: str) -> None:
    """
    Write message with time information.

    Arguments
    ---------
    label : str
        Message string.
    """
    time, diff = timer()
    print(time + diff + label)


def timer() -> Tuple[str, str]:
    """
    Return text string representation of time since previous call.

    The routine uses the global variable LAST_TIME to store the time of the
    previous call.

    Arguments
    ---------
    None

    Returns
    -------
    time_str : str
        String representing duration since first call.
    diff_str : str
        String representing duration since previous call.
    """
    global FIRST_TIME
    global LAST_TIME
    new_time = time.time()
    if "LAST_TIME" in globals():
        time_str = "{:6.2f} ".format(new_time - FIRST_TIME)
        diff_str = "{:6.2f} ".format(new_time - LAST_TIME)
    else:
        time_str = "   0.00"
        diff_str = "       "
        FIRST_TIME = new_time
    LAST_TIME = new_time
    return time_str, diff_str


def config_to_absolute_paths(
    rootdir: str, config: configparser.ConfigParser
) -> configparser.ConfigParser:
    """
    Convert a configuration object to contain absolute paths (for editing).

    Arguments
    ---------
    rootdir : str
        The path to be used as base for the absolute paths.
    config : configparser.ConfigParser
        Configuration for the D-FAST Bank Erosion analysis with absolute or relative paths.

    Returns
    -------
    config1 : configparser.ConfigParser
        Configuration for the D-FAST Bank Erosion analysis with only absolute paths.
    """
    if "General" in config:
        config = parameter_absolute_path(config, "General", "RiverKM", rootdir)
        config = parameter_absolute_path(config, "General", "BankDir", rootdir)
        config = parameter_absolute_path(config, "General", "FigureDir", rootdir)

    if "Detect" in config:
        config = parameter_absolute_path(config, "Detect", "SimFile", rootdir)
        i = 0
        while True:
            i = i + 1
            line = "Line" + str(i)
            if line in config["Detect"]:
                config = parameter_absolute_path(config, "Detect", line, rootdir)
            else:
                break

    if "Erosion" in config:
        config = parameter_absolute_path(config, "Erosion", "RiverAxis", rootdir)
        config = parameter_absolute_path(config, "Erosion", "Fairway", rootdir)
        config = parameter_absolute_path(config, "Erosion", "OutputDir", rootdir)

        config = parameter_absolute_path(config, "Erosion", "ShipType", rootdir)
        config = parameter_absolute_path(config, "Erosion", "VShip", rootdir)
        config = parameter_absolute_path(config, "Erosion", "NShip", rootdir)
        config = parameter_absolute_path(config, "Erosion", "NWave", rootdir)
        config = parameter_absolute_path(config, "Erosion", "Draught", rootdir)
        config = parameter_absolute_path(config, "Erosion", "Wave0", rootdir)
        config = parameter_absolute_path(config, "Erosion", "Wave1", rootdir)

        config = parameter_absolute_path(config, "Erosion", "BankType", rootdir)
        config = parameter_absolute_path(config, "Erosion", "ProtectionLevel", rootdir)
        config = parameter_absolute_path(config, "Erosion", "Slope", rootdir)
        config = parameter_absolute_path(config, "Erosion", "Reed", rootdir)

        n_level = df_io.config_get_int(config, "Erosion", "NLevel", default=0)
        for i in range(n_level):
            istr = str(i + 1)
            config = parameter_absolute_path(
                config, "Erosion", "SimFile" + istr, rootdir
            )
            config = parameter_absolute_path(
                config, "Erosion", "ShipType" + istr, rootdir
            )
            config = parameter_absolute_path(config, "Erosion", "VShip" + istr, rootdir)
            config = parameter_absolute_path(config, "Erosion", "NShip" + istr, rootdir)
            config = parameter_absolute_path(config, "Erosion", "NWave" + istr, rootdir)
            config = parameter_absolute_path(
                config, "Erosion", "Draught" + istr, rootdir
            )
            config = parameter_absolute_path(config, "Erosion", "Slope" + istr, rootdir)
            config = parameter_absolute_path(config, "Erosion", "Reed" + istr, rootdir)

    return config


def config_to_relative_paths(
    rootdir: str, config: configparser.ConfigParser
) -> configparser.ConfigParser:
    """
    Convert a configuration object to contain relative paths (for saving).

    Arguments
    ---------
    rootdir : str
        The path to be used as base for the relative paths.
    config : configparser.ConfigParser
        Configuration for the D-FAST Bank Erosion analysis with only absolute paths.

    Returns
    -------
    config1 : configparser.ConfigParser
        Updated configuration for D-FAST Bank Erosion analysis with as much as possible relative paths.
    """
    if "General" in config:
        config = parameter_relative_path(config, "General", "RiverKM", rootdir)
        config = parameter_relative_path(config, "General", "BankDir", rootdir)
        config = parameter_relative_path(config, "General", "FigureDir", rootdir)

    if "Detect" in config:
        config = parameter_relative_path(config, "Detect", "SimFile", rootdir)
        i = 0
        while True:
            i = i + 1
            line = "Line" + str(i)
            if line in config["Detect"]:
                config = parameter_relative_path(config, "Detect", line, rootdir)
            else:
                break

    if "Erosion" in config:
        config = parameter_relative_path(config, "Erosion", "RiverAxis", rootdir)
        config = parameter_relative_path(config, "Erosion", "Fairway", rootdir)
        config = parameter_relative_path(config, "Erosion", "OutputDir", rootdir)

        config = parameter_relative_path(config, "Erosion", "ShipType", rootdir)
        config = parameter_relative_path(config, "Erosion", "VShip", rootdir)
        config = parameter_relative_path(config, "Erosion", "NShip", rootdir)
        config = parameter_relative_path(config, "Erosion", "NWave", rootdir)
        config = parameter_relative_path(config, "Erosion", "Draught", rootdir)
        config = parameter_relative_path(config, "Erosion", "Wave0", rootdir)
        config = parameter_relative_path(config, "Erosion", "Wave1", rootdir)

        config = parameter_relative_path(config, "Erosion", "BankType", rootdir)
        config = parameter_relative_path(config, "Erosion", "ProtectionLevel", rootdir)
        config = parameter_relative_path(config, "Erosion", "Slope", rootdir)
        config = parameter_relative_path(config, "Erosion", "Reed", rootdir)

        n_level = df_io.config_get_int(config, "Erosion", "NLevel", default=0)
        for i in range(n_level):
            istr = str(i + 1)
            config = parameter_relative_path(
                config, "Erosion", "SimFile" + istr, rootdir
            )
            config = parameter_relative_path(
                config, "Erosion", "ShipType" + istr, rootdir
            )
            config = parameter_relative_path(config, "Erosion", "VShip" + istr, rootdir)
            config = parameter_relative_path(config, "Erosion", "NShip" + istr, rootdir)
            config = parameter_relative_path(config, "Erosion", "NWave" + istr, rootdir)
            config = parameter_relative_path(
                config, "Erosion", "Draught" + istr, rootdir
            )
            config = parameter_relative_path(config, "Erosion", "Slope" + istr, rootdir)
            config = parameter_relative_path(config, "Erosion", "Reed" + istr, rootdir)

    return config


def parameter_absolute_path(
    config: configparser.ConfigParser, group: str, key: str, rootdir: str
) -> configparser.ConfigParser:
    """
    Convert a parameter value to contain an absolute path.

    Determine whether the string represents a number.
    If not, try to convert to an absolute path.

    Arguments
    ---------
    config : configparser.ConfigParser
        Configuration for the D-FAST Bank Erosion analysis.
    group : str
        Name of the group in the configuration.
    key : str
        Name of the key in the configuration.
    rootdir : str
        The path to be used as base for the absolute paths.

    Returns
    -------
    config1 : configparser.ConfigParser
        Updated configuration for the D-FAST Bank Erosion analysis.
    """
    if key in config[group]:
        valstr = config[group][key]
        try:
            float(valstr)
        except:
            config[group][key] = df_io.absolute_path(rootdir, valstr)
    return config


def parameter_relative_path(
    config: configparser.ConfigParser, group: str, key: str, rootdir: str
) -> configparser.ConfigParser:
    """
    Convert a parameter value to contain a relative path.

    Determine whether the string represents a number.
    If not, try to convert to a relative path.

    Arguments
    ---------
    config : configparser.ConfigParser
        Configuration for the D-FAST Bank Erosion analysis.
    group : str
        Name of the group in the configuration.
    key : str
        Name of the key in the configuration.
    rootdir : str
        The path to be used as base for the relative paths.

    Returns
    -------
    config1 : configparser.ConfigParser
        Updated configuration for the D-FAST Bank Erosion analysis.
    """
    if key in config[group]:
        valstr = config[group][key]
        try:
            float(valstr)
        except:
            config[group][key] = df_io.relative_path(rootdir, valstr)
    return config


def get_zoom_extends(
    km_min: float,
    km_max: float,
    zoom_km_step: float,
    bank_crds: List[np.ndarray],
    bank_km: List[np.ndarray],
) -> [List[Tuple[float, float]], List[Tuple[float, float, float, float]]]:
    """Zoom.

    Arguments
    ---------
    km_min : float
        Minimum value for the chainage range of interest.
    km_max : float
        Maximum value for the chainage range of interest.
    zoom_km_step : float
        Preferred chainage length of zoom box.
    bank_crds : List[np.ndarray]
        List of N x 2 np arrays of coordinates per bank.
    bank_km : List[np.ndarray]
        List of N np arrays of chainage values per bank.

    Returns
    -------
    kmzoom : List[Tuple[float, float]]
        Zoom ranges for plots with chainage along x-axis.
    xyzoom : List[Tuple[float, float, float, float]]
        Zoom ranges for xy-plots.
    """

    zoom_km_bin = (km_min, km_max, zoom_km_step)
    zoom_km_bnd = kernel.get_km_bins(zoom_km_bin, type=0, adjust=True)
    eps = 0.1 * zoom_km_step

    kmzoom: List[Tuple[float, float]] = []
    xyzoom: List[Tuple[float, float, float, float]] = []
    inf = float('inf')
    for i in range(len(zoom_km_bnd) - 1):
        km_min = zoom_km_bnd[i] - eps
        km_max = zoom_km_bnd[i + 1] + eps
        kmzoom.append((km_min, km_max))

        xmin = inf
        xmax = -inf
        ymin = inf
        ymax = -inf
        for ib in range(len(bank_km)):
            irange = (bank_km[ib] >= km_min) & (bank_km[ib] <= km_max)
            range_crds = bank_crds[ib][irange, :]
            x = range_crds[:, 0]
            y = range_crds[:, 1]
            xmin = min(xmin, min(x))
            xmax = max(xmax, max(x))
            ymin = min(ymin, min(y))
            ymax = max(ymax, max(y))
        xyzoom.append((xmin, xmax, ymin, ymax))

    return kmzoom, xyzoom
