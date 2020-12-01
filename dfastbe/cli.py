# -*- coding: utf-8 -*-
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

from typing import Union, Dict, Any, Tuple, List

import time
import logging
import argparse
import dfastbe.kernel
import dfastbe.support
import dfastbe.io
import dfastbe.plotting
import os
import sys
import geopandas
import shapely
import pathlib
import numpy
import matplotlib
import configparser

FIRST_TIME: float
LAST_TIME: float

def banklines(filename: str = "config.ini") -> None:
    """
    Run the bank line detection analysis using a configuration specified by file name.

    Arguments
    ---------
    filename : str
        Name of the configuration file.
    """
    # read configuration file
    timedlogger("reading configuration file ...")
    config = dfastbe.io.read_config(filename)
    rootdir, config = adjust_filenames(filename, config)
    banklines_core(config, rootdir, False)


def adjust_filenames(filename: str, config: configparser.ConfigParser) -> Tuple[str, configparser.ConfigParser]:
    """
    Convert all paths to relative to current working directory

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
    rootdir = dfastbe.io.relative_path(cwd, rootdir)
    
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

    dfastbe.io.log_text(
        "header_banklines",
        dict={
            "version": dfastbe.__version__,
            "location": "https://github.com/Deltares/D-FAST_Bank_Erosion",
        },
    )
    dfastbe.io.log_text("-")

    # check output dir for bank lines
    bankdir = dfastbe.io.config_get_str(config,"General", "BankDir")
    dfastbe.io.log_text("bankdir_out", dict={"dir": bankdir})
    if os.path.exists(bankdir):
        dfastbe.io.log_text("overwrite_dir", dict={"dir": bankdir})
    else:
        os.makedirs(bankdir)

    # set plotting flags
    plotting = dfastbe.io.config_get_bool(config, "General", "Plotting", True)
    if plotting:
        saveplot = dfastbe.io.config_get_bool(config, "General", "SavePlots", True)
        closeplot = dfastbe.io.config_get_bool(config, "General", "ClosePlots", False)
    else:
        saveplot = False
        closeplot = False
    
    # as appropriate check output dir for figures and file format
    if saveplot:
        figdir = dfastbe.io.config_get_str(config,"General", "FigureDir", rootdir + os.sep + "figure")
        dfastbe.io.log_text("figure_dir", dict={"dir": figdir})
        if os.path.exists(figdir):
            dfastbe.io.log_text("overwrite_dir", dict={"dir": figdir})
        else:
            os.makedirs(figdir)
        plot_ext = dfastbe.io.config_get_str(config,"General", "FigureExt", ".png")

    # read chainage file
    xykm = dfastbe.io.config_get_xykm(config)
    xykm_numpy = numpy.array(xykm)

    # read bank search lines
    max_river_width = 1000
    search_lines = dfastbe.io.config_get_search_lines(config)
    search_lines, maxmaxd = dfastbe.support.clip_search_lines(
        search_lines, xykm, max_river_width
    )
    n_searchlines = len(search_lines)

    # convert search lines to bank polygons
    dlines = dfastbe.io.config_get_bank_search_distances(config, n_searchlines)
    bankareas = dfastbe.support.convert_search_lines_to_bank_polygons(
        search_lines, dlines
    )

    # get simulation file name
    simfile = dfastbe.io.config_get_simfile(config, "Detect", "")

    # get critical water depth used for defining bank line (default = 0.0 m)
    h0 = dfastbe.io.config_get_float(config, "Detect", "WaterDepth", default=0)

    # read simulation data and drying flooding threshold dh0
    dfastbe.io.log_text("-")
    dfastbe.io.log_text("read_simdata", dict={"file": simfile})
    dfastbe.io.log_text("-")
    sim, dh0 = dfastbe.io.read_simdata(simfile)
    dfastbe.io.log_text("-")

    # increase critical water depth h0 by flooding threshold dh0
    h0 = h0 + dh0

    # clip simulation data to boundaries ...
    dfastbe.io.log_text("clip_data")
    sim = dfastbe.support.clip_simdata(sim, xykm, maxmaxd)

    # derive bank lines (getbanklines)
    dfastbe.io.log_text("identify_banklines")
    banklines = dfastbe.support.get_banklines(sim, h0)

    # clip the set of detected bank lines to the bank areas
    dfastbe.io.log_text("simplify_banklines")
    bank = [None] * n_searchlines
    for b, bankarea in enumerate(bankareas):
        print("bank line {}".format(b + 1))
        bank[b] = dfastbe.support.clip_sort_connect_bank_lines(
            banklines, bankarea, xykm
        )
    dfastbe.io.log_text("-")

    # save bankfile
    bankname = dfastbe.io.config_get_str(config, "General", "BankFile", "bankfile")
    bankfile = bankdir + os.sep + bankname + ".shp"
    dfastbe.io.log_text("save_banklines", dict={"file": bankfile})
    geopandas.GeoSeries(bank).to_file(bankfile)

    if plotting:
        dfastbe.io.log_text("=")
        dfastbe.io.log_text("create_figures")
        ifig = 0
        bbox = get_bbox(xykm_numpy)
        
        hmax = 0.5 * sim["h_face"].max()
        fig = dfastbe.plotting.plot_detect1(bbox, xykm_numpy, bankareas, bank, sim["facenode"], sim["nnodes"], sim["x_node"], sim["y_node"], sim["h_face"], hmax, "x-coordinate [m]", "y-coordinate [m]", "water depth and detected bank lines", "water depth [m]")
        if saveplot:
            ifig = ifig + 1        
            figfile = figdir + os.sep + str(ifig) + "_banklinedetection" + plot_ext
            dfastbe.plotting.savefig(fig, figfile)

    if plotting:
        if closeplot:
            matplotlib.pyplot.close("all")
        else:
            matplotlib.pyplot.show(block=not gui)

    dfastbe.io.log_text("end_banklines")
    timedlogger("-- stop analysis --")


def bankerosion(filename="config.ini") -> None:
    """
    Run the bank erosion analysis using a configuration specified by file name.

    Arguments
    ---------
    filename : str
        Name of the configuration file.
    """
    # read configuration file
    timedlogger("reading configuration file ...")
    config = dfastbe.io.read_config(filename)
    rootdir, config = adjust_filenames(filename, config)
    bankerosion_core(config, rootdir, False)


def bankerosion_core(config: configparser.ConfigParser, rootdir: str, gui: bool) -> None:
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
    banklines: geopandas.GeoSeries
    timedlogger("-- start analysis --")

    rho = 1000  # density of water [kg/m3]
    g = 9.81  # gravititional acceleration [m/s2]
    dfastbe.io.log_text(
        "header_bankerosion",
        dict={
            "version": dfastbe.__version__,
            "location": "https://github.com/Deltares/D-FAST_Bank_Erosion",
        },
    )
    dfastbe.io.log_text("-")
	
    # check bankdir for input
    bankdir = dfastbe.io.config_get_str(config,"General", "BankDir")
    dfastbe.io.log_text("bankdir_in", dict={"dir": bankdir})
    if not os.path.exists(bankdir):
        dfastbe.io.log_text("missing_dir", dict={"dir": bankdir})
        return

    # check outputdir
    outputdir = dfastbe.io.config_get_str(config, "Erosion", "OutputDir")
    dfastbe.io.log_text("output_dir", dict={"dir": outputdir})
    if os.path.exists(outputdir):
        dfastbe.io.log_text("overwrite_dir", dict={"dir": outputdir})
    else:
        os.makedirs(outputdir)

    # set plotting flags
    plotting = dfastbe.io.config_get_bool(config, "General", "Plotting", True)
    if plotting:
        saveplot = dfastbe.io.config_get_bool(config, "General", "SavePlots", True)
        closeplot = dfastbe.io.config_get_bool(config, "General", "ClosePlots", False)
    else:
        saveplot = False
        closeplot = False
    
    # as appropriate check output dir for figures and file format
    if saveplot:
        figdir = dfastbe.io.config_get_str(config,"General", "FigureDir", rootdir + os.sep + "figure")
        dfastbe.io.log_text("figure_dir", dict={"dir": figdir})
        if os.path.exists(figdir):
            dfastbe.io.log_text("overwrite_dir", dict={"dir": figdir})
        else:
            os.makedirs(figdir)
        plot_ext = dfastbe.io.config_get_str(config,"General", "FigureExt", ".png")

    # get simulation time terosion
    Teros = dfastbe.io.config_get_int(config, "Erosion", "TErosion", positive=True)
    dfastbe.io.log_text("total_time", dict={"t": Teros})

    # read simulation data (getsimdata)
    simfile = dfastbe.io.config_get_simfile(config, "Erosion", "1")
    dfastbe.io.log_text("-")
    dfastbe.io.log_text("read_simdata", dict={"file": simfile})
    dfastbe.io.log_text("-")
    sim, dh0 = dfastbe.io.read_simdata(simfile)
    dfastbe.io.log_text("-")

    fn = sim["facenode"]
    en, ef, fe, boundary_edge = derive_topology_arrays(fn)

    # read river km file
    xykm = dfastbe.io.config_get_xykm(config)
    xykm_numpy = numpy.array(xykm)
    xy_numpy = xykm_numpy[:,:2]

    # read bank lines
    banklines = dfastbe.io.config_get_bank_lines(config, bankdir)
    n_banklines = len(banklines)

    # map bank lines to mesh cells
    dfastbe.io.log_text("intersect_bank_mesh")
    bankline_faces = [None] * n_banklines
    xf = sim["x_node"][fn]
    yf = sim["y_node"][fn]
    xe = sim["x_node"][en]
    ye = sim["y_node"][en]
    boundary_edge_nrs = numpy.nonzero(boundary_edge)[0]
    bank_crds = []
    bank_idx = []
    for ib in range(n_banklines):
        bp = numpy.array(banklines.geometry[ib])
        dfastbe.io.log_text("bank_nodes", dict={"ib": ib+1, "n": len(bp)})

        crds, idx = dfastbe.support.intersect_line_mesh(
            bp, xf, yf, xe, ye, fe, ef, fn, en, boundary_edge_nrs
        )
        bank_crds.append(crds)
        bank_idx.append(idx)

    # linking bank lines to chainage
    dfastbe.io.log_text("chainage_to_banks")
    bank_km = [None] * n_banklines
    to_right = [True] * n_banklines
    for ib, bcrds in enumerate(bank_crds):
        bank_km[ib] = dfastbe.support.project_km_on_line(bcrds, xykm_numpy)
        to_right[ib] = dfastbe.support.on_right_side(bcrds, xy_numpy)
        if to_right[ib]:
            dfastbe.io.log_text("right_side_bank", dict={"ib": ib+1})
        else:
            dfastbe.io.log_text("left_side_bank", dict={"ib": ib+1})
    
    # read river axis file
    river_axis_file = dfastbe.io.config_get_str(config, "Erosion", "RiverAxis")
    dfastbe.io.log_text("read_river_axis", dict={"file": river_axis_file})
    river_axis = dfastbe.io.read_xyc(river_axis_file)
    river_axis_numpy = numpy.array(river_axis)
    # optional sorting --> see 04_Waal_D3D example
    # check: sum all distances and determine maximum distance ... if maximum > alpha * sum then perform sort
    # Waal OK: 0.0082 ratio max/sum, Waal NotOK: 0.13 - Waal: 2500 points, so even when OK still some 21 times more than 1/2500 = 0.0004
    dist2 = (numpy.diff(river_axis_numpy, axis=0) ** 2).sum(axis=1)
    alpha = dist2.max() / dist2.sum()
    if alpha > 0.03:
        print("The river axis needs sorting!!")
        # TODO: do sorting

    # map km to axis points, further using axis
    dfastbe.io.log_text("chainage_to_axis")
    river_axis_km = dfastbe.support.project_km_on_line(
        numpy.array(river_axis.coords), xykm_numpy
    )
    max_km = numpy.where(river_axis_km == river_axis_km.max())[0]
    min_km = numpy.where(river_axis_km == river_axis_km.min())[0]
    if max_km.max() < min_km.min():
        # reverse river axis
        imin = max_km.max()
        imax = min_km.min()
        river_axis_km = river_axis_km[imin : imax + 1][::-1]
        river_axis_numpy = river_axis_numpy[imin : imax + 1, :][::-1, :]
        river_axis = shapely.geometry.LineString(river_axis_numpy)
    else:
        imin = min_km.max()
        imax = max_km.min()
        river_axis_km = river_axis_km[imin : imax + 1]
        river_axis_numpy = river_axis_numpy[imin : imax + 1, :]
        river_axis = shapely.geometry.LineString(river_axis_numpy)

    # get output interval
    km_step = dfastbe.io.config_get_float(config, "Erosion", "OutputInterval", 1.0)
    # map to output interval
    km_bin = (river_axis_km.min(), river_axis_km.max(), km_step)
    km = dfastbe.kernel.get_km_bins(km_bin)
    xykm_bin_numpy = dfastbe.support.xykm_bin(xykm_numpy, km_bin)

    # read fairway file
    fairway_file = dfastbe.io.config_get_str(config, "Erosion", "Fairway")
    dfastbe.io.log_text("read_fairway", dict={"file": fairway_file})
    fairway = dfastbe.io.read_xyc(fairway_file)
    fairway_numpy = numpy.array(fairway)
    fw_used = numpy.zeros( (len(fairway_numpy),), dtype=bool )

    # distance fairway-bankline (bankfairway)
    dfastbe.io.log_text("bank_distance_fairway")
    distance_fw = []
    ifw = []
    for ib, bcrds in enumerate(bank_crds):
        distance_fw.append(numpy.zeros(len(bcrds)))
        ifw.append(numpy.zeros(len(bcrds), dtype=numpy.int64))
        ifw_last = None
        for ip, bp in enumerate(bcrds):
            # check only fairway points starting from latest match (in MATLAB code +/-10 from latest match)
            if ifw_last is None:
                ifw[ib][ip] = numpy.argmin(((bp - fairway_numpy) ** 2).sum(axis=1))
            else:
                ifw_min = max(0, ifw_last - 10)
                ifw[ib][ip] = ifw_min + numpy.argmin(
                    ((bp - fairway_numpy[ifw_min : ifw_last + 10, :]) ** 2).sum(axis=1)
                )
            ifw_last = ifw[ib][ip]
            distance_fw[ib][ip] = ((bp - fairway_numpy[ifw_last]) ** 2).sum() ** 0.5
        fw_used[ifw[ib]] = True

    # map fairway to mesh cells
    dfastbe.io.log_text("fairway_to_mesh", dict={"nnodes": fw_used.sum()})
    fwi = -numpy.ones( (len(fairway_numpy),), dtype=numpy.int64)
    fairway_index = numpy.ma.masked_array(fwi, mask=(fwi == -1))
    fairway_index[fw_used] = dfastbe.support.map_line_mesh(
        fairway_numpy[fw_used], xf, yf, xe, ye, fe, ef, boundary_edge_nrs
    )

    # water level at fairway
    # s1 = sim["zw_face"]
    zfw_ini = []
    for ib in range(n_banklines):
        ii = fairway_index[ifw[ib]]
        zfw_ini.append(sim["zw_face"][ii])

    # wave reduction s0, s1
    dfw0 = dfastbe.io.config_get_parameter(
        config, "Erosion", "Wave0", bank_km, default=200, positive=True, onefile=True
    )
    dfw1 = dfastbe.io.config_get_parameter(
        config, "Erosion", "Wave1", bank_km, default=150, positive=True, onefile=True
    )

    # save 1_banklines

    # read vship, nship, nwave, draught (tship), shiptype (ship) ... independent of level number
    vship0 = dfastbe.io.config_get_parameter(
        config, "Erosion", "VShip", bank_km, positive=True, onefile=True
    )
    Nship0 = dfastbe.io.config_get_parameter(
        config, "Erosion", "NShip", bank_km, positive=True, onefile=True
    )
    nwave0 = dfastbe.io.config_get_parameter(
        config, "Erosion", "NWave", bank_km, default=5, positive=True, onefile=True
    )
    Tship0 = dfastbe.io.config_get_parameter(
        config, "Erosion", "Draught", bank_km, positive=True, onefile=True
    )
    ship0 = dfastbe.io.config_get_parameter(
        config, "Erosion", "ShipType", bank_km, valid=[1, 2, 3], onefile=True
    )
    parslope0 = dfastbe.io.config_get_parameter(
        config, "Erosion", "Slope", bank_km, default=20, positive=True, ext="slp"
    )
    parreed0 = dfastbe.io.config_get_parameter(
        config, "Erosion", "Reed", bank_km, default=0, positive=True, ext="rdd"
    )

    # read classes flag (yes: banktype = taucp, no: banktype = tauc) and banktype (taucp: 0-4 ... or ... tauc = critical shear value)
    classes = dfastbe.io.config_get_bool(config, "Erosion", "Classes")
    taucls = numpy.array([1e20, 95, 3.0, 0.95, 0.15])
    taucls_str = ["protected", "vegetation", "good clay", "moderate/bad clay", "sand"]
    if classes:
        banktype = dfastbe.io.config_get_parameter(
            config, "Erosion", "BankType", bank_km, default=0, ext=".btp"
        )
        tauc = []
        for ib in range(len(banktype)):
            tauc.append(taucls[banktype[ib]])
    else:
        tauc = dfastbe.io.config_get_parameter(
            config, "Erosion", "BankType", bank_km, default=0, ext=".btp"
        )
        thr = (taucls[:-1] + taucls[1:]) / 2
        banktype = [None] * len(thr)
        for ib in range(len(tauc)):
            bt = numpy.zeros(tauc[ib].size)
            for thr_i in thr:
                bt[tauc[ib] < thr_i] += 1
            banktype[ib] = bt
    
    # read bank protectlevel zss
    zss_miss = -1000
    zss = dfastbe.io.config_get_parameter(
        config, "Erosion", "ProtectLevel", bank_km, default=zss_miss, ext=".bpl"
    )
    # if zss undefined, set zss equal to zfw_ini - 1
    for ib in range(len(zss)):
        mask = zss[ib] == zss_miss
        zss[ib][mask] = zfw_ini[ib][mask] - 1
    
    # get pdischarges
    dfastbe.io.log_text("get_levels")
    num_levels = dfastbe.io.config_get_int(config, "Erosion", "NLevel")
    ref_level = dfastbe.io.config_get_int(config, "Erosion", "RefLevel") - 1
    simfiles = []
    pdischarge = []
    for iq in range(num_levels):
        iq_str = str(iq + 1)
        simfiles.append(dfastbe.io.config_get_simfile(config, "Erosion", iq_str))
        pdischarge.append(
            dfastbe.io.config_get_float(config, "Erosion", "PDischarge" + iq_str)
        )

    velocity: List[List[numpy.ndarray]] = []
    bankheight: List[numpy.ndarray] = []
    waterlevel: List[List[numpy.ndarray]] = []
    chezy: List[List[numpy.ndarray]] = []
    dv: List[List[numpy.ndarray]] = []
    
    linesize: List[numpy.ndarray] = []
    dn_flow_tot: List[numpy.ndarray] = []
    dn_ship_tot: List[numpy.ndarray] = []
    dn_tot: List[numpy.ndarray] = []
    dv_tot: List[numpy.ndarray] = []
    dn_eq: List[numpy.ndarray] = []
    dv_eq: List[numpy.ndarray] = []
    for iq in range(num_levels):
        dfastbe.io.log_text("discharge_header", dict={"i": iq + 1, "p": pdischarge[iq], "t": pdischarge[iq]*Teros})
        
        iq_str = "{}".format(iq + 1)

        dfastbe.io.log_text("read_q_params", indent="  ")
        # read vship, nship, nwave, draught, shiptype, slope, reed, fairwaydepth, ... (level specific values)
        vship = dfastbe.io.config_get_parameter(
            config,
            "Erosion",
            "VShip" + iq_str,
            bank_km,
            default=vship0,
            positive=True,
            onefile=True,
        )
        Nship = dfastbe.io.config_get_parameter(
            config,
            "Erosion",
            "NShip" + iq_str,
            bank_km,
            default=Nship0,
            positive=True,
            onefile=True,
        )
        nwave = dfastbe.io.config_get_parameter(
            config,
            "Erosion",
            "NWave" + iq_str,
            bank_km,
            default=nwave0,
            positive=True,
            onefile=True,
        )
        Tship = dfastbe.io.config_get_parameter(
            config,
            "Erosion",
            "Draught" + iq_str,
            bank_km,
            default=Tship0,
            positive=True,
            onefile=True,
        )
        ship = dfastbe.io.config_get_parameter(
            config,
            "Erosion",
            "ShipType" + iq_str,
            bank_km,
            default=ship0,
            valid=[1, 2, 3],
            onefile=True,
        )

        parslope = dfastbe.io.config_get_parameter(
            config,
            "Erosion",
            "Slope" + iq_str,
            bank_km,
            default=parslope0,
            positive=True,
            ext="slp",
        )
        parreed = dfastbe.io.config_get_parameter(
            config,
            "Erosion",
            "Reed" + iq_str,
            bank_km,
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

        dfastbe.io.log_text("-", indent="  ")
        dfastbe.io.log_text("read_simdata", dict={"file": simfiles[iq]}, indent="  ")
        dfastbe.io.log_text("-", indent="  ")
        sim, dh0 = dfastbe.io.read_simdata(simfiles[iq], indent="  ")
        dfastbe.io.log_text("-", indent="  ")
        fnc = sim["facenode"]

        dfastbe.io.log_text("bank_erosion", indent="  ")
        velocity.append([])
        waterlevel.append([])
        chezy.append([])
        dv.append([])

        dvol_bank = numpy.zeros((len(km), n_banklines))
        hfw_max = 0
        for ib, bcrds in enumerate(bank_crds):
            # determine velocity along banks ...
            dx = numpy.diff(bcrds[:, 0])
            dy = numpy.diff(bcrds[:, 1])
            if iq == 0:
                linesize.append(numpy.sqrt(dx ** 2 + dy ** 2))

            idx = bank_idx[ib]
            bank_index = idx[1:]
            velocity[iq].append(
                numpy.absolute(
                    sim["ucx_face"][bank_index] * dx + sim["ucy_face"][bank_index] * dy
                )
                / linesize[ib]
            )
            #
            if iq == 0:
                # determine velocity and bankheight along banks ...            
                # bankheight = maximum bed elevation per cell
                if sim["zb_location"] == "node":
                    bankheight.append(sim["zb_val"][fnc[bank_index, :]].max(axis=1))
                else:
                    # don't know ... need to check neighbouring cells ...
                   bankheight.append(None)
                   pass

            # get water depth along fairway
            ii = fairway_index[ifw[ib]]
            hfw = sim["h_face"][ii]
            hfw_max = max(hfw_max, hfw.max())
            waterlevel[iq].append(sim["zw_face"][ii])
            chez = sim["chz_face"][ii]
            # TODO: curious ... MATLAB: chezy{j} = 0*chezy{j}+mchez
            chezy[iq].append(0 * chez + chez.mean())
            
            if iq == ref_level:
                dn_eq1, dv_eq1 = dfastbe.kernel.comp_erosion_eq(
                    bankheight[ib],
                    linesize[ib],
                    zfw_ini[ib],
                    vship[ib],
                    ship[ib],
                    Tship[ib],
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

            displ_tauc = False  # True for Delft3D, False otherwise
            filter = False
            
            qstr = str(iq + 1)
            bstr = str(ib + 1)
            dniqib, dviqib, dnship, dnflow = dfastbe.kernel.comp_erosion(
                velocity[iq][ib],
                bankheight[ib],
                linesize[ib],
                waterlevel[iq][ib],
                zfw_ini[ib],
                tauc[ib],
                Nship[ib],
                vship[ib],
                nwave[ib],
                ship[ib],
                Tship[ib],
                Teros * pdischarge[iq],
                mu_slope[ib],
                mu_reed[ib],
                distance_fw[ib],
                dfw0[ib],
                dfw1[ib],
                hfw,
                chezy[iq][ib],
                zss[ib],
                filter,
                rho,
                g,
                displ_tauc,
            )

            # shift bank lines
            # xylines_new = dfastbe.support.move_line(bcrds, dniqib, to_right[ib])

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
            dvol = dfastbe.kernel.get_km_eroded_volume(bank_km[ib], dviqib, km_bin)
            dv[iq].append(dvol)
            dvol_bank[:, ib] += dvol

        erovol_file = dfastbe.io.config_get_str(
            config, "Erosion", "EroVol" + iq_str, default="erovolQ" + iq_str + ".evo"
        )
        dfastbe.io.log_text("save_erovol", dict={"file": erovol_file}, indent="  ")
        dfastbe.io.write_km_eroded_volumes(km, dvol_bank, outputdir + os.sep + erovol_file)

    dfastbe.io.log_text("=")
    dnav = numpy.zeros(n_banklines)
    dnmax = numpy.zeros(n_banklines)
    dnavflow = numpy.zeros(n_banklines)
    dnavship = numpy.zeros(n_banklines)
    dnaveq = numpy.zeros(n_banklines)
    dnmaxeq = numpy.zeros(n_banklines)
    vol_eq = numpy.zeros((len(km), n_banklines))
    vol_tot = numpy.zeros((len(km), n_banklines))
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

        print(
            "average erosion distance for bank line {} : {:6.2f} m".format(
                ib + 1, dnav[ib]
            )
        )
        print(
            "average erosion distance through flow    : {:6.2f} m".format(dnavflow[ib])
        )
        print(
            "average erosion distance through ships   : {:6.2f} m".format(dnavship[ib])
        )
        print("maximal erosion distance                 : {:6.2f} m".format(dnmax[ib]))
        print("average equilibrium erosion distance     : {:6.2f} m".format(dnaveq[ib]))
        print(
            "maximal equilibrium erosion distance     : {:6.2f} m".format(dnmaxeq[ib])
        )

        xyline_new = dfastbe.support.move_line(bcrds, dn_tot[ib], to_right[ib])
        xyline_new_list.append(xyline_new)
        bankline_new_list.append(shapely.geometry.LineString(xyline_new))
        
        xyline_eq = dfastbe.support.move_line(bcrds, dn_eq[ib], to_right[ib])
        xyline_eq_list.append(xyline_eq)
        bankline_eq_list.append(shapely.geometry.LineString(xyline_eq))

        dvol_eq = dfastbe.kernel.get_km_eroded_volume(bank_km[ib], dv_eq[ib], km_bin)
        vol_eq[:, ib] = dvol_eq
        dvol_tot = dfastbe.kernel.get_km_eroded_volume(bank_km[ib], dv_tot[ib], km_bin)
        vol_tot[:, ib] = dvol_tot
        if ib < n_banklines - 1:
            dfastbe.io.log_text("-")

    # write bank line files
    bankline_new_series = geopandas.geoseries.GeoSeries(bankline_new_list)
    banklines_new = geopandas.geodataframe.GeoDataFrame.from_features(bankline_new_series)
    bankname = dfastbe.io.config_get_str(config, "General", "BankFile", "bankfile")
    bankfile = outputdir + os.sep + bankname + "_new.shp"
    dfastbe.io.log_text("save_banklines", dict={"file": bankfile})
    banklines_new.to_file(bankfile)
    
    bankline_eq_series = geopandas.geoseries.GeoSeries(bankline_eq_list)
    banklines_eq = geopandas.geodataframe.GeoDataFrame.from_features(bankline_eq_series)
    bankfile = outputdir + os.sep + bankname + "_eq.shp"
    dfastbe.io.log_text("save_banklines", dict={"file": bankfile})
    banklines_eq.to_file(bankfile)

    # write eroded volumes per km (total)
    erovol_file = dfastbe.io.config_get_str(
        config, "Erosion", "EroVol", default="erovol.evo"
    )
    print("saving eroded volume in file: {}".format(erovol_file))
    dfastbe.io.write_km_eroded_volumes(km, vol_tot, outputdir + os.sep + erovol_file)

    # write eroded volumes per km (equilibrium)
    erovol_file = dfastbe.io.config_get_str(
        config, "Erosion", "EroVolEqui", default="erovol_eq.evo"
    )
    print("saving eroded volume in file: {}".format(erovol_file))
    dfastbe.io.write_km_eroded_volumes(km, vol_eq, outputdir + os.sep + erovol_file)

    # create various plots
    if plotting:
        dfastbe.io.log_text("=")
        dfastbe.io.log_text("create_figures")
        ifig = 0
        bbox = get_bbox(xykm_numpy)
        
        fig = dfastbe.plotting.plot1_waterdepth_and_banklines(bbox, xykm_numpy, banklines, fn, sim["nnodes"], sim["x_node"], sim["y_node"], sim["h_face"], hfw_max, "x-coordinate [km]", "y-coordinate [km]", "water depth and initial bank lines", "water depth [m]")
        if saveplot:
            ifig = ifig + 1        
            figfile = figdir + os.sep + str(ifig) + "_banklines" + plot_ext
            dfastbe.plotting.savefig(fig, figfile)
        
        fig = dfastbe.plotting.plot2_eroded_distance_and_equilibrium(bbox, xykm_numpy, bank_crds, dn_tot, to_right, dnav, xyline_eq_list, xe, ye, "x-coordinate [km]", "y-coordinate [km]", "eroded distance ({t} year)\n and equilibrium banks".format(t=Teros))
        if saveplot:
            ifig = ifig + 1        
            figfile = figdir + os.sep + str(ifig) + "_erosion_sensitivity" + plot_ext
            dfastbe.plotting.savefig(fig, figfile)
        
        fig = dfastbe.plotting.plot3_eroded_volume_subdivided_1(km, km_step, "river chainage [km]", dv, "eroded volume [m^3]", "eroded volume per {ds} chainage km ({t} years)".format(ds=km_step, t=Teros), "Q{iq}")
        if saveplot:
            ifig = ifig + 1        
            figfile = figdir + os.sep + str(ifig) + "_eroded_volume_per_discharge" + plot_ext
            dfastbe.plotting.savefig(fig, figfile)
        
        fig = dfastbe.plotting.plot3_eroded_volume_subdivided_2(km, km_step, "river chainage [km]", dv, "eroded volume [m^3]", "eroded volume per {ds} chainage km ({t} years)".format(ds=km_step, t=Teros), "Bank {ib}")
        if saveplot:
            ifig = ifig + 1        
            figfile = figdir + os.sep + str(ifig) + "_eroded_volume_per_bank" + plot_ext
            dfastbe.plotting.savefig(fig, figfile)
        
        fig = dfastbe.plotting.plot4_eroded_volume_eq(km, km_step, "river chainage [km]", vol_eq, "eroded volume [m^3]", "eroded volume per {ds} chainage km (equilibrium)".format(ds=km_step))
        if saveplot:
            ifig = ifig + 1        
            figfile = figdir + os.sep + str(ifig) + "_eroded_volume_eq" + plot_ext
            dfastbe.plotting.savefig(fig, figfile)
        
        figlist = dfastbe.plotting.plot5series_waterlevels_per_bank(bank_km, "river chainage [km]", waterlevel, "water level at Q{iq}", "average water level", bankheight, "level of bank", zss, "bank protection level", "elevation", "(water)levels along bank line {ib}", "[m NAP]")
        if saveplot:
            for ib,fig in enumerate(figlist):
                ifig = ifig + 1        
                figfile = figdir + os.sep + str(ifig) + "_levels_bank_" + str(ib+1) + plot_ext
                dfastbe.plotting.savefig(fig, figfile)

        figlist = dfastbe.plotting.plot6series_velocity_per_bank(bank_km, "river chainage [km]", velocity, "velocity at Q{iq}", tauc, chezy[0], rho, g, "critical velocity", "velocity", "velocity along bank line {ib}", "[m/s]")
        if saveplot:
            for ib,fig in enumerate(figlist):
                ifig = ifig + 1        
                figfile = figdir + os.sep + str(ifig) + "_velocity_bank_" + str(ib+1) + plot_ext
                dfastbe.plotting.savefig(fig, figfile)

        fig = dfastbe.plotting.plot7_banktype(bbox, xykm_numpy, bank_crds, banktype, taucls_str, "x-coordinate [km]", "y-coordinate [km]", "bank type")
        if saveplot:
            ifig = ifig + 1        
            figfile = figdir + os.sep + str(ifig) + "_banktype" + plot_ext
            dfastbe.plotting.savefig(fig, figfile)
        
        fig = dfastbe.plotting.plot8_eroded_distance(bank_km, "river chainage [km]", dn_tot, "Bank {ib}", dn_eq, "Bank {ib} (eq)", "eroded distance", "[m]")
        if saveplot:
            ifig = ifig + 1        
            figfile = figdir + os.sep + str(ifig) + "_erodis" + plot_ext
            dfastbe.plotting.savefig(fig, figfile)
    
        if closeplot:
            matplotlib.pyplot.close("all")
        else:
            matplotlib.pyplot.show(block=not gui)

    dfastbe.io.log_text("end_bankerosion")
    timedlogger("-- end analysis --")


def get_bbox(xykm: numpy.ndarray, buffer: float = 0.1) -> Tuple[float, float, float, float]:
    """
    Derive the bounding box from a line.
    
    Arguments
    ---------
    xybm : numpy.ndarray
        An N x M array containing x- and y-coordinates as first two M entries
    buffer : float
        Buffer fraction surrounding the tight bounding box
    
    Results
    -------
    bbox : Tuple[float, float, float, float]
        Tuple bounding box consisting of [min x, min y, max x, max y)
    """
    x = xykm[:,0]
    y = xykm[:,1]
    xmin = x.min()
    ymin = y.min()
    xmax = x.max()
    ymax = y.max()
    d = buffer * max(xmax - xmin, ymax - ymin)
    bbox = (xmin - d, ymin - d, xmax + d, ymax + d)
    return bbox


def derive_topology_arrays(fn: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    n_faces = fn.shape[0]
    max_n_nodes = fn.shape[1]
    tmp = numpy.repeat(fn, 2, axis=1)
    tmp = numpy.concatenate((tmp[:, 1:], tmp[:, :1]), axis=1)
    n_edges = int(tmp.size / 2)
    en = tmp.reshape(n_edges, 2)
    en.sort(axis=1)
    i2 = numpy.argsort(en[:, 1], kind="stable")
    i1 = numpy.argsort(en[i2, 0], kind="stable")
    i12 = i2[i1]
    en = en[i12, :]

    face_nr = numpy.repeat(
        numpy.arange(n_faces).reshape((n_faces, 1)), max_n_nodes, axis=1
    ).reshape((max_n_nodes * n_faces))
    face_nr = face_nr[i12]

    numpy_true = numpy.array([True])
    equal_to_previous = numpy.concatenate(
        (~numpy_true, (numpy.diff(en, axis=0) == 0).all(axis=1))
    )
    new_edge = ~equal_to_previous
    boundary_edge = new_edge & numpy.concatenate((new_edge[1:], numpy_true))
    boundary_edge = boundary_edge[new_edge]

    n_unique_edges = numpy.sum(new_edge)
    edge_nr = numpy.zeros(n_edges, dtype=numpy.int64)
    edge_nr[new_edge] = numpy.arange(n_unique_edges, dtype=numpy.int64)
    edge_nr[equal_to_previous] = edge_nr[
        numpy.concatenate((equal_to_previous[1:], equal_to_previous[:1]))
    ]
    edge_nr_unsorted = numpy.zeros(n_edges, dtype=numpy.int64)
    edge_nr_unsorted[i12] = edge_nr
    fe = edge_nr_unsorted.reshape(fn.shape)
    en = en[new_edge, :]

    ef = -numpy.ones((n_unique_edges, 2), dtype=numpy.int64)
    ef[edge_nr[new_edge], 0] = face_nr[new_edge]
    ef[edge_nr[equal_to_previous], 1] = face_nr[equal_to_previous]
    
    return en, ef, fe, boundary_edge


def debug_file(val: Union[numpy.ndarray, int, float, bool], filename: str) -> None:
    """
    Write a text file for debugging.

    Arguments
    ---------
    val : Union[numpy.ndarray, int, float, bool]
        Value(s) to be written.
    filename : str
        Name of the file to be written.
    """
    with open(filename, "w") as newfile:
        if isinstance(val, numpy.ndarray):
            for i in range(len(val)):
                newfile.write("{:g}\n".format(val[i]))
        elif isinstance(val, int) or isinstance(val, float) or isinstance(val, bool):
            newfile.write("{:g}\n".format(val))
        else:
            newfile.write(
                "Unsupported quantity type ({}) for generating debug files.\n".format(
                    str(type(val))
                )
            )


def timedlogger(label: str) -> None:
    """
    Write message with time information.

    Arguments
    ---------
    label : str
        Message string.
    """
    time, diff = timer()
    logging.info(time + diff + label)


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
            Line = "Line" + str(i)
            if Line in config["Detect"]:
                config = parameter_absolute_path(config, "Detect", Line, rootdir)
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
        config = parameter_absolute_path(config, "Erosion", "ProtectLevel", rootdir)
        config = parameter_absolute_path(config, "Erosion", "Slope", rootdir)
        config = parameter_absolute_path(config, "Erosion", "Reed", rootdir)
        
        NLevel = dfastbe.io.config_get_int(config, "Erosion", "NLevel", default=0)
        for i in range(NLevel):
            istr = str(i + 1)
            config = parameter_absolute_path(config, "Erosion", "SimFile" + istr, rootdir)
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
            Line = "Line" + str(i)
            if Line in config["Detect"]:
                config = parameter_relative_path(config, "Detect", Line, rootdir)
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
        config = parameter_relative_path(config, "Erosion", "ProtectLevel", rootdir)
        config = parameter_relative_path(config, "Erosion", "Slope", rootdir)
        config = parameter_relative_path(config, "Erosion", "Reed", rootdir)
        
        NLevel = dfastbe.io.config_get_int(config, "Erosion", "NLevel", default=0)
        for i in range(NLevel):
            istr = str(i + 1)
            config = parameter_relative_path(config, "Erosion", "SimFile" + istr, rootdir)
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
            val = float(valstr)
        except:
            config[group][key] = dfastbe.io.absolute_path(rootdir, valstr)
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
            val = float(valstr)
        except:
            config[group][key] = dfastbe.io.relative_path(rootdir, valstr)
    return config
