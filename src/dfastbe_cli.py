# coding: utf-8
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

import time
import logging
import argparse
import dfastbe_kernel
import dfastbe_support
import dfastbe_io
import os
import sys
import geopandas
import shapely
import pathlib
import numpy
import matplotlib


def banklines(filename = "config.ini"): #, variable):
    log_text("header_banklines", dict = {"version": dfastbe_kernel.program_version(), "location": "https://github.com/Deltares/D-FAST_Bank_Erosion"})

    # read configuration file
    simplelogger("reading configuration file ...")
    config = dfastbe_io.read_config(filename)

    # check bankdir for output
    # check if simulation file exists


    # read chainage file
    simplelogger("reading chainage file and selecting range of interest ...")
    xykm = dfastbe_io.config_get_xykm(config)    

    # plot chainage line
    ax = geopandas.GeoSeries(xykm).plot(edgecolor = "b")


    # read guiding bank lines
    simplelogger("reading guide lines for bank detection ...")
    max_river_width = 1000
    guide_lines = dfastbe_io.config_get_bank_guidelines(config)
    guide_lines, maxmaxd = dfastbe_support.clip_bank_guidelines(guide_lines, xykm, max_river_width)

    # convert guide lines to bank polygons
    bankareas = dfastbe_support.convert_guide_lines_to_bank_polygons(config, guide_lines)
    
    for ba in bankareas:
        geopandas.GeoSeries(ba).plot(ax = ax, alpha = 0.2, color = "k")

    # get dremove: ommiting coordinates of lines that are more than a certain distance "dremove" from neighbouring points (usually not necesary, so choose large value) (default = 5000 m)


    # get simulationfile
    simfile = dfastbe_io.config_get_simfile(config, "General", "")
    # optional plot water depth

    # get critical water depth used for defining bank line (default = 0.0 m)
    h0 = dfastbe_io.config_get_float(config, "General", "waterdepth", default = 0)

    # read simulation data and drying flooding threshold dh0
    simplelogger("reading simulation data ...")
    sim, dh0 = dfastbe_io.read_simdata(simfile)

    # increase critical water depth h0 by flooding threshold dh0
    h0 = h0 + dh0

    # clip simulation data to boundaries ...
    simplelogger("clipping simulation data ...")
    sim = dfastbe_support.clip_simdata(sim, xykm, maxmaxd)

    
    # derive bank lines (getbanklines)
    simplelogger("identifying bank lines ...")
    banklines = dfastbe_support.get_banklines(sim, h0)


    # clip the set of detected bank lines to the bank areas
    simplelogger("clipping, sorting and connecting bank lines ...")
    bank = [None]*len(bankareas)
    for b, bankarea in enumerate(bankareas):
        print("bank line {}".format(b+1))
        bank[b] = dfastbe_support.clip_sort_connect_bank_lines(banklines, bankarea, xykm)
        
        # add bank lines to plot
        geopandas.GeoSeries(bank[b]).plot(ax = ax, color = "r")

    # save bankfile
    simplelogger("saving clipped bank lines ...")
    bankfile = "banks.shp"
    geopandas.GeoSeries(bank).to_file(bankfile)
    
    # save plot as "banklinedetection"
    simplelogger("saving plot ...")
    bank_line_detection_figure = "banklinedetection.svg"
    ax.figure.savefig(bank_line_detection_figure)

    matplotlib.pyplot.show()
    
    log_text("end_banklines")

    return True


def bankerosion(filename = "config.ini"): #, variable, variable2):
    log_text("header_bankerosion", dict = {"version": dfastbe_kernel.program_version(), "location": "https://github.com/Deltares/D-FAST_Bank_Erosion"})
    rho = 1000 # density of water [kg/m3]
    g = 9.81 # gravititional accelaration [m/s2]

    # read configuration file
    simplelogger("reading configuration file ...")
    config = dfastbe_io.read_config(filename)

    # check bankdir for input
    # check localdir
    # check outputdir
    outputdir = dfastbe_io.config_get_str(config, "General", "outputdir")
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    
    # get simulation time terosion
    Teros = dfastbe_io.config_get_int(config, "General", "Terosion", positive = True)
    print("Total simulation time: {:.2f} year".format(Teros))
    
    # read bank lines
    simplelogger("reading bank lines ...")
    bankfile = "banks.shp"
    banklines = geopandas.read_file(bankfile)
    n_banklines = len(banklines)
    # optional revert direction
    
    # check if simulation file exists
    # read simulation data (getsimdata)
    simplelogger("reading simulation data ...")
    simfile = dfastbe_io.config_get_simfile(config, "General", "")
    sim, dh0 = dfastbe_io.read_simdata(simfile)

    fn = sim["facenode"]
    n_faces = fn.shape[0]
    max_n_nodes = fn.shape[1]
    tmp = numpy.repeat(fn, 2, axis = 1)
    tmp = numpy.concatenate((tmp[:, 1:], tmp[:, :1]), axis = 1)
    n_edges = int(tmp.size/2)
    en = tmp.reshape(n_edges, 2)
    en.sort(axis = 1)
    i2 = numpy.argsort(en[:,1], kind = "stable")
    i1 = numpy.argsort(en[i2,0], kind = "stable")
    i12 = i2[i1]
    en = en[i12,:]

    face_nr = numpy.repeat(numpy.arange(n_faces).reshape((n_faces, 1)), max_n_nodes, axis = 1).reshape((max_n_nodes*n_faces))
    face_nr = face_nr[i12]

    numpy_true = numpy.ones((1), dtype = numpy.bool)
    equal_to_previous = numpy.concatenate((~numpy_true, (numpy.diff(en, axis = 0) == 0).all(axis = 1)))
    new_edge = ~equal_to_previous
    boundary_edge = new_edge & numpy.concatenate((new_edge[1:], numpy_true))
    boundary_edge = boundary_edge[new_edge]

    n_unique_edges = numpy.sum(new_edge)
    edge_nr = numpy.zeros(n_edges, dtype = numpy.int64)
    edge_nr[new_edge] = numpy.arange(n_unique_edges, dtype = numpy.int64)
    edge_nr[equal_to_previous] = edge_nr[numpy.concatenate((equal_to_previous[1:], equal_to_previous[:1]))]
    edge_nr_unsorted = numpy.zeros(n_edges, dtype = numpy.int64)
    edge_nr_unsorted[i12] = edge_nr
    fe = edge_nr_unsorted.reshape(fn.shape)
    en = en[new_edge,:]
    
    ef = -numpy.ones((n_unique_edges,2), dtype = numpy.int64)
    ef[edge_nr[new_edge], 0] = face_nr[new_edge]
    ef[edge_nr[equal_to_previous], 1] = face_nr[equal_to_previous]
    
    # map bank lines to mesh cells
    simplelogger("intersect bank lines with mesh ...")
    bankline_faces = [None] * n_banklines
    xf = sim["x_node"][fn]
    yf = sim["y_node"][fn]
    xe = sim["x_node"][en]
    ye = sim["y_node"][en]
    boundary_edge_nrs = numpy.nonzero(boundary_edge)[0]
    bank_crds = [None] * n_banklines
    bank_idx = [None] * n_banklines
    for ib in range(n_banklines):
        bp = numpy.array(banklines.geometry[ib])
        print("bank line {} ({} nodes)".format(ib+1, len(bp)))
        
        crds, idx = dfastbe_support.intersect_line_mesh(bp, xf, yf, xe, ye, fe, ef, boundary_edge_nrs)
        bank_crds[ib] = crds
        bank_idx[ib] = idx

    # plot water depth

    # optional write banklines.deg for waqview (arcungenerate)

    # read river axis file
    simplelogger("reading river axis file ...")
    river_axis_file = dfastbe_io.config_get_str(config, "General", "riveraxis")
    river_axis = dfastbe_io.read_xyc(river_axis_file)
    river_axis_numpy = numpy.array(river_axis)
    # optional sorting --> see 04_Waal_D3D example
    # check: sum all distances and determine maximum distance ... if maximum > alpha * sum then perform sort
    # Waal OK: 0.0082 ratio max/sum, Waal NotOK: 0.13 - Waal: 2500 points, so even when OK still some 21 times more than 1/2500 = 0.0004
    dist2 = (numpy.diff(river_axis_numpy, axis = 0)**2).sum(axis = 1)
    alpha = dist2.max()/dist2.sum()
    if alpha>0.03:
        print("The river axis needs sorting!!")
        # TODO: do sorting

    # read river km file
    simplelogger("reading chainage file and selecting range of interest ...")
    xykm = dfastbe_io.config_get_xykm(config)
    xykm_numpy = numpy.array(xykm)

    # map km to axis points, further using axis
    simplelogger("selecting river axis range of interest ...")
    river_axis_km = dfastbe_support.project_km_on_line(numpy.array(river_axis.coords), xykm_numpy)
    max_km = numpy.where(river_axis_km == river_axis_km.max())[0]
    min_km = numpy.where(river_axis_km == river_axis_km.min())[0]
    if max_km.max() < min_km.min():
        # reverse river axis
        imin = max_km.max()
        imax = min_km.min()
        river_axis_km = river_axis_km[imin:imax+1][::-1]
        river_axis_numpy = river_axis_numpy[imin:imax+1,:][::-1,:]
        river_axis = shapely.geometry.LineString(river_axis_numpy)
    else:
        imin = min_km.max()
        imax = max_km.min()
        river_axis_km = river_axis_km[imin:imax+1]
        river_axis_numpy = river_axis_numpy[imin:imax+1,:]
        river_axis = shapely.geometry.LineString(river_axis_numpy)

    # get output interval
    km_step = dfastbe_io.config_get_float(config, "General", "outputinterval", 1.0)
    # map to output interval
    km_bin = (river_axis_km.min(), river_axis_km.max(), km_step)
    km = dfastbe_kernel.get_km_bins(km_bin)

    # read fairway file
    simplelogger("reading fairway file ...")
    fairway_file = dfastbe_io.config_get_str(config, "General", "fairway")
    fairway = dfastbe_io.read_xyc(fairway_file)
    fairway_numpy = numpy.array(fairway)
    
    # optional write fairway,mnf file --> no M,N coordinates possible --> single M index or can we speed up such that there is no need to buffer?
    # map fairway to mesh cells
    simplelogger("determine mesh cells for fairway nodes ... ({} nodes)".format(len(fairway_numpy)))
    fairway_index = dfastbe_support.map_line_mesh(fairway_numpy, xf, yf, xe, ye, fe, ef, boundary_edge_nrs)

    # linking bank lines to chainage
    simplelogger("mapping chainage to bank segments ...")
    bank_km = [None] * n_banklines
    for ib, bcrds in enumerate(bank_crds):
        bank_km[ib] = dfastbe_support.project_km_on_line(bcrds, xykm_numpy)

    # distance fairway-bankline (bankfairway)
    simplelogger("computing distance between bank lines and fairway ...")
    distance_fw = [None] * n_banklines
    ifw = [None] * n_banklines
    for ib, bcrds in enumerate(bank_crds):
        distance_fw[ib] = numpy.zeros(len(bcrds))
        ifw[ib] = numpy.zeros(len(bcrds), dtype = numpy.int64)
        ifw_last = None
        for ip, bp in enumerate(bcrds):
            # check only fairway points starting from latest match (in MATLAB code +/-10 from latest match)
            if ifw_last is None:
                ifw[ib][ip] = numpy.argmin(((bp - fairway_numpy)**2).sum(axis = 1))
            else:
                ifw_min = max(0, ifw_last - 10)
                ifw[ib][ip] = ifw_min + numpy.argmin(((bp - fairway_numpy[ifw_min:ifw_last+10, :])**2).sum(axis = 1))
            ifw_last = ifw[ib][ip]
            distance_fw[ib][ip] = ((bp - fairway_numpy[ifw_last])**2).sum()**0.5

    # water level at fairway
    # s1 = sim["zw_face"]
    zfw_ini = [None] * n_banklines
    for ib in range(n_banklines):
        ii = fairway_index[ifw[ib]]
        zfw_ini[ib] = sim["zw_face"][ii]

    # wave reduction s0, s1
    dfw0  = dfastbe_io.config_get_parameter(config, "General", "wave0", bank_km, default = 200, positive = True, onefile = True)
    dfw1  = dfastbe_io.config_get_parameter(config, "General", "wave1", bank_km, default = 150, positive = True, onefile = True)

    # save 1_banklines
    
    # read vship, nship, nwave, draught (tship), shiptype (ship) ... independent of level number
    vship0 = dfastbe_io.config_get_parameter(config, "General", "vship", bank_km, positive = True, onefile = True)
    Nship0 = dfastbe_io.config_get_parameter(config, "General", "Nship", bank_km, positive = True, onefile = True)
    nwave0 = dfastbe_io.config_get_parameter(config, "General", "nwave", bank_km, default = 5, positive = True, onefile = True)
    Tship0 = dfastbe_io.config_get_parameter(config, "General", "Draught", bank_km, positive = True, onefile = True)
    ship0 = dfastbe_io.config_get_parameter(config, "General", "shiptype", bank_km, valid = [1, 2, 3], onefile = True)
    parslope0 = dfastbe_io.config_get_parameter(config, "General", "slope", bank_km, default = 20, positive = True, ext = "slp")
    parreed0 = dfastbe_io.config_get_parameter(config, "General", "reed", bank_km, default = 0, positive = True, ext = "rdd")
    
    # read classes flag (yes: banktype = taucp, no: banktype = tauc) and banktype (taucp: 0-4 ... or ... tauc = critical shear value)
    classes = dfastbe_io.config_get_bool(config, "General", "Classes")
    taucls = numpy.array([1e20, 95, 3.0, 0.95, 0.15])
    taucls_str = ["protected", "vegetation", "good clay", "moderate/bad clay", "sand"]
    if classes:
        banktype = dfastbe_io.config_get_parameter(config, "General", "Banktype", bank_km, default = 0, ext = ".btp")
        tauc = [None] * len(banktype)
        for ib in range(len(banktype)):
            tauc[ib] = taucls[banktype[ib]]
    else:
        tauc = dfastbe_io.config_get_parameter(config, "General", "Banktype", bank_km, default = 0, ext = ".btp")
        thr = (taucls[:-1] + taucls[1:])/2
        banktype = [None] * len(thr)
        for ib in range(len(tauc)):
            bt = numpy.zeros(tauc[ib].size)
            for thr_i in thr:
                bt[tauc[ib] > thr_i] += 1
            banktype[ib] = bt
    # plot bank strength 
    # read bank protectlevel zss
    zss = dfastbe_io.config_get_parameter(config, "General", "protectlevel", bank_km, default = -1000, ext = ".bpl")
    # if zss undefined, set zss equal to zfw_ini - 1
    for ib in range(len(zss)):
        mask = zss[ib] == -999
        zss[ib][mask] = zfw_ini[ib][mask] - 1

    # get pdischarges
    simplelogger("processing level information ...")
    num_levels = dfastbe_io.config_get_int(config, "General", "NLevel")
    ref_level = dfastbe_io.config_get_int(config, "General", "RefLevel") - 1
    simfiles = [""]*num_levels
    pdischarge = [0]*num_levels
    for iq in range(num_levels):
        iq_str = str(iq + 1)
        simfiles[iq] = dfastbe_io.config_get_simfile(config, "General", iq_str)
        pdischarge[iq] = dfastbe_io.config_get_float(config, "General", "PDischarge" + iq_str)

    velocity = [None] * num_levels
    bankheight = [None] * num_levels
    linesize = [None] * num_levels
    dn_flow_tot = [None] * n_banklines
    dn_ship_tot = [None] * n_banklines
    dn_tot = [None] * n_banklines
    dv_tot = [None] * n_banklines
    dn_eq = [None] * n_banklines
    dv_eq = [None] * n_banklines
    for iq in range(num_levels):
        simplelogger("processing level {} of {} ...".format(iq+1, num_levels))
        iq_str = "{}".format(iq+1)

        simplelogger("  reading parameters ...")
        # read vship, nship, nwave, draught, shiptype, slope, reed, fairwaydepth, ... (level specific values)
        vship = dfastbe_io.config_get_parameter(config, "General", "vship" + iq_str, bank_km, default = vship0, positive = True, onefile = True)
        Nship = dfastbe_io.config_get_parameter(config, "General", "Nship" + iq_str, bank_km, default = Nship0, positive = True, onefile = True)
        nwave = dfastbe_io.config_get_parameter(config, "General", "nwave" + iq_str, bank_km, default = nwave0, positive = True, onefile = True)
        Tship = dfastbe_io.config_get_parameter(config, "General", "Draught" + iq_str, bank_km, default = Tship0, positive = True, onefile = True)
        ship = dfastbe_io.config_get_parameter(config, "General", "shiptype" + iq_str, bank_km, default = ship0, valid = [1, 2, 3], onefile = True)

        parslope = dfastbe_io.config_get_parameter(config, "General", "slope" + iq_str, bank_km, default = parslope0, positive = True, ext = "slp")
        parreed = dfastbe_io.config_get_parameter(config, "General", "reed" + iq_str, bank_km, default = parreed0, positive = True, ext = "rdd")
        mu_slope = [None] * n_banklines
        mu_reed = [None] * n_banklines
        for ib in range(n_banklines):
            mus = parslope[ib].copy()
            mus[mus > 0] = 1 / mus[mus > 0]
            mu_slope[ib] = mus
            mu_reed[ib] = 8.5e-4 * parreed[ib]**0.8

        simplelogger("  reading simulation data ...")
        sim, dh0 = dfastbe_io.read_simdata(simfiles[iq])
        fnc = sim["facenode"]

        simplelogger("  computing bank erosion ...")
        velocity[iq] = [None] * n_banklines
        bankheight[iq] = [None] * n_banklines
        linesize[iq] = [None] * n_banklines

        vol = numpy.zeros(len(km))
        for ib, bcrds in enumerate(bank_crds):
            # determine velocity and bankheight along banks ...
            dx = numpy.diff(bcrds[:, 0])
            dy = numpy.diff(bcrds[:, 1])
            linesize[iq][ib] = numpy.sqrt(dx**2 + dy**2)

            idx = bank_idx[ib]
            bank_index = idx[1:]
            velocity[iq][ib] = (sim["ucx_face"][bank_index] * dx + sim["ucy_face"][bank_index] * dy) / linesize[iq][ib]
            # bankheight = maximum water depth per cell
            if sim["zb_location"] == "node":
                bankheight[iq][ib] = sim["zw_face"][bank_index] - sim["zb_val"][fnc[bank_index,:]].min(axis = 1)
            else:
                bankheight[iq][ib] = (sim["zw_face"] - sim["zb_val"])[bank_index]

            # [hfw,zfw,chezy] = fairwaydepth(mnfwfile,sim,nbank,xlines,ylines,x_fw,y_fw,mlim,nlim);
            ii = fairway_index[ifw[ib]]
            hfw = sim["h_face"][ii]
            zfw = sim["zw_face"][ii]
            chez = sim["chz_face"][ii]
            chezy = 0 * chez + chez.mean() # TODO: curious ... MATLAB: chezy{j} = 0*chezy{j}+mchez

            if iq == ref_level:
                dn_eq[ib], dv_eq[ib] = dfastbe_kernel.comp_erosion_eq(bankheight[iq][ib],
                    linesize[iq][ib],
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
                    g)

            displ_tauc = False # TODO: input parameter (True for Delft3D, False otherwise)
            dn, dv, dnship, dnflow = dfastbe_kernel.comp_erosion(velocity[iq][ib],
                bankheight[iq][ib],
                linesize[iq][ib],
                zfw,
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
                chezy,
                zss[ib],
                filter,
                rho,
                g,
                displ_tauc)

            # shift bank lines
            xlines_new, ylines_new = dfastbe_support.move_line(bcrds[:,0], bcrds[:,1], dn)

            if dn_tot[ib] is None:
                dn_flow_tot[ib] = dnflow.copy()
                dn_ship_tot[ib] = dnship.copy()
                dn_tot[ib] = dn.copy()
                dv_tot[ib] = dv.copy()
            else:
                dn_flow_tot[ib] += dnflow
                dn_ship_tot[ib] += dnship
                dn_tot[ib] += dn
                dv_tot[ib] += dv

            # accumulate eroded volumes per km
            vol = dfastbe_kernel.get_km_eroded_volume(bank_km[ib], dv, km_bin, vol)

        erovol_file = dfastbe_io.config_get_str(config, "General", "erovol" + iq_str, default = "erovolQ" + iq_str + ".evo")
        print("  saving eroded volume in file: {}".format(erovol_file))
        dfastbe_io.write_km_eroded_volumes(km, vol, outputdir + os.sep + erovol_file)

    print("=====================================================")
    dnav = numpy.zeros(n_banklines)
    dnmax = numpy.zeros(n_banklines)
    dnavflow = numpy.zeros(n_banklines)
    dnavship = numpy.zeros(n_banklines)
    dnaveq = numpy.zeros(n_banklines)
    dnmaxeq = numpy.zeros(n_banklines)
    vol_eq = numpy.zeros(len(km))
    vol_tot = numpy.zeros(len(km))
    for ib, bcrds in enumerate(bank_crds):
        dnav[ib] = dn_tot[ib].mean()
        dnmax[ib] = dn_tot[ib].max()
        dnavflow[ib]= dn_flow_tot[ib].mean()
        dnavship[ib]= dn_ship_tot[ib].mean()
        dnaveq[ib] = dn_eq[ib].mean()
        dnmaxeq[ib] = dn_eq[ib].max()

        print("average erosion distance for bank line {} : {:6.2f} m".format(ib + 1, dnav[ib]))
        print("average erosion distance through flow    : {:6.2f} m".format(dnavflow[ib]))
        print("average erosion distance through ships   : {:6.2f} m".format(dnavship[ib]))
        print("maximal erosion distance                 : {:6.2f} m".format(dnmax[ib]))
        print("average equilibrium erosion distance     : {:6.2f} m".format(dnaveq[ib]))
        print("maximal equilibrium erosion distance     : {:6.2f} m".format(dnmaxeq[ib]))

        xlines_new, ylines_new = dfastbe_support.move_line(bcrds[:,0], bcrds[:,1], dn_tot[ib])
        xlines_eq, ylines_eq = dfastbe_support.move_line(bcrds[:,0], bcrds[:,1], dn_eq[ib])

        vol_eq = dfastbe_kernel.get_km_eroded_volume(bank_km[ib], dv_eq[ib], km_bin, vol_eq)
        vol_tot = dfastbe_kernel.get_km_eroded_volume(bank_km[ib], dv_tot[ib], km_bin, vol_tot)
        if ib < n_banklines - 1:
            print("-----------------------------------------------------")

    # write bank line files
    # write eroded volumes per km

    # compute and write eroded volumes per km
    erovol_file = dfastbe_io.config_get_str(config, "General", "erovol", default = "erovol.evo")
    print("saving eroded volume in file: {}".format(erovol_file))
    dfastbe_io.write_km_eroded_volumes(km, vol_tot, outputdir + os.sep + erovol_file)

    # create various plots

    log_text("end_bankerosion")
    return True


def log_text(key, file=None, dict={}, repeat=1):
    str = get_program_text(key)
    for r in range(repeat):
        if file is None:
            for s in str:
                logging.info(s.format(**dict))
        else:
            for s in str:
                file.write(s.format(**dict) + "\n")

def simplelogger(str):
    timer()
    logging.info(str)


def timer():
    global LAST_TIME
    new_time = time.time()
    if "LAST_TIME" in globals():
        print(new_time - LAST_TIME)
    LAST_TIME = new_time


def get_program_text(key):
    try:
        str = PROGTEXTS[key]
    except:
        str = ["No message found for " + key]
    return str


def parse_arguments():
    parser = argparse.ArgumentParser(description="D-FAST Bank Erosion.")
    parser.add_argument(
        "-m",
        "--mode",
        default="bankerosion",
        required=False,
        help='execution mode "banklines" or "bankerosion"',
        dest="mode",
    )
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        help="name of configuration file",
        dest="configfile",
    )
    args = parser.parse_args()

    mode = args.__dict__["mode"].lower()
    if mode not in ["banklines", "bankerosion"]:
        raise Exception(
            'Incorrect mode "{}" specified. Should read "banklines" or "bankerosion".'.format(mode)
        )
    
    configfile = args.__dict__["configfile"]
    return mode, configfile


if __name__ == "__main__":
    mode, configfile = parse_arguments()

    logging.basicConfig(level="INFO", format="%(message)s")

    global PROGTEXTS
    progloc = str(pathlib.Path(__file__).parent.absolute())
    PROGTEXTS = dfastbe_io.read_program_texts(progloc + os.path.sep + "messages.NL.ini")

    logging.info(sys.version)
    if mode == "banklines":
        banklines(configfile)
    elif mode == "clean":
        # clean option for localdir - removed from bankerosion
        raise Exception("Clean-up action not yet implemented")
    else:
        bankerosion(configfile)
