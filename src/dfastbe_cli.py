# coding: utf-8

import time
import logging
import argparse
import configparser
import dfastbe_kernel
import dfastbe_io
import os
import sys
import pandas
import geopandas
import shapely
import pathlib
import numpy
import matplotlib


def bankerosion(filename = "config.ini"): #, variable, variable2):
    log_text("header_bankerosion", dict = {"version": dfastbe_kernel.program_version(), "location": "https://github.com/Deltares/D-FAST_Bank_Erosion"})
    rho = 1000 # density of water [kg/m3]
    g = 9.81 # gravititional accelaration [m/s2]

    # read configuration file
    mytimer = timer(None)
    logging.info("reading configuration file ...")
    config = read_config_file(filename)

    # check bankdir for input
    # check localdir
    # check outputdir
    outputdir = get_str(config, "General", "outputdir")
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    
    # get simulation time terosion
    Teros = get_int(config, "General", "Terosion", positive = True)
    print("Total simulation time: {} year".format(Teros))
    
    # read bank lines
    mytimer = timer(mytimer)
    logging.info("reading bank lines ...")
    bankfile = "banks.shp"
    banklines = geopandas.read_file(bankfile)
    n_banklines = len(banklines)
    # optional revert direction
    
    # check if simulation file exists
    # read simulation data (getsimdata)
    mytimer = timer(mytimer)
    logging.info("reading simulation data ...")
    simfile = get_simfile(config, "General", "")
    sim, dh0 = get_simdata(simfile)

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
    mytimer = timer(mytimer)
    logging.info("intersect bank lines with mesh ...")
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
        
        crds, idx = intersect_line_mesh(bp, xf, yf, xe, ye, fe, ef, boundary_edge_nrs)
        bank_crds[ib] = crds
        bank_idx[ib] = idx

    # plot water depth

    # optional write banklines.deg for waqview (arcungenerate)

    # read river axis file
    mytimer = timer(mytimer)
    logging.info("reading river axis file ...")
    river_axis_file = get_str(config, "General", "riveraxis")
    river_axis = read_xyc(river_axis_file)
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
    mytimer = timer(mytimer)
    logging.info("reading chainage file and selecting range of interest ...")
    xykm = get_xykm(config)
    xykm_numpy = numpy.array(xykm)

    # map km to axis points, further using axis
    mytimer = timer(mytimer)
    logging.info("selecting river axis range of interest ...")
    river_axis_km = get_km(numpy.array(river_axis.coords), xykm_numpy)
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
    km_step = get_float(config, "General", "outputinterval", 1.0)
    # map to output interval
    km_bin = (river_axis_km.min(), river_axis_km.max(), km_step)
    km = get_km_bins(km_bin)

    # read fairway file
    mytimer = timer(mytimer)
    logging.info("reading fairway file ...")
    fairway_file = get_str(config, "General", "fairway")
    fairway = read_xyc(fairway_file)
    fairway_numpy = numpy.array(fairway)
    
    # optional write fairway,mnf file --> no M,N coordinates possible --> single M index or can we speed up such that there is no need to buffer?
    # map fairway to mesh cells
    mytimer = timer(mytimer)
    logging.info("determine mesh cells for fairway nodes ... ({} nodes)".format(len(fairway_numpy)))
    fairway_index = map_line_mesh(fairway_numpy, xf, yf, xe, ye, fe, ef, boundary_edge_nrs)

    # linking bank lines to chainage
    mytimer = timer(mytimer)
    logging.info("mapping chainage to bank segments ...")
    bank_km = [None] * n_banklines
    for ib, bcrds in enumerate(bank_crds):
        bank_km[ib] = get_km(bcrds, xykm_numpy)

    # distance fairway-bankline (bankfairway)
    mytimer = timer(mytimer)
    logging.info("computing distance between bank lines and fairway ...")
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
    dfw0  = get_parameter(config, "General", "wave0", bank_km, default = 200, positive = True, onefile = True)
    dfw1  = get_parameter(config, "General", "wave1", bank_km, default = 150, positive = True, onefile = True)

    # save 1_banklines
    
    # read vship, nship, nwave, draught (tship), shiptype (ship) ... independent of level number
    vship0 = get_parameter(config, "General", "vship", bank_km, positive = True, onefile = True)
    Nship0 = get_parameter(config, "General", "Nship", bank_km, positive = True, onefile = True)
    nwave0 = get_parameter(config, "General", "nwave", bank_km, default = 5, positive = True, onefile = True)
    Tship0 = get_parameter(config, "General", "Draught", bank_km, positive = True, onefile = True)
    ship0 = get_parameter(config, "General", "shiptype", bank_km, valid = [1, 2, 3], onefile = True)
    parslope0 = get_parameter(config, "General", "slope", bank_km, default = 20, positive = True, ext = "slp")
    parreed0 = get_parameter(config, "General", "reed", bank_km, default = 0, positive = True, ext = "rdd")
    
    # read classes flag (yes: banktype = taucp, no: banktype = tauc) and banktype (taucp: 0-4 ... or ... tauc = critical shear value)
    classes = get_bool(config, "General", "Classes")
    taucls = numpy.array([1e20, 95, 3.0, 0.95, 0.15])
    taucls_str = ["protected", "vegetation", "good clay", "moderate/bad clay", "sand"]
    if classes:
        banktype = get_parameter(config, "General", "Banktype", bank_km, default = 0, ext = ".btp")
        tauc = [None] * len(banktype)
        for ib in range(len(banktype)):
            tauc[ib] = taucls[banktype[ib]]
    else:
        tauc = get_parameter(config, "General", "Banktype", bank_km, default = 0, ext = ".btp")
        thr = (taucls[:-1] + taucls[1:])/2
        banktype = [None] * len(thr)
        for ib in range(len(tauc)):
            bt = numpy.zeros(tauc[ib].size)
            for thr_i in thr:
                bt[tauc[ib] > thr_i] += 1
            banktype[ib] = bt
    # plot bank strength 
    # read bank protectlevel zss
    zss = get_parameter(config, "General", "protectlevel", bank_km, default = -1000, ext = ".bpl")
    # if zss undefined, set zss equal to zfw_ini - 1
    for ib in range(len(zss)):
        mask = zss[ib] == -999
        zss[ib][mask] = zfw_ini[ib][mask] - 1

    # get pdischarges
    mytimer = timer(mytimer)
    logging.info("processing level information ...")
    num_levels = get_int(config, "General", "NLevel")
    ref_level = get_int(config, "General", "RefLevel") - 1
    simfiles = [""]*num_levels
    pdischarge = [0]*num_levels
    for iq in range(num_levels):
        iq_str = str(iq + 1)
        simfiles[iq] = get_simfile(config, "General", iq_str)
        pdischarge[iq] = get_float(config, "General", "PDischarge" + iq_str)

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
        mytimer = timer(mytimer)
        logging.info("processing level {} of {} ...".format(iq+1, num_levels))
        iq_str = "{}".format(iq+1)

        mytimer = timer(mytimer)
        logging.info("  reading parameters ...")
        # read vship, nship, nwave, draught, shiptype, slope, reed, fairwaydepth, ... (level specific values)
        vship = get_parameter(config, "General", "vship" + iq_str, bank_km, default = vship0, positive = True, onefile = True)
        Nship = get_parameter(config, "General", "Nship" + iq_str, bank_km, default = Nship0, positive = True, onefile = True)
        nwave = get_parameter(config, "General", "nwave" + iq_str, bank_km, default = nwave0, positive = True, onefile = True)
        Tship = get_parameter(config, "General", "Draught" + iq_str, bank_km, default = Tship0, positive = True, onefile = True)
        ship = get_parameter(config, "General", "shiptype" + iq_str, bank_km, default = ship0, valid = [1, 2, 3], onefile = True)

        parslope = get_parameter(config, "General", "slope" + iq_str, bank_km, default = parslope0, positive = True, ext = "slp")
        parreed = get_parameter(config, "General", "reed" + iq_str, bank_km, default = parreed0, positive = True, ext = "rdd")
        mu_slope = [None] * n_banklines
        mu_reed = [None] * n_banklines
        for ib in range(n_banklines):
            mus = parslope[ib].copy()
            mus[mus > 0] = 1 / mus[mus > 0]
            mu_slope[ib] = mus
            mu_reed[ib] = 8.5e-4 * parreed[ib]**0.8

        mytimer = timer(mytimer)
        logging.info("  reading simulation data ...")
        sim, dh0 = get_simdata(simfiles[iq])
        fnc = sim["facenode"]

        mytimer = timer(mytimer)
        logging.info("  computing bank erosion ...")
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

            # compute ship induced wave height h0 at bank
            # TODO: obtain from comp_erosion call?
            # H0 = comp_hw_ship_at_bank(distance_fw, dfw0, dfw1, hfw, shiptype, Tship, vship, g)

            if iq == ref_level:
                dn_eq[ib], dv_eq[ib] = comp_erosion_eq(bankheight[iq][ib],
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
            dn, dv, dnship, dnflow = comp_erosion(velocity[iq][ib],
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
            xlines_new, ylines_new = move_line(bcrds[:,0], bcrds[:,1], dn)

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
            vol = get_km_eroded_volume(bank_km[ib], dv, km_bin, vol)

        erovol_file = get_str(config, "General", "erovol" + iq_str, default = "erovolQ" + iq_str + ".evo")
        print("  saving eroded volume in file: {}".format(erovol_file))
        write_km_eroded_volumes(km, vol, outputdir + os.sep + erovol_file)

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

        xlines_new, ylines_new = move_line(bcrds[:,0], bcrds[:,1], dn_tot[ib])
        xlines_eq, ylines_eq = move_line(bcrds[:,0], bcrds[:,1], dn_eq[ib])

        vol_eq = get_km_eroded_volume(bank_km[ib], dv_eq[ib], km_bin, vol_eq)
        vol_tot = get_km_eroded_volume(bank_km[ib], dv_tot[ib], km_bin, vol_tot)
        if ib < n_banklines - 1:
            print("-----------------------------------------------------")

    # write bank line files
    # write eroded volumes per km

    # compute and write eroded volumes per km
    erovol_file = get_str(config, "General", "erovol", default = "erovol.evo")
    print("saving eroded volume in file: {}".format(erovol_file))
    write_km_eroded_volumes(km, vol_tot, outputdir + os.sep + erovol_file)

    # create various plots

    log_text("end_bankerosion")
    return True


def get_km_bins(km_bin):
    km_step = km_bin[2]
    # km = km_bin[0] + numpy.arange(0, int(round((km_bin[1] - km_bin[0]) / km_bin[2]) + 1)) * km_bin[2] # bin bounds
    km = km_bin[0] + km_bin[2]/2 + numpy.arange(0, int(round((km_bin[1] - km_bin[0]) / km_bin[2]))) * km_bin[2] # bin midpoints

    return km


def get_km_eroded_volume(bank_km, dv, km_bin, vol):
    bank_km_mid = (bank_km[:-1] + bank_km[1:])/2
    bin_idx = numpy.rint((bank_km_mid - km_bin[0] - km_bin[2]/2) / km_bin[2]).astype(numpy.int64)
    dvol = numpy.bincount(bin_idx, weights = dv)
    nbin = len(dvol)
    vol[:nbin] += dvol
    return vol


def write_km_eroded_volumes(km, vol, filename):
    with open(filename, "w") as erofile:
        for i in range(len(km)):
            erofile.write("{:.2f} {:.2f}\n".format(km[i], vol[i]))


def get_km(line_xy, xykm_numpy):
    line_km = numpy.zeros(line_xy.shape[0])
    xykm_numpy2 = xykm_numpy[:,:2]
    last_xykm = xykm_numpy.shape[0] - 1
    for i, rp_numpy in enumerate(line_xy):
        # find closest point to rp on xykm
        imin = numpy.argmin(((rp_numpy - xykm_numpy2)**2).sum(axis = 1))
        p0 = xykm_numpy2[imin]
        dist2 = ((rp_numpy - p0)**2).sum()
        # print(i, rp_numpy, imin, dist2)
        km = xykm_numpy[imin, 2]
        
        # check if closest point is on left link, right link, or node
        if imin > 0:
            p1 = xykm_numpy2[imin - 1]
            alpha = ((p1[0] - p0[0]) + (p1[1] - p0[1])) / ((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
            if alpha > 0 and alpha < 1:
                dist2link = (rp_numpy[0] - p0[0] - alpha * (p1[0] - p0[0]))**2 + (rp_numpy[1] - p0[1] - alpha * (p1[1] - p0[1]))**2
                if dist2link < dist2:
                    dist2 = dist2link
                    km = xykm_numpy[imin, 2] + alpha * (xykm_numpy[imin - 1, 2] - xykm_numpy[imin, 2]) 
        if imin < last_xykm:
            p1 = xykm_numpy2[imin + 1]
            alpha = ((p1[0] - p0[0]) + (p1[1] - p0[1])) / ((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
            if alpha > 0 and alpha < 1:
                dist2link = (rp_numpy[0] - p0[0] - alpha * (p1[0] - p0[0]))**2 + (rp_numpy[1] - p0[1] - alpha * (p1[1] - p0[1]))**2
                if dist2link < dist2:
                    dist2 = dist2link
                    km = xykm_numpy[imin, 2] + alpha * (xykm_numpy[imin + 1, 2] - xykm_numpy[imin, 2]) 
        line_km[i] = km
    return line_km


def intersect_line_mesh(bp, xf, yf, xe, ye, fe, ef, boundary_edge_nrs, d_thresh = 0.001):
    crds = numpy.zeros((len(bp),2))
    idx  = numpy.zeros(len(bp), dtype = numpy.int64)
    l = 0
    #
    for j, bpj in enumerate(bp):
        if j == 0:
            # first bp inside or outside?
            dx = xf - bpj[0]
            dy = yf - bpj[1]
            possible_cells = numpy.nonzero(~((dx < 0).all(axis = 1) | (dx > 0).all(axis = 1) | (dy < 0).all(axis = 1) | (dy > 0).all(axis = 1)))[0]
            if len(possible_cells) == 0:
                # no cells found ... it must be outside
                index = -1
                # print("Starting outside mesh")
            else:
                # one or more possible cells, check whether it's really inside one of them
                # using numpy math might be faster, but since it's should only be for a few points let's using shapely
                pnt = shapely.geometry.Point(bp[0])
                for k in possible_cells:
                    polygon_k = shapely.geometry.Polygon(numpy.concatenate((xf[k:k+1], yf[k:k+1]), axis = 0).T)
                    if polygon_k.contains(pnt):
                        index = k
                        # print("Starting in {}".format(index))
                        break
                else:
                    index = -1
                    # print("Starting outside mesh")
            crds[l,:] = bpj
            idx[l] = index
            l += 1
        else:
            # second or later point
            bpj1 = bp[j - 1]
            prev_b = 0
            prev_pnt = bpj1
            while True:
                if index < 0:
                    edges = boundary_edge_nrs
                else:
                    edges = fe[index]
                X0 = xe[edges,0]
                dX = xe[edges,1] - X0
                Y0 = ye[edges,0]
                dY = ye[edges,1] - Y0
                xi0 = bpj1[0]
                dxi = bpj[0] - xi0
                yi0 = bpj1[1]
                dyi = bpj[1] - yi0
                det = dX * dyi - dY * dxi
                a = (dyi * (xi0 - X0) - dxi * (yi0 - Y0)) / det # along mesh edge
                b = (dY * (xi0 - X0) - dX * (yi0 - Y0)) / det # along bank line
                slices = numpy.nonzero((b > prev_b) & (b <= 1) & (a >= 0) & (a <= 1))[0]
                # print("number of slices: ", len(slices))
                if len(slices) == 0:
                    # rest of segment associated with same face
                    # print("{}: -- no slice --".format(j))
                    if l == crds.shape[0]:
                        crds.resize((2*l,2))
                        idx.resize(2*l)
                    crds[l,:] = bpj
                    idx[l] = index
                    l += 1
                    break
                else:
                    if len(slices) > 1:
                        # crossing multiple edges, when and how?
                        # - crossing at a corner point?
                        # - going out and in again for cell seems unlogical
                        # - going in and out again for boundary seems possible [check: encountered]
                        # print("multiple intersections at ", b[slices])
                        bmin = numpy.amin(b[slices])
                        slices = slices[b[slices] == bmin]
                    # len(slices) == 1
                    edge = edges[slices[0]]
                    faces = ef[edge]
                    prev_b = b[slices[0]]
                    if index < 0:
                        index = faces[0]
                        # print("{}: Moving into {} via edge {} at b = {}".format(j, index, edge, prev_b))
                    else:
                        if faces[0] == index:
                            index = faces[1]
                            # if index < 0:
                            #     print("{}: Moving outside mesh via edge {} at b = {}".format(j, edge, prev_b))
                            # else:
                            #     print("{}: Moving to {} via edge {} at b = {}".format(j, index, edge, prev_b))
                        elif faces[1] == index:
                            index = faces[0]
                            # print("{}: Moving to {} via edge {} at b = {}".format(j, index, edge, prev_b))
                        else:
                            raise Exception("Shouldn't come here .... index {} differs from both faces {} and {} associated with slicing edge {}".format(index, faces[0], faces[1], edge))
                    if l == crds.shape[0]:
                        crds.resize((2*l, 2))
                        idx.resize(2*l)
                    crds[l, :] = bpj1 + prev_b * (bpj - bpj1)
                    idx[l] = index
                    l += 1
    # clip to actual length
    crds = crds[:l, :]
    idx = idx[:l]
    # remove tiny segments (about 35% is less than 1 mm)
    d = numpy.sqrt((numpy.diff(crds, axis = 0)**2).sum(axis = 1))
    mask = numpy.concatenate((numpy.ones((1), dtype = 'bool'), d > d_thresh))
    return crds[mask, :], idx[mask]


def map_line_mesh(bp, xf, yf, xe, ye, fe, ef, boundary_edge_nrs):
    idx  = numpy.zeros(len(bp), dtype = numpy.int64)
    #
    for j, bpj in enumerate(bp):
        if j == 0:
            # first bp inside or outside?
            dx = xf - bpj[0]
            dy = yf - bpj[1]
            possible_cells = numpy.nonzero(~((dx < 0).all(axis = 1) | (dx > 0).all(axis = 1) | (dy < 0).all(axis = 1) | (dy > 0).all(axis = 1)))[0]
            if len(possible_cells) == 0:
                # no cells found ... it must be outside
                index = -1
                # print("Starting outside mesh")
            else:
                # one or more possible cells, check whether it's really inside one of them
                # using numpy math might be faster, but since it's should only be for a few points let's using shapely
                pnt = shapely.geometry.Point(bp[0])
                for k in possible_cells:
                    polygon_k = shapely.geometry.Polygon(numpy.concatenate((xf[k:k+1], yf[k:k+1]), axis = 0).T)
                    if polygon_k.contains(pnt):
                        index = k
                        # print("Starting in {}".format(index))
                        break
                else:
                    index = -1
                    # print("Starting outside mesh")
            idx[j] = index
        else:
            # second or later point
            bpj1 = bp[j - 1]
            prev_b = 0
            prev_pnt = bpj1
            while True:
                if index < 0:
                    edges = boundary_edge_nrs
                else:
                    edges = fe[index]
                X0 = xe[edges,0]
                dX = xe[edges,1] - X0
                Y0 = ye[edges,0]
                dY = ye[edges,1] - Y0
                xi0 = bpj1[0]
                dxi = bpj[0] - xi0
                yi0 = bpj1[1]
                dyi = bpj[1] - yi0
                det = dX * dyi - dY * dxi
                a = (dyi * (xi0 - X0) - dxi * (yi0 - Y0)) / det # along mesh edge
                b = (dY * (xi0 - X0) - dX * (yi0 - Y0)) / det # along bank line
                slices = numpy.nonzero((b > prev_b) & (b <= 1) & (a >= 0) & (a <= 1))[0]
                # print("number of slices: ", len(slices))
                if len(slices) == 0:
                    # rest of segment associated with same face
                    # print("{}: -- no slice --".format(j))
                    idx[j] = index
                    break
                else:
                    if len(slices) > 1:
                        # crossing multiple edges, when and how?
                        # - crossing at a corner point?
                        # - going out and in again for cell seems unlogical
                        # - going in and out again for boundary seems possible [check: encountered]
                        # print("multiple intersections at ", b[slices])
                        bmin = numpy.amin(b[slices])
                        slices = slices[b[slices] == bmin]
                    # len(slices) == 1
                    edge = edges[slices[0]]
                    faces = ef[edge]
                    prev_b = b[slices[0]]
                    if index < 0:
                        index = faces[0]
                        # print("{}: Moving into {} via edge {} at b = {}".format(j, index, edge, prev_b))
                    else:
                        if faces[0] == index:
                            index = faces[1]
                            # if index < 0:
                            #     print("{}: Moving outside mesh via edge {} at b = {}".format(j, edge, prev_b))
                            # else:
                            #     print("{}: Moving to {} via edge {} at b = {}".format(j, index, edge, prev_b))
                        elif faces[1] == index:
                            index = faces[0]
                            # print("{}: Moving to {} via edge {} at b = {}".format(j, index, edge, prev_b))
                        else:
                            raise Exception("Shouldn't come here .... index {} differs from both faces {} and {} associated with slicing edge {}".format(index, faces[0], faces[1], edge))
    masked_idx = numpy.ma.masked_array(idx, mask = (idx == -1))
    return masked_idx


def move_line(xlines, ylines, dn):
    eps = sys.float_info.epsilon
    lenx = len(xlines)

    seg1x0 = xlines[:-2]
    seg1y0 = ylines[:-2]
    seg1dx = xlines[1:-1] - seg1x0
    seg1dy = ylines[1:-1] - seg1y0
    seg1mg = numpy.sqrt(seg1dx**2 + seg1dy**2)
    seg1x0 = seg1x0 + dn[:-1] * seg1dy / seg1mg
    seg1y0 = seg1y0 - dn[:-1] * seg1dx / seg1mg

    seg2x0 = xlines[1:-1]
    seg2y0 = ylines[1:-1]
    seg2dx = xlines[2:] - seg2x0
    seg2dy = ylines[2:] - seg2y0
    seg2mg = numpy.sqrt(seg2dx**2 + seg2dy**2)
    seg2x0 = seg2x0 + dn[1:] * seg2dy / seg2mg
    seg2y0 = seg2y0 - dn[1:] * seg2dx / seg2mg

    dsegx0 = seg2x0 - seg1x0
    dsegy0 = seg2y0 - seg1y0

    # seg1x0 + alpda * seg1dx = seg2x0 + beta * seg2dx
    # seg1y0 + alpda * seg1dy = seg2y0 + beta * seg2dy

    # alpda * seg1dx - beta * seg2dx = seg2x0 - seg1x0 = dsegx0
    # alpda * seg1dy - beta * seg2dy = seg2y0 - seg1y0 = dsegy0

    # [ seg1dx   - seg2dx ] [ alpha ] = [ dsegx0 ]
    # [ seg1dy   - seg2dy ] [ beta  ] = [ dsegy0 ]
    det = numpy.maximum(-seg2dy * seg1dx + seg2dx * seg1dy, eps)

    # [ alpha ] = [- seg2dy  seg2dx ] [ dsegx0 ]
    # [ beta  ] = [- seg1dy  seg1dx ] [ dsegy0 ] / det

    alpha = ( - seg2dy * dsegx0 + seg2dx * dsegy0 ) / det
    beta  = ( - seg1dy * dsegx0 + seg1dx * dsegy0 ) / det
    alpha[alpha<0.5] = numpy.nan
    alpha[beta>0.5] = numpy.nan
    # TODO: add limiter for sharp angles?
    # TODO: make shift "area" conservative in general?

    # lenx - 1 segment midpoints
    # lenx - 2 internal nodes
    # 2 end nodes
    # total: 2 * lenx - 1

    xlines_new = numpy.zeros(2 * lenx - 1) # lenx = 50 -> length = 99 -> 0:98
    xlines_new[0] = seg1x0[0]
    xlines_new[1:-2:2] = seg1x0 + 0.5 * seg1dx
    xlines_new[2:-2:2] = seg1x0 + alpha * seg1dx
    xlines_new[-2] = seg2x0[-1] + 0.5 * seg2dx[-1]
    xlines_new[-1] = seg2x0[-1] + seg2dx[-1]

    ylines_new = numpy.zeros(2 * lenx - 1)
    ylines_new[0] = seg1y0[0]
    ylines_new[1:-2:2] = seg1y0 + 0.5 * seg1dy
    ylines_new[2:-2:2] = seg1y0 + alpha * seg1dy
    ylines_new[-2] = seg2y0[-1] + 0.5 * seg2dy[-1]
    ylines_new[-1] = seg2y0[-1] + seg2dy[-1]

    mask = ~numpy.isnan(xlines_new)
    return xlines_new[mask], ylines_new[mask]


def comp_erosion_eq(bankheight, linesize, zfw_ini, vship, ship_type, Tship, mu_slope, distance_fw, dfw0, dfw1, hfw, zss, g):
    eps = sys.float_info.epsilon

    muslope = edge_mean(mu_slope)
    zssline = edge_mean(zss)
    wlline = edge_mean(zfw_ini) # original water level at fairway

    # ship induced wave height at the beginning of the foreshore
    H0 = comp_hw_ship_at_bank(distance_fw, dfw0, dfw1, hfw, ship_type, Tship, vship, g)
    H0 = numpy.maximum(edge_mean(H0), eps)

    zup = numpy.minimum(bankheight, wlline + 2 * H0)
    zdo = numpy.maximum(wlline - 2 * H0, zssline)
    ht  = numpy.maximum(zup - zdo, 0)
    hs = numpy.maximum(bankheight - wlline + 2 * H0, 0)
    dn_eq = ht / muslope
    dv_eq = (0.5 * ht + hs) * dn_eq * linesize

    return dn_eq, dv_eq


def comp_erosion(velocity, bankheight, linesize, zfw, zfw_ini, tauc, Nship, vship, nwave, ship_type, Tship, Teros, mu_slope, mu_reed, distance_fw, dfw0, dfw1, hfw, chezy, zss, filter, rho, g, displ_tauc):
    eps = sys.float_info.epsilon
    sec_year = 3600 * 24 *365

    # period of ship waves [s]
    T = 0.51 * vship / g
    # [s]
    ts = (T * Nship * nwave)[:-1] # TODO: check for better solution to shorten ts by one ... edge_mean?

    # number of line segments
    xlen = len(velocity)
    # total erosion per segment
    dn = numpy.zeros(xlen)
    # erosion volume per segment
    dv = numpy.zeros(xlen)
    # total wave damping coefficient
    mu_tot = numpy.zeros(xlen)

    taucline = edge_mean(tauc)
    muslope = edge_mean(mu_slope)
    mureed = edge_mean(mu_reed)
    fwd = edge_mean(hfw)
    zssline = edge_mean(zss)
    Cline = edge_mean(chezy)
    wlline = edge_mean(zfw_ini) # original water level at fairway
    z_line = edge_mean(zfw) # water level at fairway

    # Average velocity with values of neighbouring lines
    if filter:
        vel = numpy.concatenate((velocity[:1], 0.5 * velocity[1:-1] + 0.25 * velocity[:-2] + 0.25 * velocity[2:], velocity[-1:]))
    else:
        vel = velocity

    # ship induced wave height at the beginning of the foreshore
    H0 = comp_hw_ship_at_bank(distance_fw, dfw0, dfw1, hfw, ship_type, Tship, vship, g)
    H0 = numpy.maximum(edge_mean(H0), eps)

    # compute erosion parameters for each line part

    # Erosion coefficient of linesegements
    E = 0.2 * numpy.sqrt(taucline) * 1e-6

    # critical velocity along linesegements
    velc = numpy.sqrt(taucline / rho * Cline**2 / g)

    # strength of linesegements
    cE = 1.85e-4 / taucline

    # total wavedamping coefficient
    mu_tot = (muslope / H0) + mureed
    # water level along bank line
    ho_line_ship = numpy.minimum(z_line - zssline, 2 * H0)
    ho_line_flow = numpy.minimum(z_line - zssline, fwd)
    h_line_ship = numpy.maximum(bankheight - z_line + ho_line_ship, 0)
    h_line_flow = numpy.maximum(bankheight - z_line + ho_line_flow, 0)

    # compute displacement due to flow
    crit_ratio = numpy.ones(velc.shape)
    mask = (vel > velc) & (z_line > zssline)
    if displ_tauc:
        # displacement calculated based on critical shear stress
        crit_ratio[mask] = Cline[mask] / taucline[mask]
    else:
        # displacement calculated based on critical flow velocity
        crit_ratio[mask] = (vel[mask] / velc[mask])**2
    dn_flow = E * (crit_ratio - 1) * Teros * sec_year

    # compute displacement due to shipwaves
    mask = ((z_line - 2 * H0) < wlline) & (wlline < (z_line + 0.5 * H0))
    # limit mu -> 0
    
    dn_ship = cE * H0**2 * ts * Teros
    dn_ship[~mask] = 0
    #dn_ship = dn_ship[0] #TODO: this selects only the first value ... correct? MATLAB compErosion: dn_ship=dn_ship(1);

    # compute erosion volume
    mask = (h_line_ship > 0) & (z_line > zssline)
    dv_ship = dn_ship * linesize * h_line_ship
    dv_ship[~mask] = 0
    dn_ship[~mask] = 0

    mask = (h_line_flow > 0) & (z_line > zssline)
    dv_flow = dn_flow * linesize * h_line_flow
    dv_flow[~mask] = 0
    dn_flow[~mask] = 0

    dn = dn_ship + dn_flow
    dv = dv_ship + dv_flow
    
    print("  dv_flow total = ", dv_flow.sum())
    print("  dv_ship total = ", dv_ship.sum())

    return dn, dv, dn_ship, dn_flow


def edge_mean(a):
    return 0.5 * (a[:-1] + a[1:])


def comp_hw_ship_at_bank(distance_fw, dfw0, dfw1, h_input, shiptype, Tship, vship, g):
    h = numpy.copy(h_input)

    a1 = numpy.zeros(len(distance_fw))
    # multiple barge convoy set
    a1[shiptype == 1] = 0.5
    # RHK ship / motorship
    a1[shiptype == 2] = 0.28 * Tship[shiptype == 2]**1.25
    # towboat
    a1[shiptype == 3] = 1

    Froude   = vship / numpy.sqrt(h * g)
    Froude_limit = 0.8
    high_Froude = Froude > Froude_limit
    h[high_Froude] = ((vship[high_Froude] / Froude_limit)**2) / g
    Froude[high_Froude] = Froude_limit

    A = 0.5 * (1 + numpy.cos((distance_fw - dfw1) / (dfw0 - dfw1) * numpy.pi))
    A[distance_fw < dfw1] = 1
    A[distance_fw > dfw0] = 0

    h0  = a1 * h * (distance_fw / h)**(-1/3) * Froude**4 * A
    return h0


def banklines(filename = "config.ini"): #, variable):
    log_text("header_banklines", dict = {"version": dfastbe_kernel.program_version(), "location": "https://github.com/Deltares/D-FAST_Bank_Erosion"})

    # read configuration file
    logging.info("reading configuration file ...")
    config = read_config_file(filename)

    # check bankdir for output
    # check if simulation file exists


    # read chainage file
    logging.info("reading chainage file and selecting range of interest ...")
    xykm = get_xykm(config)    

    # plot chainage line
    ax = geopandas.GeoSeries(xykm).plot(edgecolor = "b")


    # read guiding bank lines
    logging.info("reading guide lines for bank detection ...")
    max_river_width = 1000
    guide_lines, maxmaxd = get_bank_guidelines(config, xykm, max_river_width)

    # convert guide lines to bank polygons
    bankareas = convert_guide_lines_to_bank_polygons(config, guide_lines)
    
    for ba in bankareas:
        geopandas.GeoSeries(ba).plot(ax = ax, alpha = 0.2, color = "k")

    # get dremove: ommiting coordinates of lines that are more than a certain distance "dremove" from neighbouring points (usually not necesary, so choose large value) (default = 5000 m)


    # get simulationfile
    simfile = get_simfile(config, "General", "")
    # optional plot water depth

    # get critical water depth used for defining bank line (default = 0.0 m)
    h0 = get_float(config, "General", "waterdepth", default = 0)

    # read simulation data and drying flooding threshold dh0
    logging.info("reading simulation data ...")
    sim, dh0 = get_simdata(simfile)

    # increase critical water depth h0 by flooding threshold dh0
    h0 = h0 + dh0

    # clip simulation data to boundaries ...
    logging.info("clipping simulation data ...")
    sim = clipsimdata(sim, xykm, maxmaxd)

    
    # derive bank lines (getbanklines)
    logging.info("identifying bank lines ...")
    banklines = get_banklines(sim, h0)


    # clip the set of detected bank lines to the bank areas
    logging.info("clipping, sorting and connecting bank lines ...")
    bank = [None]*len(bankareas)
    for b, bankarea in enumerate(bankareas):
        print("bank line {}".format(b+1))
        bank[b] = clip_sort_connect_bank_lines(banklines, bankarea, xykm)
        
        # add bank lines to plot
        geopandas.GeoSeries(bank[b]).plot(ax = ax, color = "r")

    # save bankfile
    logging.info("saving clipped bank lines ...")
    bankfile = "banks.shp"
    geopandas.GeoSeries(bank).to_file(bankfile)
    
    # save plot as "banklinedetection"
    logging.info("saving plot ...")
    bank_line_detection_figure = "banklinedetection.svg"
    ax.figure.savefig(bank_line_detection_figure)

    matplotlib.pyplot.show()
    
    log_text("end_banklines")

    return True


def clip_sort_connect_bank_lines(banklines, bankarea, xykm):
    clipped_banklines = banklines.intersection(bankarea)[0] # one MultiLineString object
    clipped_banklines = [line for line in clipped_banklines] # convert MultiLineString into list of LineStrings that can be modified later
    # loop over banklines and determine minimum/maximum projected length
    # print("numpy init")
    minlocs = numpy.zeros(len(clipped_banklines))
    maxlocs = numpy.zeros(len(clipped_banklines))
    lengths = numpy.zeros(len(clipped_banklines))
    keep = lengths == 1
    # print("loop {} bank lines".format(len(clipped_banklines)))
    for i, bl in enumerate(clipped_banklines):
        minloc = 1e20
        maxloc = -1
        for j, p in enumerate(bl.coords):
            loc = xykm.project(shapely.geometry.Point(p))
            if loc < minloc:
                minloc = loc
                minj = j
            if loc > maxloc:
                maxloc = loc
                maxj = j
        minlocs[i] = minloc # at minj
        maxlocs[i] = maxloc # at maxj
        if minj < maxj:
            clipped_banklines[i] = shapely.geometry.LineString(bl.coords[minj:maxj+1])
        elif minj > maxj:
            clipped_banklines[i] = shapely.geometry.LineString(bl.coords[maxj:minj+1][::-1])
        else:
            pass # if minj == maxj then minloc == maxloc and thus lengths == 0 and will be removed anyway
        lengths[i] = maxloc - minloc
    # print("select lines by length")
    while True:
        maxl = lengths.max()
        if maxl == 0:
            break
        iarray = numpy.nonzero(lengths == maxl)
        i = iarray[0][0]
        # print("i={}, length={}, minlocs={}, maxlocs={}".format(i, lengths[i], minlocs[i], maxlocs[i]))
        keep[i] = True
        # remove lines that are a subset
        lengths[(minlocs >= minlocs[i]) & (maxlocs <= maxlocs[i])] = 0
        # print("lengths[i] set to {}".format(lengths[i]))
        # if line partially overlaps ... but stick out on the high side
        jarray = numpy.nonzero((minlocs > minlocs[i]) & (minlocs < maxlocs[i]) & (maxlocs > maxlocs[i]))[0]
        if jarray.size > 0:
            for j in jarray:
                bl = clipped_banklines[j]
                kmax = len(bl.coords) - 1
                for k, p in enumerate(bl.coords):
                    if k == kmax:
                        # a line string of a single point would remain
                        lengths[j] = 0
                        break
                    loc = xykm.project(shapely.geometry.Point(p))
                    if loc >= maxlocs[i]:
                        clipped_banklines[j] = shapely.geometry.LineString(bl.coords[k:])
                        minlocs[j] = loc
                        break
        # if line partially overlaps ... but stick out on the low side
        jarray = numpy.nonzero((minlocs < minlocs[i]) & (maxlocs > minlocs[i]) & (maxlocs < maxlocs[i]))[0]
        if jarray.size > 0:
            for j in jarray:
                bl = clipped_banklines[j]
                kmax = len(bl.coords) - 1
                for k, p in zip(range(-1, -kmax, -1), bl.coords[:-1][::-1]):
                    if k == kmax+1:
                        # a line string of a single point would remain
                        lengths[j] = 0
                        break
                    loc = xykm.project(shapely.geometry.Point(p))
                    if loc <= minlocs[i]:
                        clipped_banklines[j] = shapely.geometry.LineString(bl.coords[:k])
                        maxlocs[j] = loc
                        break
    # select banks in order of projected length
    idx = numpy.argsort(minlocs[keep])
    idx2 = numpy.nonzero(keep)[0]
    new_bank_coords = []
    for i in idx2[idx]:
        new_bank_coords.extend(clipped_banklines[i].coords)
    bank = shapely.geometry.LineString(new_bank_coords)
    
    return bank


def read_config_file(filename):
    try:
        config = configparser.ConfigParser( comment_prefixes=("%") )
        with open(filename, "r") as configfile:
            config.read_file(configfile)
    except:
        config = configparser.ConfigParser()
        config["General"] = {}
        all_lines = open(filename, "r").read().splitlines()
        for line in all_lines:
            perc = line.find("%")
            if perc >= 0:
                line = line[:perc]
            data = line.split()
            if len(data) >= 3:
                config["General"][data[0]] = data[2]
    return config


def get_xykm(config):
    # get km bounds
    kmbounds = get_range(config, "General", "Boundaries")
    if kmbounds[0] > kmbounds[1]:
        kmbounds = kmbounds[::-1]
    
    # get the chainage file
    kmfile = get_str(config, "General", "RiverKM")
    xykm = read_xyc(kmfile, ncol = 3)
    
    # make sure that chainage is increasing with node index
    if xykm.coords[0][2] > xykm.coords[1][2]:
        xykm = shapely.geometry.asLineString(xykm.coords[::-1])

    # clip the chainage path to the range of chainages of interest
    xykm = clip_chainage_path(xykm, kmbounds)
    
    return xykm


def get_bank_guidelines(config, xykm = None, max_river_width = 1000):
    # read guiding bank line
    nbank = get_int(config, "General", "NBank")
    line = [None] * nbank
    for b in range(nbank):
        bankfile = config["General"]["Line{}".format(b+1)]
        line[b] = read_xyc(bankfile)
    
    # clip guiding bank lines to area of interest
    # using simplified geometries to speed up identifying the appropriate buffer size
    xy_simplified = xykm.simplify(1)
    maxmaxd = 0
    for b in range(nbank):
        # bank lines will probably extend beyond the xykm reach of interest, so clip them since points (far) beyond the end will ruin the computation of maxd below
        # computing the distance from xykm to the bank line may not detect local maxima in bank distance
        line[b] = line[b].intersection(xykm.buffer(max_river_width, cap_style = 2))
        if line[b].geom_type == "MultiLineString":
            L = []
            for i in range(len(line[b])):
                 L.extend(line[b][i].coords)
            line[b] = shapely.geometry.LineString(L)
        line_simplified = line[b].simplify(1)
        maxd = max([shapely.geometry.Point(c).distance(xy_simplified) for c in line_simplified.coords])
        maxmaxd = max(maxmaxd, maxd+2)
    
    return line, maxmaxd


def convert_guide_lines_to_bank_polygons(config, guide_lines):
    nbank = len(guide_lines)
    # get dlines: distance from pre defined lines used for determining bank lines (default = 50 m)
    dlines = config["General"].get("dlines", None)
    if dlines is None:
        dlines = [50]*nbank
    elif dlines[0] == "[" and dlines[-1] == "]":
        dlines = dlines[1:-1].split(",")
        dlines = [float(d) for d in dlines]
        if not all([d > 0 for d in dlines]):
            raise Exception("keyword DLINES should contain positive values in file: {}".format(filename))
        if len(dlines) != nbank:
            raise Exception("keyword DLINES should contain NBANK values in file: {}".format(filename))
    
    nbank = len(guide_lines)
    bankareas = [None]*nbank
    for b, distance in enumerate(dlines):
        bankareas[b] = guide_lines[b].buffer(distance, cap_style = 2)
        
    return bankareas
    

def clipsimdata(sim, xykm, maxmaxd):
    maxdx = maxmaxd
    xybuffer = xykm.buffer(maxmaxd + maxdx)
    bbox = xybuffer.envelope.exterior
    xmin = bbox.coords[0][0]
    xmax = bbox.coords[1][0]
    ymin = bbox.coords[0][1]
    ymax = bbox.coords[2][1]

    xybprep = shapely.prepared.prep(xybuffer)
    x = sim["x_node"]
    y = sim["y_node"]
    nnodes = x.shape
    keep = (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax)
    for i in range(x.size):
        if keep[i] and not xybprep.contains(shapely.geometry.Point((x[i],y[i]))):
            keep[i] = False

    fnc = sim["facenode"]
    keepface = keep[fnc].all(axis=1)
    renum = numpy.zeros(nnodes, dtype=numpy.int)
    renum[keep]= range(sum(keep))
    sim["facenode"] = renum[fnc[keepface]]

    sim["x_node"] = x[keep]
    sim["y_node"] = y[keep]
    if sim["zb_location"] == "node":
        sim["zb_val"] = sim["zb_val"][keep]
    else:
        sim["zb_val"] = sim["zb_val"][keepface]
    
    sim["nnodes"] = sim["nnodes"][keepface]
    sim["zw_face"] = sim["zw_face"][keepface]
    sim["h_face"] = sim["h_face"][keepface]
    sim["ucx_face"] = sim["ucx_face"][keepface]
    sim["ucy_face"] = sim["ucy_face"][keepface]
    sim["chz_face"] = sim["chz_face"][keepface]

    return sim


def clip_chainage_path(xykm, kmbounds):
    start_i = None
    end_i = None
    for i,c in enumerate(xykm.coords):
        if start_i is None:
            if c[2] >= kmbounds[0]:
                start_i = i
        if c[2] >= kmbounds[1]:
            end_i = i
            break
    
    if start_i is None:
        raise Exception('Start chainage {} is larger than the maximum chainage {} listed in "{}"'.format(kmbounds[0], xykm.coords[-1][2], kmfile))
    elif start_i == 0:
        # lower bound (potentially) clipped to available reach
        if xykm.coords[0][2] - kmbounds[0] > 0.1:
            raise Exception('Start chainage {} is smaller than the minimum chainage {} listed in "{}"'.format(kmbounds[0], xykm.coords[0][2], kmfile))
        x0 = None
    else:
        alpha = (kmbounds[0] - xykm.coords[start_i - 1][2])/(xykm.coords[start_i][2] - xykm.coords[start_i - 1][2])
        x0 = tuple((c1 + alpha * (c2-c1)) for c1,c2 in zip(xykm.coords[start_i - 1],xykm.coords[start_i]))
        if alpha > 0.9:
            # value close to first node (start_i), so let's skip that one
            start_i = start_i + 1
    
    if end_i is None:
        if kmbounds[1] - xykm.coords[-1][2] > 0.1:
            raise Exception('End chainage {} is larger than the maximum chainage {} listed in "{}"'.format(kmbounds[1], xykm.coords[-1][2], kmfile))
        # else kmbounds[1] matches chainage of last point
        if x0 is None:
            # whole range available selected
            pass
        else:
            xykm = shapely.geometry.LineString([x0] + xykm.coords[start_i:])
    elif end_i == 0:
        raise Exception('End chainage {} is smaller than the minimum chainage {} listed in "{}"'.format(kmbounds[1], xykm.coords[0][2], kmfile))
    else:
        alpha = (kmbounds[1] - xykm.coords[end_i - 1][2])/(xykm.coords[end_i][2] - xykm.coords[end_i - 1][2])
        x1 = tuple((c1 + alpha * (c2-c1)) for c1,c2 in zip(xykm.coords[end_i - 1],xykm.coords[end_i]))
        if alpha < 0.1:
            # value close to previous point (end_i - 1), so let's skip that one
            end_i = end_i - 1
        if x0 is None:
            xykm = shapely.geometry.LineString(xykm.coords[:end_i] + [x1])
        else:
            xykm = shapely.geometry.LineString([x0] + xykm.coords[start_i:end_i] + [x1])
    return xykm


def get_simfile(config, group, istr):
    simfile = config[group].get("Delft3Dfile"+istr, "")
    simfile = config[group].get("SDSfile"+istr, simfile)
    simfile = config[group].get("simfile"+istr, simfile)
    return simfile


def get_range(config, group, key):
    str = get_str(config, group, key)
    try:
        val = [float(fstr) for fstr in str.split(":")]
    except:
        raise Exception('No range specified for required keyword "{}" in block "{}".'.format(key, group))
    return val


def get_bool(config, group, key, default = None):
    try:
        str = config[group][key].lower()
        val = (str == "yes") or (str == "y") or (str == "true") or (str == "t") or (str == "1")
    except:
        if not default is None:
            val = default
        else:
            raise Exception('No integer value specified for required keyword "{}" in block "{}".'.format(key, group))
    return val


def get_int(config, group, key, default = None, positive = False):
    try:
        val = int(config[group][key])
        if positive:
            if val <= 0:
                raise Exception('Value for "{}" in block "{}" must be positive, not {}.'.format(key, group, val))
    except:
        if not default is None:
            val = default
        else:
            raise Exception('No integer value specified for required keyword "{}" in block "{}".'.format(key, group))
    return val


def get_float(config, group, key, default = None):
    try:
        val = float(config[group][key])
    except:
        if not default is None:
            val = default
        else:
            raise Exception('No floating point value specified for required keyword "{}" in block "{}".'.format(key, group))
    return val


def get_str(config, group, key, default = None):
    try:
        val = config[group][key]
    except:
        if not default is None:
            val = default
        else:
            raise Exception('No value specified for required keyword "{}" in block "{}".'.format(key, group))
    return val


def get_parameter(config, group, key, bank_km, default = None, ext = "", positive = False, valid = None, onefile = False):
    try:
        filename = config[group][key]
        use_default = False
    except:
        if default is None:
            raise Exception('No value specified for required keyword "{}" in block "{}".'.format(key, group))
        use_default = True

    # if val is value then use that value globally
    parfield = [None] * len(bank_km)
    try:
        if use_default:
            if default.__class__() == []:
                return default
            rval = default
        else:
            rval = float(filename)
            if positive:
                if rval < 0:
                    raise Exception('Value of "{}" should be positive, not {}.'.format(key, rval))
            if not valid is None:
                if valid.count(rval) == 0:
                    raise Exception('Value of "{}" should be in {}, not {}.'.format(key, valid, rval))
        for ib, bkm in enumerate(bank_km):
            parfield[ib] = numpy.zeros(len(bkm)) + rval
    except:
        if onefile:
            km_thr, val = get_kmval(filename, key, positive, valid)
        for ib, bkm in enumerate(bank_km):
            if not onefile:
                filename_i = filename + "_{}".format(ib+1) + ext
                km_thr, val = get_kmval(filename_i, key, positive, valid)
            if km_thr is None:
                parfield[ib] = numpy.zeros(len(bkm)) + val[0]
            else:
                idx = numpy.zeros(len(bkm), dtype = numpy.int64)
                for thr in km_thr:
                    idx[bkm >= thr] += 1
                parfield[ib] = val[idx]
            #print("Min/max of data: ", parfield[ib].min(), parfield[ib].max())
    return parfield


def get_kmval(filename, key, positive, valid):
    #print("Trying to read: ",filename)
    P = pandas.read_csv(filename, names = ["Chainage", "Val"], skipinitialspace = True, delim_whitespace = True)
    nPnts = len(P.Chainage)
    km = P.Chainage.to_numpy()
    val = P.Val.to_numpy()
    if len(km.shape) == 0:
        km = km[None]
        val = val[None]
    if positive:
        if (val < 0).any():
            raise Exception('Values of "{}" in "{}" should be positive. Negative value read for chainage(s): {}'.format(key, filename, km[val < 0]))
    if not valid is None:
        isvalid = False
        for valid_val in valid:
            isvalid = isvalid | (val == valid_val)
        if not isvalid.all():
            raise Exception('Value of "{}" in "{}" should be in {}. Invalid value read for chainage(s): {}.'.format(key, filename, km[~isvalid]))
    if len(km) == 1:
        km_thr = None
    else:
        if not (km[1:] > km[:-1]).all():
            raise Exception('Chainage values are not increasing in the file "{}" read for "{}".'.format(filename, key))
        km_thr = (km[:-1] + km[1:])/2
    return km_thr, val


def read_xyc(filename, ncol = 2):
    fileroot, ext = os.path.splitext(filename)
    if ext.lower() == ".xyc":
        if ncol == 3:
            colnames = ["Val","X","Y"]
        else:
            colnames = ["X","Y"]
        P = pandas.read_csv(filename, names = colnames, skipinitialspace = True, delim_whitespace = True)
        nPnts = len(P.X)
        x = P.X.to_numpy().reshape((nPnts,1))
        y = P.Y.to_numpy().reshape((nPnts,1))
        if ncol == 3:
            z = P.Val.to_numpy().reshape((nPnts,1))
            LC = numpy.concatenate((x, y, z), axis = 1)
        else:
            LC = numpy.concatenate((x, y), axis = 1)
        L = shapely.geometry.asLineString(LC)
    else:
        GEO = geopandas.read_file(filename)["geometry"]
        L = [object for object in GEO]
    return L


def get_simdata(filename):
    sim = {}
    # determine file type
    path, name = os.path.split(filename)
    if name == "dummy":
        sim["x_node"] = numpy.array([0.0, 2.0, 2.0, 0.0, 1.0])
        sim["y_node"] = numpy.array([0.0, 0.0, 2.0, 2.0, 3.0])
        sim["facenode"] = numpy.array([[0, 1, 2, 3], [2, 3, 4, 0]])
        sim["nnodes"] = numpy.array([4, 3], dtype = numpy.int)
        sim["zb_location"] = "node"
        sim["zb_val"] = numpy.array([0.0, 1.0, 1.0, 0.0, 1.0])
        sim["zw_face"] = numpy.array([0.5, 0.5])
        sim["h_face"] = numpy.array([1, 2])
        sim["ucx_face"] = numpy.array([1, 0])
        sim["ucy_face"] = numpy.array([0, 1])
        sim["chz_face"] = numpy.array([60, 60])
        #sim["kfu_edge"] = [1, 1, 0, 0, 0]
        #sim["velo_edge"] = [1, 1, 0, 0, 0]
    elif name[-6:] == "map.nc":
        sim["x_node"] = dfastbe_io.read_fm_map(filename, "x", location = "node")
        sim["y_node"] = dfastbe_io.read_fm_map(filename, "y", location = "node")
        FNC = dfastbe_io.read_fm_map(filename, "face_node_connectivity")
        if FNC.mask.shape == ():
            # all faces have the same number of nodes
            sim["nnodes"] = numpy.ones(FNC.data.shape[0], dtype = numpy.int) * FNC.data.shape[1]
        else:
            # varying number of nodes
            sim["nnodes"] = FNC.mask.shape[1] - FNC.mask.sum(axis=1)
        FNC.data[FNC.mask] = 0
        sim["facenode"] = FNC.data
        sim["zb_location"] = "node"
        sim["zb_val"] = dfastbe_io.read_fm_map(filename, "altitude", location = "node")
        sim["zw_face"] = dfastbe_io.read_fm_map(filename, "Water level")
        sim["h_face"] = dfastbe_io.read_fm_map(filename, "sea_floor_depth_below_sea_surface")
        sim["ucx_face"] = dfastbe_io.read_fm_map(filename, "sea_water_x_velocity")
        sim["ucy_face"] = dfastbe_io.read_fm_map(filename, "sea_water_y_velocity")
        sim["chz_face"] = 0 * sim["ucy_face"] + 60 # TODO: read from file ... if written ...
        dh0 = 0.1 #TODO: should be derived from netCDF file ... for now WAQUA setting
    elif name[:3] == "SDS":
        dh0 = 0.1
        raise Exception('WAQUA output files not yet supported. Unable to process "{}"'.format(name))
    elif name[:4] == "trim":
        dh0 = 0.01
        raise Exception('Delft3D map files not yet supported. Unable to process "{}"'.format(name))
    else:
        raise Exception('Unable to determine file type for "{}"'.format(name))
    return sim, dh0


def get_banklines(sim, h0):
    FNC = sim["facenode"]
    NNODES = sim["nnodes"]
    max_nnodes = FNC.shape[1]
    X = sim["x_node"][FNC]
    Y = sim["y_node"][FNC]
    ZB = sim["zb_val"][FNC]
    ZW = sim["zw_face"]
    H_face = sim["h_face"]
    WET_face = H_face > h0
    #
    nnodes_total = len(sim["x_node"])
    try:
        mask = ~FNC.mask
        nonmasked = sum(mask.reshape(FNC.size))
        FNCm = FNC[mask]
        ZWm = numpy.repeat(ZW, max_nnodes)[mask]
    except:
        mask = numpy.repeat(True, FNC.size)
        nonmasked = FNC.size
        FNCm = FNC.reshape(nonmasked)
        ZWm = numpy.repeat(ZW, max_nnodes).reshape(nonmasked)
    ZW_node = numpy.bincount(FNCm, weights = ZWm, minlength = nnodes_total)
    NVal = numpy.bincount(FNCm, weights = numpy.ones(nonmasked), minlength = nnodes_total)
    ZW_node = ZW_node / numpy.maximum(NVal,1)
    ZW_node[NVal == 0] = sim["zb_val"][NVal == 0]
    #
    H_node = ZW_node[FNC] - ZB
    WET_node = H_node > h0
    NWET = WET_node.sum(axis = 1)
    MASK = NWET.mask.size > 1
    #
    nfaces = len(FNC)
    Lines = [None]*nfaces
    frac = 0
    for i in range(nfaces):
        if i >= frac*(nfaces-1)/10:
            print("{}%".format(int(frac*10)))
            frac = frac+1
        nnodes = NNODES[i]
        nwet = NWET[i]
        if (MASK and nwet.mask) or nwet == 0 or nwet == nnodes:
            # all dry or all wet
            pass
        else:
            # some nodes dry and some nodes wet: determine the line
            if nnodes == 3:
                Lines[i] = tri_to_line(X[i], Y[i], WET_node[i], H_node[i], h0)
            else:
                Lines[i] = poly_to_line(nnodes, X[i], Y[i], WET_node[i], H_node[i], h0)
    Lines = [line for line in Lines if not line is None and not line.is_empty]
    multi_line = shapely.ops.cascaded_union(Lines)
    merged_line = shapely.ops.linemerge(multi_line)
    return geopandas.GeoSeries(merged_line)


def poly_to_line(nnodes, x, y, wet_node, h_node, h0):
    Lines = [None] * (nnodes-2)
    for i in range(nnodes-2):
        iv = [0, i+1, i+2]
        nwet = sum(wet_node[iv])
        if nwet == 1 or nwet == 2:
            # print("x: ",x[iv]," y: ",y[iv], " w: ", wet_node[iv], " d: ", h_node[iv])
            Lines[i] = tri_to_line(x[iv], y[iv], wet_node[iv], h_node[iv], h0)
    Lines = [line for line in Lines if not line is None]
    if len(Lines) == 0:
        return None
    else:
        multi_line = shapely.geometry.MultiLineString(Lines)
        merged_line = shapely.ops.linemerge(multi_line)
        return merged_line


def tri_to_line(x, y, wet_node, h_node, h0):
    if wet_node[0] and wet_node[1]:
        A = 0
        B = 2
        C = 1
        D = 2
    elif wet_node[0] and wet_node[2]:
        A = 0
        B = 1
        C = 2
        D = 1
    elif wet_node[0]:
        A = 0
        B = 1
        C = 0
        D = 2
    elif wet_node[1] and wet_node[2]:
        A = 2
        B = 0
        C = 1
        D = 0
    elif wet_node[1]:
        A = 1
        B = 0
        C = 1
        D = 2
    else: #wet_node[2]
        A = 2
        B = 0
        C = 2
        D = 1
    facAB = (h_node[A] - h0) / (h_node[A] - h_node[B]) # large facAB -> close to B
    xl = x[A] + facAB * (x[B]-x[A])
    yl = y[A] + facAB * (y[B]-y[A])
    facCD = (h_node[C] - h0) / (h_node[C] - h_node[D]) # large facCD -> close to D
    xr = x[C] + facCD * (x[D]-x[C])
    yr = y[C] + facCD * (y[D]-y[C])
    if xl == xr and yl == yr:
        Lines = None
    else:
        Lines = shapely.geometry.asLineString([[xl, yl], [xr, yr]])
    return Lines


def log_text(key, file=None, dict={}, repeat=1):
    str = program_texts(key)
    for r in range(repeat):
        if file is None:
            for s in str:
                logging.info(s.format(**dict))
        else:
            for s in str:
                file.write(s.format(**dict) + "\n")


def timer(last_time):
    new_time = time.time()
    if not last_time is None:
        print(new_time - last_time)
    return new_time


def program_texts(key):
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
