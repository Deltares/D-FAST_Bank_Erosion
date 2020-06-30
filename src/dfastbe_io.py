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

from typing import Union, Dict, List
import numpy
import netCDF4
import configparser
import os
import pandas
import geopandas
import shapely


def read_program_texts(filename: str):
    """Read program dialog texts"""
    text: List[str]
    dict: Dict[str, List[str]]
    
    all_lines = open(filename, "r").read().splitlines()
    dict = {}
    text = []
    key = None
    for line in all_lines:
        rline = line.strip()
        if rline.startswith("[") and rline.endswith("]"):
            if not key is None:
                dict[key] = text
            key = rline[1:-1]
            text = []
        else:
            text.append(line)
    if key in dict.keys():
        raise Exception('Duplicate entry for "{}" in "{}".'.format(key,filename))
    dict[key] = text
    return dict


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


def write_km_eroded_volumes(km, vol, filename):
    with open(filename, "w") as erofile:
        for i in range(len(km)):
            erofile.write("{:.2f} {:.2f}\n".format(km[i], vol[i]))


def read_config(filename: str):
    """Read a configParser object (configuration file).
    
    This function ...
        reads the config file using the standard configParser.
        falls back to a dedicated reader compatible with old waqbank files.
    """
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


def write_config(filename: str, config):
    """Pretty print a configParser object (configuration file) to file.
    
    This function ...
        aligns the equal signs for all keyword/value pairs.
        adds a two space indentation to all keyword lines.
        adds an empty line before the start of a new block.
    """
    sections = config.sections()
    ml = 0
    for s in sections:
        options = config.options(s)
        if len(options) > 0:
            ml = max(ml, max([len(x) for x in options]))

    OPTIONLINE = "  {{:{}s}} = {{}}\n".format(ml)
    with open(filename, "w") as configfile:
        first = True
        for s in sections:
            if first:
                first = False
            else:
                configfile.write("\n")
            configfile.write("[{}]\n".format(s))
            options = config.options(s)
            for o in options:
                configfile.write(OPTIONLINE.format(o, config[s][o]))


def config_get_xykm(config):
    # get km bounds
    kmbounds = config_get_range(config, "General", "Boundaries")
    if kmbounds[0] > kmbounds[1]:
        kmbounds = kmbounds[::-1]
    
    # get the chainage file
    kmfile = config_get_str(config, "General", "RiverKM")
    xykm = read_xyc(kmfile, ncol = 3)
    
    # make sure that chainage is increasing with node index
    if xykm.coords[0][2] > xykm.coords[1][2]:
        xykm = shapely.geometry.asLineString(xykm.coords[::-1])

    # clip the chainage path to the range of chainages of interest
    xykm = clip_chainage_path(xykm, kmbounds)

    return xykm


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


def config_get_bank_guidelines(config):
    # read guiding bank line
    nbank = config_get_int(config, "General", "NBank")
    line = [None] * nbank
    for b in range(nbank):
        bankfile = config["General"]["Line{}".format(b+1)]
        line[b] = read_xyc(bankfile)
    return line
    

def config_get_simfile(config, group, istr):
    simfile = config[group].get("Delft3Dfile"+istr, "")
    simfile = config[group].get("SDSfile"+istr, simfile)
    simfile = config[group].get("simfile"+istr, simfile)
    return simfile


def config_get_range(config, group, key):
    str = config_get_str(config, group, key)
    try:
        val = [float(fstr) for fstr in str.split(":")]
    except:
        raise Exception('No range specified for required keyword "{}" in block "{}".'.format(key, group))
    return val


def config_get_bool(config, group, key, default = None):
    try:
        str = config[group][key].lower()
        val = (str == "yes") or (str == "y") or (str == "true") or (str == "t") or (str == "1")
    except:
        if not default is None:
            val = default
        else:
            raise Exception('No integer value specified for required keyword "{}" in block "{}".'.format(key, group))
    return val


def config_get_int(config, group, key, default = None, positive = False):
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


def config_get_float(config, group, key, default = None):
    try:
        val = float(config[group][key])
    except:
        if not default is None:
            val = default
        else:
            raise Exception('No floating point value specified for required keyword "{}" in block "{}".'.format(key, group))
    return val


def config_get_str(config, group, key, default = None):
    try:
        val = config[group][key]
    except:
        if not default is None:
            val = default
        else:
            raise Exception('No value specified for required keyword "{}" in block "{}".'.format(key, group))
    return val


def config_get_parameter(config, group, key, bank_km, default = None, ext = "", positive = False, valid = None, onefile = False):
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


def read_simdata(filename):
    sim = {}
    # determine file type
    path, name = os.path.split(filename)
    if name[-6:] == "map.nc":
        sim["x_node"] = read_fm_map(filename, "x", location = "node")
        sim["y_node"] = read_fm_map(filename, "y", location = "node")
        FNC = read_fm_map(filename, "face_node_connectivity")
        if FNC.mask.shape == ():
            # all faces have the same number of nodes
            sim["nnodes"] = numpy.ones(FNC.data.shape[0], dtype = numpy.int) * FNC.data.shape[1]
        else:
            # varying number of nodes
            sim["nnodes"] = FNC.mask.shape[1] - FNC.mask.sum(axis=1)
        FNC.data[FNC.mask] = 0
        sim["facenode"] = FNC.data
        sim["zb_location"] = "node"
        sim["zb_val"] = read_fm_map(filename, "altitude", location = "node")
        sim["zw_face"] = read_fm_map(filename, "Water level")
        sim["h_face"] = read_fm_map(filename, "sea_floor_depth_below_sea_surface")
        sim["ucx_face"] = read_fm_map(filename, "sea_water_x_velocity")
        sim["ucy_face"] = read_fm_map(filename, "sea_water_y_velocity")
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


def read_fm_map(filename: str, varname: str, location: str = "face"):
    """Read the last time step of any quantity defined at faces from a D-Flow FM map-file"""
    # open file
    rootgrp = netCDF4.Dataset(filename)
    # locate 2d mesh variable
    mesh2d = rootgrp.get_variables_by_attributes(cf_role = "mesh_topology", topology_dimension = 2)
    if len(mesh2d) != 1:
        raise Exception(
            'Currently only one 2D mesh supported ... this file contains {} 2D meshes.'.format(len(mesh2d))
        )
    meshname = mesh2d[0].name
    start_index = 0
    if varname == "x":
        crdnames = mesh2d[0].getncattr(location+"_coordinates").split()
        for n in crdnames:
            stdname = rootgrp.variables[n].standard_name
            if stdname == "projection_x_coordinate" or stdname == "longitude":
                var = rootgrp.variables[n]
                break
    elif varname == "y":
        crdnames = mesh2d[0].getncattr(location+"_coordinates").split()
        for n in crdnames:
            stdname = rootgrp.variables[n].standard_name
            if stdname == "projection_y_coordinate" or stdname == "latitude":
                var = rootgrp.variables[n]
                break
    elif varname[-12:] == "connectivity":
        varname = mesh2d[0].getncattr(varname)
        var = rootgrp.variables[varname]
        if "start_index" in var.ncattrs():
            start_index = var.getncattr("start_index")
    else:
        var = rootgrp.get_variables_by_attributes(standard_name = varname, mesh = meshname, location = location)
        if len(var) == 0:
            var = rootgrp.get_variables_by_attributes(long_name = varname, mesh = meshname, location = location)
        if len(var) != 1:
            raise Exception(
                'Expected one variable for "{}", but obtained {}.'.format(varname, len(var))
            )
        var = var[0]
    dims = var.dimensions
    if var.get_dims()[0].isunlimited():
        # assume that time dimension is unlimited and is the first dimension
        # slice to obtain last time step
        data = var[-1, :]
    else:
        data = var[...] - start_index
    # close file
    rootgrp.close()
    # return data
    return data


def copy_ugrid(src, meshname: str, dst):
    """Copy UGRID mesh data from source file to destination file"""
    # if src is string, then open the file
    if isinstance(src, str):
        src = netCDF4.Dataset(src)
        srcclose = True
    else:
        srcclose = False

    # locate source mesh
    mesh = src.variables[meshname]

    # if dst is string, then open the file
    if isinstance(dst, str):
        dst = netCDF4.Dataset(dst, "w", format="NETCDF4")
        dstclose = True
    else:
        dstclose = False

    # copy mesh variable
    copy_var(src, meshname, dst)
    atts = [
        "face_node_connectivity",
        "edge_node_connectivity",
        "edge_face_connectivity",
        "face_coordinates",
        "edge_coordinates",
        "node_coordinates",
    ]
    for att in atts:
        try:
            varlist = mesh.getncattr(att).split()
        except:
            varlist = []
        for varname in varlist:
            copy_var(src, varname, dst)

            # check if variable has bounds attribute, if so copy those as well
            var = src.variables[varname]
            atts2 = ["bounds"]
            for att2 in atts2:
                try:
                    varlist2 = var.getncattr(att2).split()
                except:
                    varlist2 = []
                for varname2 in varlist2:
                    copy_var(src, varname2, dst)

    # close files if strings where provided
    if srcclose:
        src.close()
    if dstclose:
        dst.close()


def copy_var(src, varname: str, dst):
    """Copy a single NetCDF variable including attributes from source file to destination file. Create dimensions as necessary."""
    srcvar = src.variables[varname]
    # copy dimensions
    for name in srcvar.dimensions:
        dimension = src.dimensions[name]
        if name not in dst.dimensions.keys():
            dst.createDimension(
                name, (len(dimension) if not dimension.isunlimited() else None)
            )

    # copy variable
    dstvar = dst.createVariable(varname, srcvar.datatype, srcvar.dimensions)

    # copy variable attributes all at once via dictionary
    dstvar.setncatts(srcvar.__dict__)
    dstvar[:] = srcvar[:]


def ugrid_add(dstfile: str, varname: str, ldata, meshname: str, facedim: str):
    """Add a new variable defined at faces to an existing UGRID NetCDF file"""
    # open destination file
    dst = netCDF4.Dataset(dstfile, "a")
    # check if face dimension exists
    dim = dst.dimensions[facedim]
    # add variable and write data
    var = dst.createVariable(varname, "f8", (facedim,))
    var.mesh = meshname
    var.location = "face"
    var[:] = ldata[:]
    # close destination file
    dst.close()


def read_waqua_xyz(filename, cols=(2)):
    """Read data columns from a SIMONA XYZ file (legacy function)"""
    data = numpy.genfromtxt(filename, delimiter=",", skip_header=1, usecols=cols)
    return data


def write_simona_box(filename, rdata, firstm, firstn):
    """Write a SIMONA BOX file (legacy function)"""
    boxfile = open(filename, "w")
    shp = numpy.shape(rdata)
    mmax = shp[0]
    nmax = shp[1]
    boxheader = "      BOX MNMN=({m1:4d},{n1:5d},{m2:5d},{n2:5d}), VARIABLE_VAL=\n"
    nstep = 10
    for j in range(firstn, nmax, nstep):
        k = min(nmax, j + nstep)
        boxfile.write(boxheader.format(m1=firstm + 1, n1=j + 1, m2=mmax, n2=k))
        nvalues = (mmax - firstm) * (k - j)
        boxdata = ("   " + "{:12.3f}" * (k - j) + "\n") * (mmax - firstm)
        values = tuple(rdata[:, j:k].reshape(nvalues))
        boxfile.write(boxdata.format(*values))

    boxfile.close()
