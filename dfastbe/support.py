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

from typing import Dict, List, Tuple
from dfastbe.io import SimulationObject

import numpy
import shapely
import geopandas
import sys


def project_km_on_line(
    line_xy: numpy.ndarray, xykm_numpy: numpy.ndarray
) -> numpy.ndarray:
    """
    Project chainage values onto another line.

    The chainage values are giving along a line. For each node of the line the
    closest chainage value is located and subsequently the exact chainage is
    obtained by means of interpolation between that value and the neighbouring
    values.

    Arguments
    ---------
    line_xy : numpy.ndarray
        Array containing the x,y coordinates of a line.
    xykm_numpy : numpy.ndarray
        Array containing the x,y,chainage data.

    Results
    -------
    line_km : numpy.ndarray
        Array containing the chainage for every coordinate specified in line_xy.
    """
    line_km = numpy.zeros(line_xy.shape[0])
    xykm_numpy2 = xykm_numpy[:, :2]
    last_xykm = xykm_numpy.shape[0] - 1
    for i, rp_numpy in enumerate(line_xy):
        # find closest point to rp on xykm
        imin = numpy.argmin(((rp_numpy - xykm_numpy2) ** 2).sum(axis=1))
        p0 = xykm_numpy2[imin]
        dist2 = ((rp_numpy - p0) ** 2).sum()
        # print(i, rp_numpy, imin, dist2)
        km = xykm_numpy[imin, 2]

        # check if closest point is on left link, right link, or node
        if imin > 0:
            p1 = xykm_numpy2[imin - 1]
            alpha = ((p1[0] - p0[0]) + (p1[1] - p0[1])) / (
                (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2
            )
            if alpha > 0 and alpha < 1:
                dist2link = (rp_numpy[0] - p0[0] - alpha * (p1[0] - p0[0])) ** 2 + (
                    rp_numpy[1] - p0[1] - alpha * (p1[1] - p0[1])
                ) ** 2
                if dist2link < dist2:
                    dist2 = dist2link
                    km = xykm_numpy[imin, 2] + alpha * (
                        xykm_numpy[imin - 1, 2] - xykm_numpy[imin, 2]
                    )
        if imin < last_xykm:
            p1 = xykm_numpy2[imin + 1]
            alpha = ((p1[0] - p0[0]) + (p1[1] - p0[1])) / (
                (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2
            )
            if alpha > 0 and alpha < 1:
                dist2link = (rp_numpy[0] - p0[0] - alpha * (p1[0] - p0[0])) ** 2 + (
                    rp_numpy[1] - p0[1] - alpha * (p1[1] - p0[1])
                ) ** 2
                if dist2link < dist2:
                    dist2 = dist2link
                    km = xykm_numpy[imin, 2] + alpha * (
                        xykm_numpy[imin + 1, 2] - xykm_numpy[imin, 2]
                    )
        line_km[i] = km
    return line_km


def intersect_line_mesh(bp: numpy.ndarray, xf: numpy.ndarray, yf: numpy.ndarray, xe: numpy.ndarray, ye: numpy.ndarray, fe: numpy.ndarray, ef: numpy.ndarray, boundary_edge_nrs: numpy.ndarray, d_thresh: float = 0.001) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Intersect a (bank) line with an unstructured mesh and return the intersection coordinates and mesh face indices.

    Arguments
    ---------
    bp : numpy.ndarray
        Array containing the x,y-coordinates of the (bank) line.
    xf : numpy.ndarray
        Array containing the x-coordinates of the corner points of each mesh face.
    yf : numpy.ndarray
        Array containing the y-coordinates of the corner points of each mesh face.
    xe : numpy.ndarray
        Array containing the x-coordinates of the end points of each mesh edge.
    ye : numpy.ndarray
        Array containing the y-coordinates of the end points of each mesh edge.
    ef : numpy.ndarray
        Array containg the mesh edge-face connectivity.
    boundary_edge_nrs : numpy.ndarray
        Array containing the indices of the domain boundary edges.
    d_thresh : float
        Distance threshold.
        
    Returns
    -------
    crds : numpy.ndarray
        Array containing the x,y-coordinates of the (bank) line intersected by the mesh.
    idx : numpy.ndarray
        Array containing the indices of the mesh faces in which each line segment of crds is located.
    """
    crds = numpy.zeros((len(bp), 2))
    idx = numpy.zeros(len(bp), dtype=numpy.int64)
    l = 0
    #
    for j, bpj in enumerate(bp):
        if j == 0:
            # first bp inside or outside?
            dx = xf - bpj[0]
            dy = yf - bpj[1]
            possible_cells = numpy.nonzero(
                ~(
                    (dx < 0).all(axis=1)
                    | (dx > 0).all(axis=1)
                    | (dy < 0).all(axis=1)
                    | (dy > 0).all(axis=1)
                )
            )[0]
            if len(possible_cells) == 0:
                # no cells found ... it must be outside
                index = -1
                # print("Starting outside mesh")
            else:
                # one or more possible cells, check whether it's really inside one of them
                # using numpy math might be faster, but since it's should only be for a few points let's using shapely
                pnt = shapely.geometry.Point(bp[0])
                for k in possible_cells:
                    polygon_k = shapely.geometry.Polygon(
                        numpy.concatenate((xf[k : k + 1], yf[k : k + 1]), axis=0).T
                    )
                    if polygon_k.contains(pnt):
                        index = k
                        # print("Starting in {}".format(index))
                        break
                else:
                    index = -1
                    # print("Starting outside mesh")
            crds[l, :] = bpj
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
                X0 = xe[edges, 0]
                dX = xe[edges, 1] - X0
                Y0 = ye[edges, 0]
                dY = ye[edges, 1] - Y0
                xi0 = bpj1[0]
                dxi = bpj[0] - xi0
                yi0 = bpj1[1]
                dyi = bpj[1] - yi0
                det = dX * dyi - dY * dxi
                a = (dyi * (xi0 - X0) - dxi * (yi0 - Y0)) / det  # along mesh edge
                b = (dY * (xi0 - X0) - dX * (yi0 - Y0)) / det  # along bank line
                slices = numpy.nonzero((b > prev_b) & (b <= 1) & (a >= 0) & (a <= 1))[0]
                # print("number of slices: ", len(slices))
                if len(slices) == 0:
                    # rest of segment associated with same face
                    # print("{}: -- no slice --".format(j))
                    if l == crds.shape[0]:
                        crds.resize((2 * l, 2))
                        idx.resize(2 * l)
                    crds[l, :] = bpj
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
                            raise Exception(
                                "Shouldn't come here .... index {} differs from both faces {} and {} associated with slicing edge {}".format(
                                    index, faces[0], faces[1], edge
                                )
                            )
                    if l == crds.shape[0]:
                        crds.resize((2 * l, 2))
                        idx.resize(2 * l)
                    crds[l, :] = bpj1 + prev_b * (bpj - bpj1)
                    idx[l] = index
                    l += 1
    # clip to actual length
    crds = crds[:l, :]
    idx = idx[:l]
    # remove tiny segments (about 35% is less than 1 mm)
    d = numpy.sqrt((numpy.diff(crds, axis=0) ** 2).sum(axis=1))
    mask = numpy.concatenate((numpy.ones((1), dtype="bool"), d > d_thresh))
    return crds[mask, :], idx[mask]


def map_line_mesh(bp: numpy.ndarray, xf: numpy.ndarray, yf: numpy.ndarray, xe: numpy.ndarray, ye: numpy.ndarray, fe: numpy.ndarray, ef: numpy.ndarray, boundary_edge_nrs: numpy.ndarray) -> numpy.ndarray:
    """
    Determine for each point of a line in which mesh face it is located.

    Arguments
    ---------
    bp : numpy.ndarray
        Array containing the x,y-coordinates of the (bank) line.
    xf : numpy.ndarray
        Array containing the x-coordinates of the corner points of each mesh face.
    yf : numpy.ndarray
        Array containing the y-coordinates of the corner points of each mesh face.
    xe : numpy.ndarray
        Array containing the x-coordinates of the end points of each mesh edge.
    ye : numpy.ndarray
        Array containing the y-coordinates of the end points of each mesh edge.
    fe : numpy.ndarray
        Array containg the mesh face-edge connectivity.
    ef : numpy.ndarray
        Array containg the mesh edge-face connectivity.
    boundary_edge_nrs : numpy.ndarray
        Array containing the indices of the domain boundary edges.
    
    Returns
    -------
    masked_idx : numpy.ndarray
        Array containing the indices of the mesh faces in which each point of bp is located.
    """
    idx = numpy.zeros(len(bp), dtype=numpy.int64)
    #
    for j, bpj in enumerate(bp):
        if j == 0:
            # first bp inside or outside?
            dx = xf - bpj[0]
            dy = yf - bpj[1]
            possible_cells = numpy.nonzero(
                ~(
                    (dx < 0).all(axis=1)
                    | (dx > 0).all(axis=1)
                    | (dy < 0).all(axis=1)
                    | (dy > 0).all(axis=1)
                )
            )[0]
            if len(possible_cells) == 0:
                # no cells found ... it must be outside
                index = -1
                # print("Starting outside mesh")
            else:
                # one or more possible cells, check whether it's really inside one of them
                # using numpy math might be faster, but since it's should only be for a few points let's using shapely
                pnt = shapely.geometry.Point(bp[0])
                for k in possible_cells:
                    polygon_k = shapely.geometry.Polygon(
                        numpy.concatenate((xf[k : k + 1], yf[k : k + 1]), axis=0).T
                    )
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
                X0 = xe[edges, 0]
                dX = xe[edges, 1] - X0
                Y0 = ye[edges, 0]
                dY = ye[edges, 1] - Y0
                xi0 = bpj1[0]
                dxi = bpj[0] - xi0
                yi0 = bpj1[1]
                dyi = bpj[1] - yi0
                det = dX * dyi - dY * dxi
                a = (dyi * (xi0 - X0) - dxi * (yi0 - Y0)) / det  # along mesh edge
                b = (dY * (xi0 - X0) - dX * (yi0 - Y0)) / det  # along bank line
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
                            raise Exception(
                                "Shouldn't come here .... index {} differs from both faces {} and {} associated with slicing edge {}".format(
                                    index, faces[0], faces[1], edge
                                )
                            )
    masked_idx = numpy.ma.masked_array(idx, mask=(idx == -1))
    return masked_idx


def move_line(xlines: numpy.ndarray, ylines: numpy.ndarray, dn: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Shift a line of a variable distance sideways (positive shift towards the right).

    Arguments
    ---------
    xlines : numpy.ndarray
        Array containing the x-coordinates of the line to be moved.
    ylines : numpy.ndarray
        Array containing the y-coordinates of the line to be moved.
    dn : numpy.ndarray
        Distance over which to move the line sideways. A positive shift is
        defined towards the right when looking along the line.
        
    Returns
    -------
    xlines_new : umpy.ndarray
        Array containing the x-coordinates of the moved line.
    ylines : numpy.ndarray
        Array containing the y-coordinates of the moved line.
    """
    eps = sys.float_info.epsilon
    lenx = len(xlines)

    seg1x0 = xlines[:-2]
    seg1y0 = ylines[:-2]
    seg1dx = xlines[1:-1] - seg1x0
    seg1dy = ylines[1:-1] - seg1y0
    seg1mg = numpy.sqrt(seg1dx ** 2 + seg1dy ** 2)
    seg1x0 = seg1x0 + dn[:-1] * seg1dy / seg1mg
    seg1y0 = seg1y0 - dn[:-1] * seg1dx / seg1mg

    seg2x0 = xlines[1:-1]
    seg2y0 = ylines[1:-1]
    seg2dx = xlines[2:] - seg2x0
    seg2dy = ylines[2:] - seg2y0
    seg2mg = numpy.sqrt(seg2dx ** 2 + seg2dy ** 2)
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

    alpha = (-seg2dy * dsegx0 + seg2dx * dsegy0) / det
    beta = (-seg1dy * dsegx0 + seg1dx * dsegy0) / det
    alpha[alpha < 0.5] = numpy.nan
    alpha[beta > 0.5] = numpy.nan
    # TODO: add limiter for sharp angles?
    # TODO: make shift "area" conservative in general?

    # lenx - 1 segment midpoints
    # lenx - 2 internal nodes
    # 2 end nodes
    # total: 2 * lenx - 1

    xlines_new = numpy.zeros(2 * lenx - 1)  # lenx = 50 -> length = 99 -> 0:98
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


def clip_sort_connect_bank_lines(banklines, bankarea, xykm):
    """
    Connect the bank line segments to bank lines and clip to the area of interest.

    Arguments
    ---------
    banklines : 
        Unordered set of bank line segments.
    bankarea : List[]
        A list of search areas surrounding the bank guide lines.
    xykm : numpy.ndarray
        Array containing x,y,chainage values.
    
    Returns
    -------
    bank : 
        The detected bank lines.
        
    """
    clipped_banklines = banklines.intersection(bankarea)[
        0
    ]  # one MultiLineString object
    clipped_banklines = [
        line for line in clipped_banklines
    ]  # convert MultiLineString into list of LineStrings that can be modified later
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
        minlocs[i] = minloc  # at minj
        maxlocs[i] = maxloc  # at maxj
        if minj < maxj:
            clipped_banklines[i] = shapely.geometry.LineString(
                bl.coords[minj : maxj + 1]
            )
        elif minj > maxj:
            clipped_banklines[i] = shapely.geometry.LineString(
                bl.coords[maxj : minj + 1][::-1]
            )
        else:
            pass  # if minj == maxj then minloc == maxloc and thus lengths == 0 and will be removed anyway
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
        jarray = numpy.nonzero(
            (minlocs > minlocs[i]) & (minlocs < maxlocs[i]) & (maxlocs > maxlocs[i])
        )[0]
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
                        clipped_banklines[j] = shapely.geometry.LineString(
                            bl.coords[k:]
                        )
                        minlocs[j] = loc
                        break
        # if line partially overlaps ... but stick out on the low side
        jarray = numpy.nonzero(
            (minlocs < minlocs[i]) & (maxlocs > minlocs[i]) & (maxlocs < maxlocs[i])
        )[0]
        if jarray.size > 0:
            for j in jarray:
                bl = clipped_banklines[j]
                kmax = len(bl.coords) - 1
                for k, p in zip(range(-1, -kmax, -1), bl.coords[:-1][::-1]):
                    if k == kmax + 1:
                        # a line string of a single point would remain
                        lengths[j] = 0
                        break
                    loc = xykm.project(shapely.geometry.Point(p))
                    if loc <= minlocs[i]:
                        clipped_banklines[j] = shapely.geometry.LineString(
                            bl.coords[:k]
                        )
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


def clip_bank_guidelines(line, xykm, max_river_width: float = 1000):
    """

    Arguments
    ---------
    line
        ...
    xykm
        ...
    
    Returns
    -------
    line
        ...
    maxmaxd
        ...
    """
    nbank = len(line)
    # clip guiding bank lines to area of interest
    # using simplified geometries to speed up identifying the appropriate buffer size
    xy_simplified = xykm.simplify(1)
    maxmaxd = 0
    for b in range(nbank):
        # bank lines will probably extend beyond the xykm reach of interest, so clip them since points (far) beyond the end will ruin the computation of maxd below
        # computing the distance from xykm to the bank line may not detect local maxima in bank distance
        line[b] = line[b].intersection(xykm.buffer(max_river_width, cap_style=2))
        if line[b].geom_type == "MultiLineString":
            L = []
            for i in range(len(line[b])):
                L.extend(line[b][i].coords)
            line[b] = shapely.geometry.LineString(L)
        line_simplified = line[b].simplify(1)
        maxd = max(
            [
                shapely.geometry.Point(c).distance(xy_simplified)
                for c in line_simplified.coords
            ]
        )
        maxmaxd = max(maxmaxd, maxd + 2)

    return line, maxmaxd


def convert_guide_lines_to_bank_polygons(guide_lines: List[numpy.ndarray], dlines: List[float]):
    """
    Construct a series of polygons surrounding the bank guide lines.

    Arguments
    ---------
    guide_lines : List[numpy.ndarray]
        List of arrays containing the x,y-coordinates of a bank guide lines.
    dlines : List[float]
        Array containing the search distance value per bank line.
        
    Results
    -------
    bankareas
        Array containing the areas of interest surrounding the bank guide lines.
    """
    nbank = len(guide_lines)
    bankareas = [None] * nbank
    for b, distance in enumerate(dlines):
        bankareas[b] = guide_lines[b].buffer(distance, cap_style=2)

    return bankareas


def clip_simdata(sim: SimulationObject, xykm: numpy.ndarray, maxmaxd: float) -> SimulationObject:
    """
    Clip the simulation mesh and data to the area of interest sufficiently close to the reference line.

    Arguments
    ---------
    sim : SimulationObject
        Simulation data: mesh, bed levels, water levels, velocities, etc.
    xykm : numpy.ndarray
        Reference line.
    maxmaxd : float
        Maximum distance between the reference line and a point in the area of
        interest defined based on the guide lines for the banks and the search
        distance.
    
    Returns
    -------
    sim1 : SimulationObject
        Clipped simulation data: mesh, bed levels, water levels, velocities, etc.
    """
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
        if keep[i] and not xybprep.contains(shapely.geometry.Point((x[i], y[i]))):
            keep[i] = False

    fnc = sim["facenode"]
    keepface = keep[fnc].all(axis=1)
    renum = numpy.zeros(nnodes, dtype=numpy.int)
    renum[keep] = range(sum(keep))
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


def get_banklines(sim: SimulationObject, h0: float) -> geopandas.GeoSeries:
    """
    Detect the bank lines based on a simulation and critical water depth h0.

    Arguments
    ---------
    sim : SimulationObject
        Simulation data: mesh, bed levels, water levels, velocities, etc.
    h0 : float
        Critical water depth for determining the banks.
    
    Returns
    -------
    banklines : geopandas.GeoSeries
        The collection of all detected bank segments in the remaining model area.
    """
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
    ZW_node = numpy.bincount(FNCm, weights=ZWm, minlength=nnodes_total)
    NVal = numpy.bincount(FNCm, weights=numpy.ones(nonmasked), minlength=nnodes_total)
    ZW_node = ZW_node / numpy.maximum(NVal, 1)
    ZW_node[NVal == 0] = sim["zb_val"][NVal == 0]
    #
    H_node = ZW_node[FNC] - ZB
    WET_node = H_node > h0
    NWET = WET_node.sum(axis=1)
    MASK = NWET.mask.size > 1
    #
    nfaces = len(FNC)
    Lines = [None] * nfaces
    frac = 0
    for i in range(nfaces):
        if i >= frac * (nfaces - 1) / 10:
            print("{}%".format(int(frac * 10)))
            frac = frac + 1
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


def poly_to_line(nnodes: int, x: numpy.ndarray, y: numpy.ndarray, wet_node: numpy.ndarray, h_node: numpy.ndarray, h0: float):
    """
    Detect the bank line segments inside an individual mesh face.

    Arguments
    ---------
    nnode : int
        Number of nodes of mesh face.
    x : numpy.ndarray
        Array of x-coordinates of the nodes making up the mesh face.
    y : numpy.ndarray
        Array of y-coordinates of the nodes making up the mesh face.
    wet_node : numpy.ndarray
        Array of booleans indicating whether nodes are wet.
    h_node : numpy.ndarray
        Array of water depths (negative for dry) at the mesh nodes.
    h0 : float
        Critical water depth for determining the banks.
    
    Results
    -------
    lines : Optional[...]
        Optional bank line segments detected within the mesh face.
    """
    Lines = [None] * (nnodes - 2)
    for i in range(nnodes - 2):
        iv = [0, i + 1, i + 2]
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


def tri_to_line(x: numpy.ndarray, y: numpy.ndarray, wet_node: numpy.ndarray, h_node: numpy.ndarray, h0: float):
    """
    Detect the bank line segments inside an individual triangle.

    Arguments
    ---------
    x : numpy.ndarray
        Array of x-coordinates of the nodes making up the mesh face.
    y : numpy.ndarray
        Array of y-coordinates of the nodes making up the mesh face.
    wet_node : numpy.ndarray
        Array of booleans indicating whether nodes are wet.
    h_node : numpy.ndarray
        Array of water depths (negative for dry) at the mesh nodes.
    h0 : float
        Critical water depth for determining the banks.
        
    Returns
    -------
    Line : Optional[]
        Optional bank line segment detected within the triangle.
    """
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
    else:  # wet_node[2]
        A = 2
        B = 0
        C = 2
        D = 1
    facAB = (h_node[A] - h0) / (h_node[A] - h_node[B])  # large facAB -> close to B
    xl = x[A] + facAB * (x[B] - x[A])
    yl = y[A] + facAB * (y[B] - y[A])
    facCD = (h_node[C] - h0) / (h_node[C] - h_node[D])  # large facCD -> close to D
    xr = x[C] + facCD * (x[D] - x[C])
    yr = y[C] + facCD * (y[D] - y[C])
    if xl == xr and yl == yr:
        Line = None
    else:
        Line = shapely.geometry.asLineString([[xl, yl], [xr, yr]])
    return Line
