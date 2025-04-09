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

from typing import Dict, List, Tuple, Union
from dfastbe.io import SimulationObject

# import matplotlib
# import matplotlib.pyplot

import numpy
import math
from shapely.geometry import (
    LineString,
    Point,
    Polygon,
    MultiLineString,
)
import shapely
import geopandas
import sys


def project_km_on_line(
    line_xy: numpy.ndarray, xykm_numpy: numpy.ndarray
) -> numpy.ndarray:
    """
    Project chainage values from source line L1 onto another line L2.

    The chainage values are giving along a line L1 (xykm_numpy). For each node
    of the line L2 (line_xy) on which we would like to know the chainage, first
    the closest node (discrete set of nodes) on L1 is determined and
    subsequently the exact chainage isobtained by determining the closest point
    (continuous line) on L1 for which the chainage is determined using by means
    of interpolation.

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
    # pre-allocate the array for the mapped chainage values
    line_km = numpy.zeros(line_xy.shape[0])

    # get an array with only the x,y coordinates of line L1
    xy_numpy = xykm_numpy[:, :2]
    last_xykm = xykm_numpy.shape[0] - 1

    # for each node rp on line L2 get the chainage ...
    for i, rp_numpy in enumerate(line_xy):
        # find the node on L1 closest to rp
        imin = numpy.argmin(((rp_numpy - xy_numpy) ** 2).sum(axis=1))
        p0 = xy_numpy[imin]

        # determine the distance between that node and rp
        dist2 = ((rp_numpy - p0) ** 2).sum()

        # chainage value of that node
        km = xykm_numpy[imin, 2]
        # print("chainage closest node: ", km)

        # if we didn't get the first node
        if imin > 0:
            # project rp onto the line segment before this node
            p1 = xy_numpy[imin - 1]
            alpha = (
                (p1[0] - p0[0]) * (rp_numpy[0] - p0[0])
                + (p1[1] - p0[1]) * (rp_numpy[1] - p0[1])
            ) / ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
            # if there is a closest point not coinciding with the nodes ...
            if alpha > 0 and alpha < 1:
                dist2link = (rp_numpy[0] - p0[0] - alpha * (p1[0] - p0[0])) ** 2 + (
                    rp_numpy[1] - p0[1] - alpha * (p1[1] - p0[1])
                ) ** 2
                # if it's actually closer than the node ...
                if dist2link < dist2:
                    # update the closest point information
                    dist2 = dist2link
                    km = xykm_numpy[imin, 2] + alpha * (
                        xykm_numpy[imin - 1, 2] - xykm_numpy[imin, 2]
                    )
                    # print("chainage of projection 1: ", km)

        # if we didn't get the last node
        if imin < last_xykm:
            # project rp onto the line segment after this node
            p1 = xy_numpy[imin + 1]
            alpha = (
                (p1[0] - p0[0]) * (rp_numpy[0] - p0[0])
                + (p1[1] - p0[1]) * (rp_numpy[1] - p0[1])
            ) / ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
            # if there is a closest point not coinciding with the nodes ...
            if alpha > 0 and alpha < 1:
                dist2link = (rp_numpy[0] - p0[0] - alpha * (p1[0] - p0[0])) ** 2 + (
                    rp_numpy[1] - p0[1] - alpha * (p1[1] - p0[1])
                ) ** 2
                # if it's actually closer than the previous value ...
                if dist2link < dist2:
                    # update the closest point information
                    dist2 = dist2link
                    km = xykm_numpy[imin, 2] + alpha * (
                        xykm_numpy[imin + 1, 2] - xykm_numpy[imin, 2]
                    )
                    # print("chainage of projection 2: ", km)

        # store the chainage value, loop ... and return
        line_km[i] = km
    return line_km


def on_right_side(line_xy: numpy.ndarray, ref_xy: numpy.ndarray) -> bool:
    """
    Determine whether line_xy is to the left or right of ref_xy.

    Left and right are relative to the path along ref_xy from the first to the
    last node. It is assumed that line_xy can be uniquely identified as on the
    left or right side of ref_xy, i.e. the lines may not cross each other or
    themselves. Also line_xy should be alongside ref_xy and not "before" or
    "after" ref_xy. The typical use case is to relate a bank line line_xy to a
    centre line ref_xy.

    Arguments
    ---------
    line_xy : numpy.ndarray
        Array containing the x,y coordinates of a line.
    ref_xy : numpy.ndarray
        Array containing the x,y,chainage data.

    Results
    -------
    right_side : bool
        Flag indicating whether line is on the right side.
    """

    # determine the reference point based on the line with the fewest points
    ref_npnt = ref_xy.shape[0]
    npnt = line_xy.shape[0]
    if ref_npnt < npnt:
        # determine the mid-point p0 of ref_xy
        p0 = (ref_xy[0] + ref_xy[1]) / 2
        if ref_npnt == 2:
            imin = 0
            imind = 0
            iminu = 1
            p0 = (ref_xy[0] + ref_xy[1]) / 2
        else:
            imin = int(ref_npnt / 2)
            imind = imin - 1
            iminu = imin + 1
            p0 = ref_xy[imin]

        # find the node on line_xy closest to p0
        hpnt = numpy.argmin(((p0 - line_xy) ** 2).sum(axis=1))
        hpxy = line_xy[hpnt]
    else:
        # determine the mid-point hpxy of line_xy
        hpnt = int(npnt / 2)
        hpxy = line_xy[hpnt]

        # find the node on ref_xy closest to hpxy
        imin = numpy.argmin(((hpxy - ref_xy) ** 2).sum(axis=1))
        imind = imin - 1
        iminu = imin + 1
        p0 = ref_xy[imin]

    # direction to the midpoint of line_xy
    theta = math.atan2(hpxy[1] - p0[1], hpxy[0] - p0[0])

    # direction from which ref_xy comes
    if ref_xy.shape[0] == 1:
        raise Exception("One point is not a reference line.")
    elif imin > 0:
        phi1 = math.atan2(ref_xy[imind, 1] - p0[1], ref_xy[imind, 0] - p0[0])
        # direction to which ref_xy goes
        if imin < ref_xy.shape[0] - 1:
            phi2 = math.atan2(ref_xy[iminu, 1] - p0[1], ref_xy[iminu, 0] - p0[0])
        else:
            phi2 = -phi1
    else:
        # direction to which ref_xy goes
        phi2 = math.atan2(ref_xy[iminu, 1] - p0[1], ref_xy[iminu, 0] - p0[0])
        phi1 = -phi2

    # adjust the directions of ref_xy such that both are larger than the
    # angle of the direction towards the midpoint of line_xy
    if phi1 < theta:
        phi1 = phi1 + 2 * math.pi
    if phi2 < theta:
        phi2 = phi2 + 2 * math.pi

    # theta points to the right relative to a line coming from phi1 and going
    # to phi2 if we encounter the to direction phi2 before the from direction
    # phi1 when searching from theta in counter clockwise direction.
    right_side = phi2 < phi1

    return right_side


def xykm_bin(xykm: numpy.ndarray, km_bin: Tuple[float, float, float]) -> numpy.ndarray:
    """
    Resample a georeferenced chainage line to bin boundaries. 
    
    Arguments
    ---------
    xykm : numpy.ndarray
        N x 3 array representing the basic georeferenced chainage information.
    km_bin : Tuple[float, float, float]
        Tuple containing start, end, and step for chainage bins.
        
    Returns
    -------
    xykm1 : numpy.ndarray
        M x 3 array representing the resampled georeferenced chainage information.
    """
    length = xykm.shape[0]
    length1 = int((km_bin[1] - km_bin[0]) / km_bin[2]) + 1
    xykm1 = numpy.zeros((length1, 3))
    j = 0
    for i in range(length1):
        km = km_bin[0] + i * km_bin[2]
        while xykm[j, 2] < km:
            j = j + 1
            if j == length:
                break
        if j == 0:
            xykm1[i, :] = xykm[0, :]
        elif j == length:
            xykm1[i, :] = xykm[-1, :]
        else:
            alpha = (km - xykm[j - 1, 2]) / (xykm[j, 2] - xykm[j - 1, 2])
            xykm1[i, :] = (1 - alpha) * xykm[j - 1, :] + alpha * xykm[j, :]
    return xykm1


def intersect_line_mesh(
    bp: numpy.ndarray,
    xf: numpy.ma.masked_array,
    yf: numpy.ma.masked_array,
    xe: numpy.ndarray,
    ye: numpy.ndarray,
    fe: numpy.ma.masked_array,
    ef: numpy.ndarray,
    fn: numpy.ma.masked_array,
    en: numpy.ndarray,
    nnodes: numpy.ndarray,
    boundary_edge_nrs: numpy.ndarray,
    d_thresh: float = 0.001,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Intersect a (bank) line with an unstructured mesh and return the intersection coordinates and mesh face indices.

    Arguments
    ---------
    bp : numpy.ndarray
        Array containing the x,y-coordinates of the (bank) line.
    xf : numpy.ma.masked_array
        Array containing the x-coordinates of the corner points of each mesh face.
    yf : numpy.ma.masked_array
        Array containing the y-coordinates of the corner points of each mesh face.
    xe : numpy.ndarray
        Array containing the x-coordinates of the end points of each mesh edge.
    ye : numpy.ndarray
        Array containing the y-coordinates of the end points of each mesh edge.
    fe : numpy.ma.masked_array
        Array containg the mesh face-edge connectivity.
    ef : numpy.ndarray
        Array containg the mesh edge-face connectivity.
    fn : numpy.ma.masked_array
        Array containg the mesh face-node connectivity.
    en : numpy.ndarray
        Array containg the mesh edge-node connectivity.
    nnodes : numpy.ndarray
        Array containg the number of nodes/edges per face.
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
    verbose = False
    l = 0
    #
    index: int
    vindex: numpy.ndarray
    nprint = 0
    for j, bpj in enumerate(bp):
        if verbose:
            print("Current location: {}, {}".format(bpj[0], bpj[1]))
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
                if verbose:
                    print("starting outside mesh")
            else:
                # one or more possible cells, check whether it's really inside one of them
                # using numpy math might be faster, but since it's should only be for a few points let's use Shapely
                # a point on the edge of a polygon is not contained in the polygon.
                # a point on the edge of two polygons will thus be considered outside the mesh whereas it actually isn't.
                pnt = Point(bp[0])
                for k in possible_cells:
                    polygon_k = Polygon(
                        numpy.concatenate(
                            (xf[k : k + 1, : nnodes[k]], yf[k : k + 1, : nnodes[k]]),
                            axis=0,
                        ).T
                    )
                    if polygon_k.contains(pnt):
                        index = k
                        if verbose:
                            print("starting in {}".format(index))
                        break
                else:
                    on_edge: List[int] = []
                    for k in possible_cells:
                        nd = numpy.concatenate(
                            (xf[k : k + 1, : nnodes[k]], yf[k : k + 1, : nnodes[k]]),
                            axis=0,
                        ).T
                        line_k = LineString(numpy.concatenate(nd, nd[0:1], axis=0))
                        if line_k.contains(pnt):
                            on_edge.append(k)
                    if on_edge == []:
                        index = -1
                        if verbose:
                            print("starting outside mesh")
                    else:
                        if len(on_edge) == 1:
                            index = on_edge[0]
                        else:
                            index = -2
                            vindex = on_edge
                        if verbose:
                            print("starting on edge of {}".format(on_edge))
                        raise Exception("determine direction!")
            crds[l] = bpj
            if index == -2:
                idx[l] = vindex[0]
            else:
                idx[l] = index
            l += 1
        else:
            # second or later point
            bpj1 = bp[j - 1]
            prev_b = 0
            prev_pnt = bpj1
            while True:
                if index == -2:
                    b = numpy.zeros(0)
                    edges = numpy.zeros(0, dtype=numpy.int64)
                    nodes = numpy.zeros(0, dtype=numpy.int64)
                    index_src = numpy.zeros(0, dtype=numpy.int64)
                    for i in vindex:
                        b1, edges1, nodes1 = get_slices(
                            i,
                            prev_b,
                            bpj,
                            bpj1,
                            xe,
                            ye,
                            fe,
                            nnodes,
                            en,
                            boundary_edge_nrs,
                        )
                        b = numpy.concatenate((b, b1), axis=0)
                        edges = numpy.concatenate((edges, edges1), axis=0)
                        nodes = numpy.concatenate((nodes, nodes1), axis=0)
                        index_src = numpy.concatenate(
                            (index_src, i + 0 * edges1), axis=0
                        )
                    edges, id_edges = numpy.unique(edges, return_index=True)
                    b = b[id_edges]
                    nodes = nodes[id_edges]
                    index_src = index_src[id_edges]
                    if len(index_src) == 1:
                        index = index_src[0]
                        vindex = index_src[0:1]
                elif (bpj == bpj1).all():
                    # this is a segment of length 0, skip it since it takes us nowhere
                    break
                else:
                    b, edges, nodes = get_slices(
                        index,
                        prev_b,
                        bpj,
                        bpj1,
                        xe,
                        ye,
                        fe,
                        nnodes,
                        en,
                        boundary_edge_nrs,
                    )

                if len(edges) == 0:
                    # rest of segment associated with same face
                    if verbose:
                        if prev_b > 0:
                            print(
                                "{}: -- no further slices along this segment --".format(
                                    j
                                )
                            )
                        else:
                            print("{}: -- no slices along this segment --".format(j))
                        if index >= 0:
                            pnt = Point(bpj)
                            polygon_k = Polygon(
                                numpy.concatenate(
                                    (
                                        xf[index : index + 1, : nnodes[index]],
                                        yf[index : index + 1, : nnodes[index]],
                                    ),
                                    axis=0,
                                ).T
                            )
                            if not polygon_k.contains(pnt):
                                raise Exception(
                                    "{}: ERROR: point actually not contained within {}!".format(
                                        j, index
                                    )
                                )
                    if l == crds.shape[0]:
                        crds = enlarge(crds, (2 * l, 2))
                        idx = enlarge(idx, (2 * l,))
                    crds[l] = bpj
                    idx[l] = index
                    l += 1
                    break
                else:
                    index0 = None
                    if len(edges) > 1:
                        # line segment crosses the edge list multiple times
                        # - moving out of a cell at a corner node
                        # - moving into and out of the mesh from outside
                        # select first crossing ...
                        bmin = b == numpy.amin(b)
                        b = b[bmin]
                        edges = edges[bmin]
                        nodes = nodes[bmin]

                    # slice location identified ...
                    node = nodes[0]
                    edge = edges[0]
                    faces = ef[edge]
                    prev_b = b[0]

                    if node >= 0:
                        # if we slice at a node ...
                        if verbose:
                            print(
                                "{}: moving via node {} on edges {} at {}".format(
                                    j, node, edges, b[0]
                                )
                            )
                        # figure out where we will be heading afterwards ...
                        all_node_edges = numpy.nonzero((en == node).any(axis=1))[0]
                        all_node_faces = numpy.unique(ef[all_node_edges])
                        if b[0] < 1.0:
                            # segment passes through node and enter non-neighbouring cell ...
                            # direction of current segment from bpj1 to bpj
                            theta = math.atan2(bpj[1] - bpj1[1], bpj[0] - bpj1[0])
                        else:
                            if b[0] == 1.0 and j == len(bp) - 1:
                                # catch case of last segment
                                if verbose:
                                    print("{}: last point ends in a node".format(j))
                                if l == crds.shape[0]:
                                    crds = enlarge(crds, (l + 1, 2))
                                    idx = enlarge(idx, (l + 1,))
                                crds[l] = bpj
                                if index == -2:
                                    idx[l] = vindex[0]
                                else:
                                    idx[l] = index
                                l += 1
                                break
                            else:
                                # this segment ends in the node, so check next segment ...
                                # direction of next segment from bpj to bp[j+1]
                                theta = math.atan2(
                                    bp[j + 1][1] - bpj[1], bp[j + 1][0] - bp[j][0]
                                )
                        if verbose:
                            print("{}: moving in direction theta = {}".format(j, theta))
                        twopi = 2 * math.pi
                        left_edge = -1
                        left_dtheta = twopi
                        right_edge = -1
                        right_dtheta = twopi
                        if verbose:
                            print(
                                "{}: the edges connected to node {} are {}".format(
                                    j, node, all_node_edges
                                )
                            )
                        for ie in all_node_edges:
                            if en[ie, 0] == node:
                                theta_edge = math.atan2(
                                    ye[ie, 1] - ye[ie, 0], xe[ie, 1] - xe[ie, 0]
                                )
                            else:
                                theta_edge = math.atan2(
                                    ye[ie, 0] - ye[ie, 1], xe[ie, 0] - xe[ie, 1]
                                )
                            if verbose:
                                print(
                                    "{}: edge {} connects {}".format(j, ie, en[ie, :])
                                )
                                print(
                                    "{}: edge {} theta is {}".format(j, ie, theta_edge)
                                )
                            dtheta = theta_edge - theta
                            if dtheta > 0:
                                if dtheta < left_dtheta:
                                    left_edge = ie
                                    left_dtheta = dtheta
                                if twopi - dtheta < right_dtheta:
                                    right_edge = ie
                                    right_dtheta = twopi - dtheta
                            elif dtheta < 0:
                                dtheta = -dtheta
                                if twopi - dtheta < left_dtheta:
                                    left_edge = ie
                                    left_dtheta = twopi - dtheta
                                if dtheta < right_dtheta:
                                    right_edge = ie
                                    right_dtheta = dtheta
                            else:
                                # aligned with edge
                                if verbose:
                                    print(
                                        "{}: line is aligned with edge {}".format(j, ie)
                                    )
                                left_edge = ie
                                right_edge = ie
                                break
                        if verbose:
                            print(
                                "{}: the edge to the left is edge {}".format(
                                    j, left_edge
                                )
                            )
                            print(
                                "{}: the edge to the right is edge {}".format(
                                    j, left_edge
                                )
                            )
                        if left_edge == right_edge:
                            if verbose:
                                print("{}: continue along edge {}".format(j, left_edge))
                            index0 = ef[left_edge, :]
                        else:
                            if verbose:
                                print(
                                    "{}: continue between edges {} on the left and {} on the right".format(
                                        j, left_edge, right_edge
                                    )
                                )
                            left_faces = ef[left_edge, :]
                            right_faces = ef[right_edge, :]
                            if (
                                left_faces[0] in right_faces
                                and left_faces[1] in right_faces
                            ):
                                # the two edges are shared by two faces ... check first face
                                fn1 = fn[left_faces[0]]
                                fe1 = fe[left_faces[0]]
                                if verbose:
                                    print(
                                        "{}: those edges are shared by two faces: {}".format(
                                            j, left_faces
                                        )
                                    )
                                    print(
                                        "{}: face {} has nodes: {}".format(
                                            j, left_faces[0], fn1
                                        )
                                    )
                                    print(
                                        "{}: face {} has edges: {}".format(
                                            j, left_faces[0], fe1
                                        )
                                    )
                                # here we need that the nodes of the face are listed in clockwise order
                                # and that edges[i] is the edge connecting node[i-1] with node[i]
                                # the latter is guaranteed by batch.derive_topology_arrays
                                if fe1[fn1 == node] == right_edge:
                                    index0 = left_faces[0]
                                else:
                                    index0 = left_faces[1]
                            elif left_faces[0] in right_faces:
                                index0 = left_faces[0]
                            elif left_faces[1] in right_faces:
                                index0 = left_faces[1]
                            else:
                                raise Exception(
                                    "Shouldn't come here .... left edge {} and right edge {} don't share any face".format(
                                        left_edge, right_edge
                                    )
                                )

                    elif b[0] == 1:
                        # ending at slice point, so ending on an edge ...
                        if verbose:
                            print("{}: ending on edge {} at {}".format(j, edge, b[0]))
                        # figure out where we will be heading afterwards ...
                        if j == len(bp) - 1:
                            # catch case of last segment
                            if verbose:
                                print("{}: last point ends on an edge".format(j))
                            if l == crds.shape[0]:
                                crds = enlarge(crds, (l + 1, 2))
                                idx = enlarge(idx, (l + 1,))
                            crds[l] = bpj
                            if index == -2:
                                idx[l] = vindex[0]
                            else:
                                idx[l] = index
                            l += 1
                            break
                        else:
                            # this segment ends on the edge, so check next segment ...
                            # direction of next segment from bpj to bp[j+1]
                            theta = math.atan2(
                                bp[j + 1][1] - bpj[1], bp[j + 1][0] - bp[j][0]
                            )
                        if verbose:
                            print("{}: moving in direction theta = {}".format(j, theta))
                        theta_edge = math.atan2(
                            ye[edge, 1] - ye[edge, 0], xe[edge, 1] - xe[edge, 0]
                        )
                        if theta == theta_edge or theta == -theta_edge:
                            # aligned with edge
                            if verbose:
                                print("{}: continue along edge {}".format(j, edge))
                            index0 = faces
                        else:
                            # check whether the (extended) segment slices any edge of faces[0]
                            fe1 = fe[faces[0]]
                            a, b, edges = get_slices_core(
                                fe1, xe, ye, bpj, bp[j + 1], 0.0, False
                            )
                            if len(edges) > 0:
                                # yes, a slice (typically 1, but could be 2 if it slices at a node
                                # but that doesn't matter) ... so, we continue towards faces[0]
                                index0 = faces[0]
                            else:
                                # no slice for faces[0], so we must be going in the other direction
                                index0 = faces[1]

                    if not index0 is None:
                        if verbose:
                            if index == -1:
                                print(
                                    "{}: moving from outside via node {} to {} at b = {}".format(
                                        j, node, index0, prev_b
                                    )
                                )
                            elif index == -2:
                                print(
                                    "{}: moving from edge between {} via node {} to {} at b = {}".format(
                                        j, vindex, node, index0, prev_b
                                    )
                                )
                            else:
                                print(
                                    "{}: moving from {} via node {} to {} at b = {}".format(
                                        j, index, node, index0, prev_b
                                    )
                                )
                        if type(index0) == int or type(index0) == numpy.int64:
                            index = index0
                        elif len(index0) == 1:
                            index = index0[0]
                        else:
                            index = -2
                            vindex = index0
                    elif faces[0] == index:
                        if verbose:
                            print(
                                "{}: moving from {} via edge {} to {} at b = {}".format(
                                    j, index, edge, faces[1], prev_b
                                )
                            )
                        index = faces[1]
                    elif faces[1] == index:
                        if verbose:
                            print(
                                "{}: moving from {} via edge {} to {} at b = {}".format(
                                    j, index, edge, faces[0], prev_b
                                )
                            )
                        index = faces[0]
                    else:
                        raise Exception(
                            "Shouldn't come here .... index {} differs from both faces {} and {} associated with slicing edge {}".format(
                                index, faces[0], faces[1], edge
                            )
                        )
                    if l == crds.shape[0]:
                        crds = enlarge(crds, (2 * l, 2))
                        idx = enlarge(idx, (2 * l,))
                    crds[l] = bpj1 + prev_b * (bpj - bpj1)
                    if index == -2:
                        idx[l] = vindex[0]
                    else:
                        idx[l] = index
                    l += 1
                    if prev_b == 1:
                        break

    # clip to actual length (idx refers to segments, so we can ignore the last value)
    crds = crds[:l]
    idx = idx[: l - 1]

    # remove tiny segments
    d = numpy.sqrt((numpy.diff(crds, axis=0) ** 2).sum(axis=1))
    mask = numpy.concatenate((numpy.ones((1), dtype="bool"), d > d_thresh))
    crds = crds[mask, :]
    idx = idx[mask[1:]]

    # since index refers to segments, don't return the first one
    return crds, idx


def get_slices(
    index: int,
    prev_b: float,
    bpj: numpy.ndarray,
    bpj1: numpy.ndarray,
    xe: numpy.ndarray,
    ye: numpy.ndarray,
    fe: numpy.ndarray,
    nnodes: numpy.ndarray,
    en: numpy.ndarray,
    boundary_edge_nrs: numpy.ndarray,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Intersect a (bank) line with an unstructured mesh and return the intersection coordinates and mesh face indices.

    Arguments
    ---------
    index : int
        index of the current face.
    prev_b : float
        offset of previous slice.
    bpj : numpy.ndarray
        1x2 array containing the x,y-coordinates of a bank point.
    bpj : numpy.ndarray
        1x2 array containing the x,y-coordinates of the previous bank point.
    xe : numpy.ndarray
        Array containing the x-coordinates of the end points of each mesh edge.
    ye : numpy.ndarray
        Array containing the y-coordinates of the end points of each mesh edge.
    fe : numpy.ndarray
        Array containg the mesh face-edge connectivity.
    nnodes: numpy.ndarray
        Array containing the number of nodes per face.
    en : numpy.ndarray
        Array containg the mesh edge-node connectivity.
    boundary_edge_nrs : numpy.ndarray
        Array containing the indices of the domain boundary edges.
        
    Returns
    -------
    b : numpy.ndarray
        Array containing relative distance along the segment bpj1-bpj at which the slice occurs.
    edges : numpy.ndarray
        Array containing all considered edges.
    nodes : numpu.ndarray
        Array containing flags indicating whether the slices was at a node.
    """
    if index < 0:
        edges = boundary_edge_nrs
    else:
        edges = fe[index, : nnodes[index]]
    a, b, edges = get_slices_core(edges, xe, ye, bpj1, bpj, prev_b, True)
    nodes = -numpy.ones(a.shape, dtype=numpy.int64)
    nodes[a == 0] = en[edges[a == 0], 0]
    nodes[a == 1] = en[edges[a == 1], 1]
    return b, edges, nodes


def get_slices_core(
    edges: numpy.ndarray,
    xe: numpy.ndarray,
    ye: numpy.ndarray,
    bpj1: numpy.ndarray,
    bpj: numpy.ndarray,
    bmin: float,
    bmax1: bool = True,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Intersect a (bank) line with an unstructured mesh and return the intersection coordinates and mesh face indices.

    Arguments
    ---------
    edges : numpy.ndarray
        Array containing indices of the edges to check for slicing.
    xe : numpy.ndarray
        Array containing the x-coordinates of the end points of each mesh edge.
    ye : numpy.ndarray
        Array containing the y-coordinates of the end points of each mesh edge.
    bpj1 : numpy.ndarray
        1x2 array containing the x,y-coordinates of the previous bank point.
    bpj : numpy.ndarray
        1x2 array containing the x,y-coordinates of a bank point.
    bmin : float
        Minimum relative distance from bpj1 at which slice should occur.
    bmax1 : bool
        Flag indicating whether the the relative distance along the segment bpj1-bpj should be limited to 1.
        
    Returns
    -------
    a : numpy.ndarray
        Array containing relative distance along the edges at which the slice occurs.
    b : numpy.ndarray
        Array containing relative distance along the segment bpj1-bpj at which the slice occurs.
    edges : numpy.ndarray
        Reduced array containing only the indices of the sliced edges.
    """
    a, b, slices = get_slices_ab(
        xe[edges, 0],
        ye[edges, 0],
        xe[edges, 1],
        ye[edges, 1],
        bpj1[0],
        bpj1[1],
        bpj[0],
        bpj[1],
        bmin,
        bmax1,
    )
    edges = edges[slices]
    return a, b, edges


def get_slices_ab(
    X0: numpy.ndarray,
    Y0: numpy.ndarray,
    X1: numpy.ndarray,
    Y1: numpy.ndarray,
    xi0: float,
    yi0: float,
    xi1: float,
    yi1: float,
    bmin: float,
    bmax1: bool = True,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Get the relative locations a and b at which a segment intersects/slices a number of edges.

    Arguments
    ---------
    X0 : numpy.ndarray
        Array containing the x-coordinates of the start point of each edge.
    X1 : numpy.ndarray
        Array containing the x-coordinates of the end point of each edge.
    Y0 : numpy.ndarray
        Array containing the y-coordinates of the start point of each edge.
    Y1 : numpy.ndarray
        Array containing the y-coordinates of the end point of each edge.
    xi0 : float
        x-coordinate of start point of the segment.
    xi1 : float
        x-coordinate of end point of the segment.
    yi0 : float
        y-coordinate of start point of the segment.
    yi1 : float
        y-coordinate of end point of the segment.
    bmin : float
        Minimum relative distance from bpj1 at which slice should occur.
    bmax1 : bool
        Flag indicating whether the the relative distance along the segment bpj1-bpj should be limited to 1.
        
    Returns
    -------
    a : numpy.ndarray
        Array containing relative distance along each edge.
    b : numpy.ndarray
        Array containing relative distance along the segment for each edge.
    slices : numpy.ndarray
        Array containing a flag indicating whether the edge is sliced at a valid location.
    """
    dX = X1 - X0
    dY = Y1 - Y0
    dxi = xi1 - xi0
    dyi = yi1 - yi0
    det = dX * dyi - dY * dxi
    det[det == 0] = 1e-10
    a = (dyi * (xi0 - X0) - dxi * (yi0 - Y0)) / det  # along mesh edge
    b = (dY * (xi0 - X0) - dX * (yi0 - Y0)) / det  # along bank line
    # eps = numpy.finfo(float).eps
    if bmax1:
        slices = numpy.nonzero((b > bmin) & (b <= 1) & (a >= 0) & (a <= 1))[0]
    else:
        slices = numpy.nonzero((b > bmin) & (a >= 0) & (a <= 1))[0]
    a = a[slices]
    b = b[slices]
    return a, b, slices


def map_line_mesh(
    bp: numpy.ndarray,
    xf: numpy.ma.masked_array,
    yf: numpy.ma.masked_array,
    nnodes: numpy.ndarray,
    xe: numpy.ndarray,
    ye: numpy.ndarray,
    fe: numpy.ma.masked_array,
    ef: numpy.ndarray,
    boundary_edge_nrs: numpy.ndarray,
) -> numpy.ndarray:
    """
    Determine for each point of a line in which mesh face it is located.

    Arguments
    ---------
    bp : numpy.ndarray
        Array containing the x,y-coordinates of the (bank) line.
    xf : numpy.ma.masked_array
        Array containing the x-coordinates of the corner points of each mesh face.
    yf : numpy.ma.masked_array
        Array containing the y-coordinates of the corner points of each mesh face.
    nnodes : numpy.ndarray
        Array containing the number of nodes/edges per mesh face.
    xe : numpy.ndarray
        Array containing the x-coordinates of the end points of each mesh edge.
    ye : numpy.ndarray
        Array containing the y-coordinates of the end points of each mesh edge.
    fe : numpy.ma.masked_array
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
                pnt = Point(bp[0])
                for k in possible_cells:
                    polygon_k = Polygon(
                        numpy.concatenate(
                            (xf[k : k + 1, : nnodes[k]], yf[k : k + 1, : nnodes[k]]),
                            axis=0,
                        ).T
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
                    edges = fe[index, : nnodes[k]]
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


def move_line(
    xylines: numpy.ndarray, dn: numpy.ndarray, right_bank: bool
) -> numpy.ndarray:
    """
    Shift a line of a variable distance sideways (positive shift away from centre line).

    Chainage must be increasing along all lines. For a bank on the right side a
    positive shift will move the line to the right. For a bank on the left side
    a positive shift will move the line to the left.
    
    Arguments
    ---------
    xylines : numpy.ndarray
        Nx2 array containing the x- and y-coordinates of the line to be moved.
    dn : numpy.ndarray
        Distance over which to move the line sideways. A positive shift is
        defined towards the right for the right bank, and towards the left for
        the left bank.
    right_bank : bool
        Flag indicating whether line is on the right (or not).
        
    Returns
    -------
    xylines_new : umpy.ndarray
        Nx2 array containing the x- and y-coordinates of the moved line.
    """
    if right_bank:
        xylines_new = move_line_right(xylines, dn)
    else:
        xylines_rev = xylines[::-1, :]
        dn_rev = dn[::-1]
        xylines_new_rev = move_line_right(xylines_rev, dn_rev)
        xylines_new = xylines_new_rev[::-1, :]
    return xylines_new


def move_line_right(xylines: numpy.ndarray, dn: numpy.ndarray) -> numpy.ndarray:
    """
    Shift a line of a variable distance sideways (positive shift to the right).

    Arguments
    ---------
    xylines : numpy.ndarray
        Nx2 array containing the x- and y-coordinates of the line to be moved.
    dn0 : numpy.ndarray
        Distance over which to move the line sideways. A positive shift is
        defined towards the right when looking along the line.
        
    Returns
    -------
    xylines_new : umpy.ndarray
        Nx2 array containing the x- and y-coordinates of the moved line.
    """
    nsegments = len(dn)
    colvec = (nsegments, 1)

    # determine segment angle
    dxy = xylines[1:, :] - xylines[:-1, :]
    theta = numpy.arctan2(dxy[:, 1], dxy[:, 0])

    # determine shift vector nxy for each segment
    ds = numpy.sqrt((dxy ** 2).sum(axis=1))
    nxy = dxy[:, ::-1] * [1, -1] * (dn / ds).reshape(colvec)

    xylines_new = numpy.zeros((100, 2))
    xylines_new[0] = xylines[0] + nxy[0]
    ixy, xylines_new = add_point(0, xylines_new, xylines[1] + nxy[0])
    ixy, xylines_new = add_point(ixy, xylines_new, xylines[1])

    verbose = False
    prec = 0.000001
    ixy1: int
    for iseg in range(1, nsegments):
        dtheta = theta[iseg] - theta[iseg - 1]
        if dtheta > math.pi:
            dtheta = dtheta - 2 * math.pi
        if verbose:
            print("{}: current length of new bankline is {}".format(iseg, ixy))
            print(
                "{}: segment starting at {} to be shifted by {}".format(
                    iseg, xylines[iseg], dn[iseg]
                )
            )
            print("{}: change in direction quantified as {}".format(iseg, dtheta))

        # create a polyline for the outline of the new segment
        if dn[iseg] < prec:
            # no erosion, so just a linear extension
            if verbose:
                print("{}: no shifting, just linear extension".format(iseg))
            poly = numpy.row_stack([xylines[iseg + 1], xylines[iseg],])
        elif dtheta <= 0:
            # right bend
            if -0.001 * math.pi < dtheta:
                # almost straight
                if verbose:
                    print("{}: slight bend to right".format(iseg))
                if dn[iseg] > dn[iseg]:
                    poly = numpy.row_stack(
                        [
                            xylines[iseg + 1],
                            xylines[iseg + 1] + nxy[iseg],
                            xylines[iseg] + nxy[iseg],
                            xylines[iseg] + nxy[iseg - 1],
                            xylines[iseg - 1],
                        ]
                    )
                else:
                    poly = numpy.row_stack(
                        [
                            xylines[iseg + 1],
                            xylines[iseg + 1] + nxy[iseg],
                            xylines[iseg] + nxy[iseg],
                            xylines[iseg - 1],
                        ]
                    )
            else:
                # more significant bend
                if verbose:
                    print("{}: bend to right".format(iseg))
                poly = numpy.row_stack(
                    [
                        xylines[iseg + 1],
                        xylines[iseg + 1] + nxy[iseg],
                        xylines[iseg] + nxy[iseg],
                        xylines[iseg],
                    ]
                )
        elif dn[iseg - 1] < prec:
            # left bend: previous segment isn't eroded, so nothing to connect to
            if verbose:
                print("{}: bend to left".format(iseg))
            poly = numpy.row_stack(
                [
                    xylines[iseg + 1],
                    xylines[iseg + 1] + nxy[iseg],
                    xylines[iseg] + nxy[iseg],
                    xylines[iseg],
                ]
            )
        else:
            # left bend: connect it to the previous segment to avoid non eroded wedges
            if verbose:
                print("{}: bend to left".format(iseg))
            poly = numpy.row_stack(
                [
                    xylines[iseg + 1],
                    xylines[iseg + 1] + nxy[iseg],
                    xylines[iseg] + nxy[iseg],
                    xylines[iseg] + nxy[iseg - 1],
                    xylines[iseg - 1],
                ]
            )

        nedges = poly.shape[0] - 1

        # make a temporary copy of the last 20 nodes of the already shifted bankline
        if ixy > 20:
            X0 = xylines_new[(ixy - 20) : ixy, 0].copy()
            Y0 = xylines_new[(ixy - 20) : ixy, 1].copy()
            X1 = xylines_new[(ixy - 19) : (ixy + 1), 0].copy()
            Y1 = xylines_new[(ixy - 19) : (ixy + 1), 1].copy()
            ixy0 = ixy - 20
        else:
            X0 = xylines_new[:ixy, 0].copy()
            Y0 = xylines_new[:ixy, 1].copy()
            X1 = xylines_new[1 : ixy + 1, 0].copy()
            Y1 = xylines_new[1 : ixy + 1, 1].copy()
            ixy0 = 0

        a = []
        b = []
        slices = []
        n = []
        # for each edge of the new polyline collect all intersections with the
        # already shifted bankline ...
        for i in range(nedges):
            if (poly[i + 1] == poly[i]).all():
                # polyline segment has no actual length, so skip it
                pass
            else:
                # check for intersection
                a2, b2, slices2 = get_slices_ab(
                    X0,
                    Y0,
                    X1,
                    Y1,
                    poly[i, 0],
                    poly[i, 1],
                    poly[i + 1, 0],
                    poly[i + 1, 1],
                    0,
                    True,
                )
                # exclude the intersection if it's only at the very last point
                # of the last segment
                if i == nedges - 1:
                    keep_mask = a2 < 1 - prec
                    a2 = a2[keep_mask]
                    b2 = b2[keep_mask]
                    slices2 = slices2[keep_mask]
                a.append(a2)
                b.append(b2)
                slices.append(slices2)
                n.append(slices2 * 0 + i)

        s = numpy.concatenate(slices)
        if verbose:
            print("{}: {} intersections detected".format(iseg, len(s)))
        if len(s) == 0:
            # no intersections found
            if dtheta < 0:
                # right bend (not straight)
                if dn[iseg] > 0:
                    cross = (xylines_new[ixy, 0] - xylines_new[ixy - 1, 0]) * nxy[
                        iseg, 1
                    ] - (xylines_new[ixy, 1] - xylines_new[ixy - 1, 1]) * nxy[iseg, 0]
                else:
                    cross = (xylines_new[ixy, 0] - xylines_new[ixy - 1, 0]) * dxy[
                        iseg, 1
                    ] - (xylines_new[ixy, 1] - xylines_new[ixy - 1, 1]) * dxy[iseg, 0]
                if cross <= 0.0:
                    # extended path turns right ... always add
                    pass
                else:
                    # extended path turns left
                    # we can probably ignore it, let's do so...
                    # the only exception would be an eroded patch encompassing
                    # all of the eroded bank line
                    if verbose:
                        print("{}: ignoring segment".format(iseg))
                    continue
            else:
                # left bend or straight: always add ... just the rectangle of eroded material
                pass
            ixy1 = ixy
            for n2 in range(min(nedges, 2), -1, -1):
                if verbose:
                    print("  adding point {}".format(poly[n2]))
                ixy1, xylines_new = add_point(ixy1, xylines_new, poly[n2])
            ixy = ixy1

        else:
            # one or more intersections found
            a = numpy.concatenate(a)
            b = numpy.concatenate(b)
            n = numpy.concatenate(n)

            # sort the intersections by distance along the already shifted bank line
            d = s + a
            sorted = numpy.argsort(d)
            s = s[sorted] + ixy0
            a = a[sorted]
            b = b[sorted]
            d = d[sorted]
            n = n[sorted]

            ixy1 = s[0]
            if verbose:
                print("{}: continuing new path at point {}".format(iseg, ixy1))
            xytmp = xylines_new[ixy1 : ixy + 1].copy()
            ixytmp = ixy1

            inside = False
            s_last = s[0]
            n_last = nedges
            for i in range(len(s)):
                if verbose:
                    print(
                        "- intersection {}: new polyline edge {} crosses segment {} at {}".format(
                            i, n[i], s[i], a[i]
                        )
                    )
                if i == 0 or n[i] != nedges - 1:
                    if inside:
                        if verbose:
                            print("  existing line is inside the new polygon")
                        for n2 in range(n_last, n[i], -1):
                            if verbose:
                                print("  adding new point {}".format(poly[n2]))
                            ixy1, xylines_new = add_point(ixy1, xylines_new, poly[n2])
                    else:
                        if verbose:
                            print("  existing line is outside the new polygon")
                        for s2 in range(s_last, s[i]):
                            if verbose:
                                print(
                                    "  re-adding old point {}".format(
                                        xytmp[s2 - ixytmp + 1]
                                    )
                                )
                            ixy1, xylines_new = add_point(
                                ixy1, xylines_new, xytmp[s2 - ixytmp + 1]
                            )
                    pnt_intersect = poly[n[i]] + b[i] * (poly[n[i] + 1] - poly[n[i]])
                    if verbose:
                        print("  adding intersection point {}".format(pnt_intersect))
                    ixy1, xylines_new = add_point(ixy1, xylines_new, pnt_intersect,)
                    n_last = n[i]
                    s_last = s[i]
                    if a[i] < prec:
                        dPy = poly[n[i] + 1, 1] - poly[n[i], 1]
                        dPx = poly[n[i] + 1, 0] - poly[n[i], 0]
                        s2 = s[i] - ixy0
                        dBy = Y1[s2] - Y0[s2]
                        dBx = X1[s2] - X0[s2]
                        inside = dPy * dBx - dPx * dBy > 0
                    elif a[i] > 1 - prec:
                        dPy = poly[n[i] + 1, 1] - poly[n[i], 1]
                        dPx = poly[n[i] + 1, 0] - poly[n[i], 0]
                        s2 = s[i] - ixy0 + 1
                        if s2 > len(X0) - 1:
                            inside = True
                        else:
                            dBy = Y1[s2] - Y0[s2]
                            dBx = X1[s2] - X0[s2]
                            inside = dPy * dBx - dPx * dBy > 0
                    else:
                        # line segment slices the edge somewhere in the middle
                        inside = not inside
                    if verbose:
                        if inside:
                            print("  existing line continues inside")
                        else:
                            print("  existing line continues outside")

            if verbose:
                print("- wrapping up after last intersection")
            if inside:
                if verbose:
                    print("  existing line is inside the new polygon")
                for n2 in range(n_last, -1, -1):
                    if verbose:
                        print("  adding new point {}".format(poly[n2]))
                    ixy1, xylines_new = add_point(ixy1, xylines_new, poly[n2])
            else:
                if verbose:
                    print("  existing line is inside the new polygon")
                for s2 in range(s_last, len(xytmp) + ixytmp - 1):
                    if verbose:
                        print("  re-adding old point {}".format(xytmp[s2 - ixytmp + 1]))
                    ixy1, xylines_new = add_point(
                        ixy1, xylines_new, xytmp[s2 - ixytmp + 1]
                    )
            ixy = ixy1
        # if iseg == isegstop:
        #     break
    xylines_new = xylines_new[:ixy, :]

    return xylines_new


def add_point(
    ixy1: int, xy_in: numpy.ndarray, point: numpy.ndarray
) -> Tuple[int, numpy.ndarray]:
    """
    Add the x,y-coordinates of a point to an array of x,y-coordinates if it differs from the last point.
    
    Arguments
    ---------
    ixy1 : int
        Index of last point in xy_in array
    xy_in : numpy.ndarray
        N x 2 array containing the x- and y-coordinates of points (partially filled)
    point : numpy.ndarray
        1 x 2 array containing the x- and y-coordinates of one point
    
    Results
    -------
    ixy1 : int
        Index of the new point in the xy_out array
    xy_out : numpy.ndarray
        Possibly extended copy of xy_in that includes the coordinates of point at ixy1
    """
    if (xy_in[ixy1] - point != 0).any():
        ixy1 = ixy1 + 1
        if ixy1 >= len(xy_in):
            xy_out = enlarge(xy_in, (2 * ixy1, 2))
        else:
            xy_out = xy_in
        xy_out[ixy1] = point
    else:
        xy_out = xy_in
    return ixy1, xy_out


def clip_bank_lines(
    banklines: geopandas.geoseries.GeoSeries, bankarea: Polygon
) -> MultiLineString:
    """
    Clip the bank line segments to the area of interest.

    Arguments
    ---------
    banklines : geopandas.geoseries.GeoSeries
        Unordered set of bank line segments.
    bankarea : Polygon
        A search area corresponding to one of the bank search lines.

    Returns
    -------
    clipped_banklines : MultiLineString
        Unordered set of bank line segments, clipped to bank area.
    """
    # intersection returns one MultiLineString object
    clipped_banklines = banklines.intersection(bankarea)[0]

    return clipped_banklines


def sort_connect_bank_lines(
    banklines: MultiLineString,
    xykm: LineString,
    right_bank: bool,
) -> LineString:
    """
    Connect the bank line segments to bank lines.

    Arguments
    ---------
    banklines : shapely.geometry.multilinestring.MultiLineString
        Unordered set of bank line segments.
    xykm : shapely.geometry.linestring.LineString
        Array containing x,y,chainage values.
    right_bank : bool
        Flag indicating whether line is on the right (or not).
    
    Returns
    -------
    bank : shapely.geometry.linestring.LineString
        The detected bank line.
    """

    # convert MultiLineString into list of LineStrings that can be modified later
    banklines_list = [line for line in banklines]

    # loop over banklines and determine minimum/maximum projected length
    # print("numpy init")
    minlocs = numpy.zeros(len(banklines_list))
    maxlocs = numpy.zeros(len(banklines_list))
    lengths = numpy.zeros(len(banklines_list))
    keep = lengths == 1
    # print("loop {} bank lines".format(len(banklines_list)))
    for i, bl in enumerate(banklines_list):
        minloc = 1e20
        maxloc = -1
        for j, p in enumerate(bl.coords):
            loc = xykm.project(Point(p))
            if loc < minloc:
                minloc = loc
                minj = j
            if loc > maxloc:
                maxloc = loc
                maxj = j
        minlocs[i] = minloc  # at minj
        maxlocs[i] = maxloc  # at maxj
        if minj == maxj:
            pass  # if minj == maxj then minloc == maxloc and thus lengths == 0 and will be removed anyway
        elif bl.coords[0] == bl.coords[-1]:
            # print(i,"cyclic", minloc, maxloc)
            crd_numpy = numpy.array(bl.coords)
            ncrd = len(crd_numpy)
            if minj < maxj:
                op1 = numpy.array(bl.coords[minj : maxj + 1])
                op2 = numpy.zeros((ncrd + minj - maxj, 2))
                op2[0 : ncrd - maxj - 1] = crd_numpy[maxj:-1]
                op2[ncrd - maxj - 1 : ncrd - maxj + minj] = crd_numpy[: minj + 1]
                op2 = op2[::-1]
            else:  # minj > maxj
                op1 = numpy.array(bl.coords[maxj : minj + 1][::-1])
                op2 = numpy.zeros((ncrd + maxj - minj, 2))
                op2[0 : ncrd - minj - 1] = crd_numpy[minj:-1]
                op2[ncrd - minj - 1 : ncrd - minj + maxj] = crd_numpy[: maxj + 1]
            op1_right_of_op2 = on_right_side(op1, op2)
            if (right_bank and op1_right_of_op2) or (
                (not right_bank) and (not op1_right_of_op2)
            ):
                op = op2
            else:
                op = op1
            banklines_list[i] = LineString(op)
        else:
            if minj < maxj:
                banklines_list[i] = LineString(bl.coords[minj : maxj + 1])
            else:  # minj > maxj
                banklines_list[i] = LineString(bl.coords[maxj : minj + 1][::-1])
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
                bl = banklines_list[j]
                kmax = len(bl.coords) - 1
                for k, p in enumerate(bl.coords):
                    if k == kmax:
                        # a line string of a single point would remain
                        lengths[j] = 0
                        break
                    loc = xykm.project(Point(p))
                    if loc >= maxlocs[i]:
                        banklines_list[j] = LineString(bl.coords[k:])
                        minlocs[j] = loc
                        break
        # if line partially overlaps ... but stick out on the low side
        jarray = numpy.nonzero(
            (minlocs < minlocs[i]) & (maxlocs > minlocs[i]) & (maxlocs < maxlocs[i])
        )[0]
        if jarray.size > 0:
            for j in jarray:
                bl = banklines_list[j]
                kmax = len(bl.coords) - 1
                for k, p in zip(range(-1, -kmax, -1), bl.coords[:-1][::-1]):
                    if k == kmax + 1:
                        # a line string of a single point would remain
                        lengths[j] = 0
                        break
                    loc = xykm.project(Point(p))
                    if loc <= minlocs[i]:
                        banklines_list[j] = LineString(bl.coords[:k])
                        maxlocs[j] = loc
                        break

    # select banks in order of projected length
    idx = numpy.argsort(minlocs[keep])
    idx2 = numpy.nonzero(keep)[0]
    new_bank_coords = []
    for i in idx2[idx]:
        new_bank_coords.extend(banklines_list[i].coords)
    bank = LineString(new_bank_coords)

    return bank


def clip_search_lines(
    line: List[LineString],
    xykm: LineString,
    max_river_width: float = 1000,
) -> Tuple[List[LineString], float]:
    """
    Clip the list of lines to the envelope of certain size surrounding a reference line.

    Arguments
    ---------
    line : List[LineString]
        List of search lines to be clipped.
    xykm : LineString
        Reference line.
    max_river_width: float
        Maximum distance away from xykm.

    Returns
    -------
    line : List[LineString]
        List of clipped search lines.
    maxmaxd: float
        Maximum distance from any point within line to reference line.
    """
    nbank = len(line)
    kmbuffer = xykm.buffer(max_river_width, cap_style=2)

    # The algorithm uses simplified geometries for determining the distance between lines for speed.
    # Stay accurate to within about 1 m
    xy_simplified = xykm.simplify(1)

    maxmaxd = 0
    for b in range(nbank):
        # Clip the bank search lines to the reach of interest (indicated by the reference line).
        line[b] = line[b].intersection(kmbuffer)

        # If the bank search line breaks into multiple parts, select the part closest to the reference line.
        if line[b].geom_type == "MultiLineString":
            dmin = max_river_width
            imin = 0
            for i in range(len(line[b])):
                line_simplified = line[b][i].simplify(1)
                dmin_i = line_simplified.distance(xy_simplified)
                if dmin_i < dmin:
                    dmin = dmin_i
                    imin = i
            line[b] = line[b][imin]

        # Determine the maximum distance from a point on this line to the reference line.
        line_simplified = line[b].simplify(1)
        maxd = max([Point(c).distance(xy_simplified) for c in line_simplified.coords])

        # Increase the value of maxd by 2 to account for error introduced by using simplified lines.
        maxmaxd = max(maxmaxd, maxd + 2)

    return line, maxmaxd


def convert_search_lines_to_bank_polygons(
    search_lines: List[numpy.ndarray], dlines: List[float]
):
    """
    Construct a series of polygons surrounding the bank search lines.

    Arguments
    ---------
    search_lines : List[numpy.ndarray]
        List of arrays containing the x,y-coordinates of a bank search lines.
    dlines : List[float]
        Array containing the search distance value per bank line.
        
    Results
    -------
    bankareas
        Array containing the areas of interest surrounding the bank search lines.
    """
    nbank = len(search_lines)
    bankareas = [None] * nbank
    for b, distance in enumerate(dlines):
        bankareas[b] = search_lines[b].buffer(distance, cap_style=2)

    return bankareas


def clip_simdata(
    sim: SimulationObject, xykm: numpy.ndarray, maxmaxd: float
) -> SimulationObject:
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
        interest defined based on the search lines for the banks and the search
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
        if keep[i] and not xybprep.contains(Point((x[i], y[i]))):
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
    Detect all possible bank line segments based on simulation data.
    
    Use a critical water depth h0 as water depth threshold for dry/wet boundary.

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


def poly_to_line(
    nnodes: int,
    x: numpy.ndarray,
    y: numpy.ndarray,
    wet_node: numpy.ndarray,
    h_node: numpy.ndarray,
    h0: float,
):
    """
    Detect the bank line segments inside an individual face of arbitrary (convex) polygonal shape.

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
        multi_line = MultiLineString(Lines)
        merged_line = shapely.ops.linemerge(multi_line)
        return merged_line


def tri_to_line(
    x: numpy.ndarray,
    y: numpy.ndarray,
    wet_node: numpy.ndarray,
    h_node: numpy.ndarray,
    h0: float,
):
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
        Line = LineString([[xl, yl], [xr, yr]])
    return Line


def enlarge(
    old_array: numpy.ndarray,
    new_shape: Tuple
):
    """
    Copy the values of the old array to a new, larger array of specified shape.

    Arguments
    ---------
    old_array : numpy.ndarray
        Array containing the values.
    new_shape : Tuple
        New shape of the array.
        
    Returns
    -------
    new_array : numpy.ndarray
        Array of shape "new_shape" with the 'first entries filled by the same
        values as contained in "old_array". The data type of the new array is
        equal to that of the old array.
    """
    old_shape = old_array.shape
    print("old: ", old_shape)
    print("new: ", new_shape)
    new_array = numpy.zeros(new_shape, dtype=old_array.dtype)
    if len(new_shape)==1:
        new_array[:old_shape[0]] = old_array
    elif len(new_shape)==2:
        new_array[:old_shape[0], :old_shape[1]] = old_array
    return new_array
