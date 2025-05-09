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

import math
from typing import List, Tuple

import numpy
import numpy as np
from shapely import line_merge
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from dfastbe.bank_erosion.data_models import MeshData


def on_right_side(line_xy: numpy.ndarray, ref_xy: numpy.ndarray) -> bool:
    """
    Determine whether line_xy is to the left or right of ref_xy.

    Left and right are relative to the path along ref_xy from the first to the
    last node. It is assumed that line_xy can be uniquely identified as on the
    left or right side of ref_xy, i.e., the lines may not cross each other or
    themselves. Also, line_xy should be alongside ref_xy and not "before" or
    "after" ref_xy. The typical use case is to relate a bank line line_xy to a
    center line ref_xy.

    Args:
        line_xy : numpy.ndarray
            Array containing the x,y coordinates of a line.
        ref_xy : numpy.ndarray
            Array containing the x,y,chainage data.

    Returns:
        right_side : bool
            Flag indicating whether the line is on the right side.
    """

    # determine the reference point based on the line with the fewest points
    ref_npnt = ref_xy.shape[0]
    npnt = line_xy.shape[0]
    if ref_npnt < npnt:
        # determine the mid-point p0 of ref_xy
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


def move_line(
    xylines: numpy.ndarray, erosion_distance: numpy.ndarray, right_bank: bool
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
    erosion_distance : numpy.ndarray
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
        xylines_new = _move_line_right(xylines, erosion_distance)
    else:
        xylines_rev = xylines[::-1, :]
        dn_rev = erosion_distance[::-1]
        xylines_new_rev = _move_line_right(xylines_rev, dn_rev)
        xylines_new = xylines_new_rev[::-1, :]
    return xylines_new


def _move_line_right(xylines: numpy.ndarray, erosion_distance: numpy.ndarray) -> numpy.ndarray:
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
    nsegments = len(erosion_distance)
    colvec = (nsegments, 1)

    # determine segment angle
    dxy = xylines[1:, :] - xylines[:-1, :]
    theta = numpy.arctan2(dxy[:, 1], dxy[:, 0])

    # determine shift vector nxy for each segment
    ds = numpy.sqrt((dxy ** 2).sum(axis=1))
    nxy = dxy[:, ::-1] * [1, -1] * (erosion_distance / ds).reshape(colvec)

    xylines_new = numpy.zeros((100, 2))
    xylines_new[0] = xylines[0] + nxy[0]
    ixy, xylines_new = _add_point(0, xylines_new, xylines[1] + nxy[0])
    ixy, xylines_new = _add_point(ixy, xylines_new, xylines[1])

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
                    iseg, xylines[iseg], erosion_distance[iseg]
                )
            )
            print("{}: change in direction quantified as {}".format(iseg, dtheta))

        # create a polyline for the outline of the new segment
        if erosion_distance[iseg] < prec:
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
                if erosion_distance[iseg] > erosion_distance[iseg]:
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
        elif erosion_distance[iseg - 1] < prec:
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
                a2, b2, slices2 = _get_slices_ab(
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
                if erosion_distance[iseg] > 0:
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
                ixy1, xylines_new = _add_point(ixy1, xylines_new, poly[n2])
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
                            ixy1, xylines_new = _add_point(ixy1, xylines_new, poly[n2])
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
                            ixy1, xylines_new = _add_point(
                                ixy1, xylines_new, xytmp[s2 - ixytmp + 1]
                            )
                    pnt_intersect = poly[n[i]] + b[i] * (poly[n[i] + 1] - poly[n[i]])
                    if verbose:
                        print("  adding intersection point {}".format(pnt_intersect))
                    ixy1, xylines_new = _add_point(ixy1, xylines_new, pnt_intersect, )
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
                    ixy1, xylines_new = _add_point(ixy1, xylines_new, poly[n2])
            else:
                if verbose:
                    print("  existing line is inside the new polygon")
                for s2 in range(s_last, len(xytmp) + ixytmp - 1):
                    if verbose:
                        print("  re-adding old point {}".format(xytmp[s2 - ixytmp + 1]))
                    ixy1, xylines_new = _add_point(
                        ixy1, xylines_new, xytmp[s2 - ixytmp + 1]
                    )
            ixy = ixy1
        # if iseg == isegstop:
        #     break
    xylines_new = xylines_new[:ixy, :]

    return xylines_new


def _add_point(
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
