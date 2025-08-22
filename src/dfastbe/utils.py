"""
Copyright (C) 2025 Stichting Deltares.

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

import numpy as np


def get_zoom_extends(
    km_min: float,
    km_max: float,
    zoom_step_km: float,
    bank_crds: List[np.ndarray],
    bank_km: List[np.ndarray],
) -> tuple[list[tuple[float, float]], list[tuple[float, float, float, float]]]:
    """Zoom.

    Args:
        km_min (float):
            Minimum value for the chainage range of interest.
        km_max (float):
            Maximum value for the chainage range of interest.
        zoom_step_km (float):
            Preferred chainage length of zoom box.
        bank_crds (List[np.ndarray]):
            List of N x 2 np arrays of coordinates per bank.
        bank_km (List[np.ndarray]):
            List of N np arrays of chainage values per bank.

    Returns:
        station_zoom (List[Tuple[float, float]]):
            Zoom ranges for plots with chainage along x-axis.
        coords_zoom (List[Tuple[float, float, float, float]]):
            Zoom ranges for xy-plots.
    """
    from dfastbe.bank_erosion.utils import get_km_bins

    zoom_km_bin = (km_min, km_max, zoom_step_km)
    zoom_km_bnd = get_km_bins(zoom_km_bin, station_type="all", adjust=True)
    eps = 0.1 * zoom_step_km

    station_zoom: List[Tuple[float, float]] = []
    coords_zoom: List[Tuple[float, float, float, float]] = []
    inf = float('inf')
    for i in range(len(zoom_km_bnd) - 1):
        km_min = zoom_km_bnd[i] - eps
        km_max = zoom_km_bnd[i + 1] + eps
        station_zoom.append((km_min, km_max))

        x_min = inf
        x_max = -inf
        y_min = inf
        y_max = -inf
        for ib in range(len(bank_km)):
            ind = (bank_km[ib] >= km_min) & (bank_km[ib] <= km_max)
            range_crds = bank_crds[ib][ind, :]
            x = range_crds[:, 0]
            y = range_crds[:, 1]
            if len(x) > 0:
                x_min = min(x_min, min(x))
                x_max = max(x_max, max(x))
                y_min = min(y_min, min(y))
                y_max = max(y_max, max(y))
        coords_zoom.append((x_min, x_max, y_min, y_max))

    return station_zoom, coords_zoom


def on_right_side(line_xy: np.ndarray, ref_xy: np.ndarray) -> bool:
    """
    Determine whether line_xy is to the left or right of ref_xy.

    Left and right are relative to the path along ref_xy from the first to the
    last node. It is assumed that line_xy can be uniquely identified as on the
    left or right side of ref_xy, i.e., the lines may not cross each other or
    themselves. Also, line_xy should be alongside ref_xy and not "before" or
    "after" ref_xy. The typical use case is to relate a bank line line_xy to a
    center line ref_xy.

    Args:
        line_xy : np.ndarray
            Array containing the x,y coordinates of a line.
        ref_xy : np.ndarray
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
        hpnt = np.argmin(((p0 - line_xy) ** 2).sum(axis=1))
        hpxy = line_xy[hpnt]
    else:
        # determine the mid-point hpxy of line_xy
        hpnt = int(npnt / 2)
        hpxy = line_xy[hpnt]

        # find the node on ref_xy closest to hpxy
        imin = np.argmin(((hpxy - ref_xy) ** 2).sum(axis=1))
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
