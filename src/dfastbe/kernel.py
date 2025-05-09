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

from typing import List, Tuple

import numpy as np

from dfastbe.bank_erosion.utils import get_km_bins


def get_zoom_extends(
    km_min: float,
    km_max: float,
    zoom_km_step: float,
    bank_crds: List[np.ndarray],
    bank_km: List[np.ndarray],
) -> tuple[list[tuple[float, float]], list[tuple[float, float, float, float]]]:
    """Zoom.

    Args:
        km_min (float):
            Minimum value for the chainage range of interest.
        km_max (float):
            Maximum value for the chainage range of interest.
        zoom_km_step (float):
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

    zoom_km_bin = (km_min, km_max, zoom_km_step)
    zoom_km_bnd = get_km_bins(zoom_km_bin, station_type="all", adjust=True)
    eps = 0.1 * zoom_km_step

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
