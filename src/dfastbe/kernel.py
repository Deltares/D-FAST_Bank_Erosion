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

from typing import Tuple, List
from dfastbe.structures import ErosionInputs

import numpy
import math
import sys


def comp_erosion_eq(
    bankheight: numpy.ndarray,
    linesize: numpy.ndarray,
    zfw_ini: numpy.ndarray,
    vship: numpy.ndarray,
    ship_type: numpy.ndarray,
    Tship: numpy.ndarray,
    mu_slope: numpy.ndarray,
    distance_fw: numpy.ndarray,
    hfw: numpy.ndarray,
    erosion_inputs: "ErosionInputs",
    ib: int,
    g: float,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Compute the equilibrium bank erosion.
    
    Arguments
    ---------
    bankheight : numpy.ndarray
        Array containing bank height [m]
    linesize : numpy.ndarray
        Array containing length of line segment [m]
    zfw_ini : numpy.ndarray
        Array containing water level at fairway [m]
    vship : numpy.ndarray
        Array containing ship velocity [m/s]
    ship_type : numpy.ndarray
        Array containing ship type [-]
    Tship : numpy.ndarray
        Array containing ship draught [m]
    mu_slope : numpy.ndarray
        Array containing slope [-]
    distance_fw : numpy.ndarray
        Array containing distance from bank to fairway [m]
    dfw0 : numpy.ndarray
        Array containing distance from fairway at which wave reduction starts [m]
    dfw1 : numpy.ndarray
        Array containing distance from fairway at which all waves are gone [m]
    hfw : numpy.ndarray
        Array containing water depth at the fairway [m]
    zss : numpy.ndarray
        Array containing bank protection height [m]
    g : float
        Gravitational acceleration [m/s2]
    
    Returns
    -------
    dn_eq : numpy.ndarray
         Equilibrium bank erosion distance [m]
    dv_eq : numpy.ndarray
         Equilibrium bank erosion volume [m]
    """
    eps = sys.float_info.epsilon

    # ship induced wave height at the beginning of the foreshore
    H0 = comp_hw_ship_at_bank(
        distance_fw,
        erosion_inputs.wave_fairway_distance_0[ib],
        erosion_inputs.wave_fairway_distance_1[ib],
        hfw,
        ship_type,
        Tship,
        vship,
        g,
    )
    H0 = numpy.maximum(H0, eps)

    zup = numpy.minimum(bankheight, zfw_ini + 2 * H0)
    zdo = numpy.maximum(zfw_ini - 2 * H0, erosion_inputs.bank_protection_level[ib])
    ht = numpy.maximum(zup - zdo, 0)
    hs = numpy.maximum(bankheight - zfw_ini + 2 * H0, 0)
    dn_eq = ht / mu_slope
    dv_eq = (0.5 * ht + hs) * dn_eq * linesize

    return dn_eq, dv_eq


def comp_erosion(
    velocity: numpy.ndarray,
    bankheight: numpy.ndarray,
    linesize: numpy.ndarray,
    zfw: numpy.ndarray,
    zfw_ini: numpy.ndarray,
    Nship: numpy.ndarray,
    vship: numpy.ndarray,
    nwave: numpy.ndarray,
    ship_type: numpy.ndarray,
    Tship: numpy.ndarray,
    Teros: float,
    mu_slope: numpy.ndarray,
    mu_reed: numpy.ndarray,
    distance_fw: numpy.ndarray,
    hfw: numpy.ndarray,
    chezy: numpy.ndarray,
    erosion_inputs: "ErosionInputs",
    rho: float,
    g: float,
    ib: int,
) -> [
    numpy.ndarray,
    numpy.ndarray,
    numpy.ndarray,
    numpy.ndarray,
    numpy.ndarray,
    numpy.ndarray,
]:
    """
    Compute the bank erosion during a specific discharge level.
    
    Arguments
    ---------
    velocity : numpy.ndarray
        Array containing flow velocity magnitude [m/s]
    bankheight : numpy.ndarray
        Array containing bank height
    linesize : numpy.ndarray
        Array containing length of line segment [m]
    zfw : numpy.ndarray
        Array containing water levels at fairway [m]
    zfw_ini : numpy.ndarray
        Array containing reference water levels at fairway [m]
    tauc : numpy.ndarray
        Array containing critical shear stress [N/m2]
    Nship : numpy.ndarray
        Array containing number of ships [-]
    vship : numpy.ndarray
        Array containing ship velocity [m/s]
    nwave : numpy.ndarray
        Array containing number of waves per ship [-]
    ship_type : numpy.ndarray
        Array containing ship type [-]
    Tship : numpy.ndarray
        Array containing ship draught [m]
    Teros : float
        Erosion period [yr]
    mu_slope : numpy.ndarray
        Array containing 
    mu_reed : numpy.ndarray
        Array containing 
    distance_fw : numpy.ndarray
        Array containing distance from bank to fairway [m]
    dfw0 : numpy.ndarray
        Array containing distance from fairway at which wave reduction starts [m]
    dfw1 : numpy.ndarray
        Array containing distance from fairway at which all waves are gone [m]
    hfw : numpy.ndarray
        Array containing water depth at fairway [m]
    chezy : numpy.ndarray
        Array containing Chezy values [m0.5/s]
    zss : numpy.ndarray
        Array containing bank protection height [m]
    rho : float
        Water density [kg/m3]
    g : float
        Gravitational acceleration [m/s2]
        
    Returns
    -------
    dn : numpy.ndarray
        Total bank erosion distance [m]
    dv : numpy.ndarray
        Total bank erosion volume [m]
    dn_ship : numpy.ndarray
        Bank erosion distance due to shipping [m]
    dn_flow : numpy.ndarray
        Bank erosion distance due to current [m]
    shipmwavemax : numpy.ndarray
        Maximum bank level subject to ship waves [m]
    shipwavemin : numpy.ndarray
        Minimum bank level subject to ship waves [m]
    """
    eps = sys.float_info.epsilon
    sec_year = 3600 * 24 * 365

    # period of ship waves [s]
    T = 0.51 * vship / g
    # [s]
    ts = T * Nship * nwave

    # number of line segments
    xlen = len(velocity)
    # total erosion per segment
    dn = numpy.zeros(xlen)
    # erosion volume per segment
    dv = numpy.zeros(xlen)
    # total wave damping coefficient
    mu_tot = numpy.zeros(xlen)

    vel = velocity

    # ship induced wave height at the beginning of the foreshore
    H0 = comp_hw_ship_at_bank(
        distance_fw,
        erosion_inputs.wave_fairway_distance_0[ib],
        erosion_inputs.wave_fairway_distance_1[ib],
        hfw,
        ship_type,
        Tship,
        vship,
        g,
    )
    H0 = numpy.maximum(H0, eps)

    # compute erosion parameters for each line part

    # erosion coefficient
    E = 0.2 * numpy.sqrt(erosion_inputs.tauc[ib]) * 1e-6

    # critical velocity
    velc = numpy.sqrt(erosion_inputs.tauc[ib] / rho * chezy**2 / g)

    # strength
    cE = 1.85e-4 / erosion_inputs.tauc[ib]

    # total wavedamping coefficient
    mu_tot = (mu_slope / H0) + mu_reed
    # water level along bank line
    ho_line_ship = numpy.minimum(zfw - erosion_inputs.bank_protection_level[ib], 2 * H0)
    ho_line_flow = numpy.minimum(zfw - erosion_inputs.bank_protection_level[ib], hfw)
    h_line_ship = numpy.maximum(bankheight - zfw + ho_line_ship, 0)
    h_line_flow = numpy.maximum(bankheight - zfw + ho_line_flow, 0)

    # compute displacement due to flow
    crit_ratio = numpy.ones(velc.shape)
    mask = (vel > velc) & (zfw > erosion_inputs.bank_protection_level[ib])
    crit_ratio[mask] = (vel[mask] / velc[mask]) ** 2
    dn_flow = E * (crit_ratio - 1) * Teros * sec_year

    # compute displacement due to shipwaves
    shipwavemax = zfw + 0.5 * H0
    shipwavemin = zfw - 2 * H0
    mask = (shipwavemin < zfw_ini) & (zfw_ini < shipwavemax)
    # limit mu -> 0

    dn_ship = cE * H0 ** 2 * ts * Teros
    dn_ship[~mask] = 0

    # compute erosion volume
    mask = (h_line_ship > 0) & (zfw > erosion_inputs.bank_protection_level[ib])
    dv_ship = dn_ship * linesize * h_line_ship
    dv_ship[~mask] = 0.0
    dn_ship[~mask] = 0.0

    mask = (h_line_flow > 0) & (zfw > erosion_inputs.bank_protection_level[ib])
    dv_flow = dn_flow * linesize * h_line_flow
    dv_flow[~mask] = 0.0
    dn_flow[~mask] = 0.0

    dn = dn_ship + dn_flow
    dv = dv_ship + dv_flow

    # print("  dv_flow total = ", dv_flow.sum())
    # print("  dv_ship total = ", dv_ship.sum())
    return dn, dv, dn_ship, dn_flow, shipwavemax, shipwavemin


def comp_hw_ship_at_bank(
    distance_fw: numpy.ndarray,
    dfw0: numpy.ndarray,
    dfw1: numpy.ndarray,
    h_input: numpy.ndarray,
    ship_type: numpy.ndarray,
    Tship: numpy.ndarray,
    vship: numpy.ndarray,
    g: float,
) -> numpy.ndarray:
    """
    Compute wave heights at bank due to passing ships.
    
    Arguments
    ---------
    distance_fw : numpy.ndarray
        Array containing distance from bank to fairway [m]
    dfw0 : numpy.ndarray
        Array containing distance from fairway at which wave reduction starts [m]
    dfw1 : numpy.ndarray
        Array containing distance from fairway at which all waves are gone [m]
    h_input : numpy.ndarray
        Array containing the water depth at the fairway [m]
    ship_type : numpy.ndarray
        Array containing the ship type [-]
    Tship : numpy.ndarray
        Array containing draught of the ships [m]
    vship : numpy.ndarray
        Array containing velocity of the ships [m/s]
    g : float
        Gravitational acceleration [m/s2]
    
    Returns
    -------
    h0 : numpy.ndarray
        Array containing wave height at the bank [m]
    """
    h = numpy.copy(h_input)

    a1 = numpy.zeros(len(distance_fw))
    # multiple barge convoy set
    a1[ship_type == 1] = 0.5
    # RHK ship / motorship
    a1[ship_type == 2] = 0.28 * Tship[ship_type == 2] ** 1.25
    # towboat
    a1[ship_type == 3] = 1

    Froude = vship / numpy.sqrt(h * g)
    Froude_limit = 0.8
    high_Froude = Froude > Froude_limit
    h[high_Froude] = ((vship[high_Froude] / Froude_limit) ** 2) / g
    Froude[high_Froude] = Froude_limit

    A = 0.5 * (1 + numpy.cos((distance_fw - dfw1) / (dfw0 - dfw1) * numpy.pi))
    A[distance_fw < dfw1] = 1
    A[distance_fw > dfw0] = 0

    h0 = a1 * h * (distance_fw / h) ** (-1 / 3) * Froude ** 4 * A
    return h0


def get_km_bins(km_bin: Tuple[float, float, float], type: int = 2, adjust: bool = False) -> numpy.ndarray:
    """
    Get an array of representative chainage values.
    
    Arguments
    ---------
    km_bin : Tuple[float, float, float]
        Tuple containing (start, end, step) for the chainage bins
    type : int
        Type of characteristic chainage values returned
            0: all bounds (N+1 values)
            1: lower bounds (N values)
            2: upper bounds (N values) - default
            3: mid points (N values)
    adjust : bool
        Flag indicating whether the step size should be adjusted to include an integer number of steps
    
    Returns
    -------
    km : numpy.ndarray
        Array containing the chainage bin upper bounds
    """
    km_step = km_bin[2]
    nbins = int(math.ceil((km_bin[1] - km_bin[0]) / km_step))
    
    lb = 0
    ub = nbins + 1
    dx = 0.0
    
    if adjust:
        km_step = (km_bin[1] - km_bin[0]) / nbins

    if type == 0:
        # all bounds
        pass
    elif type == 1:
        # lower bounds
        ub = ub - 1
    elif type == 2:
        # upper bounds
        lb = lb + 1
    elif type == 3:
        # midpoint values
        ub = ub - 1
        dx = km_bin[2] / 2

    km = km_bin[0] + dx + numpy.arange(lb, ub) * km_step

    return km


def get_km_eroded_volume(
    bank_km_mid: numpy.ndarray, dv: numpy.ndarray, km_bin: Tuple[float, float, float]
) -> numpy.ndarray:
    """
    Accumulate the erosion volumes per chainage bin.
    
    Arguments
    ---------
    bank_km_mid : numpy.ndarray
        Array containing the chainage per bank segment [km]
    dv : numpy.ndarray
        Array containing the eroded volume per bank segment [m3]
    km_bin : Tuple[float, float, float]
        Tuple containing (start, end, step) for the chainage bins
        
    Returns
    -------
    dvol : numpy.ndarray
        Array containing the accumulated eroded volume per chainage bin.
    """
    km_step = km_bin[2]
    nbins = int(math.ceil((km_bin[1] - km_bin[0]) / km_step))
    
    bin_idx = numpy.rint((bank_km_mid - km_bin[0] - km_step / 2.0) / km_step).astype(
        numpy.int64
    )
    dvol_temp = numpy.bincount(bin_idx, weights=dv)
    length = int((km_bin[1] - km_bin[0]) / km_bin[2])
    if len(dvol_temp) == length:
       dvol = dvol_temp
    else:
       dvol = numpy.zeros((length,))
       dvol[:len(dvol_temp)] = dvol_temp
    return dvol


def moving_avg(xi: numpy.ndarray, yi: numpy.ndarray, dx: float) -> numpy.ndarray:
    """
    Perform a moving average for given averaging distance.
    
    Arguments
    ---------
    xi : numpy.ndarray
        Array containing the distance - should be monotonically increasing or decreasing [m or equivalent]
    yi : numpy.ndarray
        Array containing the values to be average [arbitrary]
    dx: float
        Averaging distance [same unit as x]
        
    Returns
    -------
    yo : numpy.ndarray
        Array containing the averaged values [same unit as y].
    """
    dx2 = dx / 2.0
    nx = len(xi)
    if xi[0] < xi[-1]:
        x = xi
        y = yi
    else:
        x = xi[::-1]
        y = yi[::-1]
    ym = numpy.zeros(y.shape)
    di = numpy.zeros(y.shape)
    j0 = 1
    for i in range(nx):
        for j in range(j0, nx):
            dxj = x[j] - x[j - 1]
            if x[i] - x[j] > dx2:
                # point j is too far back for point i and further
                j0 = j + 1
            elif x[j] - x[i] > dx2:
                # point j is too far ahead; wrap up and continue
                d0 = (x[i] + dx2) - x[j - 1]
                ydx2 = y[j - 1] + (y[j] - y[j - 1]) * d0 / dxj
                ym[i] += (y[j - 1] + ydx2) / 2.0 * d0
                di[i] += d0
                break
            elif x[i] - x[j - 1] > dx2:
                # point j is ok, but j-1 is too far back, so let's start
                d0 = x[j] - (x[i] - dx2)
                ydx2 = y[j] + (y[j - 1] - y[j]) * d0 / dxj
                ym[i] += (y[j] + ydx2) / 2.0 * d0
                di[i] += d0
            else:
                # segment right in the middle
                ym[i] += (y[j] + y[j - 1]) / 2.0 * dxj
                di[i] += dxj
    yo = ym / di
    if xi[0] < xi[-1]:
        return yo
    else:
        return yo[::-1]

def get_zoom_extends(km_min: float, km_max: float, zoom_km_step: float, bank_crds: List[numpy.ndarray], bank_km: List[numpy.ndarray]) -> [List[Tuple[float, float]], List[Tuple[float, float, float, float]]]:
    """
    Zoom .

    Arguments
    ---------
    km_min : float
        Minimum value for the chainage range of interest.
    km_max : float
        Maximum value for the chainage range of interest.
    zoom_km_step : float
        Preferred chainage length of zoom box.
    bank_crds : List[numpy.ndarray]
        List of N x 2 numpy arrays of coordinates per bank.
    bank_km : List[numpy.ndarray]
        List of N numpy arrays of chainage values per bank.

    Returns
    -------
    kmzoom : List[Tuple[float, float]]
        Zoom ranges for plots with chainage along x-axis.
    xyzoom : List[Tuple[float, float, float, float]]
        Zoom ranges for xy-plots.
    """

    zoom_km_bin = (km_min, km_max, zoom_km_step)
    zoom_km_bnd = get_km_bins(zoom_km_bin, type=0, adjust=True)
    eps = 0.1 * zoom_km_step

    kmzoom: List[Tuple[float, float]] = []
    xyzoom: List[Tuple[float, float, float, float]] = []
    inf = float('inf')
    for i in range(len(zoom_km_bnd)-1):
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
            if len(x) > 0:
                xmin = min(xmin, min(x))
                xmax = max(xmax, max(x))
                ymin = min(ymin, min(y))
                ymax = max(ymax, max(y))
        xyzoom.append((xmin, xmax, ymin, ymax))

    return kmzoom, xyzoom

