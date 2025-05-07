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
from dfastbe.bank_erosion.data_models import ErosionInputs, SingleErosion, ParametersPerBank

import numpy as np
import math
import sys

EPS = sys.float_info.epsilon
water_density = 1000  # density of water [kg/m3]
g = 9.81  # gravitational acceleration [m/s2]


def comp_erosion_eq(
    bank_height: np.ndarray,
    segment_length: np.ndarray,
    water_level_fairway_ref: np.ndarray,
    discharge_level_pars: ParametersPerBank,
    bank_fairway_dist: np.ndarray,
    water_depth_fairway: np.ndarray,
    erosion_inputs: SingleErosion,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the equilibrium bank erosion.
    
    Args:
        bank_height : np.ndarray
            Array containing bank height [m]
        segment_length : np.ndarray
            Array containing length of line segment [m]
        water_level_fairway_ref : np.ndarray
            Array containing water level at fairway [m]
        ship_velocity : np.ndarray
            Array containing ship velocity [m/s]
        ship_type : np.ndarray
            Array containing ship type [-]
        ship_draught : np.ndarray
            Array containing ship draught [m]
        mu_slope : np.ndarray
            Array containing slope [-]
        bank_fairway_dist : np.ndarray
            Array containing distance from bank to fairway [m]
        water_depth_fairway : np.ndarray
            Array containing water depth at the fairway [m]
        erosion_inputs (ErosionInputs):
            ErosionInputs object.
        bank_index: int
            bank_i = 0: left bank, bank_i = 1: right bank
    
    Returns:
        dn_eq : np.ndarray
             Equilibrium bank erosion distance [m]
        dv_eq : np.ndarray
             Equilibrium bank erosion volume [m]
    """
    # ship induced wave height at the beginning of the foreshore
    h0 = comp_hw_ship_at_bank(
        bank_fairway_dist,
        erosion_inputs.wave_fairway_distance_0,
        erosion_inputs.wave_fairway_distance_1,
        water_depth_fairway,
        discharge_level_pars.ship_type,
        discharge_level_pars.ship_draught,
        discharge_level_pars.ship_velocity,
    )
    h0 = np.maximum(h0, EPS)

    zup = np.minimum(bank_height, water_level_fairway_ref + 2 * h0)
    zdo = np.maximum(water_level_fairway_ref - 2 * h0, erosion_inputs.bank_protection_level)
    ht = np.maximum(zup - zdo, 0)
    hs = np.maximum(bank_height - water_level_fairway_ref + 2 * h0, 0)
    dn_eq = ht / discharge_level_pars.mu_slope
    dv_eq = (0.5 * ht + hs) * dn_eq * segment_length

    return dn_eq, dv_eq


def compute_bank_erosion_dynamics(
    velocity: np.ndarray,
    bank_height: np.ndarray,
    water_level_fairway: np.ndarray,
    chezy: np.ndarray,
    segment_length: np.ndarray,
    bank_fairway_dist: np.ndarray,
    water_level_fairway_ref: np.ndarray,
    discharge_level_pars: ParametersPerBank,
    time_erosion: float,
    water_depth_fairway: np.ndarray,
    erosion_inputs: SingleErosion,
) -> Tuple[np.ndarray]:
    """
    Compute the bank erosion during a specific discharge level.
    
    Arguments
    ---------
    velocity : np.ndarray
        Array containing flow velocity magnitude [m/s]
    bank_height : np.ndarray
        Array containing bank height
    segment_length : np.ndarray
        Array containing length of line segment [m]
    water_level_fairway : np.ndarray
        Array containing water levels at fairway [m]
    water_level_fairway_ref : np.ndarray
        Array containing reference water levels at fairway [m]
    tauc : np.ndarray
        Array containing critical shear stress [N/m2]
    discharge_level_pars: DischargeLevelParameters,
        num_ship : np.ndarray
            Array containing number of ships [-]
        ship_velocity : np.ndarray
            Array containing ship velocity [m/s]
        num_waves_per_ship : np.ndarray
            Array containing number of waves per ship [-]
        ship_type : np.ndarray
            Array containing ship type [-]
        ship_draught : np.ndarray
            Array containing ship draught [m]
    time_erosion : float
        Erosion period [yr]
    bank_fairway_dist : np.ndarray
        Array containing distance from bank to fairway [m]
    fairway_wave_reduction_distance : np.ndarray
        Array containing distance from fairway at which wave reduction starts [m]
    fairway_wave_disappear_distance : np.ndarray
        Array containing distance from fairway at which all waves are gone [m]
    water_depth_fairway : np.ndarray
        Array containing water depth at fairway [m]
    chezy : np.ndarray
        Array containing Chezy values [m0.5/s]
    dike_height : np.ndarray
        Array containing bank protection height [m]
    water_density : float
        Water density [kg/m3]
        
    Returns
    -------
    erosion_distance : np.ndarray
        Total bank erosion distance [m]
    erosion_volume : np.ndarray
        Total bank erosion volume [m]
    erosion_distance_shipping : np.ndarray
        Bank erosion distance due to shipping [m]
    erosion_distance_flow : np.ndarray
        Bank erosion distance due to current [m]
    ship_wave_max : np.ndarray
        Maximum bank level subject to ship waves [m]
    ship_wave_min : np.ndarray
        Minimum bank level subject to ship waves [m]
    """
    sec_year = 3600 * 24 * 365

    # period of ship waves [s]
    ship_wave_period = 0.51 * discharge_level_pars.ship_velocity / g
    ts = ship_wave_period * discharge_level_pars.num_ship * discharge_level_pars.num_waves_per_ship
    vel = velocity

    # the ship induced wave height at the beginning of the foreshore
    wave_height = comp_hw_ship_at_bank(
        bank_fairway_dist,
        erosion_inputs.wave_fairway_distance_0,
        erosion_inputs.wave_fairway_distance_1,
        water_depth_fairway,
        discharge_level_pars.ship_type,
        discharge_level_pars.ship_draught,
        discharge_level_pars.ship_velocity,
    )
    wave_height = np.maximum(wave_height, EPS)

    # compute erosion parameters for each line part
    # erosion coefficient
    erosion_coef = 0.2 * np.sqrt(erosion_inputs.tauc) * 1e-6

    # critical velocity
    critical_velocity = np.sqrt(erosion_inputs.tauc / water_density * chezy ** 2 / g)

    # strength
    cE = 1.85e-4 / erosion_inputs.tauc

    # total wave damping coefficient
    # mu_tot = (mu_slope / H0) + mu_reed
    # water level along bank line
    ho_line_ship = np.minimum(water_level_fairway - erosion_inputs.bank_protection_level, 2 * wave_height)
    ho_line_flow = np.minimum(water_level_fairway - erosion_inputs.bank_protection_level, water_depth_fairway)
    h_line_ship = np.maximum(bank_height - water_level_fairway + ho_line_ship, 0)
    h_line_flow = np.maximum(bank_height - water_level_fairway + ho_line_flow, 0)

    # compute displacement due to flow
    crit_ratio = np.ones(critical_velocity.shape)
    mask = (vel > critical_velocity) & (water_level_fairway > erosion_inputs.bank_protection_level)
    crit_ratio[mask] = (vel[mask] / critical_velocity[mask]) ** 2
    erosion_distance_flow = erosion_coef * (crit_ratio - 1) * time_erosion * sec_year

    # compute displacement due to ship waves
    ship_wave_max = water_level_fairway + 0.5 * wave_height
    ship_wave_min = water_level_fairway - 2 * wave_height
    mask = (ship_wave_min < water_level_fairway_ref) & (water_level_fairway_ref < ship_wave_max)
    # limit mu -> 0

    erosion_distance_shipping = cE * wave_height ** 2 * ts * time_erosion
    erosion_distance_shipping[~mask] = 0

    # compute erosion volume
    mask = (h_line_ship > 0) & (water_level_fairway > erosion_inputs.bank_protection_level)
    dv_ship = erosion_distance_shipping * segment_length * h_line_ship
    dv_ship[~mask] = 0.0
    erosion_distance_shipping[~mask] = 0.0

    mask = (h_line_flow > 0) & (water_level_fairway > erosion_inputs.bank_protection_level)
    dv_flow = erosion_distance_flow * segment_length * h_line_flow
    dv_flow[~mask] = 0.0
    erosion_distance_flow[~mask] = 0.0

    erosion_distance = erosion_distance_shipping + erosion_distance_flow
    erosion_volume = dv_ship + dv_flow

    return erosion_distance, erosion_volume, erosion_distance_shipping, erosion_distance_flow, ship_wave_max, ship_wave_min


def comp_hw_ship_at_bank(
    bank_fairway_dist: np.ndarray,
    fairway_wave_reduction_distance: np.ndarray,
    fairway_wave_disappear_distance: np.ndarray,
    water_depth_fairway: np.ndarray,
    ship_type: np.ndarray,
    ship_draught: np.ndarray,
    ship_velocity: np.ndarray,
) -> np.ndarray:
    """
    Compute wave heights at bank due to passing ships.
    
    Arguments
    ---------
    bank_fairway_dist : np.ndarray
        Array containing distance from bank to fairway [m]
    fairway_wave_reduction_distance : np.ndarray
        Array containing distance from fairway at which wave reduction starts [m]
    fairway_wave_disappear_distance : np.ndarray
        Array containing distance from fairway at which all waves are gone [m]
    water_depth_fairway : np.ndarray
        Array containing the water depth at the fairway [m]
    ship_type : np.ndarray
        Array containing the ship type [-]
    ship_draught : np.ndarray
        Array containing draught of the ships [m]
    ship_velocity : np.ndarray
        Array containing velocity of the ships [m/s]
    g : float
        Gravitational acceleration [m/s2]
    
    Returns
    -------
    h0 : np.ndarray
        Array containing wave height at the bank [m]
    """
    h = np.copy(water_depth_fairway)

    a1 = np.zeros(len(bank_fairway_dist))
    # multiple barge convoy set
    a1[ship_type == 1] = 0.5
    # RHK ship / motor ship
    a1[ship_type == 2] = 0.28 * ship_draught[ship_type == 2] ** 1.25
    # towboat
    a1[ship_type == 3] = 1

    froude = ship_velocity / np.sqrt(h * g)
    froude_limit = 0.8
    high_froude = froude > froude_limit
    h[high_froude] = ((ship_velocity[high_froude] / froude_limit) ** 2) / g
    froude[high_froude] = froude_limit

    A = 0.5 * (1 + np.cos((bank_fairway_dist - fairway_wave_disappear_distance) / (fairway_wave_reduction_distance - fairway_wave_disappear_distance) * np.pi))
    A[bank_fairway_dist < fairway_wave_disappear_distance] = 1
    A[bank_fairway_dist > fairway_wave_reduction_distance] = 0

    h0 = a1 * h * (bank_fairway_dist / h) ** (-1 / 3) * froude ** 4 * A
    return h0


def get_km_bins(km_bin: Tuple[float, float, float], type: int = 2, adjust: bool = False) -> np.ndarray:
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
    km : np.ndarray
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

    km = km_bin[0] + dx + np.arange(lb, ub) * km_step

    return km


def get_km_eroded_volume(
    bank_km_mid: np.ndarray, erosion_volume: np.ndarray, km_bin: Tuple[float, float, float]
) -> np.ndarray:
    """
    Accumulate the erosion volumes per chainage bin.
    
    Arguments
    ---------
    bank_km_mid : np.ndarray
        Array containing the chainage per bank segment [km]
    erosion_volume : np.ndarray
        Array containing the eroded volume per bank segment [m3]
    km_bin : Tuple[float, float, float]
        Tuple containing (start, end, step) for the chainage bins
        
    Returns
    -------
    dvol : np.ndarray
        Array containing the accumulated eroded volume per chainage bin.
    """
    km_step = km_bin[2]
    
    bin_idx = np.rint((bank_km_mid - km_bin[0] - km_step / 2.0) / km_step).astype(
        np.int64
    )
    dvol_temp = np.bincount(bin_idx, weights=erosion_volume)
    length = int((km_bin[1] - km_bin[0]) / km_bin[2])
    if len(dvol_temp) == length:
       dvol = dvol_temp
    else:
       dvol = np.zeros((length,))
       dvol[:len(dvol_temp)] = dvol_temp
    return dvol


def moving_avg(xi: np.ndarray, yi: np.ndarray, dx: float) -> np.ndarray:
    """
    Perform a moving average for given averaging distance.
    
    Arguments
    ---------
    xi : np.ndarray
        Array containing the distance - should be monotonically increasing or decreasing [m or equivalent]
    yi : np.ndarray
        Array containing the values to be average [arbitrary]
    dx: float
        Averaging distance [same unit as x]
        
    Returns
    -------
    yo : np.ndarray
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
    ym = np.zeros(y.shape)
    di = np.zeros(y.shape)
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

def get_zoom_extends(km_min: float, km_max: float, zoom_km_step: float, bank_crds: List[np.ndarray], bank_km: List[np.ndarray]) -> [List[Tuple[float, float]], List[Tuple[float, float, float, float]]]:
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
    bank_crds : List[np.ndarray]
        List of N x 2 np arrays of coordinates per bank.
    bank_km : List[np.ndarray]
        List of N np arrays of chainage values per bank.

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

