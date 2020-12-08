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

from typing import Tuple

import numpy
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
    dfw0: numpy.ndarray,
    dfw1: numpy.ndarray,
    hfw: numpy.ndarray,
    zss: numpy.ndarray,
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

    muslope = edge_mean(mu_slope)
    zssline = edge_mean(zss)
    wlline = edge_mean(zfw_ini)  # original water level at fairway

    # ship induced wave height at the beginning of the foreshore
    H0 = comp_hw_ship_at_bank(distance_fw, dfw0, dfw1, hfw, ship_type, Tship, vship, g)
    H0 = numpy.maximum(edge_mean(H0), eps)

    zup = numpy.minimum(bankheight, wlline + 2 * H0)
    zdo = numpy.maximum(wlline - 2 * H0, zssline)
    ht = numpy.maximum(zup - zdo, 0)
    hs = numpy.maximum(bankheight - wlline + 2 * H0, 0)
    dn_eq = ht / muslope
    dv_eq = (0.5 * ht + hs) * dn_eq * linesize

    return dn_eq, dv_eq


def comp_erosion(
    velocity: numpy.ndarray,
    bankheight: numpy.ndarray,
    linesize: numpy.ndarray,
    zfw: numpy.ndarray,
    zfw_ini: numpy.ndarray,
    tauc: numpy.ndarray,
    Nship: numpy.ndarray,
    vship: numpy.ndarray,
    nwave: numpy.ndarray,
    ship_type: numpy.ndarray,
    Tship: numpy.ndarray,
    Teros: float,
    mu_slope: numpy.ndarray,
    mu_reed: numpy.ndarray,
    distance_fw: numpy.ndarray,
    dfw0: numpy.ndarray,
    dfw1: numpy.ndarray,
    hfw: numpy.ndarray,
    chezy: numpy.ndarray,
    zss: numpy.ndarray,
    filter: bool,
    rho: float,
    g: float,
    displ_tauc: bool,
):
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
    filter : bool
        Flag indicating whether velocities should be smoothened slightly
    rho : float
        Water density [kg/m3]
    g : float
        Gravitational acceleration [m/s2]
    displ_tauc : bool
        Flag indicating whether displacement should be calculated based on critical shear stress (True) or velocity (False)
        
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
    """
    eps = sys.float_info.epsilon
    sec_year = 3600 * 24 * 365

    # period of ship waves [s]
    T = 0.51 * vship / g
    # [s]
    ts = (T * Nship * nwave)[
        :-1
    ]  # TODO: check for better solution to shorten ts by one ... edge_mean?

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
    wlline = edge_mean(zfw_ini)  # original water level at fairway
    z_line = edge_mean(zfw)  # water level at fairway

    # Average velocity with values of neighbouring lines
    if filter:
        vel = numpy.concatenate(
            (
                velocity[:1],
                0.5 * velocity[1:-1] + 0.25 * velocity[:-2] + 0.25 * velocity[2:],
                velocity[-1:],
            )
        )
    else:
        vel = velocity

    # ship induced wave height at the beginning of the foreshore
    H0 = comp_hw_ship_at_bank(distance_fw, dfw0, dfw1, hfw, ship_type, Tship, vship, g)
    H0 = numpy.maximum(edge_mean(H0), eps)

    # compute erosion parameters for each line part

    # Erosion coefficient of linesegements
    E = 0.2 * numpy.sqrt(taucline) * 1e-6

    # critical velocity along linesegements
    velc = numpy.sqrt(taucline / rho * Cline ** 2 / g)

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
        crit_ratio[mask] = (vel[mask] / velc[mask]) ** 2
    dn_flow = E * (crit_ratio - 1) * Teros * sec_year

    # compute displacement due to shipwaves
    mask = ((z_line - 2 * H0) < wlline) & (wlline < (z_line + 0.5 * H0))
    # limit mu -> 0

    dn_ship = cE * H0 ** 2 * ts * Teros
    dn_ship[~mask] = 0
    # dn_ship = dn_ship[0] #TODO: this selects only the first value ... correct? MATLAB compErosion: dn_ship=dn_ship(1);

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

    # print("  dv_flow total = ", dv_flow.sum())
    # print("  dv_ship total = ", dv_ship.sum())
    return dn, dv, dn_ship, dn_flow


def edge_mean(a_pnt: numpy.ndarray) -> numpy.ndarray:
    """
    Average values from bank points to bank segments.
    
    Arguments
    ---------
    a_pnt : numpy.ndarray
        Array containing N values for each bank point
    
    Returns
    -------
    a_sgm : numpy.ndarray
        Array containing N-1 values for each bank segment
    """
    a_sgm = 0.5 * (a_pnt[:-1] + a_pnt[1:])
    return a_sgm


def comp_hw_ship_at_bank(distance_fw: numpy.ndarray, dfw0: numpy.ndarray, dfw1: numpy.ndarray, h_input: numpy.ndarray, shiptype: numpy.ndarray, Tship: numpy.ndarray, vship: numpy.ndarray, g: float) -> numpy.ndarray:
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
    shiptype : numpy.ndarray
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
    a1[shiptype == 1] = 0.5
    # RHK ship / motorship
    a1[shiptype == 2] = 0.28 * Tship[shiptype == 2] ** 1.25
    # towboat
    a1[shiptype == 3] = 1

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


def get_km_bins(km_bin: Tuple[float, float, float], type: int = 2) -> numpy.ndarray:
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
    
    Returns
    -------
    km : numpy.ndarray
        Array containing the chainage bin upper bounds
    """
    km_step = km_bin[2]
    
    lb = 0
    ub = int(round((km_bin[1] - km_bin[0]) / km_bin[2])) + 1
    dx = 0.0
    
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
        dx = km_bin[2]/2
    
    km = km_bin[0] + dx + numpy.arange(lb, ub) * km_bin[2]

    return km


def get_km_eroded_volume(bank_km: numpy.ndarray, dv: numpy.ndarray, km_bin: Tuple[float, float, float]) -> numpy.ndarray:
    """
    Accumulate the erosion volumes per chainage bin.
    
    Arguments
    ---------
    bank_km : numpy.ndarray
        Array containing the chainage per bank point [km]
    dv : numpy.ndarray
        Array containing the eroded volume per bank segment [m3]
    km_bin : Tuple[float, float, float]
        Tuple containing (start, end, step) for the chainage bins
        
    Returns
    -------
    dvol : numpy.ndarray
        Array containing the accumulated eroded volume per chainage bin.
    """
    bank_km_mid = (bank_km[:-1] + bank_km[1:]) / 2
    bin_idx = numpy.rint((bank_km_mid - km_bin[0] - km_bin[2] / 2) / km_bin[2]).astype(
        numpy.int64
    )
    dvol = numpy.bincount(bin_idx, weights=dv)
    length = int((km_bin[1]-km_bin[0])/km_bin[2])
    dvol.resize((length,))
    return dvol
