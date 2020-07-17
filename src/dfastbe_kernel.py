# coding: utf-8
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

import numpy
import sys

def program_version():
    return 'PRE-ALPHA'


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
    print("----")
    print("E[1559] = {}".format(E[1559]))
    print("crit_ratio[1559] = {}".format(crit_ratio[1559]))
    print(" displ_tauc = {}".format(displ_tauc))
    print(" *:mask[1559] = {}".format(mask[1559]))
    print(" T:Cline[1559] = {}".format(Cline[1559]))
    print(" T:taucline[1559] = {}".format(taucline[1559]))
    print(" F:vel[1559] = {}".format(vel[1559]))
    print(" F:velc[1559] = {}".format(velc[1559]))
    print("dn_flow[1559] = {}".format(dn_flow[1559]))
    print("linesize[1559] = {}".format(linesize[1559]))

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
    
    #print("  dv_flow total = ", dv_flow.sum())
    #print("  dv_ship total = ", dv_ship.sum())
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


def get_km_bins(km_bin):
    km_step = km_bin[2]
    # km = km_bin[0] + numpy.arange(0, int(round((km_bin[1] - km_bin[0]) / km_bin[2]) + 1)) * km_bin[2] # bin bounds
    km = km_bin[0] + numpy.arange(1, int(round((km_bin[1] - km_bin[0]) / km_bin[2]) + 1)) * km_bin[2] # bin upper bounds
    # km = km_bin[0] + km_bin[2]/2 + numpy.arange(0, int(round((km_bin[1] - km_bin[0]) / km_bin[2]))) * km_bin[2] # bin midpoints

    return km


def get_km_eroded_volume(bank_km, dv, km_bin, col, vol):
    bank_km_mid = (bank_km[:-1] + bank_km[1:])/2
    bin_idx = numpy.rint((bank_km_mid - km_bin[0] - km_bin[2]/2) / km_bin[2]).astype(numpy.int64)
    dvol = numpy.bincount(bin_idx, weights = dv)
    nbin = len(dvol)
    vol[:nbin, col] += dvol
    return vol
