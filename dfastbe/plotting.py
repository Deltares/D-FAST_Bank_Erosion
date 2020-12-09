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

from typing import List, Tuple
import shapely

import matplotlib
import matplotlib.pyplot
import geopandas
import numpy


def savefig(fig: matplotlib.figure.Figure, filename: str) -> None:
    """
    Save a single figure to file.
    
    Arguments
    ---------
    fig : matplotlib.figure.Figure
        Figure to a be saved.
    filename : str
        Name of the file to be written.
    """
    print("saving figure {file}".format(file=filename))
    matplotlib.pyplot.show(block=False)
    fig.savefig(filename, dpi=600)


def setsize(fig: matplotlib.figure.Figure) -> None:
    """
    Set the size of a figure.
    
    Currently the size is hardcoded, but functionality may be extended in the
    future.
    
    Arguments
    ---------
    fig : matplotlib.figure.Figure
        Figure to a be saved.
    """
    fig.set_size_inches(11.75, 8.25)  # a4
    # fig.set_size_inches(16.5, 11.75) # a3


def set_bbox(
    ax: matplotlib.axes.Axes,
    bbox: Tuple[float, float, float, float],
    scale: float = 1000,
) -> None:
    """
    Specify the bounding limits of an axes object.
    
    Arguments
    ---------
    ax : matplotlib.axes.Axes
        Axes object to be adjusted. 
    bbox : Tuple[float, float, float, float]
        Tuple containing boundary limits (xmin, ymin, xmax, ymax); unit m.
    scale: float
        Indicates whether the axes are in m (1) or km (1000).
    """
    ax.set_xlim(xmin=bbox[0] / scale, xmax=bbox[2] / scale)
    ax.set_ylim(ymin=bbox[1] / scale, ymax=bbox[3] / scale)


def chainage_markers(
    xykm: numpy.ndarray, ax: matplotlib.axes.Axes, ndec: int = 1, scale: float = 1000
) -> None:
    """
    Add markers indicating the river chainage to a plot.
    
    Arguments
    ---------
    xykm : numpy.ndarray
        Array containing the x, y, and chainage; unit m for x and y, km for chainage.
    ax : matplotlib.axes.Axes
        Axes object in which to add the markers.
    ndec : int
        Number of decimals used for marks.
    scale: float
        Indicates whether the axes are in m (1) or km (1000).
    """
    step = 10 ** (-ndec)
    labelstr = " {:." + str(ndec) + "f}"
    km_rescaled = xykm[:, 2] / step
    mask = numpy.isclose(numpy.round(km_rescaled), km_rescaled)
    ax.plot(
        xykm[mask, 0] / scale,
        xykm[mask, 1] / scale,
        linestyle="None",
        marker="+",
        color="k",
    )
    for i in numpy.nonzero(mask)[0]:
        ax.text(
            xykm[i, 0] / scale,
            xykm[i, 1] / scale,
            labelstr.format(xykm[i, 2]),
            fontsize="x-small",
            clip_on=True,
        )


def plot_mesh(
    ax: matplotlib.axes.Axes, xe: numpy.ndarray, ye: numpy.ndarray, scale: float = 1000
) -> None:
    """
    Add a mesh to a plot.
    
    Arguments
    ---------
    ax : matplotlib.axes.Axes
        Axes object in which to add the mesh.
    xe : numpy.ndarray
        M x 2 array of begin/end x-coordinates of mesh edges.
    ye : numpy.ndarray
        M x 2 array of begin/end y-coordinates of mesh edges.
    scale : float
        Indicates whether the axes are in m (1) or km (1000).
    """
    xe1 = xe[:, (0, 1, 1)] / scale
    xe1[:, 2] = numpy.nan
    xev = xe1.reshape((xe1.size,))

    ye1 = ye[:, (0, 1, 1)] / scale
    ye1[:, 2] = numpy.nan
    yev = ye1.reshape((ye1.size,))

    # to avoid OverflowError: In draw_path: Exceeded cell block limit
    # plot the data in chunks ...
    matplotlib.rcParams["agg.path.chunksize"] = 10000
    ax.plot(xev, yev, color=(0.5, 0.5, 0.5), linewidth=0.25)


def plot_mesh_patches(
    ax: matplotlib.axes.Axes,
    fn: numpy.ndarray,
    nnodes: numpy.ndarray,
    xn: numpy.ndarray,
    yn: numpy.ndarray,
    val: numpy.ndarray,
    minval: float,
    maxval: float,
    scale: float = 1000,
) -> matplotlib.collections.PolyCollection:
    """
    Add a collection of patches to the plot one for every face of the mesh.
    
    Arguments
    ---------
    ax : matplotlib.axes.Axes
        Axes object in which to add the mesh.
    fn : numpy.ndarray
        N x M array listing the nodes (max M) per face (total N) of the mesh.
    nnodes : numpy.ndarray
        Number of nodes per face (max M).
    xn : numpy.ndarray
        X-coordinates of the mesh nodes.
    yn : numpy.ndarray
        Y-coordinates of the mesh nodes.
    val : numpy.ndarray
        Array of length N containing the value per face.
    minval : float
        Lower limit for the color scale.
    maxval : float
        Upper limit for the color scale.
    scale : float
        Indicates whether the axes are in m (1) or km (1000).
    
    Returns
    -------
    p : matplotlib.collections.PolyCollection
        Patches object.
    """
    tfn_list = []
    tval_list = []
    for n in range(3, max(nnodes) + 1):
        mask = nnodes >= n
        fn_masked = fn[mask, :]
        tfn_list.append(fn_masked[:, (0, n - 2, n - 1)])
        tval_list.append(val[mask])
    tfn = numpy.concatenate(tfn_list, axis=0)
    tval = numpy.concatenate(tval_list, axis=0)
    # cmap = matplotlib.pyplot.get_cmap('Spectral')
    p = ax.tripcolor(
        xn / scale,
        yn / scale,
        tfn,
        facecolors=tval,
        cmap="Spectral",
        vmin=minval,
        vmax=maxval,
    )
    return p


def plot_detect1(
    bbox: Tuple[float, float, float, float],
    xykm: numpy.ndarray,
    bankareas: List[shapely.geometry.polygon.Polygon],
    bank: List[shapely.geometry.linestring.LineString],
    fn: numpy.ndarray,
    nnodes: numpy.ndarray,
    xn: numpy.ndarray,
    yn: numpy.ndarray,
    h: numpy.ndarray,
    hmax: float,
    xlabel_txt: str,
    ylabel_txt: str,
    title_txt: str,
    waterdepth_txt: str,
) -> matplotlib.figure.Figure:
    """
    Create the bank line detection plot.
    
    The figure contains a map of the water depth, the chainage, and detected
    bank lines.
    
    Arguments
    ---------
    bbox : Tuple[float, float, float, float]
        Tuple containing boundary limits (xmin, ymin, xmax, ymax); unit m.
    xykm : numpy.ndarray
        Array containing the x, y, and chainage; unit m for x and y, km for chainage.
    bankareas : List[shapely.geometry.polygon.Polygon]
        List of bank polygons.
    bank : List[shapely.geometry.linestring.LineString]
        List of bank lines.
    fn : numpy.ndarray
        N x M array listing the nodes (max M) per face (total N) of the mesh.
    nnodes : numpy.ndarray
        Number of nodes per face (max M).
    xn : numpy.ndarray
        X-coordinates of the mesh nodes.
    yn : numpy.ndarray
        Y-coordinates of the mesh nodes.
    h : numpy.ndarray
        Array of water depth values.
    hmax : float
        Water depth value to be used as upper limit for coloring.
    xlabel_txt : str
        Label for the x-axis.
    ylabel_txt : str
        Label for the y-axis.
    title_txt : str
        Label for the axes title.
    waterdepth_txt : str
        Label for the color bar.
    
    Returns
    -------
    fig : matplotlib.figure.Figure:
        Figure object.
    """
    fig, ax = matplotlib.pyplot.subplots()
    setsize(fig)
    ax.set_aspect(1)
    #
    chainage_markers(xykm, ax, ndec=0, scale=1)
    p = plot_mesh_patches(ax, fn, nnodes, xn, yn, h, 0, hmax, scale=1)
    for b, bankarea in enumerate(bankareas):
        geopandas.GeoSeries(bankarea).plot(ax=ax, alpha=0.2, color="k")
        geopandas.GeoSeries(bank[b]).plot(ax=ax, color="r")
    cbar = fig.colorbar(p, ax=ax, shrink=0.5, drawedges=False, label=waterdepth_txt)
    #
    set_bbox(ax, bbox, scale=1)
    ax.set_xlabel(xlabel_txt)
    ax.set_ylabel(ylabel_txt)
    ax.grid(True)
    ax.set_title(title_txt)
    return fig


def plot1_waterdepth_and_banklines(
    bbox: Tuple[float, float, float, float],
    xykm: numpy.ndarray,
    banklines: geopandas.geodataframe.GeoDataFrame,
    fn: numpy.ndarray,
    nnodes: numpy.ndarray,
    xn: numpy.ndarray,
    yn: numpy.ndarray,
    h: numpy.ndarray,
    hfw_max: float,
    xlabel_txt: str,
    ylabel_txt: str,
    title_txt: str,
    waterdepth_txt: str,
) -> matplotlib.figure.Figure:
    """
    Create the bank erosion plot with water depths and initial bank lines.
    
    bbox : Tuple[float, float, float, float]
        Tuple containing boundary limits (xmin, ymin, xmax, ymax); unit m.
    xykm : numpy.ndarray
        Array containing the x, y, and chainage; unit m for x and y, km for chainage.
    banklines : geopandas.geodataframe.GeoDataFrame
        Pandas object containing the bank lines.
        
    fn : numpy.ndarray
        N x M array listing the nodes (max M) per face (total N) of the mesh.
    nnodes : numpy.ndarray
        Number of nodes per face (max M).
    xn : numpy.ndarray
        X-coordinates of the mesh nodes.
    yn : numpy.ndarray
        Y-coordinates of the mesh nodes.
    h : numpy.ndarray
        Array of water depth values.
    hfw_max : float
        Water depth value to be used as upper limit for coloring.
    xlabel_txt : str
        Label for the x-axis.
    ylabel_txt : str
        Label for the y-axis.
    title_txt : str
        Label for the axes title.
    waterdepth_txt : str
        Label for the color bar.
    
    Results
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    """
    fig, ax = matplotlib.pyplot.subplots()
    setsize(fig)
    ax.set_aspect(1)
    #
    scale = 1000
    chainage_markers(xykm, ax, ndec=0, scale=scale)
    ax.plot(xykm[:, 0] / scale, xykm[:, 1] / scale, linestyle="--", color="k")
    for bl in banklines.geometry:
        bp = numpy.array(bl)
        ax.plot(bp[:, 0] / scale, bp[:, 1] / scale, color="k")
    p = plot_mesh_patches(ax, fn, nnodes, xn, yn, h, 0, 1.1 * hfw_max)
    cbar = fig.colorbar(p, ax=ax, shrink=0.5, drawedges=False, label=waterdepth_txt)
    #
    set_bbox(ax, bbox)
    ax.set_xlabel(xlabel_txt)
    ax.set_ylabel(ylabel_txt)
    ax.grid(True)
    ax.set_title(title_txt)
    return fig


def plot2_eroded_distance_and_equilibrium(
    bbox: Tuple[float, float, float, float],
    xykm: numpy.ndarray,
    bank_crds: List[numpy.ndarray],
    dn_tot: List[numpy.ndarray],
    to_right: List[bool],
    dnav: numpy.ndarray,
    xy_eq: List[numpy.ndarray],
    xe: numpy.ndarray,
    ye: numpy.ndarray,
    xlabel_txt: str,
    ylabel_txt: str,
    title_txt: str,
) -> matplotlib.figure.Figure:
    """
    Create the bank erosion plot with predicted bank line shift and equilibrium bank line.
    
    Arguments
    ---------
    bbox : Tuple[float, float, float, float]
        Tuple containing boundary limits (xmin, ymin, xmax, ymax); unit m.
    xykm : numpy.ndarray
        Array containing the x, y, and chainage; unit m for x and y, km for chainage.
    bank_crds : List[numpy.ndarray]
        List of N arrays containing the x- and y-coordinates of the oroginal
        bank lines.
    dn_tot : List[numpy.ndarray]
        List of N arrays containing the total erosion distance values.
    to_right : List[bool]
        List of N booleans indicating whether the bank is on the right.
    dnav : numpy.ndarray
        Array of N average erosion distance values.
    xy_eq : List[numpy.ndarray]
        List of N arrays containing the x- and y-coordinates of the equilibrium
        bank line.
    xe : numpy.ndarray
        M x 2 array of begin/end x-coordinates of mesh edges.
    ye : numpy.ndarray
        M x 2 array of begin/end y-coordinates of mesh edges.
    xlabel_txt : str
        Label for the x-axis.
    ylabel_txt : str
        Label for the y-axis.
    title_txt : str
        Label for the axes title.
    
    Results
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    """
    scale = 1000
    fig, ax = matplotlib.pyplot.subplots()
    setsize(fig)
    ax.set_aspect(1)
    #
    # plot_mesh(ax, xe, ye)
    chainage_markers(xykm, ax, ndec=0, scale=scale)
    dnav_max = dnav.max()
    for ib in range(len(xy_eq)):
        ax.plot(
            xy_eq[ib][:, 0] / scale, xy_eq[ib][:, 1] / scale, linewidth=2, color="k"
        )
        #
        if to_right[ib]:
            bankc = bank_crds[ib]
            dnc = dn_tot[ib]
        else:
            bankc = bank_crds[ib][::-1]
            dnc = dn_tot[ib][::-1]
        nbp = len(bankc)
        #
        dxy = bankc[1:] - bankc[:-1]
        ds = numpy.sqrt((dxy ** 2).sum(axis=1))
        dxy = dxy * (dn_tot[ib] / ds).reshape((nbp - 1, 1))
        #
        x = numpy.zeros(((nbp - 1) * 4,))
        x[0::4] = bankc[:-1, 0]
        x[1::4] = bankc[1:, 0]
        x[2::4] = bankc[:-1, 0] + dxy[:, 1]
        x[3::4] = bankc[1:, 0] + dxy[:, 1]
        #
        y = numpy.zeros(((nbp - 1) * 4,))
        y[0::4] = bankc[:-1, 1]
        y[1::4] = bankc[1:, 1]
        y[2::4] = bankc[:-1, 1] - dxy[:, 0]
        y[3::4] = bankc[1:, 1] - dxy[:, 0]
        #
        tfn = numpy.zeros(((nbp - 1) * 2, 3))
        tfn[0::2, 0] = [4 * i for i in range(nbp - 1)]
        tfn[0::2, 1] = tfn[0::2, 0] + 1
        tfn[0::2, 2] = tfn[0::2, 0] + 2
        #
        tfn[1::2, 0] = tfn[0::2, 0] + 1
        tfn[1::2, 1] = tfn[0::2, 0] + 2
        tfn[1::2, 2] = tfn[0::2, 0] + 3
        #
        tval = numpy.zeros(((nbp - 1) * 2,))
        tval[0::2] = dnc
        tval[1::2] = dnc
        #
        colors = ["lawngreen", "gold", "darkorange"]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mycmap", colors)
        p = ax.tripcolor(
            x / scale,
            y / scale,
            tfn,
            facecolors=tval,
            edgecolors="face",
            linewidth=3,
            cmap=cmap,
            vmin=0,
            vmax=2 * dnav_max,
        )
    #
    cbar = fig.colorbar(
        p, ax=ax, shrink=0.5, drawedges=False, label="eroded distance [m]"
    )
    set_bbox(ax, bbox)
    ax.set_xlabel(xlabel_txt)
    ax.set_ylabel(ylabel_txt)
    ax.grid(True)
    ax.set_title(title_txt)
    return fig


def plot3_eroded_volume_subdivided_1(
    km_mid: numpy.ndarray,
    km_step: float,
    chainage_txt: str,
    dv: List[List[numpy.ndarray]],
    ylabel_txt: str,
    title_txt: str,
    qlabel: str,
) -> matplotlib.figure.Figure:
    """
    Create the bank erosion plot with total eroded volume subdivided per discharge level.
    
    Arguments
    ---------
    km_mid : numpy.ndarray
        Array containing the mid points for the chainage bins.
    km_step : float
        Bin width.
    chainage_txt : str
        Label for the horizontal chainage axes.
    dv : List[List[numpy.ndarray]]
        List of nQ lists of N arrays containing the total erosion distance values
    ylabel_txt : str
        Label for the vertical erosion volume axes.
    title_txt : str
        Label for axes title.
    qlabel : str
        Label for discharge level.
    
    Results
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    """
    fig, ax = matplotlib.pyplot.subplots()
    setsize(fig)
    #
    n_levels = len(dv)
    clrs = get_colors("Blues", n_levels + 1)
    for iq in range(n_levels):
        for ib in range(len(dv[iq])):
            if ib == 0:
                dvq = dv[iq][ib].copy()
            else:
                dvq = dvq + dv[iq][ib]
        if iq == 0:
            ax.bar(
                km_mid,
                dvq,
                width=0.8 * km_step,
                color=clrs[iq + 1],
                label=qlabel.format(iq=iq + 1),
            )
            cumdv = dvq
        else:
            ax.bar(
                km_mid,
                dvq,
                width=0.8 * km_step,
                bottom=cumdv,
                color=clrs[iq + 1],
                label=qlabel.format(iq=iq + 1),
            )
            cumdv = cumdv + dvq
    #
    ax.set_xlabel(chainage_txt)
    ax.set_ylabel(ylabel_txt)
    ax.grid(True)
    ax.set_title(title_txt)
    ax.legend(loc="upper right")
    return fig


def plot3_eroded_volume_subdivided_2(
    km_mid: numpy.ndarray,
    km_step: float,
    chainage_txt: str,
    dv: List[List[numpy.ndarray]],
    ylabel_txt: str,
    title_txt: str,
    banklabel: str,
) -> matplotlib.figure.Figure:
    """
    Create the bank erosion plot with total eroded volume subdivided per bank.
    
    Arguments
    ---------
    km_mid : numpy.ndarray
        Array containing the mid points for the chainage bins.
    km_step : float
        Bin width.
    chainage_txt : str
        Label for the horizontal chainage axes.
    dv : List[List[numpy.ndarray]]
        List of nQ lists of N arrays containing the total erosion distance values
    ylabel_txt : str
        Label for the vertical erosion volume axes.
    title_txt : str
        Label for axes title.
    banklabel : str
        Label for bank id.
    
    Results
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    """
    fig, ax = matplotlib.pyplot.subplots()
    setsize(fig)
    #
    n_banklines = len(dv[0])
    clrs = get_colors("plasma", n_banklines + 1)
    for ib in range(n_banklines):
        for iq in range(len(dv)):
            if iq == 0:
                dvq = dv[iq][ib].copy()
            else:
                dvq = dvq + dv[iq][ib]
        if ib == 0:
            ax.bar(
                km_mid,
                dvq,
                width=0.8 * km_step,
                color=clrs[ib],
                label=banklabel.format(ib=ib + 1),
            )
            cumdv = dvq
        else:
            ax.bar(
                km_mid,
                dvq,
                width=0.8 * km_step,
                bottom=cumdv,
                color=clrs[ib],
                label=banklabel.format(ib=ib + 1),
            )
            cumdv = cumdv + dvq
    #
    ax.set_xlabel(chainage_txt)
    ax.set_ylabel(ylabel_txt)
    ax.grid(True)
    ax.set_title(title_txt)
    ax.legend(loc="upper right")
    return fig


def plot4_eroded_volume_eq(
    km_mid: numpy.ndarray,
    km_step: float,
    chainage_txt: str,
    vol_eq: numpy.ndarray,
    ylabel_txt: str,
    title_txt: str,
) -> matplotlib.figure.Figure:
    """
    Create the bank erosion plot with equilibrium eroded volume.
    
    Arguments
    ---------
    km_mid : numpy.ndarray
        Array containing the mid points for the chainage bins.
    km_step : float
        Bin width.
    chainage_txt : str
        Label for the horizontal chainage axes.
    vol_eq : numpy.ndarray
        Array containing the equilibrium eroded volume per bin.
    ylabel_txt : str
        Label for the vertical erosion volume axes.
    title_txt : str
        Label for axes title.
    
    Results
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    """
    fig, ax = matplotlib.pyplot.subplots()
    setsize(fig)
    #
    tvol = numpy.zeros(km_mid.shape)
    for i in range(len(km_mid)):
        tvol[i] = vol_eq[i].sum()
    ax.bar(km_mid, tvol, width=0.8 * km_step)
    #
    ax.set_xlabel(chainage_txt)
    ax.set_ylabel(ylabel_txt)
    ax.grid(True)
    ax.set_title(title_txt)
    return fig


def plot5series_waterlevels_per_bank(
    bank_km: List[numpy.ndarray],
    chainage_txt: str,
    waterlevel: List[List[numpy.ndarray]],
    waterlevelq_txt: str,
    avg_waterlevel_txt: str,
    bankheight: List[numpy.ndarray],
    bankheight_txt: str,
    bankprotect: List[numpy.ndarray],
    bankprotect_txt: str,
    elevation_txt: str,
    title_txt: str,
    elevation_unit: str,
) -> List[matplotlib.figure.Figure]:
    """
    Create the bank erosion plots with water levels, bank height and bank protection height along each bank.
    
    Arguments
    ---------
    bank_km : List[numpy.ndarray]
        List of arrays containing the chainage values per bank (point) [km].
    chainage_txt : str
        Label for the horizontal chainage axes.
    waterlevel : List[List[numpy.ndarray]]
        List of arrays containing the water levels per bank (point) [elevation_unit].
    waterlevelq_txt : str
        Label for the water level per discharge level.
    avg_waterlevel_txt : str
        Label for the average water level.
    bankheight : List[numpy.ndarray]
        List of arrays containing the bank heights per bank (segment) [elevation_unit].
    bankheight_txt : str
        Label for the bank height.
    bankprotect : List[numpy.ndarray]
        List of arrays containing the bank protection height per bank (point) [elevation_unit].
    bankprotect_txt : str
        Label for the bank protection height.
    elevation_txt : str
        General label for elevation data.
    title_txt : str
        Label for the axes title.
    elevation_unit : str
        Unit used for all elevation data.
    
    Results
    -------
    fig : List[matplotlib.figure.Figure]
        List of figure object, one per bank.
    """
    n_banklines = len(bank_km)
    n_levels = len(waterlevel)
    figlist = []
    clrs = get_colors("Blues", n_levels + 1)
    for ib in range(n_banklines):
        fig, ax = matplotlib.pyplot.subplots()
        setsize(fig)
        bk = (bank_km[ib][:-1] + bank_km[ib][1:]) / 2
        #
        for iq in range(n_levels):
            if iq == 0:
                wl_avg = waterlevel[iq][ib].copy()
            else:
                wl_avg = wl_avg + waterlevel[iq][ib]
        wl_avg = wl_avg / n_levels
        ax.plot(
            bank_km[ib],
            wl_avg,
            color=(0.5, 0.5, 0.5),
            linewidth=2,
            label=avg_waterlevel_txt,
        )
        ax.plot(
            bank_km[ib],
            bankprotect[ib],
            color=(0.5, 0.5, 0.5),
            linestyle="--",
            label=bankprotect_txt,
        )
        ax.plot(bk, bankheight[ib], color=(0.5, 0.5, 0.5), label=bankheight_txt)
        #
        for iq in range(n_levels):
            ax.plot(
                bank_km[ib],
                waterlevel[iq][ib],
                color=clrs[iq + 1],
                label=waterlevelq_txt.format(iq=iq + 1),
            )
        ax.set_xlabel(chainage_txt)
        ax.set_ylabel(elevation_txt + " " + elevation_unit)
        ax.grid(True)
        ax.set_title(title_txt.format(ib=ib + 1))
        ax.legend(loc="upper right")
        figlist.append(fig)
    return figlist


def plot6series_velocity_per_bank(
    bank_km: List[numpy.ndarray],
    chainage_txt: str,
    veloc: List[List[numpy.ndarray]],
    velocq_txt: str,
    tauc: List[numpy.ndarray],
    chezy: List[numpy.ndarray],
    rho: float,
    g: float,
    ucrit_txt: str,
    ylabel_txt: str,
    title_txt: str,
    veloc_unit: str,
) -> List[matplotlib.figure.Figure]:
    """
    Create the bank erosion plots with velocities and critical velocities along each bank.
    
    Arguments
    ---------
    bank_km : List[numpy.ndarray]
        List of arrays containing the chainage values per bank (point) [km].
    chainage_txt : str
        Label for the horizontal chainage axes.
    veloc: List[List[numpy.ndarray]]
        List of arrays containing the velocities per bank (segment) [m/s].
    velocq_txt: str,
        Label for the velocity per discharge level.
    tauc: List[numpy.ndarray]
        List of arrays containing the shear stresses per bank (point) [N/m2].
    chezy: List[numpy.ndarray]
        List of arrays containing the Chezy values per bank [m0.5/s].
    rho: float
        Water density [kg/m3].
    g: float
        Gravitational acceleration [m/s2].
    ucrit_txt: str
        Label for the critical velocity.
    ylabel_txt: str
        Label for the vertical (velocity) axis.
    title_txt: str
        Label for the axes title.
    veloc_unit: str
        Unit used for all velocities.
    
    Results
    -------
    fig : List[matplotlib.figure.Figure]
        List of figure object, one per bank.
    """
    n_banklines = len(bank_km)
    n_levels = len(veloc)
    figlist = []
    clrs = get_colors("Blues", n_levels + 1)
    for ib in range(n_banklines):
        fig, ax = matplotlib.pyplot.subplots()
        setsize(fig)
        bk = (bank_km[ib][:-1] + bank_km[ib][1:]) / 2
        #
        velc = numpy.sqrt(tauc[ib] * chezy[ib] ** 2 / (rho * g))
        ax.plot(bank_km[ib], velc, color="k", label=ucrit_txt)
        for iq in range(n_levels):
            ax.plot(
                bk,
                veloc[iq][ib],
                color=clrs[iq + 1],
                label=velocq_txt.format(iq=iq + 1),
            )
        #
        ax.set_xlabel(chainage_txt)
        ax.set_ylabel(ylabel_txt + " " + veloc_unit)
        ax.grid(True)
        ax.set_title(title_txt.format(ib=ib + 1))
        ax.legend(loc="upper right")
        figlist.append(fig)
    return figlist


def plot7_banktype(
    bbox: Tuple[float, float, float, float],
    xykm: numpy.ndarray,
    bank_crds: List[numpy.ndarray],
    banktype: List[numpy.ndarray],
    taucls_str: List[str],
    xlabel_txt: str,
    ylabel_txt: str,
    title_txt: str,
) -> matplotlib.figure.Figure:
    """
    Create the bank erosion plot with colour-coded bank types.
    
    Arguments
    ---------
    bbox : Tuple[float, float, float, float]
        Tuple containing boundary limits (xmin, ymin, xmax, ymax); unit m.
    xykm : numpy.ndarray
        Array containing the x, y, and chainage; unit m for x and y, km for chainage.
    bank_crds : List[numpy.ndarray]
        List of N arrays containing the x- and y-coordinates of the oroginal
        bank lines.
    banktype : List[numpy.ndarray]
        List of N arrays containing the bank type values.
    taucls_str : List[str]
        List of strings representing the distinct bank type classes.
    xlabel_txt : str
        Label for the x-axis.
    ylabel_txt : str
        Label for the y-axis.
    title_txt : str
        Label for the axes title.
    
    Results
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    """
    fig, ax = matplotlib.pyplot.subplots()
    setsize(fig)
    ax.set_aspect(1)
    #
    scale = 1000
    chainage_markers(xykm, ax, ndec=0, scale=scale)
    clrs = get_colors("plasma", len(taucls_str) + 1)
    for ib in range(len(bank_crds)):
        for ibt in range(len(taucls_str)):
            isbanktype = numpy.nonzero(banktype[ib] == ibt)[0]
            if isbanktype.any():
                x = bank_crds[ib][:, 0].copy() / scale
                y = bank_crds[ib][:, 1].copy() / scale
                #
                x[banktype[ib] != ibt] = numpy.nan
                y[banktype[ib] != ibt] = numpy.nan
                #
                if ib == 0:
                    ax.plot(x, y, color=clrs[ibt], label=taucls_str[ibt])
                else:
                    ax.plot(x, y, color=clrs[ibt])
            else:
                if ib == 0:
                    ax.plot(
                        numpy.nan, numpy.nan, color=clrs[ibt], label=taucls_str[ibt]
                    )
    #
    set_bbox(ax, bbox)
    ax.set_xlabel(xlabel_txt)
    ax.set_ylabel(ylabel_txt)
    ax.grid(True)
    ax.set_title(title_txt)
    ax.legend(loc="upper right")
    return fig


def plot8_eroded_distance(
    bank_km: List[numpy.ndarray],
    chainage_txt: str,
    dn_tot: List[numpy.ndarray],
    dn_tot_txt: str,
    dn_eq: List[numpy.ndarray],
    dn_eq_txt: str,
    dn_txt: str,
    dn_unit: str,
) -> matplotlib.figure.Figure:
    """
    Create the bank erosion plot with total and equilibrium eroded distance.
    
    Arguments
    ---------
    bank_km : List[numpy.ndarray]
        List of arrays containing the chainage values per bank (point) [km].
    chainage_txt : str
        Label for the horizontal chainage axes.
    dn_tot : List[numpy.ndarray]
        List of arrays containing the total bank erosion distance per bank (segment) [m].
    dn_tot_txt : str
        Label for the total bank erosion distance.
    dn_eq : List[numpy.ndarray]
        List of arrays containing the equilibrium bank erosion distance per bank (segment) [m].
    dn_eq_txt : str
        Label for equilibrium bank erosion distance.
    dn_txt : str
        General label for bank erosion distance.
    dn_unit: str
        Unit used for bank erosion distance.
    
    Results
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    """
    fig, ax = matplotlib.pyplot.subplots()
    setsize(fig)
    #
    n_banklines = len(dn_tot)
    clrs = get_colors("plasma", n_banklines + 1)
    for ib in range(n_banklines):
        bk = (bank_km[ib][:-1] + bank_km[ib][1:]) / 2
        ax.plot(bk, dn_tot[ib], color=clrs[ib], label=dn_tot_txt.format(ib=ib + 1))
        ax.plot(
            bk,
            dn_eq[ib],
            linestyle=":",
            color=clrs[ib],
            label=dn_eq_txt.format(ib=ib + 1),
        )
    #
    ax.set_xlabel(chainage_txt)
    ax.set_ylabel(dn_txt + " " + dn_unit)
    ax.grid(True)
    ax.set_title(dn_txt)
    ax.legend(loc="upper right")
    return fig


def get_colors(cmap_name: str, n: int) -> List[Tuple[float, float, float]]:
    """
    Obtain N colors from the specified colormap.
    
    Arguments
    ---------
    cmap_name : str
        Name of the color map.
    n : int
        Number of colors to be returned.
    
    Returns
    -------
    clrcyc : List[Tuple[float, float, float]]
        List of colour tuplets.
    """
    cmap = matplotlib.cm.get_cmap(cmap_name)
    clrs = [cmap(i / (n - 1)) for i in range(n)]
    return clrs
