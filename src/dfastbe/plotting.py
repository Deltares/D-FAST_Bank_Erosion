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

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from dfastbe.io import BaseSimulationData
from dfastbe.kernel import g, water_density


def plot3_eroded_volume_subdivided_1(
    km_mid: np.ndarray,
    km_step: float,
    chainage_txt: str,
    erosion_volume: List[List[np.ndarray]],
    ylabel_txt: str,
    title_txt: str,
    qlabel: str,
) -> [matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Create the bank erosion plot with total eroded volume subdivided per discharge level.

    Arguments
    ---------
    km_mid : np.ndarray
        Array containing the mid points for the chainage bins.
    km_step : float
        Bin width.
    chainage_txt : str
        Label for the horizontal chainage axes.
    erosion_volume : List[List[np.ndarray]]
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
    ax : matplotlib.axes.Axes
        Axes object.
    """
    fig, ax = matplotlib.pyplot.subplots()
    setsize(fig)
    #
    plot3_stacked_per_discharge(ax, km_mid, km_step, erosion_volume, qlabel, 0.8)
    #
    ax.set_xlabel(chainage_txt)
    ax.set_ylabel(ylabel_txt)
    ax.grid(True)
    ax.set_title(title_txt)
    ax.legend(loc="upper right")
    return fig, ax


def plot3_eroded_volume_subdivided_2(
    km_mid: np.ndarray,
    km_step: float,
    chainage_txt: str,
    erosion_volume: List[List[np.ndarray]],
    ylabel_txt: str,
    title_txt: str,
    banklabel: str,
) -> [matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Create the bank erosion plot with total eroded volume subdivided per bank.

    Arguments
    ---------
    km_mid : np.ndarray
        Array containing the mid points for the chainage bins.
    km_step : float
        Bin width.
    chainage_txt : str
        Label for the horizontal chainage axes.
    erosion_volume : List[List[np.ndarray]]
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
    ax : matplotlib.axes.Axes
        Axes object.
    """
    fig, ax = matplotlib.pyplot.subplots()
    setsize(fig)
    #
    plot3_stacked_per_bank(ax, km_mid, km_step, erosion_volume, banklabel, 0.8)
    #
    ax.set_xlabel(chainage_txt)
    ax.set_ylabel(ylabel_txt)
    ax.grid(True)
    ax.set_title(title_txt)
    ax.legend(loc="upper right")
    return fig, ax


def plot4_eroded_volume_eq(
    km_mid: np.ndarray,
    km_step: float,
    chainage_txt: str,
    vol_eq: np.ndarray,
    ylabel_txt: str,
    title_txt: str,
) -> [matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Create the bank erosion plot with equilibrium eroded volume.

    Arguments
    ---------
    km_mid : np.ndarray
        Array containing the mid points for the chainage bins.
    km_step : float
        Bin width.
    chainage_txt : str
        Label for the horizontal chainage axes.
    vol_eq : np.ndarray
        Array containing the equilibrium eroded volume per bin.
    ylabel_txt : str
        Label for the vertical erosion volume axes.
    title_txt : str
        Label for axes title.

    Results
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    fig, ax = matplotlib.pyplot.subplots()
    setsize(fig)
    #
    tvol = np.zeros(km_mid.shape)
    for i in range(len(km_mid)):
        tvol[i] = vol_eq[i].sum()
    ax.bar(km_mid, tvol, width=0.8 * km_step)
    #
    ax.set_xlabel(chainage_txt)
    ax.set_ylabel(ylabel_txt)
    ax.grid(True)
    ax.set_title(title_txt)
    return fig, ax


def plot5series_waterlevels_per_bank(
    bank_km_mid: List[np.ndarray],
    chainage_txt: str,
    waterlevel: List[List[np.ndarray]],
    shipwavemax: List[List[np.ndarray]],
    shipwavemin: List[List[np.ndarray]],
    waterlevelq_txt: str,
    avg_waterlevel_txt: str,
    shipwave_txt: str,
    bankheight: List[np.ndarray],
    bankheight_txt: str,
    bankprotect: List[np.ndarray],
    bankprotect_txt: str,
    elevation_txt: str,
    title_txt: str,
    elevation_unit: str,
) -> [List[matplotlib.figure.Figure], List[matplotlib.axes.Axes]]:
    """
    Create the bank erosion plots with water levels, bank height and bank protection height along each bank.

    Arguments
    ---------
    bank_km_mid : List[np.ndarray]
        List of arrays containing the chainage values per bank (segment) [km].
    chainage_txt : str
        Label for the horizontal chainage axes.
    waterlevel : List[List[np.ndarray]]
        List of arrays containing the water levels per bank (point) [elevation_unit].
    shipmwavemax : np.ndarray
        Maximum bank level subject to ship waves [m]
    shipwavemin : np.ndarray
        Minimum bank level subject to ship waves [m]
    waterlevelq_txt : str
        Label for the water level per discharge level.
    avg_waterlevel_txt : str
        Label for the average water level.
    shipwave_txt : str
        Label for the elevation range influenced by ship waves.
    bankheight : List[np.ndarray]
        List of arrays containing the bank heights per bank (segment) [elevation_unit].
    bankheight_txt : str
        Label for the bank height.
    bankprotect : List[np.ndarray]
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
    figlist : List[matplotlib.figure.Figure]
        List of figure objects, one per bank.
    axlist : List[matplotlib.axes.Axes]
        List of axes objects, one per bank.
    """
    n_banklines = len(bank_km_mid)
    n_levels = len(waterlevel)
    figlist: List[matplotlib.figure.Figure] = []
    axlist: List[matplotlib.axes.Axes] = []
    clrs = get_colors("Blues", n_levels + 1)
    for ib in range(n_banklines):
        fig, ax = matplotlib.pyplot.subplots()
        setsize(fig)
        bk = bank_km_mid[ib]
        #
        for iq in range(n_levels):
            # shaded range of influence for ship waves
            ax.fill_between(
                bk,
                shipwavemin[iq][ib],
                shipwavemax[iq][ib],
                color=clrs[iq + 1],
                alpha=0.1,
            )
            ax.plot(
                bk,
                shipwavemax[iq][ib],
                color=clrs[iq + 1],
                linestyle="--",
                linewidth=0.5,
            )
            ax.plot(
                bk,
                shipwavemin[iq][ib],
                color=clrs[iq + 1],
                linestyle="--",
                linewidth=0.5,
            )
            # water level line itself
            ax.plot(
                bk,
                waterlevel[iq][ib],
                color=clrs[iq + 1],
                label=waterlevelq_txt.format(iq=iq + 1),
            )
            if iq == 0:
                wl_avg = waterlevel[iq][ib].copy()
            else:
                wl_avg = wl_avg + waterlevel[iq][ib]
        #
        wl_avg = wl_avg / n_levels
        ax.plot(
            bk, wl_avg, color=(0.5, 0.5, 0.5), linewidth=2, label=avg_waterlevel_txt,
        )
        ax.plot(bk, bankheight[ib], color=(0.5, 0.5, 0.5), label=bankheight_txt)
        ymin, ymax = ax.get_ylim()
        #
        # bank protection is only visually included in the plot
        # if it is in the same range as the other quantities
        # don't stretch the vertical scale to squeeze in a very low value.
        #
        ax.plot(
            bk,
            bankprotect[ib],
            color=(0.5, 0.5, 0.5),
            linestyle="--",
            label=bankprotect_txt,
        )
        ax.set_ylim(ymin=ymin, ymax=ymax)
        #
        handles, labels = ax.get_legend_handles_labels()
        #
        # use a slightly higher alpha for the legend to make it stand out better.
        iq = int(n_levels/2)
        shaded = matplotlib.patches.Patch(color=clrs[iq + 1], alpha=0.2)
        handles = [*handles[:-3], shaded, *handles[-3:]]
        labels = [*labels[:-3], shipwave_txt, *labels[-3:]]
        #
        ax.set_xlabel(chainage_txt)
        ax.set_ylabel(elevation_txt + " " + elevation_unit)
        ax.grid(True)
        ax.set_title(title_txt.format(ib=ib + 1))
        ax.legend(handles, labels, loc="upper right")
        figlist.append(fig)
        axlist.append(ax)
    return figlist, axlist


def plot6series_velocity_per_bank(
    bank_km_mid: List[np.ndarray],
    chainage_txt: str,
    veloc: List[List[np.ndarray]],
    velocq_txt: str,
    tauc: List[np.ndarray],
    chezy: List[np.ndarray],
    ucrit_txt: str,
    ylabel_txt: str,
    title_txt: str,
    veloc_unit: str,
) -> [List[matplotlib.figure.Figure], List[matplotlib.axes.Axes]]:
    """
    Create the bank erosion plots with velocities and critical velocities along each bank.

    Arguments
    ---------
    bank_km_mid : List[np.ndarray]
        List of arrays containing the chainage values per bank (segment) [km].
    chainage_txt : str
        Label for the horizontal chainage axes.
    veloc: List[List[np.ndarray]]
        List of arrays containing the velocities per bank (segment) [m/s].
    velocq_txt: str,
        Label for the velocity per discharge level.
    tauc: List[np.ndarray]
        List of arrays containing the shear stresses per bank (point) [N/m2].
    chezy: List[np.ndarray]
        List of arrays containing the Chezy values per bank [m0.5/s].
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
    figlist : List[matplotlib.figure.Figure]
        List of figure objects, one per bank.
    axlist : List[matplotlib.axes.Axes]
        List of axes objects, one per bank.
    """
    n_banklines = len(bank_km_mid)
    n_levels = len(veloc)
    figlist: List[matplotlib.figure.Figure] = []
    axlist: List[matplotlib.axes.Axes] = []
    clrs = get_colors("Blues", n_levels + 1)
    for ib in range(n_banklines):
        fig, ax = matplotlib.pyplot.subplots()
        setsize(fig)
        bk = bank_km_mid[ib]
        #
        velc = np.sqrt(tauc[ib] * chezy[ib] ** 2 / (water_density * g))
        ax.plot(bank_km_mid[ib], velc, color="k", label=ucrit_txt)
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
        axlist.append(ax)
    return figlist, axlist


def plot7_banktype(
    bbox: Tuple[float, float, float, float],
    xykm: np.ndarray,
    bank_crds: List[np.ndarray],
    banktype: List[np.ndarray],
    taucls_str: List[str],
    xlabel_txt: str,
    ylabel_txt: str,
    title_txt: str,
) -> [matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Create the bank erosion plot with colour-coded bank types.

    Arguments
    ---------
    bbox : Tuple[float, float, float, float]
        Tuple containing boundary limits (xmin, ymin, xmax, ymax); unit m.
    xykm : np.ndarray
        Array containing the x, y, and chainage; unit m for x and y, km for chainage.
    bank_crds : List[np.ndarray]
        List of N arrays containing the x- and y-coordinates of the oroginal
        bank lines.
    banktype : List[np.ndarray]
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
    ax : matplotlib.axes.Axes
        Axes object.
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
            ibtEdges = np.nonzero(banktype[ib] == ibt)[0]
            if len(ibtEdges) > 0:
                nedges = len(ibtEdges)
                nx = max(3 * nedges - 1, 0)
                x = np.zeros((nx,)) + np.nan
                y = x.copy()
                x[0::3] = bank_crds[ib][ibtEdges, 0].copy() / scale
                y[0::3] = bank_crds[ib][ibtEdges, 1].copy() / scale
                x[1::3] = bank_crds[ib][ibtEdges + 1, 0].copy() / scale
                y[1::3] = bank_crds[ib][ibtEdges + 1, 1].copy() / scale
                #
                if ib == 0:
                    ax.plot(x, y, color=clrs[ibt], label=taucls_str[ibt])
                else:
                    ax.plot(x, y, color=clrs[ibt])
            else:
                if ib == 0:
                    ax.plot(np.nan, np.nan, color=clrs[ibt], label=taucls_str[ibt])
    #
    set_bbox(ax, bbox)
    ax.set_xlabel(xlabel_txt)
    ax.set_ylabel(ylabel_txt)
    ax.grid(True)
    ax.set_title(title_txt)
    ax.legend(loc="upper right")
    return fig, ax


def plot8_eroded_distance(
    bank_km_mid: List[np.ndarray],
    chainage_txt: str,
    dn_tot: List[np.ndarray],
    dn_tot_txt: str,
    dn_eq: List[np.ndarray],
    dn_eq_txt: str,
    dn_txt: str,
    dn_unit: str,
) -> [matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Create the bank erosion plot with total and equilibrium eroded distance.

    Arguments
    ---------
    bank_km_mid : List[np.ndarray]
        List of arrays containing the chainage values per bank (segment) [km].
    chainage_txt : str
        Label for the horizontal chainage axes.
    dn_tot : List[np.ndarray]
        List of arrays containing the total bank erosion distance per bank (segment) [m].
    dn_tot_txt : str
        Label for the total bank erosion distance.
    dn_eq : List[np.ndarray]
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
    ax : matplotlib.axes.Axes
        Axes object.
    """
    fig, ax = matplotlib.pyplot.subplots()
    setsize(fig)
    #
    n_banklines = len(dn_tot)
    clrs = get_colors("plasma", n_banklines + 1)
    for ib in range(n_banklines):
        bk = bank_km_mid[ib]
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
    return fig, ax


class PlottingBase:

    def savefig(self, fig: Figure, filename: Union[str, Path]) -> None:
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
        fig.savefig(filename, dpi=300)

    def setsize(self, fig: Figure) -> None:
        """
        Set the size of a figure.

        Currently the size is hardcoded, but functionality may be extended in the
        future.

        Arguments
        ---------
        fig : matplotlib.figure.Figure
            Figure to a be saved.
        """
        # the size of an a3 is (16.5, 11.75)
        # the size of an a3 is (16.5, 11.75)
        fig.set_size_inches(11.75, 8.25)  # a4

    def set_bbox(
        self,
        ax: Axes,
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
        self,
        xykm: np.ndarray,
        ax: Axes,
        ndec: int = 1,
        scale: float = 1000,
    ) -> None:
        """
        Add markers indicating the river chainage to a plot.

        Arguments
        ---------
        xykm : np.ndarray
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
        mask = np.isclose(np.round(km_rescaled), km_rescaled)
        ax.plot(
            xykm[mask, 0] / scale,
            xykm[mask, 1] / scale,
            linestyle="None",
            marker="+",
            color="k",
        )
        for i in np.nonzero(mask)[0]:
            ax.text(
                xykm[i, 0] / scale,
                xykm[i, 1] / scale,
                labelstr.format(xykm[i, 2]),
                fontsize="x-small",
                clip_on=True,
            )

    def plot_mesh(
        self,
        ax: Axes,
        xe: np.ndarray,
        ye: np.ndarray,
        scale: float = 1000,
    ) -> None:
        """
        Add a mesh to a plot.

        Arguments
        ---------
        ax : matplotlib.axes.Axes
            Axes object in which to add the mesh.
        xe : np.ndarray
            M x 2 array of begin/end x-coordinates of mesh edges.
        ye : np.ndarray
            M x 2 array of begin/end y-coordinates of mesh edges.
        scale : float
            Indicates whether the axes are in m (1) or km (1000).
        """
        xe1 = xe[:, (0, 1, 1)] / scale
        xe1[:, 2] = np.nan
        xev = xe1.reshape((xe1.size,))

        ye1 = ye[:, (0, 1, 1)] / scale
        ye1[:, 2] = np.nan
        yev = ye1.reshape((ye1.size,))

        # to avoid OverflowError: In draw_path: Exceeded cell block limit
        # plot the data in chunks ...
        for i in range(0, len(xev), 3000):
            ax.plot(
                xev[i : i + 3000],
                yev[i : i + 3000],
                color=(0.5, 0.5, 0.5),
                linewidth=0.25,
            )

    def plot_mesh_patches(
        self,
        ax: Axes,
        simulation_data: BaseSimulationData,
        minval: Optional[float] = None,
        maxval: Optional[float] = None,
        scale: float = 1000,
    ) -> matplotlib.collections.PolyCollection:
        """
        Add a collection of patches to the plot one for every face of the mesh.

        Arguments
        ---------
        ax : matplotlib.axes.Axes
            Axes object in which to add the mesh.
        minval : Optional[float]
            Lower limit for the color scale.
        maxval : Optional[float]
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
        for n in range(3, max(simulation_data.n_nodes) + 1):
            mask = simulation_data.n_nodes >= n
            fn_masked = simulation_data.face_node[mask, :]
            tfn_list.append(fn_masked[:, (0, n - 2, n - 1)])
            tval_list.append(simulation_data.water_depth_face[mask])
        tfn = np.concatenate(tfn_list, axis=0)
        tval = np.concatenate(tval_list, axis=0)
        # cmap = matplotlib.pyplot.get_cmap('Spectral')
        if minval is None:
            minval = np.min(tval)
        if maxval is None:
            maxval = np.max(tval)
        p = ax.tripcolor(
            simulation_data.x_node / scale,
            simulation_data.y_node / scale,
            tfn,
            facecolors=tval,
            cmap="Spectral",
            vmin=minval,
            vmax=maxval,
        )
        return p

    def get_colors(self, cmap_name: str, n: int) -> List[Tuple[float, float, float]]:
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

    def zoom_x_and_save(
        self,
        fig: Figure,
        ax: Axes,
        figbase: Path,
        plot_ext: str,
        xzoom: List[Tuple[float, float]],
    ) -> None:
        """
        Zoom in on subregions of the x-axis and save the figure.

        Arguments
        ---------
        fig : matplotlib.figure.Figure
            Figure to be processed.
        ax : matplotlib.axes.Axes
            Axes to be processed.
        fig_base : str
            Base name of the figure to be saved.
        plot_ext : str
            File extension of the figure to be saved.
        xzoom : List[list[float,float]]
            Values at which to split the x-axis.
        """
        xmin, xmax = ax.get_xlim()
        for ix, zoom in enumerate(xzoom):
            ax.set_xlim(xmin=zoom[0], xmax=zoom[1])
            figfile = figbase.with_name(f"{figbase.stem}.sub{str(ix + 1)}{plot_ext}")
            self.savefig(fig, figfile)
        ax.set_xlim(xmin=xmin, xmax=xmax)

    def zoom_xy_and_save(
        self,
        fig: Figure,
        ax: Axes,
        figbase: Path,
        plot_ext: str,
        xyzoom: List[Tuple[float, float, float, float]],
        scale: float = 1000,
    ) -> None:
        """
        Zoom in on subregions in x,y-space and save the figure.

        Arguments
        ---------
        fig : matplotlib.figure.Figure
            Figure to be processed.
        ax : matplotlib.axes.Axes
            Axes to be processed.
        fig_base : str
            Base name of the figure to be saved.
        plot_ext : str
            File extension of the figure to be saved.
        xyzoom : List[List[float, float, float, float]]
            List of xmin, xmax, ymin, ymax values to zoom into.
        scale: float
            Indicates whether the axes are in m (1) or km (1000).
        """
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        dx_zoom = 0
        xy_ratio = (ymax - ymin) / (xmax - xmin)
        for zoom in xyzoom:
            xmin0 = zoom[0]
            xmax0 = zoom[1]
            ymin0 = zoom[2]
            ymax0 = zoom[3]
            dx = xmax0 - xmin0
            dy = ymax0 - ymin0
            if dy < xy_ratio * dx:
                # x range limiting
                dx_zoom = max(dx_zoom, dx)
            else:
                # y range limiting
                dx_zoom = max(dx_zoom, dy / xy_ratio)
        dy_zoom = dx_zoom * xy_ratio

        for ix, zoom in enumerate(xyzoom):
            x0 = (zoom[0] + zoom[1]) / 2
            y0 = (zoom[2] + zoom[3]) / 2
            ax.set_xlim(
                xmin=(x0 - dx_zoom / 2) / scale, xmax=(x0 + dx_zoom / 2) / scale
            )
            ax.set_ylim(
                ymin=(y0 - dy_zoom / 2) / scale, ymax=(y0 + dy_zoom / 2) / scale
            )
            figfile = figbase.with_name(f"{figbase.stem}.sub{str(ix + 1)}{plot_ext}")
            self.savefig(fig, figfile)

        ax.set_xlim(xmin=xmin, xmax=xmax)
        ax.set_ylim(ymin=ymin, ymax=ymax)

    def get_bbox(
        self, coords: np.ndarray, buffer: float = 0.1
    ) -> Tuple[float, float, float, float]:
        """
        Derive the bounding box from a line.
        Args:
            coords (np.ndarray):
                An N x M array containing x- and y-coordinates as first two M entries
            buffer : float
                Buffer fraction surrounding the tight bounding box
        Returns:
            bbox (Tuple[float, float, float, float]):
                Tuple bounding box consisting of [min x, min y, max x, max y)
        """
        x = coords[:, 0]
        y = coords[:, 1]
        x_min = x.min()
        y_min = y.min()
        x_max = x.max()
        y_max = y.max()
        d = buffer * max(x_max - x_min, y_max - y_min)
        bbox = (x_min - d, y_min - d, x_max + d, y_max + d)

        return bbox

    def save_plot(
        self,
        fig: Figure,
        ax: Axes,
        figure_index: int,
        plot_name: str,
        zoom_coords: Optional[List[Tuple[float, float, float, float]]],
        plot_flags: Dict[str, Any],
        zoom_xy: bool,
    ) -> int:
        """Save the plot to a file."""
        figure_index += 1
        fig_base = Path(plot_flags['fig_dir']) / f"{figure_index}_{plot_name}"
        if plot_flags["save_plot_zoomed"] and zoom_xy:
            self.zoom_xy_and_save(
                fig, ax, fig_base, plot_flags["plot_ext"], zoom_coords
            )
        elif plot_flags["save_plot_zoomed"]:
            self.zoom_x_and_save(fig, ax, fig_base, plot_flags["plot_ext"], zoom_coords)
        fig_path = fig_base.with_suffix(plot_flags["plot_ext"])
        self.savefig(fig, fig_path)
        return figure_index
