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
from dfastbe.io.data_models import BaseSimulationData

from dfastbe.io.config import ConfigFile
from dfastbe.bank_erosion.erosion_calculator import WATER_DENSITY, g

class BasePlot:

    def __init__(self, gui, plot_flags) -> None:
        self.gui = gui
        self.flags = plot_flags

    def save_fig(self, fig: Figure, filename: Union[str, Path]) -> None:
        """
        Save a single figure to file.

        Arguments
        ---------
        fig : matplotlib.figure.Figure
            Figure to be saved.
        filename : str
            Name of the file to be written.
        """
        print("saving figure {file}".format(file=filename))
        matplotlib.pyplot.show(block=False)
        fig.savefig(filename, dpi=300)

    def set_size(self, fig: Figure) -> None:
        """
        Set the size of a figure.

        Currently, the size is hardcoded, but functionality may be extended in the
        future.

        Arguments
        ---------
        fig : matplotlib.figure.Figure
            Figure to be saved.
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

    def stations_marker(
        self,
        river_center_line_arr: np.ndarray,
        ax: Axes,
        float_format: int = 1,
        scale: float = 1000,
    ) -> None:
        """
        Add markers indicating the river chainage to a plot.

        Arguments
        ---------
        river_center_line_arr : np.ndarray
            Array containing the x, y, and chainage; unit m for x and y, km for chainage.
        ax : matplotlib.axes.Axes
            Axes object in which to add the markers.
        float_format : int
            Number of decimals used for marks.
        scale: float
            Indicates whether the axes are in m (1) or km (1000).
        """
        step = 10 ** (-float_format)
        label_str = " {:." + str(float_format) + "f}"
        km_rescaled = river_center_line_arr[:, 2] / step
        mask = np.isclose(np.round(km_rescaled), km_rescaled)
        ax.plot(
            river_center_line_arr[mask, 0] / scale,
            river_center_line_arr[mask, 1] / scale,
            linestyle="None",
            marker="+",
            color="k",
        )
        for i in np.nonzero(mask)[0]:
            ax.text(
                river_center_line_arr[i, 0] / scale,
                river_center_line_arr[i, 1] / scale,
                label_str.format(river_center_line_arr[i, 2]),
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

    def mesh_patches(
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
            self.save_fig(fig, figfile)
        ax.set_xlim(xmin=xmin, xmax=xmax)

    def zoom_xy_and_save(
        self,
        fig: Figure,
        ax: Axes,
        fig_base: Path,
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
            figfile = fig_base.with_name(f"{fig_base.stem}.sub{str(ix + 1)}{plot_ext}")
            self.save_fig(fig, figfile)

        ax.set_xlim(xmin=xmin, xmax=xmax)
        ax.set_ylim(ymin=ymin, ymax=ymax)


    def set_axes_properties(
        self,
        ax: Axes,
        chainage_txt: str,
        ylabel_txt: str,
        grid: bool,
        title_txt: str,
        handles: Optional[List[Any]] = None,
        labels: Optional[List[str]] = None,
    ) -> None:
        """
        Set the properties of the axes.

        Args:
            ax (Axes): The axes object to set properties for.
            chainage_txt (str): Label for the horizontal chainage axes.
            ylabel_txt (str): Label for the vertical axes.
            title_txt (str): Title for the plot.
        """
        ax.set_xlabel(chainage_txt)
        ax.set_ylabel(ylabel_txt)
        ax.grid(grid)
        ax.set_title(title_txt)
        if handles and labels:
            ax.legend(handles, labels, loc="upper right")
        else:
            ax.legend(loc="upper right")

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
        self.save_fig(fig, fig_path)
        return figure_index