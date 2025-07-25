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
from typing import Any, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from dfastbe.io.config import PlotProperties
from dfastbe.io.data_models import BaseSimulationData


class BasePlot:

    def __init__(self, gui, plot_flags: PlotProperties) -> None:
        self.gui = gui
        self.flags = plot_flags

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

    # def plot_mesh(
    #     self,
    #     ax: Axes,
    #     xe: np.ndarray,
    #     ye: np.ndarray,
    #     scale: float = 1000,
    # ) -> None:
    #     """
    #     Add a mesh to a plot.
    #
    #     Arguments
    #     ---------
    #     ax : matplotlib.axes.Axes
    #         Axes object in which to add the mesh.
    #     xe : np.ndarray
    #         M x 2 array of begin/end x-coordinates of mesh edges.
    #     ye : np.ndarray
    #         M x 2 array of begin/end y-coordinates of mesh edges.
    #     scale : float
    #         Indicates whether the axes are in m (1) or km (1000).
    #     """
    #     xe1 = xe[:, (0, 1, 1)] / scale
    #     xe1[:, 2] = np.nan
    #     xev = xe1.reshape((xe1.size,))
    #
    #     ye1 = ye[:, (0, 1, 1)] / scale
    #     ye1[:, 2] = np.nan
    #     yev = ye1.reshape((ye1.size,))
    #
    #     # to avoid OverflowError: In draw_path: Exceeded cell block limit
    #     # plot the data in chunks ...
    #     for i in range(0, len(xev), 3000):
    #         ax.plot(
    #             xev[i : i + 3000],
    #             yev[i : i + 3000],
    #             color=(0.5, 0.5, 0.5),
    #             linewidth=0.25,
    #         )

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


class Plot:

    def __init__(
        self,
        plot_flags: PlotProperties,
        scale: int = 1000,
        aspect: int = None,
        gui: bool = False,
    ) -> None:
        """

        Args:
            gui:
            plot_flags:
            scale: float
                Indicates whether the axes are in m (1) or km (1000).
        """
        self.gui = gui
        self.flags = plot_flags
        self._fig, self._ax = plt.subplots()
        self.set_size()
        if aspect:
            self._ax.set_aspect(aspect)
        self.scale = scale

    @property
    def fig(self) -> Figure:
        """Get the figure object."""
        return self._fig

    @property
    def ax(self) -> Axes:
        """Get the axes object."""
        return self._ax

    def set_size(self) -> None:
        """
        Set the size of a figure.

        Currently, the size is hardcoded, but functionality may be extended in the
        future.
        """
        # the size of an a3 is (16.5, 11.75)
        # the size of an a3 is (16.5, 11.75)
        self.fig.set_size_inches(11.75, 8.25)  # a4

    def save_fig(self, path: Union[str, Path]) -> None:
        """
        Save a single figure to file.

        Args:
            path : str
                Name of the file to be written.
        """
        plt.show(block=False)
        self.fig.savefig(path, dpi=300)

    def save(
        self,
        figure_index: int,
        plot_name: str,
        zoom_coords: Optional[List[Tuple[float, float, float, float]]],
        zoom_xy: bool,
    ) -> int:
        """Save the plot to a file."""
        figure_index += 1
        path = Path(self.flags.save_dir) / f"{figure_index}_{plot_name}"
        if self.flags.save_zoomed_plot and zoom_xy:
            self._zoom_xy_and_save(path, self.flags.plot_extension, zoom_coords)
        elif self.flags.save_zoomed_plot:
            self._zoom_x_and_save(path, self.flags.plot_extension, zoom_coords)

        fig_path = path.with_suffix(self.flags.plot_extension)
        self.save_fig(fig_path)
        return figure_index

    def _zoom_x_and_save(
        self,
        path: Path,
        plot_ext: str,
        xzoom: List[Tuple[float, float]],
    ) -> None:
        """
        Zoom in on subregions of the x-axis and save the figure.

        Args:
            path: Path

            plot_ext : str
                File extension of the figure to be saved.
            xzoom : List[list[float,float]]
                Values at which to split the x-axis.
        """
        x_min, x_max = self.ax.get_xlim()
        for ix, zoom in enumerate(xzoom):
            self.ax.set_xlim(xmin=zoom[0], xmax=zoom[1])
            path = path.with_name(f"{path.stem}.sub{str(ix + 1)}{plot_ext}")
            self.save_fig(path)
        self.ax.set_xlim(xmin=x_min, xmax=x_max)

    def _zoom_xy_and_save(
        self,
        fig_base: Path,
        plot_ext: str,
        xyzoom: List[Tuple[float, float, float, float]],
    ) -> None:
        """
        Zoom in on subregions in x,y-space and save the figure.

        Args:
            fig_base : str
                Base name of the figure to be saved.
            plot_ext : str
                File extension of the figure to be saved.
            xyzoom : List[List[float, float, float, float]]
                List of xmin, xmax, ymin, ymax values to zoom into.
        """
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()

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
            self.ax.set_xlim(
                xmin=(x0 - dx_zoom / 2) / self.scale, xmax=(x0 + dx_zoom / 2) / self.scale
            )
            self.ax.set_ylim(
                ymin=(y0 - dy_zoom / 2) / self.scale, ymax=(y0 + dy_zoom / 2) / self.scale
            )
            path = fig_base.with_name(f"{fig_base.stem}.sub{str(ix + 1)}{plot_ext}")
            self.save_fig(path)

        self.ax.set_xlim(xmin=xmin, xmax=xmax)
        self.ax.set_ylim(ymin=ymin, ymax=ymax)

    def set_axes_properties(
        self,
        x_label: str,
        y_label: str,
        grid: bool,
        title_txt: str,
        handles: Optional[List[Any]] = None,
        labels: Optional[List[str]] = None,
    ) -> None:
        """
        Set the properties of the axes.

        Args:
            ax (Axes): The axes object to set properties for.
            x_label (str): Label for the horizontal chainage axes.
            y_label (str): Label for the vertical axes.
            title_txt (str): Title for the plot.
        """
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.grid(grid)
        self.ax.set_title(title_txt)
        if handles and labels:
            self.ax.legend(handles, labels, loc="upper right")
        else:
            self.ax.legend(loc="upper right")

    def set_bbox(
        self,
        bbox: Tuple[float, float, float, float],
    ) -> None:
        """
        Specify the bounding limits of an axes object.

        Args:
            bbox : Tuple[float, float, float, float]
                Tuple containing boundary limits (xmin, ymin, xmax, ymax); unit m.
        """
        self.ax.set_xlim(xmin=bbox[0] / self.scale, xmax=bbox[2] / self.scale)
        self.ax.set_ylim(ymin=bbox[1] / self.scale, ymax=bbox[3] / self.scale)
