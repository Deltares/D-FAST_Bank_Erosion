from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely import LineString, Polygon

from dfastbe import plotting as df_plt
from dfastbe.io import BaseSimulationData, ConfigFile, LineGeometry, log_text
from dfastbe.kernel import get_zoom_extends


class BankLinesPlotter(df_plt.PlottingBase):

    def __init__(
        self,
        gui: bool,
        plot_flags: dict,
        config_file: ConfigFile,
        simulation_data: BaseSimulationData,
    ):
        self.gui = gui
        self.plot_flags = plot_flags
        self.config_file = config_file
        self.simulation_data = simulation_data

    def _get_zoom_extends(
        self,
        bank: List[np.ndarray],
        n_search_lines: int,
        crs: str,
        xy_km_numpy: np.ndarray,
        km_bounds: Tuple[float, float],
    ) -> Optional[np.ndarray]:
        """Get zoom extents for plotting.

        Args:
            bank (List[np.ndarray]): List of bank coordinates as NumPy arrays.
            n_search_lines (int): Number of search lines.
            crs (str): Coordinate reference system.
            xy_km_numpy (np.ndarray): Array of x, y coordinates in kilometers.
            km_bounds (Tuple[float, float]): Minimum and maximum chainage bounds.

        Returns:
            Optional[np.ndarray]: Array of zoom extents in x, y space, or None if zooming is disabled.
        """
        if not self.plot_flags["save_plot_zoomed"]:
            return None
        bank_crds: List[np.ndarray] = []
        bank_km: List[np.ndarray] = []
        for ib in range(n_search_lines):
            bcrds_numpy = np.array(bank[ib])
            line_geom = LineGeometry(bcrds_numpy, crs=crs)
            km_numpy = line_geom.intersect_with_line(xy_km_numpy)
            bank_crds.append(bcrds_numpy)
            bank_km.append(km_numpy)

        _, xy_zoom = get_zoom_extends(
            km_bounds[0],
            km_bounds[1],
            self.plot_flags["zoom_km_step"],
            bank_crds,
            bank_km,
        )
        return xy_zoom

    def plot(
        self,
        xy_km_numpy: np.ndarray,
        n_search_lines: int,
        bank: List,
        km_bounds,
        bank_areas,
    ):
        """Plot the bank lines and the simulation data."""
        log_text("=")
        log_text("create_figures")
        fig_i = 0
        bbox = self.get_bbox(xy_km_numpy)

        xy_zoom = self._get_zoom_extends(
            bank, n_search_lines, self.config_file.crs, xy_km_numpy, km_bounds
        )

        fig, ax = self.plot_detect1(
            bbox,
            xy_km_numpy,
            bank_areas,
            bank,
            "x-coordinate [m]",
            "y-coordinate [m]",
            "water depth and detected bank lines",
            "water depth [m]",
            "bank search area",
            "detected bank line",
            self.config_file,
        )
        if self.plot_flags["save_plot"]:
            fig_i = self.save_plot(
                fig, ax, fig_i, "banklinedetection", xy_zoom, self.plot_flags, True
            )

        if self.plot_flags["close_plot"]:
            plt.close("all")
        else:
            plt.show(block=not self.gui)

    def plot_detect1(
        self,
        bbox: Tuple[float, float, float, float],
        xykm: np.ndarray,
        bankareas: List[Polygon],
        bank: List[LineString],
        xlabel_txt: str,
        ylabel_txt: str,
        title_txt: str,
        waterdepth_txt: str,
        bankarea_txt: str,
        bankline_txt: str,
        config_file: ConfigFile,
    ) -> Tuple[Figure, Axes]:
        """
        Create the bank line detection plot.

        The figure contains a map of the water depth, the chainage, and detected
        bank lines.

        Arguments
        ---------
        bbox : Tuple[float, float, float, float]
            Tuple containing boundary limits (xmin, ymin, xmax, ymax); unit m.
        xykm : np.ndarray
            Array containing the x, y, and chainage; unit m for x and y, km for chainage.
        bankareas : List[Polygon]
            List of bank polygons.
        bank : List[LineString]
            List of bank lines.
        xlabel_txt : str
            Label for the x-axis.
        ylabel_txt : str
            Label for the y-axis.
        title_txt : str
            Label for the axes title.
        waterdepth_txt : str
            Label for the color bar.
        bankarea_txt : str
            Label for the bank search areas.
        bankline_txt : str
            Label for the identified bank lines.

        Returns
        -------
        fig : matplotlib.figure.Figure:
            Figure object.
        ax : matplotlib.axes.Axes
            Axes object.
        """
        fig, ax = plt.subplots()
        self.setsize(fig)
        ax.set_aspect(1)
        #
        scale = 1  # using scale 1 here because of the geopandas plot commands
        maximum_water_depth = 1.1 * self.simulation_data.water_depth_face.max()
        self.chainage_markers(xykm, ax, ndec=0, scale=scale)
        p = self.plot_mesh_patches(
            ax,
            self.simulation_data.face_node,
            self.simulation_data.n_nodes,
            self.simulation_data.x_node,
            self.simulation_data.y_node,
            self.simulation_data.water_depth_face,
            0,
            maximum_water_depth,
            scale=scale,
        )
        for b, bankarea in enumerate(bankareas):
            gpd.GeoSeries(bankarea, crs=config_file.crs).plot(
                ax=ax, alpha=0.2, color="k"
            )
            gpd.GeoSeries(bank[b], crs=config_file.crs).plot(ax=ax, color="r")
        fig.colorbar(p, ax=ax, shrink=0.5, drawedges=False, label=waterdepth_txt)
        #
        shaded = Patch(color="k", alpha=0.2)
        bankln = Line2D([], [], color="r")
        handles = [shaded, bankln]
        labels = [bankarea_txt, bankline_txt]
        #
        self.set_bbox(ax, bbox, scale=scale)
        ax.set_xlabel(xlabel_txt)
        ax.set_ylabel(ylabel_txt)
        ax.grid(True)
        ax.set_title(title_txt)
        ax.legend(handles, labels, loc="upper right")
        return fig, ax
