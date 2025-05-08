from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely import LineString, Polygon

from dfastbe.io import BaseSimulationData, ConfigFile, LineGeometry, log_text
from dfastbe.kernel import get_zoom_extends
from dfastbe.plotting import PlottingBase


class BankLinesPlotter(PlottingBase):

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
        bank: List[LineString],
        num_search_lines: int,
        crs: str,
        station_coords: np.ndarray,
        station_bounds: Tuple[float, float],
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
        for ib in range(num_search_lines):
            bcrds_numpy = np.array(bank[ib].coords)
            line_geom = LineGeometry(bcrds_numpy, crs=crs)
            km_numpy = line_geom.intersect_with_line(station_coords)
            bank_crds.append(bcrds_numpy)
            bank_km.append(km_numpy)

        _, xy_zoom = get_zoom_extends(
            station_bounds[0],
            station_bounds[1],
            self.plot_flags["zoom_km_step"],
            bank_crds,
            bank_km,
        )
        return xy_zoom

    def plot(
        self,
        station_coords: np.ndarray,
        num_search_lines: int,
        bank: List[LineString],
        stations_bounds: Tuple[float, float],
        bank_areas: List[Polygon],
    ):
        """Plot the bank lines and the simulation data.

        Args:
            station_coords (np.ndarray):
                Array of x and y coordinates in km.
            num_search_lines (int):
                Number of search lines.
            bank (List):
                List of bank lines.
            stations_bounds (Tuple[float, float]):
                Minimum and maximum km bounds.
            bank_areas (List[Polygon]):
                A search area corresponding to one of the bank search lines.
            config_file (ConfigFile):
                Configuration file object.

        Examples:
            ```python
            >>> import matplotlib
            >>> from dfastbe.bank_lines.bank_lines import BankLines
            >>> matplotlib.use('Agg')
            >>> config_file = ConfigFile.read("tests/data/bank_lines/meuse_manual.cfg")  # doctest: +ELLIPSIS
            >>> bank_lines = BankLines(config_file)
            N...e
            >>> bank_lines.plot_flags["save_plot"] = False
            >>> station_coords = np.array([[0, 0, 0], [1, 1, 0]])
            >>> num_search_lines = 1
            >>> bank = [LineString([(0, 0), (1, 1)])]
            >>> stations_bounds = (0, 1)
            >>> bank_areas = [Polygon([(0, 0), (1, 1), (1, 0)])]
            >>> bank_lines_plotter = BankLinesPlotter(False, bank_lines.plot_flags, config_file, bank_lines.simulation_data)
            >>> bank_lines_plotter.plot(station_coords, num_search_lines, bank, stations_bounds, bank_areas)
            N...s

            ```
        """
        log_text("=")
        log_text("create_figures")
        fig_i = 0
        bbox = self.get_bbox(station_coords)

        xy_zoom = self._get_zoom_extends(
            bank,
            num_search_lines,
            self.config_file.crs,
            station_coords,
            stations_bounds,
        )

        fig, ax = self.plot_detect1(
            bbox,
            station_coords,
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
            ax, self.simulation_data, 0, maximum_water_depth, scale=scale
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
