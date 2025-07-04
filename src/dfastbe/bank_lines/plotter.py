from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely import LineString, Polygon

from dfastbe.io.config import ConfigFile
from dfastbe.io.logger import log_text
from dfastbe.io.data_models import BaseSimulationData, LineGeometry
from dfastbe.utils import get_zoom_extends
from dfastbe.plotting import BasePlot

# Texts for the plot
PLOT_TEXTS = {
    "x_label": "x-coordinate [m]",
    "y_label": "y-coordinate [m]",
    "title": "water depth and detected bank lines", # Label for the axes title.
    "color_bar_label": "water depth [m]",  # Label for the color bar.
    "bank_area_txt": "bank search area", # Label for the bank search areas.
    "bankline_txt": "detected bank line", # Label for the identified bank lines.
}



class BankLinesPlotter(BasePlot):

    def __init__(
        self,
        gui: bool,
        plot_flags: dict,
        crs: str,
        simulation_data: BaseSimulationData,
        river_center_line: LineGeometry,
        stations_bounds: Tuple[float, float],
    ):
        """

        Args:
            gui:
            plot_flags:
            crs:
            simulation_data:
            river_center_line:
            stations_bounds (Tuple[float, float]):
                Minimum and maximum chainage bounds.
        """
        self.gui = gui
        self.flags = plot_flags
        self.crs = crs
        self.simulation_data = simulation_data
        self.river_center_line = river_center_line
        self.bbox = river_center_line.get_bbox()
        self.stations_bounds = stations_bounds

    def _get_zoom_extends(
        self,
        bank: List[LineString],
        num_search_lines: int,
        crs: str,
    ) -> Optional[np.ndarray]:
        """Get zoom extents for plotting.

        Args:
            bank (List[np.ndarray]):
                List of bank coordinates as NumPy arrays.
            num_search_lines (int):
                Number of search lines.
            crs (str):
                Coordinate reference system.

        Returns:
            np.ndarray:
                Array of zoom extents in x, y space.
        """
        banks_coords: List[np.ndarray] = []
        banks_station: List[np.ndarray] = []
        river_center_line = self.river_center_line.as_array()
        for ib in range(num_search_lines):
            bank_line_geom = LineGeometry(bank[ib], crs=crs)
            bank_stations = bank_line_geom.intersect_with_line(river_center_line)
            banks_coords.append(bank_line_geom.as_array())
            banks_station.append(bank_stations)

        _, xy_zoom = get_zoom_extends(
            self.stations_bounds[0],
            self.stations_bounds[1],
            self.flags["zoom_km_step"],
            banks_coords,
            banks_station,
        )
        return xy_zoom

    def plot(
        self,
        num_search_lines: int,
        bank: List[LineString],
        bank_areas: List[Polygon],
    ):
        """Plot the bank lines and the simulation data.

        Args:
            num_search_lines (int):
                Number of search lines.
            bank (List):
                List of bank lines.
            bank_areas (List[Polygon]):
                A search area corresponding to one of the bank search lines.

        Examples:
            ```python
            >>> import matplotlib
            >>> from dfastbe.bank_lines.bank_lines import BankLines
            >>> matplotlib.use('Agg')
            >>> config_file = ConfigFile.read("tests/data/bank_lines/meuse_manual.cfg")  # doctest: +ELLIPSIS
            >>> bank_lines = BankLines(config_file)
            N...e
            >>> bank_lines.flags["save_plot"] = False
            >>> station_coords = np.array([[0, 0, 0], [1, 1, 0]])
            >>> num_search_lines = 1
            >>> bank = [LineString([(0, 0), (1, 1)])]
            >>> stations_bounds = (0, 1)
            >>> bank_areas = [Polygon([(0, 0), (1, 1), (1, 0)])]
            >>> bank_lines_plotter = BankLinesPlotter(
            ...     False, bank_lines.flags, config_file.crs, bank_lines.simulation_data, stations_bounds
            ... )
            >>> bank_lines_plotter.plot(station_coords, num_search_lines, bank, stations_bounds, bank_areas)
            N...s

            ```
        """
        log_text("=")
        log_text("create_figures")
        fig_i = 0

        if self.flags["save_plot_zoomed"]:
            xy_zoom = self._get_zoom_extends(
                bank,
                num_search_lines,
                self.crs,
            )
        else:
            xy_zoom = None

        fig, ax = self._plot_data(
            bank_areas,
            bank,
        )
        if self.flags["save_plot"]:
            self.save_plot(
                fig, ax, fig_i, "bankline-detection", xy_zoom, self.flags, True
            )

        if self.flags["close_plot"]:
            plt.close("all")
        else:
            plt.show(block=not self.gui)

    def _plot_data(
        self,
        bank_areas: List[Polygon],
        bank: List[LineString],
    ) -> Tuple[Figure, Axes]:
        """
        Create the bank line detection plot.

        The figure contains a map of the water depth, the chainage, and detected bank lines.

        Args:
            bank_areas (List[Polygon]):
                List of bank polygons.
            bank (List[LineString]):
                List of bank lines.

        Returns:
            fig : matplotlib.figure.Figure:
                Figure object.
            ax : matplotlib.axes.Axes
                Axes object.
        """
        fig, ax = plt.subplots()
        self.set_size(fig)
        ax.set_aspect(1)

        scale = 1  # using scale 1 here because of the geopandas plot commands
        maximum_water_depth = 1.1 * self.simulation_data.water_depth_face.max()
        self.stations_marker(self.river_center_line.as_array(), ax, float_format=0, scale=scale)
        patches = self.plot_mesh_patches(
            ax, self.simulation_data, 0, maximum_water_depth, scale=scale
        )
        for ind, bank_area in enumerate(bank_areas):
            gpd.GeoSeries(bank_area, crs=self.crs).plot(
                ax=ax, alpha=0.2, color="k"
            )
            gpd.GeoSeries(bank[ind], crs=self.crs).plot(ax=ax, color="r")

        fig.colorbar(patches, ax=ax, shrink=0.5, drawedges=False, label=PLOT_TEXTS["color_bar_label"])

        shaded = Patch(color="k", alpha=0.2)
        banklines = Line2D([], [], color="r")
        handles = [shaded, banklines]
        labels = [PLOT_TEXTS["bank_area_txt"], PLOT_TEXTS["bankline_txt"]]

        self.set_bbox(ax, self.bbox, scale=scale)
        self.set_axes_properties(
            ax, PLOT_TEXTS["x_label"], PLOT_TEXTS["y_label"], True, PLOT_TEXTS["title"], handles, labels
        )
        return fig, ax
