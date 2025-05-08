from typing import List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt

from dfastbe import plotting as df_plt
from dfastbe.io import BaseSimulationData, ConfigFile, LineGeometry, log_text
from dfastbe.kernel import get_zoom_extends


class BankLinesPlotter:

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
        bbox = df_plt.get_bbox(xy_km_numpy)

        xy_zoom = self._get_zoom_extends(
            bank, n_search_lines, self.config_file.crs, xy_km_numpy, km_bounds
        )

        fig, ax = df_plt.plot_detect1(
            bbox,
            xy_km_numpy,
            bank_areas,
            bank,
            self.simulation_data.face_node,
            self.simulation_data.n_nodes,
            self.simulation_data.x_node,
            self.simulation_data.y_node,
            self.simulation_data.water_depth_face,
            1.1 * self.simulation_data.water_depth_face.max(),
            "x-coordinate [m]",
            "y-coordinate [m]",
            "water depth and detected bank lines",
            "water depth [m]",
            "bank search area",
            "detected bank line",
            self.config_file,
        )
        if self.plot_flags["save_plot"]:
            fig_i = df_plt.save_plot(
                fig, ax, fig_i, "banklinedetection", xy_zoom, self.plot_flags, True
            )

        if self.plot_flags["close_plot"]:
            plt.close("all")
        else:
            plt.show(block=not self.gui)
