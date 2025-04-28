"""Bank line detection module."""

import os
from typing import List, Tuple

import geopandas as gpd
import numpy as np
from geopandas.geoseries import GeoSeries
from matplotlib import pyplot as plt
from shapely import line_merge, union_all
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.polygon import Polygon

from dfastbe import __version__
from dfastbe import plotting as df_plt
from dfastbe.bank_lines.data_models import BankLinesRiverData
from dfastbe.io import BaseSimulationData, ConfigFile, LineGeometry, get_bbox, log_text
from dfastbe.kernel import get_zoom_extends
from dfastbe.support import (
    on_right_side,
    poly_to_line,
    sort_connect_bank_lines,
    tri_to_line,
)
from dfastbe.utils import timed_logger

MAX_RIVER_WIDTH = 1000
RAW_DETECTED_BANKLINE_FRAGMENTS_FILE = "raw_detected_bankline_fragments"
BANK_AREAS_FILE = "bank_areas"
BANKLINE_FRAGMENTS_PER_BANK_AREA_FILE = "bankline_fragments_per_bank_area"
EXTENSION = ".shp"


class BankLines:
    """Bank line detection class."""

    def __init__(self, config_file: ConfigFile, gui: bool = False):
        """Bank line initializer.

        Args:
            config_file : configparser.ConfigParser
                Analysis configuration settings.
            gui : bool
                Flag indicating whether this routine is called from the GUI.

        Examples:
            ```python
            >>> from unittest.mock import patch
            >>> from dfastbe.io import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> with patch("dfastbe.io.log_text"), patch("dfastbe.bank_lines.data_models.log_text"):
            ...    bank_lines = BankLines(config_file)
            ...    isinstance(bank_lines, BankLines)
            True

            ```
        """
        # the root_dir is used to get the FigureDir in the `_get_plotting_flags`
        self.root_dir = config_file.root_dir

        self._config_file = config_file
        self.gui = gui
        self.bank_output_dir = config_file.get_output_dir("banklines")

        # set plotting flags
        self.plot_flags = config_file.get_plotting_flags(self.root_dir)
        self.river_data = BankLinesRiverData(config_file)
        self.search_lines = self.river_data.search_lines
        self.simulation_data, self.h0 = self.river_data.simulation_data()

    @property
    def config_file(self) -> ConfigFile:
        """ConfigFile: object containing the configuration settings."""
        return self._config_file

    @property
    def max_river_width(self) -> int:
        """int: Maximum river width in meters."""
        return MAX_RIVER_WIDTH

    def detect(self) -> None:
        """Run the bank line detection analysis for a specified configuration.

        Examples:
            ```python
            >>> import matplotlib
            >>> matplotlib.use('Agg')
            >>> import sys
            >>> from io import StringIO
            >>> from dfastbe.io import ConfigFile
            >>> sys.stdout = StringIO()
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> bank_lines = BankLines(config_file)
            >>> bank_lines.detect()

            ```
        """
        config_file = self.config_file
        river_data = self.river_data
        timed_logger("-- start analysis --")

        log_text(
            "header_banklines",
            data={
                "version": __version__,
                "location": "https://github.com/Deltares/D-FAST_Bank_Erosion",
            },
        )
        log_text("-")

        # clip the chainage path to the range of chainages of interest
        river_center_line = river_data.river_center_line
        station_bounds = river_center_line.station_bounds
        river_center_line_values = river_center_line.values
        center_line_arr = river_center_line.as_array()
        stations_coords = center_line_arr[:, :2]

        bank_areas: List[Polygon] = self.search_lines.to_polygons()

        # determine whether search lines are located on the left or right
        to_right = [True] * self.search_lines.size
        for ib in range(self.search_lines.size):
            to_right[ib] = on_right_side(
                np.array(self.search_lines.values[ib].coords), stations_coords
            )

        log_text("identify_banklines")
        banklines = self.detect_bank_lines(self.simulation_data, self.h0, config_file)

        # clip the set of detected bank lines to the bank areas
        log_text("simplify_banklines")
        bank = [None] * self.search_lines.size
        clipped_banklines = [None] * self.search_lines.size
        for ib, bank_area in enumerate(bank_areas):
            log_text("bank_lines", data={"ib": ib + 1})
            clipped_banklines[ib] = self.mask(banklines, bank_area)
            bank[ib] = sort_connect_bank_lines(
                clipped_banklines[ib], river_center_line_values, to_right[ib]
            )

        self.save(bank, banklines, clipped_banklines, bank_areas, config_file)

        if self.plot_flags["plot_data"]:
            self.plot(
                center_line_arr,
                self.search_lines.size,
                bank,
                station_bounds,
                bank_areas,
                config_file,
            )

        log_text("end_banklines")
        timed_logger("-- stop analysis --")

    @staticmethod
    def mask(banklines: GeoSeries, bank_area: Polygon) -> MultiLineString:
        """
        Clip the bank line segments to the area of interest.

        Args:
            banklines (GeoSeries):
                Unordered set of bank line segments.
            bank_area (Polygon):
                A search area corresponding to one of the bank search lines.

        Returns:
            MultiLineString: Un-ordered set of bank line segments, clipped to bank area.

        Examples:
            ```python
            >>> import sys
            >>> from io import StringIO
            >>> from dfastbe.io import ConfigFile
            >>> sys.stdout = StringIO()
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> river_data = BankLinesRiverData(config_file)
            >>> bank_lines = BankLines(config_file)
            >>> simulation_data, h0 = river_data.simulation_data()
            >>> banklines = bank_lines.detect_bank_lines(simulation_data, h0, config_file)
            >>> bank_area = bank_lines.search_lines.to_polygons()[0]
            >>> bank_lines.mask(banklines, bank_area)

            ```
        """
        # intersection returns one MultiLineString object
        clipped_banklines = banklines.intersection(bank_area)[0]

        return clipped_banklines

    def plot(
        self,
        xy_km_numpy: np.ndarray,
        n_search_lines: int,
        bank: List,
        km_bounds,
        bank_areas,
        config_file: ConfigFile,
    ):
        """Plot the bank lines and the simulation data."""
        log_text("=")
        log_text("create_figures")
        i_fig = 0
        bbox = get_bbox(xy_km_numpy)

        if self.plot_flags["save_plot_zoomed"]:
            bank_crds: List[np.ndarray] = []
            bank_km: List[np.ndarray] = []
            for ib in range(n_search_lines):
                bcrds_numpy = np.array(bank[ib])
                line_geom = LineGeometry(bcrds_numpy, crs=config_file.crs)
                km_numpy = line_geom.intersect_with_line(xy_km_numpy)
                bank_crds.append(bcrds_numpy)
                bank_km.append(km_numpy)
            km_zoom, xy_zoom = get_zoom_extends(
                km_bounds[0],
                km_bounds[1],
                self.plot_flags["zoom_km_step"],
                bank_crds,
                bank_km,
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
            config_file,
        )
        if self.plot_flags["save_plot"]:
            i_fig = i_fig + 1
            fig_base = (
                f"{self.plot_flags.get('fig_dir')}{os.sep}{i_fig}_banklinedetection"
            )
            if self.plot_flags["save_plot_zoomed"]:
                df_plt.zoom_xy_and_save(
                    fig, ax, fig_base, self.plot_flags.get("plot_ext"), xy_zoom, scale=1
                )
            fig_file = fig_base + self.plot_flags["plot_ext"]
            df_plt.savefig(fig, fig_file)

        if self.plot_flags["close_plot"]:
            plt.close("all")
        else:
            plt.show(block=not self.gui)

    def save(
        self,
        bank: List[LineString],
        banklines: GeoSeries,
        clipped_banklines: List[MultiLineString],
        bank_areas: List[Polygon],
        config_file: ConfigFile,
    ):
        """Save results to files.

        Args:
            bank (List[LineString]):
                List of bank lines.
            banklines (GeoSeries):
                Un-ordered set of bank line segments.
            clipped_banklines (List[MultiLineString]):
                Un-ordered set of bank line segments, clipped to bank area.
            bank_areas (List[Polygon]):
                A search area corresponding to one of the bank search lines.
            config_file (ConfigFile):
                Configuration file object.
        """
        bank_name = self.config_file.get_str("General", "BankFile", "bankfile")
        bank_file = self.bank_output_dir / f"{bank_name}.shp"
        log_text("save_banklines", data={"file": bank_file})
        gpd.GeoSeries(bank, crs=config_file.crs).to_file(bank_file)

        gpd.GeoSeries(clipped_banklines, crs=config_file.crs).to_file(
            self.bank_output_dir / f"{BANKLINE_FRAGMENTS_PER_BANK_AREA_FILE}{EXTENSION}"
        )
        banklines.to_file(
            self.bank_output_dir / f"{RAW_DETECTED_BANKLINE_FRAGMENTS_FILE}{EXTENSION}"
        )
        gpd.GeoSeries(bank_areas, crs=config_file.crs).to_file(
            self.bank_output_dir / f"{BANK_AREAS_FILE}{EXTENSION}"
        )

    @staticmethod
    def detect_bank_lines(
        simulation_data: BaseSimulationData, h0: float, config_file: ConfigFile
    ) -> gpd.GeoSeries:
        """Detect all possible bank line segments based on simulation data.

        Use a critical water depth h0 as a water depth threshold for dry/wet boundary.

        Args:
            simulation_data (BaseSimulationData):
                Simulation data: mesh, bed levels, water levels, velocities, etc.
            h0 (float):
                Critical water depth for determining the banks.

        Returns:
            geopandas.GeoSeries:
                The collection of all detected bank segments in the remaining model area.

        Examples:
            ```python
            >>> from unittest.mock import patch
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> with patch("dfastbe.io.log_text"), patch("dfastbe.bank_lines.bank_lines.print"), patch("dfastbe.bank_lines.data_models.log_text"):
            ...     river_data = BankLinesRiverData(config_file)
            ...     simulation_data, h0 = river_data.simulation_data()
            ...     BankLines.detect_bank_lines(simulation_data, h0, config_file)
            0    MULTILINESTRING ((207927.151 391960.747, 20792...
            dtype: geometry

            ```
        """
        h_node = BankLines._calculate_water_depth(simulation_data)

        wet_node = h_node > h0
        n_wet_arr = wet_node.sum(axis=1)

        lines = BankLines._generate_bank_lines(
            simulation_data, wet_node, n_wet_arr, h_node, h0
        )
        multi_line = union_all(lines)
        merged_line = line_merge(multi_line)

        return gpd.GeoSeries(merged_line, crs=config_file.crs)

    @staticmethod
    def _calculate_water_depth(
        simulation_data: BaseSimulationData,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the water depth at each node in the simulation data.

        This method computes the water depth for each node by considering the
        water levels at the faces and the bed elevation values.

        Args:
            simulation_data (BaseSimulationData):
                Simulation data containing face-node relationships, water levels,
                and bed elevation values.

        Returns:
            np.ndarray:
                An array representing the water depth at each node.
        """
        face_node = simulation_data.face_node
        max_n_nodes = simulation_data.face_node.shape[1]
        n_nodes_total = len(simulation_data.x_node)

        if hasattr(face_node, "mask"):
            mask = ~face_node.mask
            non_masked = sum(mask.reshape(face_node.size))
            f_nc_m = face_node[mask]
            zwm = np.repeat(simulation_data.water_level_face, max_n_nodes)[mask]
        else:
            mask = np.repeat(True, face_node.size)
            non_masked = face_node.size
            f_nc_m = face_node.reshape(non_masked)
            zwm = np.repeat(simulation_data.water_level_face, max_n_nodes).reshape(
                non_masked
            )

        zw_node = np.bincount(f_nc_m, weights=zwm, minlength=n_nodes_total)
        n_val = np.bincount(
            f_nc_m, weights=np.ones(non_masked), minlength=n_nodes_total
        )
        zw_node = zw_node / np.maximum(n_val, 1)
        zw_node[n_val == 0] = simulation_data.bed_elevation_values[n_val == 0]
        h_node = zw_node[face_node] - simulation_data.bed_elevation_values[face_node]
        return h_node

    @staticmethod
    def _generate_bank_lines(
        simulation_data: BaseSimulationData,
        wet_node: np.ndarray,
        n_wet_arr: np.ndarray,
        h_node: np.ndarray,
        h0: float,
    ) -> List[LineString]:
        """Detect bank lines based on wet/dry nodes.

        Args:
            simulation_data (BaseSimulationData):
                Simulation data: mesh, bed levels, water levels, velocities, etc.
            wet_node (np.ndarray):
                Wet/dry boolean array for each face node.
            n_wet_arr (np.ndarray):
                Number of wet nodes for each face.
            h_node (np.ndarray):
                Water depth at each node.
            h0 (float):
                Critical water depth for determining the banks.

        Returns:
            List[LineString or MultiLineString]:
                List of detected bank lines.
        """
        num_faces = len(simulation_data.face_node)
        x_node = simulation_data.x_node[simulation_data.face_node]
        y_node = simulation_data.y_node[simulation_data.face_node]
        mask = n_wet_arr.mask.size > 1
        lines = []

        for i in range(num_faces):
            BankLines._progress_bar(i, num_faces)

            n_wet = n_wet_arr[i]
            n_node = simulation_data.n_nodes[i]
            if (mask and n_wet.mask) or n_wet == 0 or n_wet == n_node:
                continue

            if n_node == 3:
                line = tri_to_line(x_node[i], y_node[i], wet_node[i], h_node[i], h0)
            else:
                line = poly_to_line(
                    n_node, x_node[i], y_node[i], wet_node[i], h_node[i], h0
                )

            if line is not None:
                lines.append(line)

        return lines

    @staticmethod
    def _progress_bar(current: int, total: int) -> None:
        """Print progress bar."""
        if current % 100 == 0:
            percent = (current / total) * 100
            print(f"Progress: {percent:.2f}% ({current}/{total})", end="\r")
        if current == total - 1:
            print("Progress: 100.00% (100%)")
