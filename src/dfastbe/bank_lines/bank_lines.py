"""Bank line detection module."""

from typing import List, Tuple, Dict, Any

import geopandas as gpd
import numpy as np
from geopandas.geoseries import GeoSeries
from shapely import line_merge, union_all
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.polygon import Polygon

from dfastbe import __version__
from dfastbe.bank_lines.data_models import BankLinesRiverData
from dfastbe.bank_lines.plotter import BankLinesPlotter
from dfastbe.io.data_models import BaseSimulationData
from dfastbe.io.config import ConfigFile
from dfastbe.io.logger import log_text
from dfastbe.bank_lines.utils import sort_connect_bank_lines, poly_to_line, tri_to_line
from dfastbe.utils import on_right_side
from dfastbe.io.logger import timed_logger

MAX_RIVER_WIDTH = 1000
RAW_DETECTED_BANKLINE_FRAGMENTS_FILE = "raw_detected_bankline_fragments"
BANK_AREAS_FILE = "bank_areas"
BANKLINE_FRAGMENTS_PER_BANK_AREA_FILE = "bankline_fragments_per_bank_area"
EXTENSION = ".shp"

__all__ = ["BankLines"]


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
            >>> from dfastbe.io.config import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/bank_lines/meuse_manual.cfg")
            >>> bank_lines = BankLines(config_file)  # doctest: +ELLIPSIS
            N...e
            >>> isinstance(bank_lines, BankLines)
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
        self.simulation_data, self.critical_water_depth = (
            self.river_data.simulation_data()
        )
        if self.plot_flags["plot_data"]:
            self.plotter = self.get_plotter()

        self._results = None

    @property
    def config_file(self) -> ConfigFile:
        """ConfigFile: object containing the configuration file."""
        return self._config_file

    @property
    def max_river_width(self) -> int:
        """int: Maximum river width in meters."""
        return MAX_RIVER_WIDTH

    def get_plotter(self) -> BankLinesPlotter:
        return BankLinesPlotter(
            self.gui, self.plot_flags, self.config_file.crs, self.simulation_data, self.river_data.river_center_line,
            self.river_data.river_center_line.station_bounds,
        )

    @property
    def results(self) -> Dict[str, Any]:
        """dict: Results of the bank line detection analysis."""
        return self._results

    @results.setter
    def results(self, value: Dict[str, Any]):
        """Set the results of the bank line detection analysis."""
        self._results = value

    def detect(self) -> None:
        """Run the bank line detection analysis for a specified configuration.

        This method performs bank line detection using the provided configuration file.
        It generates shapefiles that can be opened with GeoPandas or QGIS, and also
        creates a plot of the detected bank lines along with the simulation data.

        Examples:
            ```python
            >>> import matplotlib
            >>> matplotlib.use('Agg')
            >>> from dfastbe.io.config import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/bank_lines/meuse_manual.cfg")
            >>> bank_lines = BankLines(config_file)  # doctest: +ELLIPSIS
            N...e
            >>> bank_lines.detect()
               0...-

            ```
            In the BankDir directory specified in the .cfg, the following files are created:
            - "raw_detected_bankline_fragments.shp"
            - "bank_areas.shp"
            - "bankline_fragments_per_bank_area.shp"
            - "bankfile.shp"
            In the FigureDir directory specified in the .cfg, the following files are created:
            - "1_banklinedetection.png"
        """
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
        river_center_line = self.river_data.river_center_line
        river_center_line_values = river_center_line.values
        center_line_arr = river_center_line.as_array()

        bank_areas: List[Polygon] = self.search_lines.to_polygons()

        to_right = [True] * self.search_lines.size
        for ib in range(self.search_lines.size):
            to_right[ib] = on_right_side(
                np.array(self.search_lines.values[ib].coords), center_line_arr[:, :2]
            )

        log_text("identify_banklines")
        banklines = self.detect_bank_lines(
            self.simulation_data, self.critical_water_depth, self.config_file
        )

        # clip the set of detected bank lines to the bank areas
        log_text("simplify_banklines")
        bank = []
        masked_bank_lines = []
        for ib, bank_area in enumerate(bank_areas):
            log_text("bank_lines", data={"ib": ib + 1})
            masked_bank_lines.append(self.mask(banklines, bank_area))
            bank.append(sort_connect_bank_lines(masked_bank_lines[ib], river_center_line_values, to_right[ib]))

        self.save(bank, banklines, masked_bank_lines, bank_areas, self.config_file)

        self.results = {
            "bank": bank,
            "banklines": banklines,
            "masked_bank_lines": masked_bank_lines,
            "bank_areas": bank_areas,
        }

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
            >>> from dfastbe.io.config import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/bank_lines/meuse_manual.cfg")
            >>> river_data = BankLinesRiverData(config_file)  # doctest: +ELLIPSIS
            N...e
            >>> bank_lines = BankLines(config_file)
            N...e
            >>> simulation_data, critical_water_depth = river_data.simulation_data()
            N...e
            >>> banklines = bank_lines.detect_bank_lines(simulation_data, critical_water_depth, config_file)
            P...)
            >>> bank_area = bank_lines.search_lines.to_polygons()[0]
            >>> bank_lines.mask(banklines, bank_area)
            <MULTILINESTRING ((207830.389 392063.658, 2078...>

            ```
        """
        # intersection returns one MultiLineString object
        masked_bank_lines = banklines.intersection(bank_area)[0]

        return masked_bank_lines

    def plot(self):
        if self.plot_flags["plot_data"]:
            self.plotter.plot(
                self.search_lines.size,
                self.results["bank"],
                self.results["bank_areas"],
            )

    def save(
        self,
        bank: List[LineString],
        banklines: GeoSeries,
        masked_bank_lines: List[MultiLineString],
        bank_areas: List[Polygon],
        config_file: ConfigFile,
    ):
        """Save results to files.

        Args:
            bank (List[LineString]):
                List of bank lines.
            banklines (GeoSeries):
                Un-ordered set of bank line segments.
            masked_bank_lines (List[MultiLineString]):
                Un-ordered set of bank line segments, clipped to bank area.
            bank_areas (List[Polygon]):
                A search area corresponding to one of the bank search lines.
            config_file (ConfigFile):
                Configuration file object.

        Examples:
            ```python
            >>> from dfastbe.io.config import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/bank_lines/meuse_manual.cfg")  # doctest: +ELLIPSIS
            >>> bank_lines = BankLines(config_file)
            N...e
            >>> bank = [LineString([(0, 0), (1, 1)])]
            >>> banklines = gpd.GeoSeries([LineString([(0, 0), (1, 1)])])
            >>> masked_bank_lines = [MultiLineString([LineString([(0, 0), (1, 1)])])]
            >>> bank_areas = [Polygon([(0, 0), (1, 1), (1, 0)])]
            >>> bank_lines.save(bank, banklines, masked_bank_lines, bank_areas, config_file)
            No message found for save_banklines

            ```
        """
        bank_name = self.config_file.get_str("General", "BankFile", "bankfile")
        bank_file = self.bank_output_dir / f"{bank_name}.shp"
        log_text("save_banklines", data={"file": bank_file})
        gpd.GeoSeries(bank, crs=config_file.crs).to_file(bank_file)

        gpd.GeoSeries(masked_bank_lines, crs=config_file.crs).to_file(
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
        simulation_data: BaseSimulationData,
        critical_water_depth: float,
        config_file: ConfigFile,
    ) -> gpd.GeoSeries:
        """Detect all possible bank line segments based on simulation data.

        Use a critical water depth critical_water_depth as a water depth threshold for dry/wet boundary.

        Args:
            simulation_data (BaseSimulationData):
                Simulation data: mesh, bed levels, water levels, velocities, etc.
            critical_water_depth (float):
                Critical water depth for determining the banks.

        Returns:
            geopandas.GeoSeries:
                The collection of all detected bank segments in the remaining model area.

        Examples:
            ```python
            >>> config_file = ConfigFile.read("tests/data/bank_lines/meuse_manual.cfg")
            >>> river_data = BankLinesRiverData(config_file)  # doctest: +ELLIPSIS
            N...e
            >>> simulation_data, critical_water_depth = river_data.simulation_data()
            N...e
            >>> BankLines.detect_bank_lines(simulation_data, critical_water_depth, config_file)
            P...
            0    MULTILINESTRING ((207927.151 391960.747, 20792...
            dtype: geometry

            ```
        """
        h_node = BankLines._calculate_water_depth(simulation_data)

        wet_node = h_node > critical_water_depth
        num_wet_arr = wet_node.sum(axis=1)

        lines = BankLines._generate_bank_lines(
            simulation_data, wet_node, num_wet_arr, h_node, critical_water_depth
        )
        multi_line = union_all(lines)
        merged_line = line_merge(multi_line)

        return gpd.GeoSeries(merged_line, crs=config_file.crs)

    @staticmethod
    def _calculate_water_depth(
        simulation_data: BaseSimulationData,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the water depth at each node in the simulation data.

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
        max_num_nodes = simulation_data.face_node.shape[1]
        num_nodes_total = len(simulation_data.x_node)

        if hasattr(face_node, "mask"):
            mask = ~face_node.mask
            non_masked = sum(mask.reshape(face_node.size))
            f_nc_m = face_node[mask]
            zwm = np.repeat(simulation_data.water_level_face, max_num_nodes)[
                mask.flatten()
            ]
        else:
            non_masked = face_node.size
            f_nc_m = face_node.reshape(non_masked)
            zwm = np.repeat(simulation_data.water_level_face, max_num_nodes).reshape(
                non_masked
            )

        zw_node = np.bincount(f_nc_m, weights=zwm, minlength=num_nodes_total)
        num_val = np.bincount(
            f_nc_m, weights=np.ones(non_masked), minlength=num_nodes_total
        )
        zw_node = zw_node / np.maximum(num_val, 1)
        zw_node[num_val == 0] = simulation_data.bed_elevation_values[num_val == 0]
        h_node = zw_node[face_node] - simulation_data.bed_elevation_values[face_node]
        return h_node

    @staticmethod
    def _generate_bank_lines(
        simulation_data: BaseSimulationData,
        wet_node: np.ndarray,
        num_wet_arr: np.ndarray,
        h_node: np.ndarray,
        critical_water_depth: float,
    ) -> List[LineString]:
        """Detect bank lines based on wet/dry nodes.

        Args:
            simulation_data (BaseSimulationData):
                Simulation data: mesh, bed levels, water levels, velocities, etc.
            wet_node (np.ndarray):
                Wet/dry boolean array for each face node.
            num_wet_arr (np.ndarray):
                Number of wet nodes for each face.
            h_node (np.ndarray):
                Water depth at each node.
            critical_water_depth (float):
                Critical water depth for determining the banks.

        Returns:
            List[LineString or MultiLineString]:
                List of detected bank lines.
        """
        num_faces = len(simulation_data.face_node)
        x_node = simulation_data.x_node[simulation_data.face_node]
        y_node = simulation_data.y_node[simulation_data.face_node]
        mask = num_wet_arr.mask.size > 1
        lines = []

        for i in range(num_faces):
            BankLines._progress_bar(i, num_faces)

            n_wet = num_wet_arr[i]
            n_node = simulation_data.n_nodes[i]
            if (mask and n_wet.mask) or n_wet == 0 or n_wet == n_node:
                continue

            if n_node == 3:
                line = tri_to_line(
                    x_node[i], y_node[i], wet_node[i], h_node[i], critical_water_depth
                )
            else:
                line = poly_to_line(
                    n_node,
                    x_node[i],
                    y_node[i],
                    wet_node[i],
                    h_node[i],
                    critical_water_depth,
                )

            if line is not None:
                lines.append(line)

        return lines

    @staticmethod
    def _progress_bar(current: int, total: int) -> None:
        """Print progress bar.

        Args:
            current (int): Current iteration.
            total (int): Total iterations.
        """
        if current % 100 == 0:
            percent = (current / total) * 100
            print(f"Progress: {percent:.2f}% ({current}/{total})", end="\r")
        if current == total - 1:
            print("Progress: 100.00% (100%)")
