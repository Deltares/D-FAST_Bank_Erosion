"""Bank line detection module."""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon
from shapely import union_all, line_merge

from dfastbe import __version__
from dfastbe import plotting as df_plt
from dfastbe.io import (
    ConfigFile,
    RiverData,
    SimulationData,
    log_text,
    get_bbox
)
from dfastbe.kernel import get_zoom_extends
from dfastbe.support import (
    clip_bank_lines,
    on_right_side,
    poly_to_line,
    project_km_on_line,
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
        """
        # the root_dir is used to get the FigureDir in the `_get_plotting_flags`
        self.root_dir = config_file.root_dir

        self._config_file = config_file
        self.gui = gui
        self.bank_output_dir = config_file.get_output_dir("banklines")

        # set plotting flags
        self.plot_flags = config_file.get_plotting_flags(self.root_dir)
        self.river_data = RiverData(config_file)
        self.search_lines = self.river_data.search_lines

        self.simulation_data, self.h0 = self._get_simulation_data()

    def _get_simulation_data(self) -> Tuple[SimulationData, float]:
        # read simulation data and drying flooding threshold dh0
        sim_file = self.config_file.get_sim_file("Detect", "")
        log_text("read_simdata", data={"file": sim_file})
        simulation_data = SimulationData.read(sim_file)
        # increase critical water depth h0 by flooding threshold dh0
        # get critical water depth used for defining bank line (default = 0.0 m)
        critical_water_depth = self.config_file.get_float(
            "Detect", "WaterDepth", default=0
        )
        h0 = critical_water_depth + simulation_data.dry_wet_threshold
        return simulation_data, h0

    @property
    def config_file(self) -> ConfigFile:
        """Configuration file object."""
        return self._config_file

    @property
    def max_river_width(self) -> int:
        """Maximum river width in meters."""
        return MAX_RIVER_WIDTH

    def _get_bank_output_dir(self) -> Path:
        bank_output_dir = self.config_file.get_str("General", "BankDir")
        log_text("bankdir_out", data={"dir": bank_output_dir})
        if os.path.exists(bank_output_dir):
            log_text("overwrite_dir", data={"dir": bank_output_dir})
        else:
            os.makedirs(bank_output_dir)

        return Path(bank_output_dir)

    def detect(self) -> None:
        """Run the bank line detection analysis for a specified configuration."""
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

        search_lines = self.search_lines.values
        max_distance = self.search_lines.max_distance
        num_search_lines = self.search_lines.size
        # convert search lines to bank polygons
        d_lines = config_file.get_bank_search_distances(self.search_lines.size)
        bank_areas: List[Polygon] = self._convert_search_lines_to_bank_polygons(
            search_lines, d_lines
        )

        # determine whether search lines are located on the left or right
        to_right = [True] * num_search_lines
        for ib in range(num_search_lines):
            to_right[ib] = on_right_side(
                np.array(search_lines[ib].coords), stations_coords
            )

        # clip simulation data to boundaries ...
        log_text("clip_data")
        self.simulation_data.clip(river_center_line_values, max_distance)

        # derive bank lines (get_banklines)
        log_text("identify_banklines")
        banklines = self.detect_bank_lines(self.simulation_data, self.h0, config_file)

        # clip the set of detected bank lines to the bank areas
        log_text("simplify_banklines")
        bank = [None] * num_search_lines
        clipped_banklines = [None] * num_search_lines
        for ib, bank_area in enumerate(bank_areas):
            log_text("bank_lines", data={"ib": ib + 1})
            clipped_banklines[ib] = clip_bank_lines(banklines, bank_area)
            bank[ib] = sort_connect_bank_lines(
                clipped_banklines[ib], river_center_line_values, to_right[ib]
            )

        # save bank_file
        self.save(bank, banklines, clipped_banklines, bank_areas, config_file)


        if self.plot_flags["plot_data"]:
            self.plot(
                center_line_arr,
                num_search_lines,
                bank,
                station_bounds,
                bank_areas,
                config_file,
            )

        log_text("end_banklines")
        timed_logger("-- stop analysis --")

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
                km_numpy = project_km_on_line(bcrds_numpy, xy_km_numpy)
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
        self, bank, banklines, clipped_banklines, bank_areas, config_file: ConfigFile
    ):
        """Save result files."""
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
        simulation_data: SimulationData, h0: float, config_file: ConfigFile
    ) -> gpd.GeoSeries:
        """
        Detect all possible bank line segments based on simulation data.

        Use a critical water depth h0 as a water depth threshold for dry/wet boundary.

        Args:
            simulation_data (SimulationData):
                Simulation data: mesh, bed levels, water levels, velocities, etc.
            h0 (float):
                Critical water depth for determining the banks.

        Returns:
            geopandas.GeoSeries:
                The collection of all detected bank segments in the remaining model area.
        """
        fnc = simulation_data.face_node
        n_nodes = simulation_data.n_nodes
        max_nnodes = fnc.shape[1]
        x_node = simulation_data.x_node[fnc]
        y_node = simulation_data.y_node[fnc]
        zb = simulation_data.bed_elevation_values[fnc]
        zw = simulation_data.water_level_face

        nnodes_total = len(simulation_data.x_node)
        try:
            mask = ~fnc.mask
            non_masked = sum(mask.reshape(fnc.size))
            f_nc_m = fnc[mask]
            zwm = np.repeat(zw, max_nnodes)[mask]
        except:
            mask = np.repeat(True, fnc.size)
            non_masked = fnc.size
            f_nc_m = fnc.reshape(non_masked)
            zwm = np.repeat(zw, max_nnodes).reshape(non_masked)

        zw_node = np.bincount(f_nc_m, weights=zwm, minlength=nnodes_total)
        n_val = np.bincount(f_nc_m, weights=np.ones(non_masked), minlength=nnodes_total)
        zw_node = zw_node / np.maximum(n_val, 1)
        zw_node[n_val == 0] = simulation_data.bed_elevation_values[n_val == 0]

        h_node = zw_node[fnc] - zb
        wet_node = h_node > h0
        n_wet_arr = wet_node.sum(axis=1)
        mask = n_wet_arr.mask.size > 1

        n_faces = len(fnc)
        lines = [None] * n_faces
        frac = 0
        for i in range(n_faces):
            if i >= frac * (n_faces - 1) / 10:
                print(int(frac * 10))
                frac = frac + 1
            nnodes = n_nodes[i]
            n_wet = n_wet_arr[i]
            if (mask and n_wet.mask) or n_wet == 0 or n_wet == nnodes:
                # all dry or all wet
                pass
            else:
                # some nodes dry and some nodes wet: determine the line
                if nnodes == 3:
                    lines[i] = tri_to_line(
                        x_node[i], y_node[i], wet_node[i], h_node[i], h0
                    )
                else:
                    lines[i] = poly_to_line(
                        nnodes, x_node[i], y_node[i], wet_node[i], h_node[i], h0
                    )
        lines = [line for line in lines if line is not None and not line.is_empty]
        multi_line = union_all(lines)
        merged_line = line_merge(multi_line)

        return gpd.GeoSeries(merged_line, crs=config_file.crs)

    @staticmethod
    def _convert_search_lines_to_bank_polygons(
        search_lines: List[np.ndarray], d_lines: List[float]
    ) -> List[Polygon]:
        """
        Construct a series of polygons surrounding the bank search lines.

        Args:
            search_lines : List[numpy.ndarray]
                List of arrays containing the x,y-coordinates of a bank search lines.
            d_lines : List[float]
                Array containing the search distance value per bank line.

        Returns:
            bank_areas:
                Array containing the areas of interest surrounding the bank search lines.
        """
        n_bank = len(search_lines)
        bank_areas = [None] * n_bank
        for b, distance in enumerate(d_lines):
            bank_areas[b] = search_lines[b].buffer(distance, cap_style=2)

        return bank_areas
