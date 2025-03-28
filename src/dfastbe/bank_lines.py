from typing import List, Dict
import os
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.ops import cascaded_union, linemerge
from matplotlib import pyplot as plt
from dfastbe.support import SimulationObject, convert_search_lines_to_bank_polygons,\
    on_right_side, clip_simdata, clip_bank_lines, project_km_on_line, sort_connect_bank_lines, poly_to_line,\
    tri_to_line
from dfastbe import __version__
from dfastbe.io import ConfigFile, log_text, read_simdata, RiverData
from dfastbe.kernel import get_bbox, get_zoom_extends
from dfastbe.utils import timed_logger
from dfastbe import plotting as df_plt


MAX_RIVER_WIDTH = 1000
RAW_DETECTED_BANKLINE_FRAGMENTS_FILE = "raw_detected_bankline_fragments"
BANK_AREAS_FILE = "bank_areas"
BANKLINE_FRAGMENTS_PER_BANK_AREA_FILE = "bankline_fragments_per_bank_area"
EXTENSION = ".shp"


class BankLines:
    def __init__(self, config_file: ConfigFile, gui: bool = False):
        """
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
        self.plot_data = config_file.get_bool("General", "Plotting", True)
        self.bank_output_dir = self._get_bank_output_dir()
        # get simulation file name
        self.sim_file = config_file.get_sim_file("Detect", "")

        # get critical water depth used for defining bank line (default = 0.0 m)
        self.critical_water_depth = config_file.get_float("Detect", "WaterDepth", default=0)
        # set plotting flags
        self.plot_flags = self._get_plotting_flags()
        self.river_data = RiverData(config_file)

    @property
    def config_file(self) -> ConfigFile:
        return self._config_file

    @property
    def max_river_width(self) -> int:
        return MAX_RIVER_WIDTH

    def _get_bank_output_dir(self) -> Path:
        bank_output_dir = self.config_file.get_str("General", "BankDir")
        log_text("bankdir_out", dict={"dir": bank_output_dir})
        if os.path.exists(bank_output_dir):
            log_text("overwrite_dir", dict={"dir": bank_output_dir})
        else:
            os.makedirs(bank_output_dir)

        return Path(bank_output_dir)

    def _get_plotting_flags(self) -> Dict[str, bool]:
        """Get the plotting flags from the configuration file.

        Returns:
            data (Dict[str, bool]):
                Dictionary containing the plotting flags.
                save_plot (bool): Flag indicating whether to save the plot.
                save_plot_zoomed (bool): Flag indicating whether to save the zoomed plot.
                zoom_km_step (float): Step size for zooming in on the plot.
                close_plot (bool): Flag indicating whether to close the plot.
        """
        if self.plot_data:
            save_plot = self.config_file.get_bool("General", "SavePlots", True)
            save_plot_zoomed = self.config_file.get_bool("General", "SaveZoomPlots", True)
            zoom_km_step = self.config_file.get_float("General", "ZoomStepKM", 1.0)
            if zoom_km_step < 0.01:
                save_plot_zoomed = False
            close_plot = self.config_file.get_bool("General", "ClosePlots", False)
        else:
            save_plot = False
            save_plot_zoomed = False
            close_plot = False

        data = {
            "save_plot": save_plot,
            "save_plot_zoomed": save_plot_zoomed,
            "zoom_km_step": zoom_km_step,
            "close_plot": close_plot,
        }

        # as appropriate, check output dir for figures and file format
        if save_plot:
            fig_dir = self.config_file.get_str("General", "FigureDir", f"{self.root_dir}{os.sep}figure")
            log_text("figure_dir", dict={"dir": fig_dir})
            if os.path.exists(fig_dir):
                log_text("overwrite_dir", dict={"dir": fig_dir})
            else:
                os.makedirs(fig_dir)
            plot_ext = self.config_file.get_str("General", "FigureExt", ".png")
            data = data | {
                "fig_dir": fig_dir,
                "plot_ext": plot_ext,
            }

        return data

    def detect(self) -> None:
        """
        Run the bank line detection analysis for a specified configuration.
        """
        config_file = self.config_file
        river_data = self.river_data
        timed_logger("-- start analysis --")

        log_text(
            "header_banklines",
            dict={
                "version": __version__,
                "location": "https://github.com/Deltares/D-FAST_Bank_Erosion",
            },
        )
        log_text("-")

        # clip the chainage path to the range of chainages of interest
        km_bounds = river_data.station_bounds
        river_profile = river_data.masked_profile
        stations_coords = river_data.masked_profile_arr[:, :2]
        masked_search_lines, max_distance = river_data.clip_search_lines()

        # convert search lines to bank polygons
        d_lines = config_file.get_bank_search_distances(river_data.num_search_lines)
        bank_areas = convert_search_lines_to_bank_polygons(
            masked_search_lines, d_lines
        )

        # determine whether search lines are located on left or right
        to_right = [True] * river_data.num_search_lines
        for ib in range(river_data.num_search_lines):
            to_right[ib] = on_right_side(
                np.array(masked_search_lines[ib]), stations_coords
            )

        # read simulation data and drying flooding threshold dh0
        log_text("-")
        log_text("read_simdata", dict={"file": self.sim_file})
        log_text("-")
        sim, dh0 = read_simdata(self.sim_file)
        log_text("-")

        # increase critical water depth h0 by flooding threshold dh0
        h0 = self.critical_water_depth + dh0

        # clip simulation data to boundaries ...
        log_text("clip_data")
        sim = clip_simdata(sim, river_profile, max_distance)

        # derive bank lines (get_banklines)
        log_text("identify_banklines")
        banklines = self.get_banklines(sim, h0)
        banklines.to_file(self.bank_output_dir / f"{RAW_DETECTED_BANKLINE_FRAGMENTS_FILE}{EXTENSION}")
        gpd.GeoSeries(bank_areas).to_file(self.bank_output_dir / f"{BANK_AREAS_FILE}{EXTENSION}")

        # clip the set of detected bank lines to the bank areas
        log_text("simplify_banklines")
        bank = [None] * river_data.num_search_lines
        clipped_banklines = [None] * river_data.num_search_lines
        for ib, bank_area in enumerate(bank_areas):
            log_text("bank_lines", dict={"ib": ib + 1})
            clipped_banklines[ib] = clip_bank_lines(banklines, bank_area)
            bank[ib] = sort_connect_bank_lines(
                clipped_banklines[ib], river_profile, to_right[ib]
            )
        gpd.GeoSeries(clipped_banklines).to_file(
            self.bank_output_dir / f"{BANKLINE_FRAGMENTS_PER_BANK_AREA_FILE}{EXTENSION}"
        )
        log_text("-")

        # save bankfile
        self.save(bank)

        if self.plot_data:
            self.plot(river_data.masked_profile_arr, self.plot_flags, river_data.num_search_lines, bank, km_bounds, bank_areas, sim)

        if self.plot_data:
            if self.plot_flags["close_plot"]:
                plt.close("all")
            else:
                plt.show(block=not self.gui)

        log_text("end_banklines")
        timed_logger("-- stop analysis --")


    def plot(
        self, xy_km_numpy: np.ndarray, plot_flags: Dict[str, bool], n_search_lines: int, bank: List, km_bounds,
            bank_areas, sim
    ):

        log_text("=")
        log_text("create_figures")
        i_fig = 0
        bbox = get_bbox(xy_km_numpy)

        if plot_flags["save_plot_zoomed"]:
            bank_crds: List[np.ndarray] = []
            bank_km: List[np.ndarray] = []
            for ib in range(n_search_lines):
                bcrds_numpy = np.array(bank[ib])
                km_numpy = project_km_on_line(bcrds_numpy, xy_km_numpy)
                bank_crds.append(bcrds_numpy)
                bank_km.append(km_numpy)
            km_zoom, xy_zoom = get_zoom_extends(km_bounds[0], km_bounds[1], plot_flags["zoom_km_step"], bank_crds, bank_km)

        fig, ax = df_plt.plot_detect1(
            bbox,
            xy_km_numpy,
            bank_areas,
            bank,
            sim["facenode"],
            sim["nnodes"],
            sim["x_node"],
            sim["y_node"],
            sim["h_face"],
            1.1 * sim["h_face"].max(),
            "x-coordinate [m]",
            "y-coordinate [m]",
            "water depth and detected bank lines",
            "water depth [m]",
            "bank search area",
            "detected bank line"
        )
        if plot_flags["save_plot"]:
            i_fig = i_fig + 1
            fig_base = f"{plot_flags.get('fig_dir')}{os.sep}{i_fig}_banklinedetection"
            if plot_flags["save_plot_zoomed"]:
                df_plt.zoom_xy_and_save(fig, ax, fig_base, plot_flags.get("plot_ext"), xy_zoom, scale=1)
            fig_file = fig_base + plot_flags["plot_ext"]
            df_plt.savefig(fig, fig_file)

    def save(self, bank):
        bank_name = self.config_file.get_str("General", "BankFile", "bankfile")
        bank_file = self.bank_output_dir / f"{bank_name}.shp"
        log_text("save_banklines", dict={"file": bank_file})
        gpd.GeoSeries(bank).to_file(bank_file)

    @staticmethod
    def get_banklines(sim: SimulationObject, h0: float) -> gpd.GeoSeries:
        """
        Detect all possible bank line segments based on simulation data.

        Use a critical water depth h0 as water depth threshold for dry/wet boundary.

        Args:
            sim (SimulationObject):
                Simulation data: mesh, bed levels, water levels, velocities, etc.
            h0 (float):
                Critical water depth for determining the banks.

        Returns:
        banklines (geopandas.GeoSeries):
            The collection of all detected bank segments in the remaining model area.
        """
        fnc = sim["facenode"]
        n_nodes = sim["nnodes"]
        max_nnodes = fnc.shape[1]
        x_node = sim["x_node"][fnc]
        y_node = sim["y_node"][fnc]
        zb = sim["zb_val"][fnc]
        zw = sim["zw_face"]

        nnodes_total = len(sim["x_node"])
        try:
            mask = ~fnc.mask
            non_masked = sum(mask.reshape(fnc.size))
            fncm = fnc[mask]
            zwm = np.repeat(zw, max_nnodes)[mask]
        except:
            mask = np.repeat(True, fnc.size)
            non_masked = fnc.size
            fncm = fnc.reshape(non_masked)
            zwm = np.repeat(zw, max_nnodes).reshape(non_masked)
        zw_node = np.bincount(fncm, weights=zwm, minlength=nnodes_total)
        n_val = np.bincount(fncm, weights=np.ones(non_masked), minlength=nnodes_total)
        zw_node = zw_node / np.maximum(n_val, 1)
        zw_node[n_val == 0] = sim["zb_val"][n_val == 0]

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
                    lines[i] = tri_to_line(x_node[i], y_node[i], wet_node[i], h_node[i], h0)
                else:
                    lines[i] = poly_to_line(nnodes, x_node[i], y_node[i], wet_node[i], h_node[i], h0)
        lines = [line for line in lines if not line is None and not line.is_empty]
        multi_line = cascaded_union(lines)
        merged_line = linemerge(multi_line)

        return gpd.GeoSeries(merged_line)