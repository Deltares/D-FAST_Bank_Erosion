from typing import List, Dict
import os
import numpy as np
import geopandas as gpd
from matplotlib import pyplot as plt
from dfastbe import support
from dfastbe import __version__
from dfastbe.io import ConfigFile, log_text, clip_path_to_kmbounds, read_simdata
from dfastbe.kernel import get_bbox, get_zoom_extends
from dfastbe.utils import timed_logger
from dfastbe import plotting as df_plt

class BankLines:
    def __init__(self, config_file: ConfigFile, gui: bool = False):
        """
        Args:
            config_file : configparser.ConfigParser
                Analysis configuration settings.
            gui : bool
                Flag indicating whether this routine is called from the GUI.
        """
        if hasattr(config_file, "path"):
            rootdir = config_file.adjust_filenames()
            self.root_dir = rootdir
        else:
            self.root_dir = config_file.root_dir

        self._config_file = config_file
        self.gui = gui
        self.plot_data = config_file.get_bool("General", "Plotting", True)

    @property
    def config_file(self) -> ConfigFile:
        return self._config_file

    def _get_bank_output_dir(self) -> str:
        bank_output_dir = self.config_file.get_str("General", "BankDir")
        log_text("bankdir_out", dict={"dir": bank_output_dir})
        if os.path.exists(bank_output_dir):
            log_text("overwrite_dir", dict={"dir": bank_output_dir})
        else:
            os.makedirs(bank_output_dir)
        return bank_output_dir

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

    def banklines_core(self) -> None:
        """
        Run the bank line detection analysis for a specified configuration.
        """
        config_file = self.config_file
        timed_logger("-- start analysis --")

        log_text(
            "header_banklines",
            dict={
                "version": __version__,
                "location": "https://github.com/Deltares/D-FAST_Bank_Erosion",
            },
        )
        log_text("-")
        bank_output_dir = self._get_bank_output_dir()

        # set plotting flags
        plot_flags = self._get_plotting_flags()

        # read the chainage path
        xykm = config_file.get_xy_km()

        # clip the chainage path to the range of chainages of interest
        kmbounds = config_file.get_km_bounds()
        log_text("clip_chainage", dict={"low": kmbounds[0], "high": kmbounds[1]})
        xykm = clip_path_to_kmbounds(xykm, kmbounds)
        xykm_numpy = np.array(xykm)
        xy_numpy = xykm_numpy[:, :2]

        # read bank search lines
        max_river_width = 1000
        search_lines = config_file.get_search_lines()
        search_lines, maxmaxd = support.clip_search_lines(
            search_lines, xykm, max_river_width
        )
        n_searchlines = len(search_lines)

        # convert search lines to bank polygons
        dlines = config_file.get_bank_search_distances(n_searchlines)
        bankareas = support.convert_search_lines_to_bank_polygons(
            search_lines, dlines
        )

        # determine whether search lines are located on left or right
        to_right = [True] * n_searchlines
        for ib in range(n_searchlines):
            to_right[ib] = support.on_right_side(
                np.array(search_lines[ib]), xy_numpy
            )

        # get simulation file name
        simfile = config_file.get_sim_file("Detect", "")

        # get critical water depth used for defining bank line (default = 0.0 m)
        h0 = config_file.get_float("Detect", "WaterDepth", default=0)

        # read simulation data and drying flooding threshold dh0
        log_text("-")
        log_text("read_simdata", dict={"file": simfile})
        log_text("-")
        sim, dh0 = read_simdata(simfile)
        log_text("-")

        # increase critical water depth h0 by flooding threshold dh0
        h0 = h0 + dh0

        # clip simulation data to boundaries ...
        log_text("clip_data")
        sim = support.clip_simdata(sim, xykm, maxmaxd)

        # derive bank lines (getbanklines)
        log_text("identify_banklines")
        banklines = support.get_banklines(sim, h0)
        banklines.to_file(bank_output_dir + os.sep + "raw_detected_bankline_fragments.shp")
        gpd.GeoSeries(bankareas).to_file(bank_output_dir + os.sep + "bank_areas.shp")

        # clip the set of detected bank lines to the bank areas
        log_text("simplify_banklines")
        bank = [None] * n_searchlines
        clipped_banklines = [None] * n_searchlines
        for ib, bankarea in enumerate(bankareas):
            log_text("bank_lines", dict={"ib": ib + 1})
            clipped_banklines[ib] = support.clip_bank_lines(banklines, bankarea)
            bank[ib] = support.sort_connect_bank_lines(
                clipped_banklines[ib], xykm, to_right[ib]
            )
        gpd.GeoSeries(clipped_banklines).to_file(
            bank_output_dir + os.sep + "bankline_fragments_per_bank_area.shp"
        )
        log_text("-")

        # save bankfile
        bankname = config_file.get_str("General", "BankFile", "bankfile")
        bankfile = bank_output_dir + os.sep + bankname + ".shp"
        log_text("save_banklines", dict={"file": bankfile})
        gpd.GeoSeries(bank).to_file(bankfile)

        if self.plot_data:
            log_text("=")
            log_text("create_figures")
            ifig = 0
            bbox = get_bbox(xykm_numpy)

            if plot_flags["save_plot_zoomed"]:
                bank_crds: List[np.ndarray] = []
                bank_km: List[np.ndarray] = []
                for ib in range(n_searchlines):
                    bcrds_numpy = np.array(bank[ib])
                    km_numpy = support.project_km_on_line(bcrds_numpy, xykm_numpy)
                    bank_crds.append(bcrds_numpy)
                    bank_km.append(km_numpy)
                kmzoom, xyzoom = get_zoom_extends(kmbounds[0], kmbounds[1], plot_flags["zoom_km_step"], bank_crds, bank_km)

            fig, ax = df_plt.plot_detect1(
                bbox,
                xykm_numpy,
                bankareas,
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
                ifig = ifig + 1
                fig_base = f"{plot_flags.get('fig_dir')}{os.sep}{ifig}_banklinedetection"
                if plot_flags["save_plot_zoomed"]:
                    df_plt.zoom_xy_and_save(fig, ax, fig_base, plot_flags.get("plot_ext"), xyzoom, scale=1)
                fig_file = fig_base + plot_flags["plot_ext"]
                df_plt.savefig(fig, fig_file)

        if self.plot_data:
            if plot_flags["close_plot"]:
                plt.close("all")
            else:
                plt.show(block=not self.gui)

        log_text("end_banklines")
        timed_logger("-- stop analysis --")

