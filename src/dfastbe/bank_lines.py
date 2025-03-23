from typing import List
import os
import configparser
import numpy as np
import geopandas as gpd
from matplotlib import pyplot as plt
from dfastbe import support
from dfastbe import __version__
from dfastbe.io import ConfigFile, log_text, config_get_xykm, \
                        config_get_kmbounds, clip_path_to_kmbounds, config_get_bank_search_distances, \
                        config_get_simfile, read_simdata, config_get_search_lines
from dfastbe.kernel import get_bbox, get_zoom_extends
from dfastbe.utils import timed_logger
from dfastbe import plotting as df_plt


def banklines_core(config: configparser.ConfigParser, rootdir: str, gui: bool) -> None:
    """
    Run the bank line detection analysis for a specified configuration.

    Arguments
    ---------
    config : configparser.ConfigParser
        Analysis configuration settings.
    rootdir : str
        Root folder for the analysis (may be relative to current work directory).
    gui : bool
        Flag indicating whether this routine is called from the GUI.
    """
    config_file = ConfigFile(config)
    timed_logger("-- start analysis --")

    log_text(
        "header_banklines",
        dict={
            "version": __version__,
            "location": "https://github.com/Deltares/D-FAST_Bank_Erosion",
        },
    )
    log_text("-")

    # check output dir for bank lines
    bank_output_dir = config_file.get_str("General", "BankDir")
    log_text("bankdir_out", dict={"dir": bank_output_dir})
    if os.path.exists(bank_output_dir):
        log_text("overwrite_dir", dict={"dir": bank_output_dir})
    else:
        os.makedirs(bank_output_dir)

    # set plotting flags
    plotting = config_file.get_bool("General", "Plotting", True)
    if plotting:
        saveplot = config_file.get_bool("General", "SavePlots", True)
        saveplot_zoomed = config_file.get_bool("General", "SaveZoomPlots", True)
        zoom_km_step = config_file.get_float("General", "ZoomStepKM", 1.0)
        if zoom_km_step < 0.01:
            saveplot_zoomed = False
        closeplot = config_file.get_bool("General", "ClosePlots", False)
    else:
        saveplot = False
        saveplot_zoomed = False
        closeplot = False

    # as appropriate check output dir for figures and file format
    if saveplot:
        figdir = config_file.get_str("General", "FigureDir", f"{rootdir}{os.sep}figure")
        log_text("figure_dir", dict={"dir": figdir})
        if os.path.exists(figdir):
            log_text("overwrite_dir", dict={"dir": figdir})
        else:
            os.makedirs(figdir)
        plot_ext = config_file.get_str("General", "FigureExt", ".png")

    # read chainage path
    xykm = config_get_xykm(config)

    # clip the chainage path to the range of chainages of interest
    kmbounds = config_get_kmbounds(config)
    log_text("clip_chainage", dict={"low": kmbounds[0], "high": kmbounds[1]})
    xykm = clip_path_to_kmbounds(xykm, kmbounds)
    xykm_numpy = np.array(xykm)
    xy_numpy = xykm_numpy[:, :2]

    # read bank search lines
    max_river_width = 1000
    search_lines = config_get_search_lines(config)
    search_lines, maxmaxd = support.clip_search_lines(
        search_lines, xykm, max_river_width
    )
    n_searchlines = len(search_lines)

    # convert search lines to bank polygons
    dlines = config_get_bank_search_distances(config, n_searchlines)
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
    simfile = config_get_simfile(config, "Detect", "")

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

    if plotting:
        log_text("=")
        log_text("create_figures")
        ifig = 0
        bbox = get_bbox(xykm_numpy)

        if saveplot_zoomed:
            bank_crds: List[np.ndarray] = []
            bank_km: List[np.ndarray] = []
            for ib in range(n_searchlines):
                bcrds_numpy = np.array(bank[ib])
                km_numpy = support.project_km_on_line(bcrds_numpy, xykm_numpy)
                bank_crds.append(bcrds_numpy)
                bank_km.append(km_numpy)
            kmzoom, xyzoom = get_zoom_extends(kmbounds[0], kmbounds[1], zoom_km_step, bank_crds, bank_km)

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
        if saveplot:
            ifig = ifig + 1
            figbase = figdir + os.sep + str(ifig) + "_banklinedetection"
            if saveplot_zoomed:
                df_plt.zoom_xy_and_save(fig, ax, figbase, plot_ext, xyzoom, scale=1)
            figfile = figbase + plot_ext
            df_plt.savefig(fig, figfile)

    if plotting:
        if closeplot:
            plt.close("all")
        else:
            plt.show(block=not gui)

    log_text("end_banklines")
    timed_logger("-- stop analysis --")


def banklines(filename: str = "dfastbe.cfg") -> None:
    """
    Run the bank line detection analysis using a configuration specified by file name.

    Arguments
    ---------
    filename : str
        Name of the configuration file.
    """
    # read configuration file
    timed_logger("reading configuration file ...")

    config = ConfigFile.read(filename)
    rootdir = config.adjust_filenames()
    config = config.config
    banklines_core(config, rootdir, False)
