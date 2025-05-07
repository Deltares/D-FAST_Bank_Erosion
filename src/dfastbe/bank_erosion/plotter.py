import os
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from dfastbe import plotting as df_plt
from dfastbe.bank_erosion.data_models import (
    BankData,
    ErosionInputs,
    ErosionResults,
    ErosionSimulationData,
    MeshData,
    WaterLevelData,
)
from dfastbe.io import log_text
from dfastbe.kernel import get_zoom_extends

X_AXIS_TITLE = "x-coordinate [km]"
Y_AXIS_TITLE = "y-coordinate [km]"
class ErosionPlotter:
    """class to plot the results of the bank erosion analysis."""

    def __init__(
        self,
        gui: bool,
        plot_flags: Dict[str, Any],
        erosion_results: ErosionResults,
        bank_data: BankData,
        water_level_data: WaterLevelData,
        erosion_inputs: ErosionInputs,
    ):
        """Initialize the ErosionPlotter with the required data.
        
        Args:
            erosion_results (ErosionResults):
                The results of the erosion analysis.
            bank_data (BankData):
                The bank data used in the analysis.
            water_level_data (WaterLevelData):
                The water level data used in the analysis.
            erosion_inputs (ErosionInputs):
                The inputs for the erosion analysis.
            midpoint_chainages (np.ndarray):
                The midpoint chainages for the analysis.
        """
        self._gui = gui
        self._plot_flags = plot_flags
        self._erosion_results = erosion_results
        self._bank_data = bank_data
        self._water_level_data = water_level_data
        self._erosion_inputs = erosion_inputs

    @property
    def gui(self) -> bool:
        """bool: whether to use the GUI for plotting."""
        return self._gui

    @property
    def erosion_results(self) -> ErosionResults:
        """ErosionResults: the results from the erosion analysis."""
        return self._erosion_results

    @property
    def bank_data(self) -> BankData:
        """BankData: the bank data used in the analysis."""
        return self._bank_data

    @property
    def water_level_data(self) -> WaterLevelData:
        """WaterLevelData: the water level data used in the analysis."""
        return self._water_level_data

    @property
    def erosion_inputs(self) -> ErosionInputs:
        """ErosionInputs: the inputs for the erosion analysis."""
        return self._erosion_inputs

    @property
    def plot_flags(self) -> dict:
        """dict: the flags for plotting."""
        return self._plot_flags

    def plot_all(
        self,
        river_axis_km,
        xy_line_eq_list,
        km_mid,
        km_step,
        river_center_line_arr: np.ndarray,
        mesh_data: MeshData,
        simulation_data: ErosionSimulationData,
    ):
        """Plot all the results of the bank erosion analysis.

        Args:
            river_axis_km (np.ndarray):
                The river axis in km.
            xy_line_eq_list (list):
                The equilibrium line coordinates.
            km_mid (np.ndarray):
                The midpoint chainages for the analysis.
            km_step (float):
                The step size for the analysis.
            river_center_line_arr (np.ndarray):
                The river center line coordinates.
            mesh_data (MeshData):
                The mesh data used in the analysis.
            simulation_data (ErosionSimulationData):
                The simulation data used in the analysis.
        """
        if not self.plot_flags["plot_data"]:
            return
        log_text("=")
        log_text("create_figures")
        fig_i = 0
        bbox = ErosionPlotter.get_bbox(river_center_line_arr)

        km_zoom, xy_zoom = self._generate_zoomed_coordinates(river_axis_km)

        fig, ax = df_plt.plot1_waterdepth_and_banklines(
            bbox,
            river_center_line_arr,
            self.bank_data.bank_lines,
            simulation_data.face_node,
            simulation_data.n_nodes,
            simulation_data.x_node,
            simulation_data.y_node,
            simulation_data.water_depth_face,
            1.1 * self.water_level_data.hfw_max,
            X_AXIS_TITLE,
            Y_AXIS_TITLE,
            "water depth and initial bank lines",
            "water depth [m]",
        )
        if self.plot_flags["save_plot"]:
            fig_i = fig_i + 1
            fig_base = f"{self.plot_flags['fig_dir']}{os.sep}{fig_i}_banklines"

            if self.plot_flags["save_plot_zoomed"]:
                df_plt.zoom_xy_and_save(
                    fig,
                    ax,
                    fig_base,
                    self.plot_flags["plot_ext"],
                    xy_zoom,
                )

            fig_path = fig_base + self.plot_flags["plot_ext"]
            df_plt.savefig(fig, fig_path)

        fig, ax = df_plt.plot2_eroded_distance_and_equilibrium(
            bbox,
            river_center_line_arr,
            self.bank_data.bank_line_coords,
            self.erosion_results.total_erosion_dist,
            self.bank_data.is_right_bank,
            self.erosion_results.avg_erosion_rate,
            xy_line_eq_list,
            mesh_data.x_edge_coords,
            mesh_data.y_edge_coords,
            X_AXIS_TITLE,
            Y_AXIS_TITLE,
            "eroded distance and equilibrium bank location",
            f"eroded during {self.erosion_results.erosion_time} year",
            "eroded distance [m]",
            "equilibrium location",
        )
        if self.plot_flags["save_plot"]:
            fig_i = fig_i + 1
            fig_base = (
                f"{self.plot_flags['fig_dir']}{os.sep}{fig_i}_erosion_sensitivity"
            )

            if self.plot_flags["save_plot_zoomed"]:
                df_plt.zoom_xy_and_save(
                    fig,
                    ax,
                    fig_base,
                    self.plot_flags["plot_ext"],
                    xy_zoom,
                )

            fig_path = fig_base + self.plot_flags["plot_ext"]
            df_plt.savefig(fig, fig_path)

        fig, ax = df_plt.plot3_eroded_volume(
            km_mid,
            km_step,
            "river chainage [km]",
            self.erosion_results.vol_per_discharge,
            "eroded volume [m^3]",
            f"eroded volume per {km_step} chainage km ({self.erosion_results.erosion_time} years)",
            "Q{iq}",
            "Bank {ib}",
        )
        if self.plot_flags["save_plot"]:
            fig_i = fig_i + 1
            fig_base = f"{self.plot_flags['fig_dir']}{os.sep}{fig_i}_eroded_volume"

            if self.plot_flags["save_plot_zoomed"]:
                df_plt.zoom_x_and_save(
                    fig,
                    ax,
                    fig_base,
                    self.plot_flags["plot_ext"],
                    km_zoom,
                )

            fig_path = fig_base + self.plot_flags["plot_ext"]
            df_plt.savefig(fig, fig_path)

        fig, ax = df_plt.plot3_eroded_volume_subdivided_1(
            km_mid,
            km_step,
            "river chainage [km]",
            self.erosion_results.vol_per_discharge,
            "eroded volume [m^3]",
            f"eroded volume per {km_step} chainage km ({self.erosion_results.erosion_time} years)",
            "Q{iq}",
        )
        if self.plot_flags["save_plot"]:
            fig_i = fig_i + 1
            fig_base = (
                self.plot_flags["fig_dir"]
                + os.sep
                + str(fig_i)
                + "_eroded_volume_per_discharge"
            )
            if self.plot_flags["save_plot_zoomed"]:
                df_plt.zoom_x_and_save(
                    fig,
                    ax,
                    fig_base,
                    self.plot_flags["plot_ext"],
                    km_zoom,
                )
            fig_path = fig_base + self.plot_flags["plot_ext"]
            df_plt.savefig(fig, fig_path)

        fig, ax = df_plt.plot3_eroded_volume_subdivided_2(
            km_mid,
            km_step,
            "river chainage [km]",
            self.erosion_results.vol_per_discharge,
            "eroded volume [m^3]",
            f"eroded volume per {km_step} chainage km ({self.erosion_results.erosion_time} years)",
            "Bank {ib}",
        )
        if self.plot_flags["save_plot"]:
            fig_i = fig_i + 1
            fig_base = (
                self.plot_flags["fig_dir"]
                + os.sep
                + str(fig_i)
                + "_eroded_volume_per_bank"
            )
            if self.plot_flags["save_plot_zoomed"]:
                df_plt.zoom_x_and_save(
                    fig,
                    ax,
                    fig_base,
                    self.plot_flags["plot_ext"],
                    km_zoom,
                )
            fig_path = fig_base + self.plot_flags["plot_ext"]
            df_plt.savefig(fig, fig_path)

        fig, ax = df_plt.plot4_eroded_volume_eq(
            km_mid,
            km_step,
            "river chainage [km]",
            self.erosion_results.eq_eroded_vol_per_km,
            "eroded volume [m^3]",
            f"eroded volume per {km_step} chainage km (equilibrium)",
        )
        if self.plot_flags["save_plot"]:
            fig_i = fig_i + 1
            fig_base = (
                self.plot_flags["fig_dir"] + os.sep + str(fig_i) + "_eroded_volume_eq"
            )
            if self.plot_flags["save_plot_zoomed"]:
                df_plt.zoom_x_and_save(
                    fig,
                    ax,
                    fig_base,
                    self.plot_flags["plot_ext"],
                    km_zoom,
                )
            fig_path = fig_base + self.plot_flags["plot_ext"]
            df_plt.savefig(fig, fig_path)

        figlist, axlist = df_plt.plot5series_waterlevels_per_bank(
            self.bank_data.bank_chainage_midpoints,
            "river chainage [km]",
            self.water_level_data.water_level,
            self.water_level_data.ship_wave_max,
            self.water_level_data.ship_wave_min,
            "water level at Q{iq}",
            "average water level",
            "wave influenced range",
            self.water_level_data.bank_height,
            "level of bank",
            self.erosion_inputs.bank_protection_level,
            "bank protection level",
            "elevation",
            "(water)levels along bank line {ib}",
            "[m NAP]",
        )
        if self.plot_flags["save_plot"]:
            for ib, fig in enumerate(figlist):
                fig_i = fig_i + 1
                fig_base = f"{self.plot_flags['fig_dir']}/{fig_i}_levels_bank_{ib + 1}"

                if self.plot_flags["save_plot_zoomed"]:
                    df_plt.zoom_x_and_save(
                        fig,
                        axlist[ib],
                        fig_base,
                        self.plot_flags["plot_ext"],
                        km_zoom,
                    )
                fig_file = f"{fig_base}{self.plot_flags['plot_ext']}"
                df_plt.savefig(fig, fig_file)

        figlist, axlist = df_plt.plot6series_velocity_per_bank(
            self.bank_data.bank_chainage_midpoints,
            "river chainage [km]",
            self.water_level_data.velocity,
            "velocity at Q{iq}",
            self.erosion_inputs.tauc,
            self.water_level_data.chezy[0],
            "critical velocity",
            "velocity",
            "velocity along bank line {ib}",
            "[m/s]",
        )
        if self.plot_flags["save_plot"]:
            for ib, fig in enumerate(figlist):
                fig_i = fig_i + 1
                fig_base = f"{self.plot_flags['fig_dir']}{os.sep}{fig_i}_velocity_bank_{ib + 1}"

                if self.plot_flags["save_plot_zoomed"]:
                    df_plt.zoom_x_and_save(
                        fig,
                        axlist[ib],
                        fig_base,
                        self.plot_flags["plot_ext"],
                        km_zoom,
                    )

                fig_file = fig_base + self.plot_flags["plot_ext"]
                df_plt.savefig(fig, fig_file)

        fig, ax = df_plt.plot7_banktype(
            bbox,
            river_center_line_arr,
            self.bank_data.bank_line_coords,
            self.erosion_inputs.bank_type,
            self.erosion_inputs.taucls_str,
            X_AXIS_TITLE,
            Y_AXIS_TITLE,
            "bank type",
        )
        if self.plot_flags["save_plot"]:
            fig_i = fig_i + 1
            fig_base = self.plot_flags["fig_dir"] + os.sep + str(fig_i) + "_banktype"
            if self.plot_flags["save_plot_zoomed"]:
                df_plt.zoom_xy_and_save(
                    fig,
                    ax,
                    fig_base,
                    self.plot_flags["plot_ext"],
                    xy_zoom,
                )
            fig_file = fig_base + self.plot_flags["plot_ext"]
            df_plt.savefig(fig, fig_file)

        fig, ax = df_plt.plot8_eroded_distance(
            self.bank_data.bank_chainage_midpoints,
            "river chainage [km]",
            self.erosion_results.total_erosion_dist,
            "Bank {ib}",
            self.erosion_results.eq_erosion_dist,
            "Bank {ib} (eq)",
            "eroded distance",
            "[m]",
        )
        if self.plot_flags["save_plot"]:
            fig_i = fig_i + 1
            fig_base = self.plot_flags["fig_dir"] + os.sep + str(fig_i) + "_erodis"
            if self.plot_flags["save_plot_zoomed"]:
                df_plt.zoom_x_and_save(
                    fig,
                    ax,
                    fig_base,
                    self.plot_flags["plot_ext"],
                    km_zoom,
                )
            fig_file = fig_base + self.plot_flags["plot_ext"]
            df_plt.savefig(fig, fig_file)

        if self.plot_flags["close_plot"]:
            plt.close("all")
        else:
            plt.show(block=not self.gui)

    def _generate_zoomed_coordinates(self, river_axis_km):
        if not self.plot_flags["save_plot_zoomed"]:
            return None, None

        bank_coords_mid = [
            (coords[:-1, :] + coords[1:, :]) / 2
            for coords in self.bank_data.bank_line_coords
        ]
        return get_zoom_extends(
            river_axis_km.min(),
            river_axis_km.max(),
            self.plot_flags["zoom_km_step"],
            bank_coords_mid,
            self.bank_data.bank_chainage_midpoints,
        )

    @staticmethod
    def get_bbox(
        coords: np.ndarray, buffer: float = 0.1
    ) -> Tuple[float, float, float, float]:
        """
        Derive the bounding box from a line.

        Args:
            coords (np.ndarray):
                An N x M array containing x- and y-coordinates as first two M entries
            buffer : float
                Buffer fraction surrounding the tight bounding box

        Returns:
            bbox (Tuple[float, float, float, float]):
                Tuple bounding box consisting of [min x, min y, max x, max y)
        """
        return df_plt.get_bbox(coords, buffer)
