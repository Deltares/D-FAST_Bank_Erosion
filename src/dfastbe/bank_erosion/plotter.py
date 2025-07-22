from logging import getLogger
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from dfastbe.bank_erosion.data_models.calculation import (
    BankData,
    ErosionInputs,
    ErosionResults,
    WaterLevelData,
)
from dfastbe.bank_erosion.data_models.inputs import ErosionSimulationData
from dfastbe.bank_erosion.erosion_calculator import WATER_DENSITY, g
from dfastbe.io.config import PlotProperties, get_bbox
from dfastbe.io.logger import DfastbeLogger
from dfastbe.plotting import BasePlot, Plot
from dfastbe.utils import get_zoom_extends

X_AXIS_TITLE = "x-coordinate [km]"
Y_AXIS_TITLE = "y-coordinate [km]"


class ErosionPlotter(BasePlot):
    """class to plot the results of the bank erosion analysis."""

    def __init__(
        self,
        gui: bool,
        plot_flags: PlotProperties,
        erosion_results: ErosionResults,
        bank_data: BankData,
        water_level_data: WaterLevelData,
        erosion_inputs: ErosionInputs,
    ):
        """Initialize the ErosionPlotter with the required data.

        Args:
            gui (bool):
                Whether the plot is for a GUI application.
            plot_flags (PlotProperties):
                Flags for plotting options.
            erosion_results (ErosionResults):
                The results of the erosion analysis.
            bank_data (BankData):
                The bank data used in the analysis.
            water_level_data (WaterLevelData):
                The water level data used in the analysis.
            erosion_inputs (ErosionInputs):
                The inputs for the erosion analysis.
        """
        super().__init__(gui, plot_flags)
        self._erosion_results = erosion_results
        self._bank_data = bank_data
        self._water_level_data = water_level_data
        self._erosion_inputs = erosion_inputs
        self.logger: DfastbeLogger = getLogger("dfastbe")

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

    def plot_all(
        self,
        river_axis_km,
        xy_line_eq_list,
        km_mid,
        km_step,
        river_center_line_arr: np.ndarray,
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
            simulation_data (ErosionSimulationData):
                The simulation data used in the analysis.
        """
        self.logger.log_text("=")
        self.logger.log_text("create_figures")
        fig_i = 0
        bbox = get_bbox(river_center_line_arr)

        if self.flags.save_zoomed_plot:
            km_zoom, xy_zoom = self._generate_zoomed_coordinates(river_axis_km)
        else:
            km_zoom = None
            xy_zoom = None

        fig_i = self._plot_water_level_data(
            fig_i, bbox, river_center_line_arr, simulation_data, xy_zoom
        )

        fig_i = self._plot_erosion_sensitivity(
            fig_i, bbox, river_center_line_arr, xy_line_eq_list, xy_zoom
        )

        fig_i = self._plot_eroded_volume(fig_i, km_mid, km_step, km_zoom)

        fig_i = self._plot_eroded_volume_per_discharge(fig_i, km_mid, km_step, km_zoom)

        fig_i = self._plot_eroded_volume_per_bank(fig_i, km_mid, km_step, km_zoom)

        fig_i = self._plot_eroded_volume_equilibrium(fig_i, km_mid, km_step, km_zoom)

        fig_i = self._plot_water_levels_per_bank(fig_i, km_zoom)

        fig_i = self._plot_velocity_per_bank(fig_i, km_zoom)

        fig_i = self._plot_bank_type(fig_i, bbox, river_center_line_arr, xy_zoom)

        fig_i = self._plot_eroded_distance(fig_i, km_zoom)

        self._finalize_plots()

    def _generate_zoomed_coordinates(self, river_axis_km):

        bank_coords_mid = [
            (coords[:-1, :] + coords[1:, :]) / 2
            for coords in self.bank_data.bank_line_coords
        ]
        return get_zoom_extends(
            river_axis_km.min(),
            river_axis_km.max(),
            self.flags.zoom_step_km,
            bank_coords_mid,
            self.bank_data.bank_chainage_midpoints,
        )

    def _plot_water_level_data(
        self,
        fig_i: int,
        bbox: Tuple[float, float, float, float],
        river_center_line_arr: np.ndarray,
        simulation_data: ErosionSimulationData,
        xy_zoom: List[Tuple],
    ) -> int:
        """Plot the water level data."""
        scale = 1000
        plot = Plot(self.flags, aspect=1)

        collection = self._plot_base_water_level(
            plot.ax, river_center_line_arr, simulation_data, scale
        )
        plot.fig.colorbar(
            collection, ax=plot.ax, shrink=0.5, drawedges=False, label="water depth [m]"
        )
        plot.set_bbox(bbox)
        plot.set_axes_properties(
            X_AXIS_TITLE,
            Y_AXIS_TITLE,
            True,
            "water depth and initial bank lines"
        )
        if self.flags.save_plot:
            fig_i = plot.save(fig_i, "banklines", xy_zoom, True)
        return fig_i

    def _plot_erosion_sensitivity(
        self,
        fig_i: int,
        bbox: Tuple[float, float, float, float],
        river_center_line_arr: np.ndarray,
        xy_line_eq_list: List,
        xy_zoom: List[Tuple],
    ) -> int:
        scale = 1000
        plot = Plot(self.flags,  aspect=1, scale=scale)

        self.stations_marker(river_center_line_arr, plot.ax, float_format=0, scale=scale)
        avg_erosion_rate_max = self.erosion_results.avg_erosion_rate.max()

        p = self._create_patches(
            plot.ax,
            self.bank_data.bank_line_coords,
            self.erosion_results.total_erosion_dist,
            self.bank_data.is_right_bank,
            avg_erosion_rate_max,
            xy_line_eq_list,
            plot.scale,
        )

        plot.fig.colorbar(
            p, ax=plot.ax, shrink=0.5, drawedges=False, label="eroded distance [m]"
        )
        shaded = Patch(color="gold", linewidth=0.5)
        eqbank = Line2D([], [], color="k", linewidth=1)
        handles = [shaded, eqbank]
        labels = [
            f"eroded during {self.erosion_results.erosion_time} year",
            "equilibrium location",
        ]

        plot.set_bbox(bbox)
        plot.set_axes_properties(
            X_AXIS_TITLE,
            Y_AXIS_TITLE,
            True,
            "eroded distance and equilibrium bank location",
            handles,
            labels,
        )
        if self.flags.save_plot:
            fig_i = plot.save(
                fig_i, "erosion_sensitivity", xy_zoom, True
            )
        return fig_i

    def _plot_eroded_volume(
        self,
        fig_i: int,
        km_mid: np.ndarray,
        km_step: float,
        km_zoom: List[Tuple],
    ) -> int:
        plot = Plot(self.flags)

        self._plot_stacked_per_discharge(
            plot.ax,
            km_mid + 0.2 * km_step,
            km_step,
            self.water_level_data.vol_per_discharge,
            "Q{iq}",
            0.4,
        )
        self._plot_stacked_per_bank(
            plot.ax,
            km_mid - 0.2 * km_step,
            km_step,
            self.water_level_data.vol_per_discharge,
            "Bank {ib}",
            0.4,
        )
        plot.set_axes_properties(
            "river chainage [km]",
            "eroded volume [m^3]",
            True,
            f"eroded volume per {km_step} chainage km ({self.erosion_results.erosion_time} years)",
        )
        if self.flags.save_plot:
            fig_i = plot.save(fig_i, "eroded_volume", km_zoom, False)
        return fig_i

    def _plot_eroded_volume_per_discharge(
        self,
        fig_i: int,
        km_mid: np.ndarray,
        km_step: float,
        km_zoom: List[Tuple],
    ) -> int:
        plot = Plot(self.flags)

        self._plot_stacked_per_discharge(
            plot.ax,
            km_mid,
            km_step,
            self.water_level_data.vol_per_discharge,
            "Q{iq}",
            0.8,
        )
        plot.set_axes_properties(
            "river chainage [km]",
            "eroded volume [m^3]",
            True,
            f"eroded volume per {km_step} chainage km ({self.erosion_results.erosion_time} years)",
        )
        if self.flags.save_plot:
            fig_i = plot.save(
                fig_i, "eroded_volume_per_discharge", km_zoom, False
            )
        return fig_i

    def _plot_eroded_volume_per_bank(
        self,
        fig_i: int,
        km_mid: np.ndarray,
        km_step: float,
        km_zoom: List[Tuple],
    ) -> int:
        plot = Plot(self.flags)

        self._plot_stacked_per_bank(
            plot.ax,
            km_mid,
            km_step,
            self.water_level_data.vol_per_discharge,
            "Bank {ib}",
            0.8,
        )
        plot.set_axes_properties(
            "river chainage [km]",
            "eroded volume [m^3]",
            True,
            f"eroded volume per {km_step} chainage km ({self.erosion_results.erosion_time} years)",
        )
        if self.flags.save_plot:
            fig_i = plot.save(
                fig_i, "eroded_volume_per_bank", km_zoom, False
            )
        return fig_i

    def _plot_eroded_volume_equilibrium(
        self,
        fig_i: int,
        km_mid: np.ndarray,
        km_step: float,
        km_zoom: List[Tuple],
    ) -> int:
        plot = Plot(self.flags)

        tvol = np.zeros(km_mid.shape)
        for i in range(len(km_mid)):
            tvol[i] = self.erosion_results.eq_eroded_vol_per_km[i].sum()
        plot.ax.bar(km_mid, tvol, width=0.8 * km_step)

        plot.set_axes_properties(
            "river chainage [km]",
            "eroded volume [m^3]",
            True,
            f"eroded volume per {km_step} chainage km (equilibrium)"
        )
        if self.flags.save_plot:
            fig_i = plot.save(fig_i, "eroded_volume_eq", km_zoom, False)
        return fig_i

    def _plot_water_levels_per_bank(
        self,
        fig_i: int,
        km_zoom: List[Tuple],
    ) -> int:
        plot_list = self.plot5series_waterlevels_per_bank(
            "river chainage [km]",
            "water level at Q{iq}",
            "average water level",
            "wave influenced range",
            "level of bank",
            "bank protection level",
            "elevation",
            "(water)levels along bank line {ib}",
            "[m NAP]",
        )
        if self.flags.save_plot:
            for ib, plot in enumerate(plot_list):
                fig_i = plot.save(
                    fig_i, f"levels_bank_{ib + 1}", km_zoom, False
                )
        return fig_i

    def _plot_velocity_per_bank(
        self,
        fig_i: int,
        km_zoom: List[Tuple],
    ) -> int:
        n_banklines = len(self.bank_data.bank_chainage_midpoints)
        n_levels = len(self.water_level_data.velocity)
        plot_list = []
        clrs = self.get_colors("Blues", n_levels + 1)
        for i in range(n_banklines):
            plot = self._velocity_per_bank(
                clrs,
                "river chainage [km]",
                "velocity at Q{iq}",
                "critical velocity",
                "velocity",
                "velocity along bank line {ib}",
                "[m/s]",
                i,
            )
            plot_list.append(plot)

        if self.flags.save_plot:
            for ib, plot in enumerate(plot_list):
                fig_i = plot.save(
                    fig_i, f"velocity_bank_{ib + 1}", km_zoom, False
                )
        return fig_i

    def _plot_bank_type(
        self,
        fig_i: int,
        bbox: Tuple[float, float, float, float],
        river_center_line_arr: np.ndarray,
        xy_zoom: List[Tuple],
    ) -> int:
        scale = 1000
        plot = Plot(self.flags, aspect=1, scale=scale)

        self.stations_marker(river_center_line_arr, plot.ax, float_format=0, scale=scale)
        clrs = self.get_colors("plasma", len(self.erosion_inputs.taucls_str) + 1)
        self._plot_bank_type_segments(
            plot.ax,
            self.bank_data.bank_line_coords,
            self.erosion_inputs.bank_type,
            self.erosion_inputs.taucls_str,
            clrs,
            scale,
        )

        plot.set_bbox(bbox)
        plot.set_axes_properties(X_AXIS_TITLE, Y_AXIS_TITLE, True, "bank type")
        if self.flags.save_plot:
            fig_i = plot.save(fig_i, "banktype", xy_zoom, True)
        return fig_i

    def _plot_eroded_distance(self, fig_i: int, km_zoom: List[Tuple]) -> int:
        plot = Plot(self.flags)

        n_banklines = len(self.erosion_results.total_erosion_dist)
        clrs = self.get_colors("plasma", n_banklines + 1)
        for ib in range(n_banklines):
            bk = self.bank_data.bank_chainage_midpoints[ib]
            plot.ax.plot(
                bk,
                self.erosion_results.total_erosion_dist[ib],
                color=clrs[ib],
                label=f"Bank {ib + 1}",
            )
            plot.ax.plot(
                bk,
                self.erosion_results.eq_erosion_dist[ib],
                linestyle=":",
                color=clrs[ib],
                label=f"Bank {ib + 1} (eq)",
            )

        plot.set_axes_properties(
            "river chainage [km]",
            "eroded distance [m]",
            True,
            "eroded distance",
        )
        if self.flags.save_plot:
            fig_i = plot.save(fig_i, "erodis", km_zoom, False)
        return fig_i

    def _finalize_plots(self):
        if self.flags.close_plot:
            plt.close("all")
        else:
            plt.show(block=not self.gui)

    def _plot_base_water_level(
        self,
        ax: Axes,
        river_center_line_arr: np.ndarray,
        simulation_data: ErosionSimulationData,
        scale: float,
    ) -> PolyCollection:
        """
        Helper function to plot the base water level data, including the river centerline,
        bank lines, and mesh patches.

        Args:
            ax (Axes): The axes object to plot on.
            river_center_line_arr (np.ndarray): Array of river centerline coordinates.
            bank_lines (GeoDataFrame): GeoDataFrame containing bank line geometries.
            simulation_data (ErosionSimulationData): Simulation data for water depth.
            scale (float): Scaling factor for coordinates.
        """
        self.stations_marker(river_center_line_arr, ax, float_format=0, scale=scale)
        ax.plot(
            river_center_line_arr[:, 0] / scale,
            river_center_line_arr[:, 1] / scale,
            linestyle="--",
            color="k",
        )

        for bl in self.bank_data.bank_lines.geometry:
            bp = np.array(bl.coords)
            ax.plot(bp[:, 0] / scale, bp[:, 1] / scale, color="k")

        maximum_water_depth = 1.1 * self.water_level_data.hfw_max
        return self.mesh_patches(ax, simulation_data, 0, maximum_water_depth)

    def _create_patches(self, ax, bank_coords, total_erosion_dist, is_right_bank, avg_erosion_rate_max, xy_eq, scale):
        for i, xy_eq_part in enumerate(xy_eq):
            ax.plot(
                xy_eq_part[:, 0] / scale,
                xy_eq_part[:, 1] / scale,
                linewidth=1,
                color="k",
            )
            if is_right_bank[i]:
                bankc = bank_coords[i]
                dnc = total_erosion_dist[i]
            else:
                bankc = bank_coords[i][::-1]
                dnc = total_erosion_dist[i][::-1]
            nbp = len(bankc)

            dxy = bankc[1:] - bankc[:-1]
            ds = np.sqrt((dxy**2).sum(axis=1))
            dxy = dxy * (total_erosion_dist[i] / ds).reshape((nbp - 1, 1))

            x = np.zeros(((nbp - 1) * 4,))
            x[0::4] = bankc[:-1, 0]
            x[1::4] = bankc[1:, 0]
            x[2::4] = bankc[:-1, 0] + dxy[:, 1]
            x[3::4] = bankc[1:, 0] + dxy[:, 1]

            y = np.zeros(((nbp - 1) * 4,))
            y[0::4] = bankc[:-1, 1]
            y[1::4] = bankc[1:, 1]
            y[2::4] = bankc[:-1, 1] - dxy[:, 0]
            y[3::4] = bankc[1:, 1] - dxy[:, 0]

            tfn = np.zeros(((nbp - 1) * 2, 3))
            tfn[0::2, 0] = [4 * i for i in range(nbp - 1)]
            tfn[0::2, 1] = tfn[0::2, 0] + 1
            tfn[0::2, 2] = tfn[0::2, 0] + 2

            tfn[1::2, 0] = tfn[0::2, 0] + 1
            tfn[1::2, 1] = tfn[0::2, 0] + 2
            tfn[1::2, 2] = tfn[0::2, 0] + 3

            tval = np.zeros(((nbp - 1) * 2,))
            tval[0::2] = dnc
            tval[1::2] = dnc

            colors = ["lawngreen", "gold", "darkorange"]
            cmap = LinearSegmentedColormap.from_list("mycmap", colors)
            p = ax.tripcolor(
                x / scale,
                y / scale,
                tfn,
                facecolors=tval,
                edgecolors="face",
                linewidth=0.5,
                cmap=cmap,
                vmin=0,
                vmax=2 * avg_erosion_rate_max,
            )
        return p

    def _plot_stacked_bars(
        self,
        ax: Axes,
        km_mid: np.ndarray,
        km_step: float,
        erosion_volume: List[List[np.ndarray]],
        labels: List[str],
        colors: List[str],
        wfrac: float,
        is_discharge: bool = True,
    ) -> None:
        """Generalized helper function to plot stacked bars for erosion volume."""
        n_levels = len(erosion_volume) if is_discharge else len(erosion_volume[0])
        cumdv = None

        for i in range(n_levels):
            dvq = (
                sum(erosion_volume[i])
                if is_discharge
                else sum(erosion_volume[j][i] for j in range(len(erosion_volume)))
            )

            ax.bar(
                km_mid,
                dvq,
                width=wfrac * km_step,
                bottom=cumdv,
                color=colors[i],
                label=labels[i],
            )
            cumdv = dvq if cumdv is None else cumdv + dvq

    def _plot_stacked_per_discharge(
        self,
        ax: Axes,
        km_mid: np.ndarray,
        km_step: float,
        erosion_volume: List[List[np.ndarray]],
        qlabel: str,
        wfrac: float,
    ) -> None:
        """
        Add a stacked plot of bank erosion with total eroded volume subdivided per discharge level to the selected axes.
        """
        labels = [qlabel.format(iq=ind + 1) for ind,_ in enumerate(erosion_volume)]
        colors = self.get_colors("Blues", len(erosion_volume) + 1)[1:]
        self._plot_stacked_bars(
            ax,
            km_mid,
            km_step,
            erosion_volume,
            labels,
            colors,
            wfrac,
            is_discharge=True,
        )

    def _plot_stacked_per_bank(
        self,
        ax: Axes,
        km_mid: np.ndarray,
        km_step: float,
        erosion_volume: List[List[np.ndarray]],
        banklabel: str,
        wfrac: float,
    ) -> None:
        """
        Add a stacked plot of bank erosion with total eroded volume subdivided per bank to the selected axes.
        """
        labels = [banklabel.format(ib=ind + 1) for ind, _ in enumerate(erosion_volume[0])]
        colors = self.get_colors("plasma", len(erosion_volume[0]) + 1)[:-1]
        self._plot_stacked_bars(
            ax,
            km_mid,
            km_step,
            erosion_volume,
            labels,
            colors,
            wfrac,
            is_discharge=False,
        )

    def plot5series_waterlevels_per_bank(
        self,
        chainage_txt: str,
        waterlevelq_txt: str,
        avg_waterlevel_txt: str,
        shipwave_txt: str,
        bankheight_txt: str,
        bankprotect_txt: str,
        elevation_txt: str,
        title_txt: str,
        elevation_unit: str,
    ) -> Tuple[List[Figure], List[Axes]]:
        """
        Create the bank erosion plots with water levels, bank height and bank protection height along each bank.

        Arguments
        ---------
        bank_km_mid : List[np.ndarray]
            List of arrays containing the chainage values per bank (segment) [km].
        chainage_txt : str
            Label for the horizontal chainage axes.
        waterlevel : List[List[np.ndarray]]
            List of arrays containing the water levels per bank (point) [elevation_unit].
        shipmwavemax : np.ndarray
            Maximum bank level subject to ship waves [m]
        shipwavemin : np.ndarray
            Minimum bank level subject to ship waves [m]
        waterlevelq_txt : str
            Label for the water level per discharge level.
        avg_waterlevel_txt : str
            Label for the average water level.
        shipwave_txt : str
            Label for the elevation range influenced by ship waves.
        bankheight : List[np.ndarray]
            List of arrays containing the bank heights per bank (segment) [elevation_unit].
        bankheight_txt : str
            Label for the bank height.
        bankprotect : List[np.ndarray]
            List of arrays containing the bank protection height per bank (point) [elevation_unit].
        bankprotect_txt : str
            Label for the bank protection height.
        elevation_txt : str
            General label for elevation data.
        title_txt : str
            Label for the axes title.
        elevation_unit : str
            Unit used for all elevation data.

        Results
        -------
        figlist : List[matplotlib.figure.Figure]
            List of figure objects, one per bank.
        axlist : List[matplotlib.axes.Axes]
            List of axes objects, one per bank.
        """
        n_banklines = len(self.bank_data.bank_chainage_midpoints)
        n_levels = len(self.water_level_data.water_level)
        plot_list = []

        clrs = self.get_colors("Blues", n_levels + 1)
        for ib in range(n_banklines):
            plot = Plot(self.flags)

            bk = self.bank_data.bank_chainage_midpoints[ib]

            for iq in range(n_levels):
                # shaded range of influence for ship waves
                plot.ax.fill_between(
                    bk,
                    self.water_level_data.ship_wave_min[iq][ib],
                    self.water_level_data.ship_wave_max[iq][ib],
                    color=clrs[iq + 1],
                    alpha=0.1,
                )
                plot.ax.plot(
                    bk,
                    self.water_level_data.ship_wave_max[iq][ib],
                    color=clrs[iq + 1],
                    linestyle="--",
                    linewidth=0.5,
                )
                plot.ax.plot(
                    bk,
                    self.water_level_data.ship_wave_min[iq][ib],
                    color=clrs[iq + 1],
                    linestyle="--",
                    linewidth=0.5,
                )
                # water level line itself
                plot.ax.plot(
                    bk,
                    self.water_level_data.water_level[iq][ib],
                    color=clrs[iq + 1],
                    label=waterlevelq_txt.format(iq=iq + 1),
                )
                if iq == 0:
                    wl_avg = self.water_level_data.water_level[iq][ib].copy()
                else:
                    wl_avg = wl_avg + self.water_level_data.water_level[iq][ib]

            wl_avg = wl_avg / n_levels
            plot.ax.plot(
                bk,
                wl_avg,
                color=(0.5, 0.5, 0.5),
                linewidth=2,
                label=avg_waterlevel_txt,
            )
            plot.ax.plot(
                bk,
                self.bank_data.height[ib],
                color=(0.5, 0.5, 0.5),
                label=bankheight_txt,
            )
            ymin, ymax = plot.ax.get_ylim()

            # bank protection is only visually included in the plot
            # if it is in the same range as the other quantities
            # don't stretch the vertical scale to squeeze in a very low value.

            plot.ax.plot(
                bk,
                self.erosion_inputs.bank_protection_level[ib],
                color=(0.5, 0.5, 0.5),
                linestyle="--",
                label=bankprotect_txt,
            )
            plot.ax.set_ylim(ymin=ymin, ymax=ymax)
            handles, labels = plot.ax.get_legend_handles_labels()
            # use a slightly higher alpha for the legend to make it stand out better.
            iq = int(n_levels / 2)
            shaded = Patch(color=clrs[iq + 1], alpha=0.2)
            handles = [*handles[:-3], shaded, *handles[-3:]]
            labels = [*labels[:-3], shipwave_txt, *labels[-3:]]
            plot.set_axes_properties(
                chainage_txt,
                elevation_txt + " " + elevation_unit,
                True,
                title_txt.format(ib=ib + 1),
                handles,
                labels,
            )
            plot_list.append(plot)

        return plot_list

    def _velocity_per_bank(
        self,
        clrs: List[str],
        chainage_txt: str,
        velocq_txt: str,
        ucrit_txt: str,
        ylabel_txt: str,
        title_txt: str,
        veloc_unit: str,
        index: int,
    ):
        plot = Plot(self.flags)

        bank = self.bank_data.bank_chainage_midpoints[index]
        n_levels = len(self.water_level_data.velocity)
        velocity = np.sqrt(
            self.erosion_inputs.tauc[index]
            * self.water_level_data.chezy[0][index] ** 2
            / (WATER_DENSITY * g)
        )
        plot.ax.plot(
            self.bank_data.bank_chainage_midpoints[index],
            velocity,
            color="k",
            label=ucrit_txt,
        )
        for iq in range(n_levels):
            plot.ax.plot(
                bank,
                self.water_level_data.velocity[iq][index],
                color=clrs[iq + 1],
                label=velocq_txt.format(iq=iq + 1),
            )

        plot.set_axes_properties(
            chainage_txt,
            ylabel_txt + " " + veloc_unit,
            True,
            title_txt.format(ib=index + 1),
        )
        return plot

    def _plot_bank_type_segments(
        self,
        ax: Axes,
        bank_crds: List[np.ndarray],
        banktype: List[np.ndarray],
        taucls_str: List[str],
        clrs: List[str],
        scale: float,
    ) -> None:
        """
        Helper method to plot bank type segments.

        Args:
            ax (Axes): The axes object to plot on.
            bank_crds (List[np.ndarray]): List of bank coordinates.
            banktype (List[np.ndarray]): List of bank type values.
            taucls_str (List[str]): List of bank type labels.
            clrs (List[str]): List of colors for each bank type.
            scale (float): Scaling factor for coordinates.
        """
        for ib, bank_coords in enumerate(bank_crds):
            for ibt, tau_label in enumerate(taucls_str):
                ibt_edges = np.nonzero(banktype[ib] == ibt)[0]
                if len(ibt_edges) > 0:
                    x, y = self._generate_segment_coordinates(
                        bank_coords, ibt_edges, scale
                    )
                    ax.plot(x, y, color=clrs[ibt], label=tau_label if ib == 0 else None)
                elif ib == 0:
                    ax.plot(np.nan, np.nan, color=clrs[ibt], label=tau_label)

    def _generate_segment_coordinates(
        self, bank_coords: np.ndarray, edges: np.ndarray, scale: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate x and y coordinates for bank type segments.

        Args:
            bank_coords (np.ndarray): Array of bank coordinates.
            edges (np.ndarray): Indices of edges for the bank type.
            scale (float): Scaling factor for coordinates.

        Returns:
            Tuple[np.ndarray, np.ndarray]: x and y coordinates for the segments.
        """
        nedges = len(edges)
        nx = max(3 * nedges - 1, 0)
        x = np.full(nx, np.nan)
        y = np.full(nx, np.nan)
        x[0::3] = bank_coords[edges, 0] / scale
        y[0::3] = bank_coords[edges, 1] / scale
        x[1::3] = bank_coords[edges + 1, 0] / scale
        y[1::3] = bank_coords[edges + 1, 1] / scale
        return x, y
