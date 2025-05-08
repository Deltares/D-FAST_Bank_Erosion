from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from geopandas.geodataframe import GeoDataFrame
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
from dfastbe.kernel import g, get_zoom_extends, water_density

X_AXIS_TITLE = "x-coordinate [km]"
Y_AXIS_TITLE = "y-coordinate [km]"
class ErosionPlotter(df_plt.PlottingBase):
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
        bbox = self.get_bbox(river_center_line_arr)

        km_zoom, xy_zoom = self._generate_zoomed_coordinates(river_axis_km)

        fig_i = self._plot_water_level_data(
            fig_i, bbox, river_center_line_arr, simulation_data, xy_zoom
        )

        fig_i = self._plot_erosion_sensitivity(
            fig_i, bbox, river_center_line_arr, xy_line_eq_list, mesh_data, xy_zoom
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

    def _plot_water_level_data(
        self,
        fig_i: int,
        bbox: Tuple[float, float, float, float],
        river_center_line_arr: np.ndarray,
        simulation_data: ErosionSimulationData,
        xy_zoom: List[Tuple],
    ) -> int:
        """Plot the water level data."""
        fig, ax = self.plot1_waterdepth_and_banklines(
            bbox,
            river_center_line_arr,
            self.bank_data.bank_lines,
            X_AXIS_TITLE,
            Y_AXIS_TITLE,
            "water depth and initial bank lines",
            "water depth [m]",
            simulation_data,
        )
        if self.plot_flags["save_plot"]:
            fig_i = self._save_plot(fig, ax, fig_i, "banklines", xy_zoom, True)
        return fig_i

    def _plot_erosion_sensitivity(
        self,
        fig_i: int,
        bbox: Tuple[float, float, float, float],
        river_center_line_arr: np.ndarray,
        xy_line_eq_list: List,
        mesh_data: MeshData,
        xy_zoom: List[Tuple],
    ) -> int:
        fig, ax = self.plot2_eroded_distance_and_equilibrium(
            bbox,
            river_center_line_arr,
            self.bank_data.bank_line_coords,
            self.erosion_results.total_erosion_dist,
            self.bank_data.is_right_bank,
            self.erosion_results.avg_erosion_rate,
            xy_line_eq_list,
            X_AXIS_TITLE,
            Y_AXIS_TITLE,
            "eroded distance and equilibrium bank location",
            f"eroded during {self.erosion_results.erosion_time} year",
            "eroded distance [m]",
            "equilibrium location",
        )
        if self.plot_flags["save_plot"]:
            fig_i = self._save_plot(
                fig, ax, fig_i, "erosion_sensitivity", xy_zoom, True
            )
        return fig_i

    def _plot_eroded_volume(
        self,
        fig_i: int,
        km_mid: np.ndarray,
        km_step: float,
        km_zoom: List[Tuple],
    ) -> int:
        fig, ax = self.plot3_eroded_volume(
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
            fig_i = self._save_plot(fig, ax, fig_i, "eroded_volume", km_zoom, False)
        return fig_i

    def _plot_eroded_volume_per_discharge(
        self,
        fig_i: int,
        km_mid: np.ndarray,
        km_step: float,
        km_zoom: List[Tuple],
    ) -> int:
        fig, ax = self.plot3_eroded_volume_subdivided_1(
            km_mid,
            km_step,
            "river chainage [km]",
            self.erosion_results.vol_per_discharge,
            "eroded volume [m^3]",
            f"eroded volume per {km_step} chainage km ({self.erosion_results.erosion_time} years)",
            "Q{iq}",
        )
        if self.plot_flags["save_plot"]:
            fig_i = self._save_plot(
                fig, ax, fig_i, "eroded_volume_per_discharge", km_zoom, False
            )
        return fig_i

    def _plot_eroded_volume_per_bank(
        self,
        fig_i: int,
        km_mid: np.ndarray,
        km_step: float,
        km_zoom: List[Tuple],
    ) -> int:
        fig, ax = self.plot3_eroded_volume_subdivided_2(
            km_mid,
            km_step,
            "river chainage [km]",
            self.erosion_results.vol_per_discharge,
            "eroded volume [m^3]",
            f"eroded volume per {km_step} chainage km ({self.erosion_results.erosion_time} years)",
            "Bank {ib}",
        )
        if self.plot_flags["save_plot"]:
            fig_i = self._save_plot(
                fig, ax, fig_i, "eroded_volume_per_bank", km_zoom, False
            )
        return fig_i

    def _plot_eroded_volume_equilibrium(
        self,
        fig_i: int,
        km_mid: np.ndarray,
        km_step: float,
        km_zoom: List[Tuple],
    ) -> int:
        fig, ax = self.plot4_eroded_volume_eq(
            km_mid,
            km_step,
            "river chainage [km]",
            self.erosion_results.eq_eroded_vol_per_km,
            "eroded volume [m^3]",
            f"eroded volume per {km_step} chainage km (equilibrium)",
        )
        if self.plot_flags["save_plot"]:
            fig_i = self._save_plot(fig, ax, fig_i, "eroded_volume_eq", km_zoom, False)
        return fig_i

    def _plot_water_levels_per_bank(
        self,
        fig_i: int,
        km_zoom: List[Tuple],
    ) -> int:
        figlist, axlist = self.plot5series_waterlevels_per_bank(
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
                fig_i = self._save_plot(
                    fig, axlist[ib], fig_i, f"levels_bank_{ib + 1}", km_zoom, False
                )
        return fig_i

    def _plot_velocity_per_bank(
        self,
        fig_i: int,
        km_zoom: List[Tuple],
    ) -> int:
        figlist, axlist = self.plot6series_velocity_per_bank(
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
                fig_i = self._save_plot(
                    fig, axlist[ib], fig_i, f"velocity_bank_{ib + 1}", km_zoom, False
                )
        return fig_i

    def _plot_bank_type(
        self,
        fig_i: int,
        bbox: Tuple[float, float, float, float],
        river_center_line_arr: np.ndarray,
        xy_zoom: List[Tuple],
    ) -> int:
        fig, ax = self.plot7_banktype(
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
            fig_i = self._save_plot(fig, ax, fig_i, "banktype", xy_zoom, True)
        return fig_i

    def _plot_eroded_distance(self, fig_i: int, km_zoom: List[Tuple]) -> int:
        fig, ax = self.plot8_eroded_distance(
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
            fig_i = self._save_plot(fig, ax, fig_i, "erodis", km_zoom, False)
        return fig_i

    def _save_plot(self, fig, ax, fig_i, plot_name, zoom_coords, zoom_xy) -> int:
        """Save the plot to a file."""
        return self.save_plot(
            fig, ax, fig_i, plot_name, zoom_coords, self.plot_flags, zoom_xy
        )

    def _finalize_plots(self):
        if self.plot_flags["close_plot"]:
            plt.close("all")
        else:
            plt.show(block=not self.gui)

    def plot1_waterdepth_and_banklines(
        self,
        bbox: Tuple[float, float, float, float],
        xykm: np.ndarray,
        banklines: GeoDataFrame,
        xlabel_txt: str,
        ylabel_txt: str,
        title_txt: str,
        waterdepth_txt: str,
        simulation_data: ErosionSimulationData,
    ) -> Tuple[Figure, Axes]:
        """
        Create the bank erosion plot with water depths and initial bank lines.

        bbox : Tuple[float, float, float, float]
            Tuple containing boundary limits (xmin, ymin, xmax, ymax); unit m.
        xykm : np.ndarray
            Array containing the x, y, and chainage; unit m for x and y, km for chainage.
        banklines : geopandas.geodataframe.GeoDataFrame
            Pandas object containing the bank lines.
        xlabel_txt : str
            Label for the x-axis.
        ylabel_txt : str
            Label for the y-axis.
        title_txt : str
            Label for the axes title.
        waterdepth_txt : str
            Label for the color bar.

        Results
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        ax : matplotlib.axes.Axes
            Axes object.
        """
        fig, ax = plt.subplots()
        self.setsize(fig)
        ax.set_aspect(1)
        #
        scale = 1000
        self.chainage_markers(xykm, ax, ndec=0, scale=scale)
        ax.plot(xykm[:, 0] / scale, xykm[:, 1] / scale, linestyle="--", color="k")
        for bl in banklines.geometry:
            bp = np.array(bl.coords)
            ax.plot(bp[:, 0] / scale, bp[:, 1] / scale, color="k")

        maximum_water_depth = 1.1 * self.water_level_data.hfw_max
        p = self.plot_mesh_patches(ax, simulation_data, 0, maximum_water_depth)
        cbar = fig.colorbar(p, ax=ax, shrink=0.5, drawedges=False, label=waterdepth_txt)
        #
        self.set_bbox(ax, bbox)
        ax.set_xlabel(xlabel_txt)
        ax.set_ylabel(ylabel_txt)
        ax.grid(True)
        ax.set_title(title_txt)
        return fig, ax

    def plot2_eroded_distance_and_equilibrium(
        self,
        bbox: Tuple[float, float, float, float],
        xykm: np.ndarray,
        bank_crds: List[np.ndarray],
        dn_tot: List[np.ndarray],
        to_right: List[bool],
        dnav: np.ndarray,
        xy_eq: List[np.ndarray],
        xlabel_txt: str,
        ylabel_txt: str,
        title_txt: str,
        erosion_txt: str,
        eroclr_txt: str,
        eqbank_txt: str,
    ) -> Tuple[Figure, Axes]:
        """
        Create the bank erosion plot with predicted bank line shift and equilibrium bank line.

        Arguments
        ---------
        bbox : Tuple[float, float, float, float]
            Tuple containing boundary limits (xmin, ymin, xmax, ymax); unit m.
        xykm : np.ndarray
            Array containing the x, y, and chainage; unit m for x and y, km for chainage.
        bank_crds : List[np.ndarray]
            List of N arrays containing the x- and y-coordinates of the original
            bank lines.
        dn_tot : List[np.ndarray]
            List of N arrays containing the total erosion distance values.
        to_right : List[bool]
            List of N booleans indicating whether the bank is on the right.
        dnav : np.ndarray
            Array of N average erosion distance values.
        xy_eq : List[np.ndarray]
            List of N arrays containing the x- and y-coordinates of the equilibrium
            bank line.
        xlabel_txt : str
            Label for the x-axis.
        ylabel_txt : str
            Label for the y-axis.
        title_txt : str
            Label for the axes title.
        erosion_txt : str
            Label for the shaded eroded area.
        eroclr_txt : str
            Label for the color bar.
        eqbank_txt : str
            Label for the equilibrium bank position.

        Results
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        ax : matplotlib.axes.Axes
            Axes object.
        """
        scale = 1000
        fig, ax = plt.subplots()
        self.setsize(fig)
        ax.set_aspect(1)
        #
        # plot_mesh(ax, xe, ye, scale=scale)
        self.chainage_markers(xykm, ax, ndec=0, scale=scale)
        dnav_max = dnav.max()
        for ib in range(len(xy_eq)):
            ax.plot(
                xy_eq[ib][:, 0] / scale, xy_eq[ib][:, 1] / scale, linewidth=1, color="k"
            )
            #
            if to_right[ib]:
                bankc = bank_crds[ib]
                dnc = dn_tot[ib]
            else:
                bankc = bank_crds[ib][::-1]
                dnc = dn_tot[ib][::-1]
            nbp = len(bankc)
            #
            dxy = bankc[1:] - bankc[:-1]
            ds = np.sqrt((dxy**2).sum(axis=1))
            dxy = dxy * (dn_tot[ib] / ds).reshape((nbp - 1, 1))
            #
            x = np.zeros(((nbp - 1) * 4,))
            x[0::4] = bankc[:-1, 0]
            x[1::4] = bankc[1:, 0]
            x[2::4] = bankc[:-1, 0] + dxy[:, 1]
            x[3::4] = bankc[1:, 0] + dxy[:, 1]
            #
            y = np.zeros(((nbp - 1) * 4,))
            y[0::4] = bankc[:-1, 1]
            y[1::4] = bankc[1:, 1]
            y[2::4] = bankc[:-1, 1] - dxy[:, 0]
            y[3::4] = bankc[1:, 1] - dxy[:, 0]
            #
            tfn = np.zeros(((nbp - 1) * 2, 3))
            tfn[0::2, 0] = [4 * i for i in range(nbp - 1)]
            tfn[0::2, 1] = tfn[0::2, 0] + 1
            tfn[0::2, 2] = tfn[0::2, 0] + 2
            #
            tfn[1::2, 0] = tfn[0::2, 0] + 1
            tfn[1::2, 1] = tfn[0::2, 0] + 2
            tfn[1::2, 2] = tfn[0::2, 0] + 3
            #
            tval = np.zeros(((nbp - 1) * 2,))
            tval[0::2] = dnc
            tval[1::2] = dnc
            #
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
                vmax=2 * dnav_max,
            )
        #
        cbar = fig.colorbar(p, ax=ax, shrink=0.5, drawedges=False, label=eroclr_txt)
        #
        shaded = Patch(color="gold", linewidth=0.5)
        eqbank = Line2D([], [], color="k", linewidth=1)
        handles = [shaded, eqbank]
        labels = [erosion_txt, eqbank_txt]
        #
        self.set_bbox(ax, bbox)
        ax.set_xlabel(xlabel_txt)
        ax.set_ylabel(ylabel_txt)
        ax.grid(True)
        ax.set_title(title_txt)
        ax.legend(handles, labels, loc="upper right")
        return fig, ax

    def plot3_eroded_volume(
        self,
        km_mid: np.ndarray,
        km_step: float,
        chainage_txt: str,
        erosion_volume: List[List[np.ndarray]],
        ylabel_txt: str,
        title_txt: str,
        qlabel: str,
        banklabel: str,
    ) -> Tuple[Figure, Axes]:
        """
        Create the bank erosion plot with total eroded volume subdivided per discharge level.

        Arguments
        ---------
        km_mid : np.ndarray
            Array containing the mid points for the chainage bins.
        km_step : float
            Bin width.
        chainage_txt : str
            Label for the horizontal chainage axes.
        erosion_volume : List[List[np.ndarray]]
            List of nQ lists of N arrays containing the total erosion distance values
        ylabel_txt : str
            Label for the vertical erosion volume axes.
        title_txt : str
            Label for axes title.
        qlabel : str
            Label for discharge level.
        banklabel : str
            Label for bank id.

        Results
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        ax : matplotlib.axes.Axes
            Axes object.
        """
        fig, ax = plt.subplots()
        self.setsize(fig)
        self.plot3_stacked_per_discharge(
            ax, km_mid + 0.2 * km_step, km_step, erosion_volume, qlabel, 0.4
        )
        self.plot3_stacked_per_bank(
            ax, km_mid - 0.2 * km_step, km_step, erosion_volume, banklabel, 0.4
        )
        ax.set_xlabel(chainage_txt)
        ax.set_ylabel(ylabel_txt)
        ax.grid(True)
        ax.set_title(title_txt)
        ax.legend(loc="upper right")
        return fig, ax

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

    def plot3_stacked_per_discharge(
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
        labels = [qlabel.format(iq=iq + 1) for iq in range(len(erosion_volume))]
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

    def plot3_stacked_per_bank(
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
        labels = [banklabel.format(ib=ib + 1) for ib in range(len(erosion_volume[0]))]
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

    def plot3_eroded_volume_subdivided_1(
        self,
        km_mid: np.ndarray,
        km_step: float,
        chainage_txt: str,
        erosion_volume: List[List[np.ndarray]],
        ylabel_txt: str,
        title_txt: str,
        qlabel: str,
    ) -> Tuple[Figure, Axes]:
        """
        Create the bank erosion plot with total eroded volume subdivided per discharge level.

        Arguments
        ---------
        km_mid : np.ndarray
            Array containing the mid points for the chainage bins.
        km_step : float
            Bin width.
        chainage_txt : str
            Label for the horizontal chainage axes.
        erosion_volume : List[List[np.ndarray]]
            List of nQ lists of N arrays containing the total erosion distance values
        ylabel_txt : str
            Label for the vertical erosion volume axes.
        title_txt : str
            Label for axes title.
        qlabel : str
            Label for discharge level.

        Results
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        ax : matplotlib.axes.Axes
            Axes object.
        """
        fig, ax = plt.subplots()
        self.setsize(fig)
        #
        self.plot3_stacked_per_discharge(
            ax, km_mid, km_step, erosion_volume, qlabel, 0.8
        )
        #
        ax.set_xlabel(chainage_txt)
        ax.set_ylabel(ylabel_txt)
        ax.grid(True)
        ax.set_title(title_txt)
        ax.legend(loc="upper right")
        return fig, ax

    def plot3_eroded_volume_subdivided_2(
        self,
        km_mid: np.ndarray,
        km_step: float,
        chainage_txt: str,
        erosion_volume: List[List[np.ndarray]],
        ylabel_txt: str,
        title_txt: str,
        banklabel: str,
    ) -> Tuple[Figure, Axes]:
        """
        Create the bank erosion plot with total eroded volume subdivided per bank.

        Arguments
        ---------
        km_mid : np.ndarray
            Array containing the mid points for the chainage bins.
        km_step : float
            Bin width.
        chainage_txt : str
            Label for the horizontal chainage axes.
        erosion_volume : List[List[np.ndarray]]
            List of nQ lists of N arrays containing the total erosion distance values
        ylabel_txt : str
            Label for the vertical erosion volume axes.
        title_txt : str
            Label for axes title.
        banklabel : str
            Label for bank id.

        Results
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        ax : matplotlib.axes.Axes
            Axes object.
        """
        fig, ax = plt.subplots()
        self.setsize(fig)
        #
        self.plot3_stacked_per_bank(ax, km_mid, km_step, erosion_volume, banklabel, 0.8)
        #
        ax.set_xlabel(chainage_txt)
        ax.set_ylabel(ylabel_txt)
        ax.grid(True)
        ax.set_title(title_txt)
        ax.legend(loc="upper right")
        return fig, ax

    def plot4_eroded_volume_eq(
        self,
        km_mid: np.ndarray,
        km_step: float,
        chainage_txt: str,
        vol_eq: np.ndarray,
        ylabel_txt: str,
        title_txt: str,
    ) -> Tuple[Figure, Axes]:
        """
        Create the bank erosion plot with equilibrium eroded volume.

        Arguments
        ---------
        km_mid : np.ndarray
            Array containing the mid points for the chainage bins.
        km_step : float
            Bin width.
        chainage_txt : str
            Label for the horizontal chainage axes.
        vol_eq : np.ndarray
            Array containing the equilibrium eroded volume per bin.
        ylabel_txt : str
            Label for the vertical erosion volume axes.
        title_txt : str
            Label for axes title.

        Results
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        ax : matplotlib.axes.Axes
            Axes object.
        """
        fig, ax = plt.subplots()
        self.setsize(fig)
        #
        tvol = np.zeros(km_mid.shape)
        for i in range(len(km_mid)):
            tvol[i] = vol_eq[i].sum()
        ax.bar(km_mid, tvol, width=0.8 * km_step)
        #
        ax.set_xlabel(chainage_txt)
        ax.set_ylabel(ylabel_txt)
        ax.grid(True)
        ax.set_title(title_txt)
        return fig, ax

    def plot5series_waterlevels_per_bank(
        self,
        bank_km_mid: List[np.ndarray],
        chainage_txt: str,
        waterlevel: List[List[np.ndarray]],
        shipwavemax: List[List[np.ndarray]],
        shipwavemin: List[List[np.ndarray]],
        waterlevelq_txt: str,
        avg_waterlevel_txt: str,
        shipwave_txt: str,
        bankheight: List[np.ndarray],
        bankheight_txt: str,
        bankprotect: List[np.ndarray],
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
        n_banklines = len(bank_km_mid)
        n_levels = len(waterlevel)
        figlist: List[Figure] = []
        axlist: List[Axes] = []
        clrs = self.get_colors("Blues", n_levels + 1)
        for ib in range(n_banklines):
            fig, ax = plt.subplots()
            self.setsize(fig)
            bk = bank_km_mid[ib]
            #
            for iq in range(n_levels):
                # shaded range of influence for ship waves
                ax.fill_between(
                    bk,
                    shipwavemin[iq][ib],
                    shipwavemax[iq][ib],
                    color=clrs[iq + 1],
                    alpha=0.1,
                )
                ax.plot(
                    bk,
                    shipwavemax[iq][ib],
                    color=clrs[iq + 1],
                    linestyle="--",
                    linewidth=0.5,
                )
                ax.plot(
                    bk,
                    shipwavemin[iq][ib],
                    color=clrs[iq + 1],
                    linestyle="--",
                    linewidth=0.5,
                )
                # water level line itself
                ax.plot(
                    bk,
                    waterlevel[iq][ib],
                    color=clrs[iq + 1],
                    label=waterlevelq_txt.format(iq=iq + 1),
                )
                if iq == 0:
                    wl_avg = waterlevel[iq][ib].copy()
                else:
                    wl_avg = wl_avg + waterlevel[iq][ib]
            #
            wl_avg = wl_avg / n_levels
            ax.plot(
                bk,
                wl_avg,
                color=(0.5, 0.5, 0.5),
                linewidth=2,
                label=avg_waterlevel_txt,
            )
            ax.plot(bk, bankheight[ib], color=(0.5, 0.5, 0.5), label=bankheight_txt)
            ymin, ymax = ax.get_ylim()
            #
            # bank protection is only visually included in the plot
            # if it is in the same range as the other quantities
            # don't stretch the vertical scale to squeeze in a very low value.
            #
            ax.plot(
                bk,
                bankprotect[ib],
                color=(0.5, 0.5, 0.5),
                linestyle="--",
                label=bankprotect_txt,
            )
            ax.set_ylim(ymin=ymin, ymax=ymax)
            #
            handles, labels = ax.get_legend_handles_labels()
            #
            # use a slightly higher alpha for the legend to make it stand out better.
            iq = int(n_levels / 2)
            shaded = Patch(color=clrs[iq + 1], alpha=0.2)
            handles = [*handles[:-3], shaded, *handles[-3:]]
            labels = [*labels[:-3], shipwave_txt, *labels[-3:]]
            #
            ax.set_xlabel(chainage_txt)
            ax.set_ylabel(elevation_txt + " " + elevation_unit)
            ax.grid(True)
            ax.set_title(title_txt.format(ib=ib + 1))
            ax.legend(handles, labels, loc="upper right")
            figlist.append(fig)
            axlist.append(ax)
        return figlist, axlist

    def plot6series_velocity_per_bank(
        self,
        bank_km_mid: List[np.ndarray],
        chainage_txt: str,
        veloc: List[List[np.ndarray]],
        velocq_txt: str,
        tauc: List[np.ndarray],
        chezy: List[np.ndarray],
        ucrit_txt: str,
        ylabel_txt: str,
        title_txt: str,
        veloc_unit: str,
    ) -> Tuple[List[Figure], List[Axes]]:
        """
        Create the bank erosion plots with velocities and critical velocities along each bank.

        Arguments
        ---------
        bank_km_mid : List[np.ndarray]
            List of arrays containing the chainage values per bank (segment) [km].
        chainage_txt : str
            Label for the horizontal chainage axes.
        veloc: List[List[np.ndarray]]
            List of arrays containing the velocities per bank (segment) [m/s].
        velocq_txt: str,
            Label for the velocity per discharge level.
        tauc: List[np.ndarray]
            List of arrays containing the shear stresses per bank (point) [N/m2].
        chezy: List[np.ndarray]
            List of arrays containing the Chezy values per bank [m0.5/s].
        ucrit_txt: str
            Label for the critical velocity.
        ylabel_txt: str
            Label for the vertical (velocity) axis.
        title_txt: str
            Label for the axes title.
        veloc_unit: str
            Unit used for all velocities.

        Results
        -------
        figlist : List[matplotlib.figure.Figure]
            List of figure objects, one per bank.
        axlist : List[matplotlib.axes.Axes]
            List of axes objects, one per bank.
        """
        n_banklines = len(bank_km_mid)
        n_levels = len(veloc)
        figlist: List[Figure] = []
        axlist: List[Axes] = []
        clrs = self.get_colors("Blues", n_levels + 1)
        for ib in range(n_banklines):
            fig, ax = plt.subplots()
            self.setsize(fig)
            bk = bank_km_mid[ib]
            #
            velc = np.sqrt(tauc[ib] * chezy[ib] ** 2 / (water_density * g))
            ax.plot(bank_km_mid[ib], velc, color="k", label=ucrit_txt)
            for iq in range(n_levels):
                ax.plot(
                    bk,
                    veloc[iq][ib],
                    color=clrs[iq + 1],
                    label=velocq_txt.format(iq=iq + 1),
                )
            #
            ax.set_xlabel(chainage_txt)
            ax.set_ylabel(ylabel_txt + " " + veloc_unit)
            ax.grid(True)
            ax.set_title(title_txt.format(ib=ib + 1))
            ax.legend(loc="upper right")
            figlist.append(fig)
            axlist.append(ax)
        return figlist, axlist

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

    def plot7_banktype(
        self,
        bbox: Tuple[float, float, float, float],
        xykm: np.ndarray,
        bank_crds: List[np.ndarray],
        banktype: List[np.ndarray],
        taucls_str: List[str],
        xlabel_txt: str,
        ylabel_txt: str,
        title_txt: str,
    ) -> Tuple[Figure, Axes]:
        """
        Create the bank erosion plot with colour-coded bank types.

        Arguments
        ---------
        bbox : Tuple[float, float, float, float]
            Tuple containing boundary limits (xmin, ymin, xmax, ymax); unit m.
        xykm : np.ndarray
            Array containing the x, y, and chainage; unit m for x and y, km for chainage.
        bank_crds : List[np.ndarray]
            List of N arrays containing the x- and y-coordinates of the oroginal
            bank lines.
        banktype : List[np.ndarray]
            List of N arrays containing the bank type values.
        taucls_str : List[str]
            List of strings representing the distinct bank type classes.
        xlabel_txt : str
            Label for the x-axis.
        ylabel_txt : str
            Label for the y-axis.
        title_txt : str
            Label for the axes title.

        Results
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        ax : matplotlib.axes.Axes
            Axes object.
        """
        fig, ax = plt.subplots()
        self.setsize(fig)
        ax.set_aspect(1)

        scale = 1000
        self.chainage_markers(xykm, ax, ndec=0, scale=scale)
        clrs = self.get_colors("plasma", len(taucls_str) + 1)
        self._plot_bank_type_segments(ax, bank_crds, banktype, taucls_str, clrs, scale)

        self.set_bbox(ax, bbox)
        ax.set_xlabel(xlabel_txt)
        ax.set_ylabel(ylabel_txt)
        ax.grid(True)
        ax.set_title(title_txt)
        ax.legend(loc="upper right")
        return fig, ax

    def plot8_eroded_distance(
        self,
        bank_km_mid: List[np.ndarray],
        chainage_txt: str,
        dn_tot: List[np.ndarray],
        dn_tot_txt: str,
        dn_eq: List[np.ndarray],
        dn_eq_txt: str,
        dn_txt: str,
        dn_unit: str,
    ) -> Tuple[Figure, Axes]:
        """
        Create the bank erosion plot with total and equilibrium eroded distance.

        Arguments
        ---------
        bank_km_mid : List[np.ndarray]
            List of arrays containing the chainage values per bank (segment) [km].
        chainage_txt : str
            Label for the horizontal chainage axes.
        dn_tot : List[np.ndarray]
            List of arrays containing the total bank erosion distance per bank (segment) [m].
        dn_tot_txt : str
            Label for the total bank erosion distance.
        dn_eq : List[np.ndarray]
            List of arrays containing the equilibrium bank erosion distance per bank (segment) [m].
        dn_eq_txt : str
            Label for equilibrium bank erosion distance.
        dn_txt : str
            General label for bank erosion distance.
        dn_unit: str
            Unit used for bank erosion distance.

        Results
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        ax : matplotlib.axes.Axes
            Axes object.
        """
        fig, ax = plt.subplots()
        self.setsize(fig)
        #
        n_banklines = len(dn_tot)
        clrs = self.get_colors("plasma", n_banklines + 1)
        for ib in range(n_banklines):
            bk = bank_km_mid[ib]
            ax.plot(bk, dn_tot[ib], color=clrs[ib], label=dn_tot_txt.format(ib=ib + 1))
            ax.plot(
                bk,
                dn_eq[ib],
                linestyle=":",
                color=clrs[ib],
                label=dn_eq_txt.format(ib=ib + 1),
            )
        #
        ax.set_xlabel(chainage_txt)
        ax.set_ylabel(dn_txt + " " + dn_unit)
        ax.grid(True)
        ax.set_title(dn_txt)
        ax.legend(loc="upper right")
        return fig, ax
