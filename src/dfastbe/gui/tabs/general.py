from __future__ import annotations
from PySide6.QtWidgets import (
    QTabWidget,
    QWidget,
    QFormLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QCheckBox,
    QMainWindow
)
from dfastbe.gui.base import BaseTab
from dfastbe.gui.state_management import StateStore
from dfastbe.gui.utils import addOpenFileRow, validator

__all__ = ["GeneralTab"]


class GeneralTab(BaseTab):

    def __init__(self, tabs: QTabWidget, window: QMainWindow):
        """Initializer.

        Args:
            tabs : QTabWidget
                Tabs object to which the tab should be added.
            window : QMainWindow
                Windows in which the tab item is located.
        """
        super().__init__(tabs, window)

    def create(self,) -> None:
        """Create the tab for the general settings.

        These settings are used by both the bank line detection and the bank
        erosion analysis.
        """
        state_management = StateStore.instance()
        general_widget = QWidget()
        general_layout = QFormLayout(general_widget)
        self.tabs.addTab(general_widget, "General")

        addOpenFileRow(general_layout, "chainFile", "Chain File")

        chain_range = QWidget()
        gridly = QGridLayout(chain_range)
        gridly.setContentsMargins(0, 0, 0, 0)

        gridly.addWidget(QLabel("From [km]", self.window), 0, 0)
        start_range = QLineEdit(self.window)
        state_management["startRange"] = start_range
        gridly.addWidget(start_range, 0, 1)
        gridly.addWidget(QLabel("To [km]", self.window), 0, 2)
        end_range = QLineEdit(self.window)
        state_management["endRange"] = end_range
        gridly.addWidget(end_range, 0, 3)

        general_layout.addRow("Study Range", chain_range)

        addOpenFileRow(general_layout, "bankDir", "Bank Directory")

        bank_file_name = QLineEdit(self.window)
        state_management["bankFileName"] = bank_file_name
        general_layout.addRow("Bank File Name", bank_file_name)

        add_check_box(general_layout, "makePlots", "Create Figures", True)
        state_management["makePlotsEdit"].stateChanged.connect(update_plotting)

        add_check_box(general_layout, "savePlots", "Save Figures", True)
        state_management["savePlotsEdit"].stateChanged.connect(update_plotting)

        zoom_plots = QWidget()
        gridly = QGridLayout(zoom_plots)
        gridly.setContentsMargins(0, 0, 0, 0)

        save_zoom_plots_edit = QCheckBox("", self.window)
        save_zoom_plots_edit.stateChanged.connect(update_plotting)
        save_zoom_plots_edit.setChecked(False)
        gridly.addWidget(save_zoom_plots_edit, 0, 0)
        state_management["saveZoomPlotsEdit"] = save_zoom_plots_edit

        zoom_plots_range_txt = QLabel("Zoom Range [km]", self.window)
        zoom_plots_range_txt.setEnabled(False)
        gridly.addWidget(zoom_plots_range_txt, 0, 1)
        state_management["zoomPlotsRangeTxt"] = zoom_plots_range_txt

        zoom_plots_range_edit = QLineEdit("1.0", self.window)
        zoom_plots_range_edit.setValidator(validator("positive_real"))
        zoom_plots_range_edit.setEnabled(False)
        gridly.addWidget(zoom_plots_range_edit, 0, 2)
        state_management["zoomPlotsRangeEdit"] = zoom_plots_range_edit

        save_zoom_plots = QLabel("Save Zoomed Figures", self.window)
        general_layout.addRow(save_zoom_plots, zoom_plots)
        state_management["saveZoomPlots"] = save_zoom_plots

        addOpenFileRow(general_layout, "figureDir", "Figure Directory")
        add_check_box(general_layout, "closePlots", "Close Figures")
        add_check_box(general_layout, "debugOutput", "Debug Output")


def add_check_box(
    form_layout: QFormLayout,
    key: str,
    label_string: str,
    is_checked: bool = False,
) -> None:
    """
    Add a line of with checkbox control to a form layout.

    Args:
        form_layout : QFormLayout
            Form layout object in which to position the edit controls.
        key : str
            Short name of the parameter.
        label_string : str
            String describing the parameter to be displayed as a label.
        is_checked : bool
            Initial state of the checkbox.
    """
    state_management = StateStore.instance()
    check_box = QCheckBox("")
    check_box.setChecked(is_checked)
    state_management[key + "Edit"] = check_box

    check_txt = QLabel(label_string)
    state_management[key] = check_txt
    form_layout.addRow(check_txt, check_box)


def update_plotting() -> None:
    """Update the plotting flags."""
    state_management = StateStore.instance()
    plot_flag = state_management["makePlotsEdit"].isChecked()
    state_management["savePlots"].setEnabled(plot_flag)
    state_management["savePlotsEdit"].setEnabled(plot_flag)

    save_flag = state_management["savePlotsEdit"].isChecked() and plot_flag
    state_management["saveZoomPlots"].setEnabled(save_flag)
    state_management["saveZoomPlotsEdit"].setEnabled(save_flag)

    save_zoom_flag = state_management["saveZoomPlotsEdit"].isChecked() and save_flag
    state_management["zoomPlotsRangeTxt"].setEnabled(save_zoom_flag)
    state_management["zoomPlotsRangeEdit"].setEnabled(save_zoom_flag)

    state_management["figureDir"].setEnabled(save_flag)
    state_management["figureDirEdit"].setEnabled(save_flag)
    state_management["figureDirEditFile"].setEnabled(save_flag)

    state_management["closePlots"].setEnabled(plot_flag)
    state_management["closePlotsEdit"].setEnabled(plot_flag)