"""
Copyright (C) 2025 Stichting Deltares.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation version 2.1.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, see <http://www.gnu.org/licenses/>.

contact: delft3d.support@deltares.nl
Stichting Deltares
P.O. Box 177
2600 MH Delft, The Netherlands

All indications and logos of, and references to, "Delft3D" and "Deltares"
are registered trademarks of Stichting Deltares, and remain the property of
Stichting Deltares. All rights reserved.

INFORMATION
This file is part of D-FAST Bank Erosion: https://github.com/Deltares/D-FAST_Bank_Erosion
"""
from __future__ import annotations
import sys
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QTabWidget,
    QLineEdit,
    QLabel,
    QApplication,
    QBoxLayout,
    QMainWindow,
    QFormLayout,
    QGridLayout,
    QWidget,
    QCheckBox,
)

from dfastbe.gui.utils import (
    get_icon,
    validator,
    ICONS_DIR,
    addOpenFileRow,
)
from dfastbe.gui.configs import (
    load_configuration,
)

from dfastbe.gui.tabs.detection import DetectionTab
from dfastbe.gui.tabs.erosion import ErosionTab
from dfastbe.gui.tabs.shipping import ShippingTab
from dfastbe.gui.tabs.bank import BankTab
from dfastbe.gui.tabs.main_components import ButtonBar, MenuBar
from dfastbe.gui.base import BaseTab
from dfastbe.gui.state_management import StateStore

__all__ = ["GUI", "main"]

class _StateProxy(MutableMapping[str, Any]):
    """Lazy proxy that forwards mapping operations to the StateStore singleton.

    This keeps existing `StateManagement[...]` call sites working without
    reassigning a module-level global in `GUI.__init__`. Every access resolves
    the current singleton via `StateStore.instance()`, so all reads and writes
    target the same shared dictionary created by the GUI constructor.
    """

    def _state(self) -> StateStore:
        return StateStore.instance()

    def __getitem__(self, key: str) -> Any:
        return self._state()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._state()[key] = value

    def __delitem__(self, key: str) -> None:
        del self._state()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._state())

    def __len__(self) -> int:
        return len(self._state())

    def __repr__(self) -> str:
        return repr(self._state())


StateManagement: MutableMapping[str, Any] = _StateProxy()


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
        general_widget = QWidget()
        general_layout = QFormLayout(general_widget)
        self.tabs.addTab(general_widget, "General")

        addOpenFileRow(general_layout, "chainFile", "Chain File")

        chain_range = QWidget()
        gridly = QGridLayout(chain_range)
        gridly.setContentsMargins(0, 0, 0, 0)

        gridly.addWidget(QLabel("From [km]", self.window), 0, 0)
        start_range = QLineEdit(self.window)
        StateManagement["startRange"] = start_range
        gridly.addWidget(start_range, 0, 1)
        gridly.addWidget(QLabel("To [km]", self.window), 0, 2)
        end_range = QLineEdit(self.window)
        StateManagement["endRange"] = end_range
        gridly.addWidget(end_range, 0, 3)

        general_layout.addRow("Study Range", chain_range)

        addOpenFileRow(general_layout, "bankDir", "Bank Directory")

        bank_file_name = QLineEdit(self.window)
        StateManagement["bankFileName"] = bank_file_name
        general_layout.addRow("Bank File Name", bank_file_name)

        add_check_box(general_layout, "makePlots", "Create Figures", True)
        StateManagement["makePlotsEdit"].stateChanged.connect(updatePlotting)

        add_check_box(general_layout, "savePlots", "Save Figures", True)
        StateManagement["savePlotsEdit"].stateChanged.connect(updatePlotting)

        zoom_plots = QWidget()
        gridly = QGridLayout(zoom_plots)
        gridly.setContentsMargins(0, 0, 0, 0)

        save_zoom_plots_edit = QCheckBox("", self.window)
        save_zoom_plots_edit.stateChanged.connect(updatePlotting)
        save_zoom_plots_edit.setChecked(False)
        gridly.addWidget(save_zoom_plots_edit, 0, 0)
        StateManagement["saveZoomPlotsEdit"] = save_zoom_plots_edit

        zoom_plots_range_txt = QLabel("Zoom Range [km]", self.window)
        zoom_plots_range_txt.setEnabled(False)
        gridly.addWidget(zoom_plots_range_txt, 0, 1)
        StateManagement["zoomPlotsRangeTxt"] = zoom_plots_range_txt

        zoom_plots_range_edit = QLineEdit("1.0", self.window)
        zoom_plots_range_edit.setValidator(validator("positive_real"))
        zoom_plots_range_edit.setEnabled(False)
        gridly.addWidget(zoom_plots_range_edit, 0, 2)
        StateManagement["zoomPlotsRangeEdit"] = zoom_plots_range_edit

        save_zoom_plots = QLabel("Save Zoomed Figures", self.window)
        general_layout.addRow(save_zoom_plots, zoom_plots)
        StateManagement["saveZoomPlots"] = save_zoom_plots

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
            String describing the parameter to be displayed as label.
        is_checked : bool
            Initial state of the check box.
    """
    check_box = QCheckBox("")
    check_box.setChecked(is_checked)
    StateManagement[key + "Edit"] = check_box

    check_txt = QLabel(label_string)
    StateManagement[key] = check_txt
    form_layout.addRow(check_txt, check_box)


def updatePlotting() -> None:
    """Update the plotting flags."""

    plotFlag = StateManagement["makePlotsEdit"].isChecked()
    StateManagement["savePlots"].setEnabled(plotFlag)
    StateManagement["savePlotsEdit"].setEnabled(plotFlag)

    saveFlag = StateManagement["savePlotsEdit"].isChecked() and plotFlag
    StateManagement["saveZoomPlots"].setEnabled(saveFlag)
    StateManagement["saveZoomPlotsEdit"].setEnabled(saveFlag)

    saveZoomFlag = StateManagement["saveZoomPlotsEdit"].isChecked() and saveFlag
    StateManagement["zoomPlotsRangeTxt"].setEnabled(saveZoomFlag)
    StateManagement["zoomPlotsRangeEdit"].setEnabled(saveZoomFlag)

    StateManagement["figureDir"].setEnabled(saveFlag)
    StateManagement["figureDirEdit"].setEnabled(saveFlag)
    StateManagement["figureDirEditFile"].setEnabled(saveFlag)

    StateManagement["closePlots"].setEnabled(plotFlag)
    StateManagement["closePlotsEdit"].setEnabled(plotFlag)


class GUI:

    def __init__(self):
        self.state = StateStore.initialize()

        self.app = QApplication()
        self.app.setStyle("fusion")
        StateManagement["application"] = self.app
        self.window, self.layout = self.create_window()
        StateManagement["window"] = self.window

        self.tabs = QTabWidget(self.window)
        StateManagement["tabs"] = self.tabs
        self.layout.addWidget(self.tabs)

        self.menu_Bar = self.create_menu_bar()
        self.button_bar = self.create_action_buttons()

    @staticmethod
    def create_window():
        win = QMainWindow()
        win.setWindowTitle("D-FAST Bank Erosion")
        win.setGeometry(200, 200, 600, 300)
        win.setWindowIcon(get_icon(f"{ICONS_DIR}/D-FASTBE.png"))

        # win.resize(1000, 800)

        central_widget = QWidget()
        layout = QBoxLayout(QBoxLayout.Direction.TopToBottom, central_widget)
        win.setCentralWidget(central_widget)
        return win, layout

    def create(self) -> None:
        """Construct the D-FAST Bank Erosion user interface."""
        self.general_tab = GeneralTab(self.tabs, self.window)
        self.general_tab.create()

        self.detection_tab = DetectionTab(self.tabs, self.window, self.app)
        self.detection_tab.create()

        self.erosion_tab = ErosionTab(self.tabs, self.window, self.app)
        self.erosion_tab.create()

        self.shipping_tab = ShippingTab(self.tabs)
        self.shipping_tab.create()

        self.bank_tab = BankTab(self.tabs)
        self.bank_tab.create()

    def create_menu_bar(self) -> MenuBar:
        """Add the menus to the menubar."""
        menu = MenuBar(window=self.window, app=self.app)
        menu.create()
        return menu

    def create_action_buttons(self) -> ButtonBar:
        button_bar = ButtonBar(window=self.window, app=self.app, layout=self.layout)
        button_bar.create()
        return button_bar

    def activate(self) -> None:
        """Activate the user interface and run the program."""
        self.window.show()
        sys.exit(self.app.exec())

    def close(self) -> None:
        """Close the dialog and program."""
        plt.close("all")
        self.window.close()
        self.app.closeAllWindows()
        self.app.quit()


def main(config: Optional[Path] = None) -> None:
    """
    Start the user interface using default settings or optional configuration.

    Args:
        config : Optional[str]
            Optional name of configuration file.
    """
    gui = GUI()
    gui.create()
    if config is not None:
        load_configuration(config)

    gui.activate()
