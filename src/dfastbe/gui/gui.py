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
import os
import sys
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
from PySide6.QtGui import QIntValidator
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QTabWidget,
    QSizePolicy,
    QComboBox,
    QLineEdit,
    QLabel,
    QApplication,
    QBoxLayout,
    QPushButton,
    QDialog,
    QMainWindow,
    QFormLayout,
    QGridLayout,
    QWidget,
    QSpacerItem,
    QCheckBox,
    QTreeWidget,
    QTreeWidgetItem,
    QFileDialog,
)

from dfastbe.io.config import ConfigFile
from dfastbe.gui.utils import (
    get_icon,
    gui_text,
    SHIP_TYPES,
    menu_open_manual,
    menu_about_self,
    menu_about_qt,
    validator,
    close_edit,
    ICONS_DIR
)
from dfastbe.gui.configs import (
    get_configuration,
    load_configuration,
    bankStrengthSwitch,
    typeUpdatePar,
    openFileLayout,
    addTabForLevel,
    selectFile,
)
from dfastbe.gui.analysis_runner import run_detection, run_erosion
from dfastbe.gui.state_management import StateStore


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


class BaseTab:

    def __init__(self, tabs: QTabWidget, window: QMainWindow | None = None, app: QApplication | None = None):
        self.tabs = tabs
        self.window = window
        self.app = app

    def add_remove_edit_layout(
        self, main_widget: QWidget, key: str
    ) -> QWidget:
        """
        Create a standard layout with list control and add, edit and remove buttons.

        Arguments
        ---------
        main_widget : QWidget
            Main object on which the add, edit and remove buttons should operate.
        key : str
            Short name of the parameter.

        Returns
        -------
        parent : QWidget
            Parent QtWidget that contains the add, edit and remove buttons.
        """
        parent = QWidget()
        gridly = QGridLayout(parent)
        gridly.setContentsMargins(0, 0, 0, 0)

        StateManagement[key] = main_widget
        gridly.addWidget(main_widget, 0, 0)

        button_bar = QWidget()
        button_bar_layout = QBoxLayout(QBoxLayout.Direction.TopToBottom, button_bar)
        button_bar_layout.setContentsMargins(0, 0, 0, 0)
        gridly.addWidget(button_bar, 0, 1)

        add_button = QPushButton(get_icon(f"{ICONS_DIR}/add.png"), "")
        add_button.clicked.connect(lambda: add_an_item(key))
        StateManagement[key + "Add"] = add_button
        button_bar_layout.addWidget(add_button)

        edit_button = QPushButton(get_icon(f"{ICONS_DIR}/edit.png"), "")
        edit_button.clicked.connect(lambda: self.edit_an_item(key))
        edit_button.setEnabled(False)
        StateManagement[key + "Edit"] = edit_button
        button_bar_layout.addWidget(edit_button)

        delete_button = QPushButton(get_icon(f"{ICONS_DIR}/remove.png"), "")
        delete_button.clicked.connect(lambda: remove_an_item(key))
        delete_button.setEnabled(False)
        StateManagement[key + "Remove"] = delete_button
        button_bar_layout.addWidget(delete_button)

        stretch = QSpacerItem(
            10, 10, QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )
        button_bar_layout.addItem(stretch)

        return parent

    def edit_an_item(self, key: str) -> None:
        """
        Implements the actions for the edit item button.

        Dialog implemented in separate routines.

        Arguments
        ---------
        key : str
            Short name of the parameter.
        """
        selected = StateManagement[key].selectedItems()
        # root = dialog[key].invisibleRootItem()
        if len(selected) > 0:
            istr = selected[0].text(0)
            if key == "searchLines":
                file_name = selected[0].text(1)
                dist = selected[0].text(2)
                file_name, dist = edit_search_line(key, istr, file_name=file_name, dist=dist)
                selected[0].setText(1, file_name)
                selected[0].setText(2, dist)
            elif key == "discharges":
                file_name = selected[0].text(1)
                prob = selected[0].text(2)
                file_name, prob = editADischarge(key, istr, file_name=file_name, prob=prob)
                selected[0].setText(1, file_name)
                selected[0].setText(2, prob)


def edit_search_line(
    key: str, istr: str, file_name: str = "", dist: str = "50"
) -> Tuple[str, str]:
    """
    Create an edit dialog for the search lines list.

    Arguments
    ---------
    key : str
        Short name of the parameter.
    istr : str
        String representation of the search line in the list.
    file_name : str
        Name of the search line file.
    dist : str
        String representation of the search distance.

    Returns
    -------
    fileName1 : str
        Updated name of the search line file.
    dist1 : str
        Updated string representation of the search distance.
    """

    edit_dialog = QDialog()
    set_dialog_size(edit_dialog, 600, 100)
    edit_dialog.setWindowFlags(
        Qt.WindowTitleHint | Qt.WindowSystemMenuHint
    )
    edit_dialog.setWindowTitle("Edit Search Line")
    edit_layout = QFormLayout(edit_dialog)

    label = QLabel(istr)
    edit_layout.addRow("Search Line Nr", label)

    addOpenFileRow(edit_layout, "editSearchLine", "Search Line File")
    StateManagement["editSearchLineEdit"].setText(file_name)

    search_distance = QLineEdit()
    search_distance.setText(dist)
    search_distance.setValidator(validator("positive_real"))
    edit_layout.addRow("Search Distance [m]", search_distance)

    done = QPushButton("Done")
    done.clicked.connect(lambda: close_edit(edit_dialog))
    # edit_SearchDistance.setValidator(validator("positive_real"))
    edit_layout.addRow(" ", done)

    edit_dialog.exec()

    file_name = StateManagement["editSearchLineEdit"].text()
    dist = search_distance.text()
    return file_name, dist


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


class DetectionTab(BaseTab):

    def __init__(self, tabs: QTabWidget, window: QMainWindow, app: QApplication):
        """Initialize the tab for the bank line detection settings.

        Args:
            tabs : QTabWidget
                Tabs object to which the tab should be added.
            window : QMainWindow
                The window object in which the tab item is located.
            app : QApplication
                The application object to which the window belongs, needed for font information.
        """
        super().__init__(tabs, window, app)

    def create(self) -> None:
        """Create the tab for the bank line detection settings."""
        detect_widget = QWidget()
        detect_layout = QFormLayout(detect_widget)
        self.tabs.addTab(detect_widget, "Detection")

        addOpenFileRow(detect_layout, "simFile", "Simulation File")

        water_depth = QLineEdit(self.window)
        water_depth.setValidator(validator("positive_real"))
        StateManagement["waterDepth"] = water_depth
        detect_layout.addRow("Water Depth [m]", water_depth)

        search_lines = QTreeWidget(self.window)
        search_lines.setHeaderLabels(["Index", "FileName", "Search Distance [m]"])
        search_lines.setFont(self.app.font())
        search_lines.setColumnWidth(0, 50)
        search_lines.setColumnWidth(1, 200)

        search_lines_layout = self.add_remove_edit_layout(search_lines, "searchLines")
        detect_layout.addRow("Search Lines", search_lines_layout)


class ErosionTab(BaseTab):
    def __init__(self, tabs: QTabWidget, window: QMainWindow, app: QApplication):
        """Initialize the tab for the bank erosion settings.

        Args:
            tabs : QTabWidget
                Tabs object to which the tab should be added.
            window : QMainWindow
                The window object in which the tab item is located.
            app : QApplication
                The application object to which the window belongs, needed for font information.
        """
        super().__init__(tabs, window, app)

    def create(self) -> None:
        """Create the tab for the main bank erosion settings."""
        erosion_widget = QWidget()
        erosion_layout = QFormLayout(erosion_widget)
        self.tabs.addTab(erosion_widget, "Erosion")

        erosion_time = QLineEdit(self.window)
        erosion_time.setValidator(validator("positive_real"))
        StateManagement["tErosion"] = erosion_time
        erosion_layout.addRow("Simulation Time [yr]", erosion_time)

        addOpenFileRow(erosion_layout, "riverAxis", "River Axis File")

        addOpenFileRow(erosion_layout, "fairway", "Fairway File")

        discharges = QTreeWidget(self.window)
        discharges.setHeaderLabels(["Level", "FileName", "Probability [-]"])
        discharges.setFont(self.app.font())
        discharges.setColumnWidth(0, 50)
        discharges.setColumnWidth(1, 250)
        # c1 = QTreeWidgetItem(discharges, ["0", "test\\filename", "0.5"])

        discharge_layout = self.add_remove_edit_layout(discharges, "discharges")
        erosion_layout.addRow("Discharges", discharge_layout)

        ref_level = QLineEdit(self.window)
        ref_level.setValidator(QIntValidator(1, 1))
        StateManagement["refLevel"] = ref_level
        erosion_layout.addRow("Reference Case", ref_level)

        chainage_out_step = QLineEdit(self.window)
        chainage_out_step.setValidator(validator("positive_real"))
        StateManagement["chainageOutStep"] = chainage_out_step
        erosion_layout.addRow("Chainage Output Step [km]", chainage_out_step)

        addOpenFileRow(erosion_layout, "outDir", "Output Directory")

        new_bank_file = QLineEdit(self.window)
        StateManagement["newBankFile"] = new_bank_file
        erosion_layout.addRow("New Bank File Name", new_bank_file)

        new_eq_bank_file = QLineEdit(self.window)
        StateManagement["newEqBankFile"] = new_eq_bank_file
        erosion_layout.addRow("New Eq Bank File Name", new_eq_bank_file)

        erosion_volume = QLineEdit(self.window)
        StateManagement["eroVol"] = erosion_volume
        erosion_layout.addRow("EroVol File Name", erosion_volume)

        ero_vol_eq = QLineEdit(self.window)
        StateManagement["eroVolEqui"] = ero_vol_eq
        erosion_layout.addRow("EroVolEqui File Name", ero_vol_eq)


class ShippingTab(BaseTab):
    def __init__(self, tabs: QTabWidget):
        """Initialize the tab for the bank erosion settings.

        Args:
            tabs : QTabWidget
                Tabs object to which the tab should be added.
        """
        super().__init__(tabs)

    def create(self) -> None:
        """
        Create the tab for the general shipping settings.
        """
        eParamsWidget = QWidget()
        eParamsLayout = QGridLayout(eParamsWidget)
        self.tabs.addTab(eParamsWidget, "Shipping Parameters")

        generalParLayout(eParamsLayout, 0, "shipType", "Ship Type", selectList=SHIP_TYPES)
        generalParLayout(eParamsLayout, 2, "shipVeloc", "Velocity [m/s]")
        generalParLayout(eParamsLayout, 3, "nShips", "# Ships [1/yr]")
        generalParLayout(eParamsLayout, 4, "shipNWaves", "# Waves [1/ship]")
        generalParLayout(eParamsLayout, 5, "shipDraught", "Draught [m]")
        generalParLayout(eParamsLayout, 6, "wavePar0", "Wave0 [m]")
        generalParLayout(eParamsLayout, 7, "wavePar1", "Wave1 [m]")

        stretch = QSpacerItem(
            10, 10, QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )
        eParamsLayout.addItem(stretch, 8, 0)


class BankTab(BaseTab):
    def __init__(self, tabs: QTabWidget):
        """Initialize the tab for the bank erosion settings.

        Args:
            tabs : QTabWidget
                Tabs object to which the tab should be added.
        """
        super().__init__(tabs)

    def create(self) -> None:
        """Create the tab for the general bank properties."""
        eParamsWidget = QWidget()
        eParamsLayout = QGridLayout(eParamsWidget)
        self.tabs.addTab(eParamsWidget, "Bank Parameters")

        strength = QLabel("Strength Parameter")
        eParamsLayout.addWidget(strength, 0, 0)
        strengthPar = QComboBox()
        strengthPar.addItems(("Bank Type", "Critical Shear Stress"))
        strengthPar.currentIndexChanged.connect(bankStrengthSwitch)
        StateManagement["strengthPar"] = strengthPar
        eParamsLayout.addWidget(strengthPar, 0, 1, 1, 2)

        generalParLayout(
            eParamsLayout,
            1,
            "bankType",
            "Bank Type",
            selectList=[
                "0 (Beschermde oeverlijn)",
                "1 (Begroeide oeverlijn)",
                "2 (Goede klei)",
                "3 (Matig / slechte klei)",
                "4 (Zand)",
            ],
        )
        generalParLayout(eParamsLayout, 3, "bankShear", "Critical Shear Stress [N/m2]")
        bankStrengthSwitch()
        generalParLayout(eParamsLayout, 4, "bankProtect", "Protection [m]")
        generalParLayout(eParamsLayout, 5, "bankSlope", "Slope [-]")
        generalParLayout(eParamsLayout, 6, "bankReed", "Reed [-]")

        addFilter(eParamsLayout, 7, "velFilter", "Velocity Filter [km]")
        addFilter(eParamsLayout, 8, "bedFilter", "Bank Elevation Filter [km]")

        stretch = QSpacerItem(
            10, 10, QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )
        eParamsLayout.addItem(stretch, 9, 0)


def addFilter(
    gridLayout: QGridLayout, row: int, key: str, labelString: str
) -> None:
    """
    Add a line of controls for a filter

    Arguments
    ---------
    gridLayout : QGridLayout
        Grid layout object in which to position the edit controls.
    row : int
        Grid row number to be used for this parameter.
    key : str
        Short name of the parameter.
    labelString : str
        String describing the parameter to be displayed as label.
    """

    widthEdit = QLineEdit("0.3")
    widthEdit.setValidator(validator("positive_real"))
    gridLayout.addWidget(widthEdit, row, 2)
    StateManagement[key + "Width"] = widthEdit

    useFilter = QCheckBox("")
    useFilter.setChecked(False)
    useFilter.stateChanged.connect(lambda: updateFilter(key))
    gridLayout.addWidget(useFilter, row, 1)
    StateManagement[key + "Active"] = useFilter

    filterTxt = QLabel(labelString)
    gridLayout.addWidget(filterTxt, row, 0)
    StateManagement[key + "Txt"] = filterTxt

    updateFilter(key)


def updateFilter(key: str) -> None:
    """
    Implements the dialog setting switching for both general and optional parameters.

    Arguments
    ---------
    key : str
        Short name of the parameter.
    """
    if StateManagement[key + "Active"].isChecked():
        StateManagement[key + "Width"].setEnabled(True)
    else:
        StateManagement[key + "Width"].setEnabled(False)


def generalParLayout(
    gridLayout: QGridLayout,
    row: int,
    key: str,
    labelString: str,
    selectList: Optional[List[str]] = None,
) -> None:
    """
    Add a line of controls for editing a general parameter.

    Arguments
    ---------
    gridLayout : QGridLayout
        Grid layout object in which to position the edit controls.
    row : int
        Grid row number to be used for this parameter.
    key : str
        Short name of the parameter.
    labelString : str
        String describing the parameter to be displayed as label.
    selectList : Optional[List[str]]
        In case the parameter can only have a limited number of values: a list
        of strings describing the options.
    """
    Label = QLabel(labelString)
    StateManagement[key] = Label
    gridLayout.addWidget(Label, row, 0)

    paramTypes = ("Constant", "Variable")
    Type = QComboBox()
    Type.addItems(paramTypes)
    Type.currentIndexChanged.connect(lambda: typeUpdatePar(key))
    StateManagement[key + "Type"] = Type
    gridLayout.addWidget(Type, row, 1)

    if selectList is None:
        fLayout = openFileLayout(key + "Edit", enabled=False)
        gridLayout.addWidget(fLayout, row, 2)
    else:
        Select = QComboBox()
        Select.addItems(selectList)
        StateManagement[key + "Select"] = Select
        gridLayout.addWidget(Select, row, 2)

        fLayout = openFileLayout(key + "Edit", enabled=False)
        StateManagement[key + "Edit"].setEnabled(False)
        gridLayout.addWidget(fLayout, row + 1, 2)

    typeUpdatePar(key)


def addOpenFileRow(
    formLayout: QFormLayout, key: str, labelString: str
) -> None:
    """
    Add a line of controls for selecting a file or folder in a form layout.

    Arguments
    ---------
    formLayout : QFormLayout
        Form layout object in which to position the edit controls.
    key : str
        Short name of the parameter.
    labelString : str
        String describing the parameter to be displayed as label.
    """
    Label = QLabel(labelString)
    StateManagement[key] = Label
    fLayout = openFileLayout(key + "Edit")
    formLayout.addRow(Label, fLayout)


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


def add_an_item(key: str) -> None:
    """Implements the actions for the add item button.

    Args:
        key : str
            Short name of the parameter.
    """
    n_items = StateManagement[key].invisibleRootItem().childCount()
    i = n_items + 1
    istr = str(i)
    if key == "searchLines":
        file_name, dist = edit_search_line(key, istr)
        c1 = QTreeWidgetItem(StateManagement["searchLines"], [istr, file_name, dist])
    elif key == "discharges":
        prob = str(1 / (n_items + 1))
        file_name, prob = editADischarge(key, istr, prob=prob)
        c1 = QTreeWidgetItem(StateManagement["discharges"], [istr, file_name, prob])
        addTabForLevel(istr)
        StateManagement["refLevel"].validator().setTop(i)

    StateManagement[key + "Edit"].setEnabled(True)
    StateManagement[key + "Remove"].setEnabled(True)


def set_dialog_size(edit_dialog: QDialog, width: int, height: int) -> None:
    """Set the width and height of a dialog and position it centered relative to the main window.

    Args:
        edit_dialog : QDialog
            Dialog object to be positioned correctly.
        width : int
            Desired width of the dialog.
        height : int
            Desired height of the dialog.
    """
    parent = StateManagement["window"]
    x = parent.x()
    y = parent.y()
    pw = parent.width()
    ph = parent.height()
    edit_dialog.setGeometry(
        x + pw / 2 - width / 2, y + ph / 2 - height / 2, width, height
    )


def editADischarge(key: str, istr: str, file_name: str = "", prob: str = ""):
    """Create an edit dialog for simulation file and weighing.

    Args:
        key : str
            Short name of the parameter.
        istr : str
            String representation of the simulation in the list.
        file_name : str
            Name of the simulation file.
        prob : str
            String representation of the weight for this simulation.
    """
    edit_dialog = QDialog()
    set_dialog_size(edit_dialog, 600, 100)
    edit_dialog.setWindowFlags(
        Qt.WindowTitleHint | Qt.WindowSystemMenuHint
    )
    edit_dialog.setWindowTitle("Edit Discharge")
    edit_layout = QFormLayout(edit_dialog)

    label = QLabel(istr)
    edit_layout.addRow("Level Nr", label)

    addOpenFileRow(edit_layout, "editDischarge", "Simulation File")
    StateManagement["editDischargeEdit"].setText(file_name)

    probability = QLineEdit()
    probability.setText(prob)
    probability.setValidator(validator("positive_real"))
    edit_layout.addRow("Probability [-]", probability)

    done = QPushButton("Done")
    done.clicked.connect(lambda: close_edit(edit_dialog))
    # edit_SearchDistance.setValidator(validator("positive_real"))
    edit_layout.addRow(" ", done)

    edit_dialog.exec()

    file_name = StateManagement["editDischargeEdit"].text()
    prob = probability.text()
    return file_name, prob


def remove_an_item(key: str) -> None:
    """Implements the actions for the remove item button.

    Args:
        key : str
            Short name of the parameter.
    """
    selected = StateManagement[key].selectedItems()
    root = StateManagement[key].invisibleRootItem()
    if len(selected) > 0:
        istr = selected[0].text(0)
        root.removeChild(selected[0])
        i = int(istr) - 1
        for j in range(i, root.childCount()):
            root.child(j).setText(0, str(j + 1))
    else:
        istr = ""
    if root.childCount() == 0:
        StateManagement[key + "Edit"].setEnabled(False)
        StateManagement[key + "Remove"].setEnabled(False)
    if istr == "":
        pass
    elif key == "searchLines":
        pass
    elif key == "discharges":
        tabs = StateManagement["tabs"]
        StateManagement["refLevel"].validator().setTop(root.childCount())
        dj = 0
        for j in range(tabs.count()):
            if dj > 0:
                tabs.setTabText(j - 1, "Level " + str(j + dj))
                update_tab_keys(j + dj + 1)
            elif tabs.tabText(j) == "Level " + istr:
                tabs.removeTab(j)
                dj = i - j


def update_tab_keys(i: int) -> None:
    """Renumber tab i to tab i-1.

    Args:
        i : str
            Number of the tab to be updated.
    """
    iStart = str(i) + "_"
    newStart = str(i - 1) + "_"
    N = len(iStart)
    keys = [key for key in StateManagement.keys() if key[:N] == iStart]
    for key in keys:
        obj = StateManagement.pop(key)
        if key[-4:] == "Type":
            obj.currentIndexChanged.disconnect()
            obj.currentIndexChanged.connect(
                lambda: typeUpdatePar(newStart + key[N:-4])
            )
        elif key[-4:] == "File":
            obj.clicked.disconnect()
            obj.clicked.connect(lambda: selectFile(newStart + key[N:-4]))
        StateManagement[newStart + key[N:]] = obj


def menu_load_configuration() -> None:
    """Select and load a configuration file."""

    file = QFileDialog.getOpenFileName(
        caption="Select Configuration File", filter="Config Files (*.cfg)"
    )
    filename = file[0]
    if filename != "":
        load_configuration(filename)


def menu_save_configuration() -> None:
    """Ask for a configuration file name and save GUI selection to that file."""

    fil = QFileDialog.getSaveFileName(
        caption="Save Configuration As", filter="Config Files (*.cfg)"
    )
    filename = fil[0]
    if filename != "":
        config = get_configuration()
        rootdir = os.path.dirname(filename)
        config_file = ConfigFile(config)
        config_file.relative_to(rootdir)
        config.write(filename)


class BaseBar:
    def __init__(self, *, window: QMainWindow, app: QApplication, layout: QBoxLayout | None = None):
        self.window = window
        self.layout = layout
        self.app = app

    def create(self):
        ...

    def close(self) -> None:
        """Close the dialog and program."""
        plt.close("all")
        self.window.close()
        self.app.closeAllWindows()
        self.app.quit()


class MenuBar(BaseBar):
    def __init__(self, window: QMainWindow, app: QApplication):
        super().__init__(window=window, app=app)
        self.menubar = self.window.menuBar()

    def create(self):
        menu = self.menubar.addMenu(gui_text("File"))
        item = menu.addAction(gui_text("Load"))
        item.triggered.connect(menu_load_configuration)
        item = menu.addAction(gui_text("Save"))
        item.triggered.connect(menu_save_configuration)
        menu.addSeparator()
        item = menu.addAction(gui_text("Close"))
        item.triggered.connect(self.close)

        menu = self.menubar.addMenu(gui_text("Help"))
        item = menu.addAction(gui_text("Manual"))
        item.triggered.connect(menu_open_manual)
        menu.addSeparator()
        item = menu.addAction(gui_text("Version"))
        item.triggered.connect(menu_about_self)
        item = menu.addAction(gui_text("AboutQt"))
        item.triggered.connect(menu_about_qt)


class ButtonBar(BaseBar):
    def __init__(self, window: QMainWindow, layout: QBoxLayout, app: QApplication | None = None):
        super().__init__(window=window, app=app, layout=layout)

    def create(self):
        button_bar = QWidget(self.window)
        button_bar_layout = QBoxLayout(QBoxLayout.Direction.LeftToRight, button_bar)
        button_bar_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(button_bar)

        detect = QPushButton(gui_text("action_detect"), self.window)
        detect.clicked.connect(lambda: run_detection(self.app))
        button_bar_layout.addWidget(detect)

        erode = QPushButton(gui_text("action_erode"), self.window)
        erode.clicked.connect(lambda: run_erosion(self.app))
        button_bar_layout.addWidget(erode)

        done = QPushButton(gui_text("action_close"), self.window)
        done.clicked.connect(self.close)
        button_bar_layout.addWidget(done)


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
