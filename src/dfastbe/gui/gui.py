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
import configparser
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import matplotlib.pyplot as plt
from PySide6.QtGui import QValidator, QIntValidator, QDoubleValidator
from PySide6 import QtCore
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
    QMessageBox
)

from dfastbe import __file__, __version__
from dfastbe.io.config import ConfigFile
from dfastbe.io.file_utils import absolute_path
from dfastbe.gui.utils import get_icon, gui_text, SHIP_TYPES, show_error
from dfastbe.gui.analysis_runner import run_detection, run_erosion


USER_MANUAL_FILE_NAME = "dfastbe_usermanual.pdf"
DialogObject = Dict[str, QtCore.QObject]

dialog: DialogObject

r_dir = Path(__file__).resolve().parent
ICONS_DIR = r_dir / "gui/icons"


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

        dialog[key] = main_widget
        gridly.addWidget(main_widget, 0, 0)

        button_bar = QWidget()
        button_bar_layout = QBoxLayout(QBoxLayout.Direction.TopToBottom, button_bar)
        button_bar_layout.setContentsMargins(0, 0, 0, 0)
        gridly.addWidget(button_bar, 0, 1)

        add_button = QPushButton(get_icon(f"{ICONS_DIR}/add.png"), "")
        add_button.clicked.connect(lambda: addAnItem(key))
        dialog[key + "Add"] = add_button
        button_bar_layout.addWidget(add_button)

        edit_button = QPushButton(get_icon(f"{ICONS_DIR}/edit.png"), "")
        edit_button.clicked.connect(lambda: self.edit_an_item(key))
        edit_button.setEnabled(False)
        dialog[key + "Edit"] = edit_button
        button_bar_layout.addWidget(edit_button)

        delete_button = QPushButton(get_icon(f"{ICONS_DIR}/remove.png"), "")
        delete_button.clicked.connect(lambda: removeAnItem(key))
        delete_button.setEnabled(False)
        dialog[key + "Remove"] = delete_button
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
        selected = dialog[key].selectedItems()
        # root = dialog[key].invisibleRootItem()
        if len(selected) > 0:
            istr = selected[0].text(0)
            if key == "searchLines":
                fileName = selected[0].text(1)
                dist = selected[0].text(2)
                fileName, dist = editASearchLine(key, istr, fileName=fileName, dist=dist)
                selected[0].setText(1, fileName)
                selected[0].setText(2, dist)
            elif key == "discharges":
                fileName = selected[0].text(1)
                prob = selected[0].text(2)
                fileName, prob = editADischarge(key, istr, fileName=fileName, prob=prob)
                selected[0].setText(1, fileName)
                selected[0].setText(2, prob)


def editASearchLine(
    key: str, istr: str, fileName: str = "", dist: str = "50"
) -> Tuple[str, str]:
    """
    Create an edit dialog for the search lines list.

    Arguments
    ---------
    key : str
        Short name of the parameter.
    istr : str
        String representation of the search line in the list.
    fileName : str
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

    editDialog = QDialog()
    setDialogSize(editDialog, 600, 100)
    editDialog.setWindowFlags(
        Qt.WindowTitleHint | Qt.WindowSystemMenuHint
    )
    editDialog.setWindowTitle("Edit Search Line")
    editLayout = QFormLayout(editDialog)

    label = QLabel(istr)
    editLayout.addRow("Search Line Nr", label)

    addOpenFileRow(editLayout, "editSearchLine", "Search Line File")
    dialog["editSearchLineEdit"].setText(fileName)

    searchDistance = QLineEdit()
    searchDistance.setText(dist)
    searchDistance.setValidator(validator("positive_real"))
    editLayout.addRow("Search Distance [m]", searchDistance)

    done = QPushButton("Done")
    done.clicked.connect(lambda: close_edit(editDialog))
    # edit_SearchDistance.setValidator(validator("positive_real"))
    editLayout.addRow(" ", done)

    editDialog.exec()

    fileName = dialog["editSearchLineEdit"].text()
    dist = searchDistance.text()
    return fileName, dist


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
        dialog["startRange"] = start_range
        gridly.addWidget(start_range, 0, 1)
        gridly.addWidget(QLabel("To [km]", self.window), 0, 2)
        end_range = QLineEdit(self.window)
        dialog["endRange"] = end_range
        gridly.addWidget(end_range, 0, 3)

        general_layout.addRow("Study Range", chain_range)

        addOpenFileRow(general_layout, "bankDir", "Bank Directory")

        bank_file_name = QLineEdit(self.window)
        dialog["bankFileName"] = bank_file_name
        general_layout.addRow("Bank File Name", bank_file_name)

        addCheckBox(general_layout, "makePlots", "Create Figures", True)
        dialog["makePlotsEdit"].stateChanged.connect(updatePlotting)

        addCheckBox(general_layout, "savePlots", "Save Figures", True)
        dialog["savePlotsEdit"].stateChanged.connect(updatePlotting)

        zoom_plots = QWidget()
        gridly = QGridLayout(zoom_plots)
        gridly.setContentsMargins(0, 0, 0, 0)

        save_zoom_plots_edit = QCheckBox("", self.window)
        save_zoom_plots_edit.stateChanged.connect(updatePlotting)
        save_zoom_plots_edit.setChecked(False)
        gridly.addWidget(save_zoom_plots_edit, 0, 0)
        dialog["saveZoomPlotsEdit"] = save_zoom_plots_edit

        zoom_plots_range_txt = QLabel("Zoom Range [km]", self.window)
        zoom_plots_range_txt.setEnabled(False)
        gridly.addWidget(zoom_plots_range_txt, 0, 1)
        dialog["zoomPlotsRangeTxt"] = zoom_plots_range_txt

        zoom_plots_range_edit = QLineEdit("1.0",self.window)
        zoom_plots_range_edit.setValidator(validator("positive_real"))
        zoom_plots_range_edit.setEnabled(False)
        gridly.addWidget(zoom_plots_range_edit, 0, 2)
        dialog["zoomPlotsRangeEdit"] = zoom_plots_range_edit

        save_zoom_plots = QLabel("Save Zoomed Figures", self.window)
        general_layout.addRow(save_zoom_plots, zoom_plots)
        dialog["saveZoomPlots"] = save_zoom_plots

        addOpenFileRow(general_layout, "figureDir", "Figure Directory")
        addCheckBox(general_layout, "closePlots", "Close Figures")
        addCheckBox(general_layout, "debugOutput", "Debug Output")


def addCheckBox(
    formLayout: QFormLayout,
    key: str,
    labelString: str,
    isChecked: bool = False,
) -> None:
    """
    Add a line of with checkbox control to a form layout.

    Args:
        formLayout : QFormLayout
            Form layout object in which to position the edit controls.
        key : str
            Short name of the parameter.
        labelString : str
            String describing the parameter to be displayed as label.
        isChecked : bool
            Initial state of the check box.
    """
    checkBox = QCheckBox("")
    checkBox.setChecked(isChecked)
    dialog[key + "Edit"] = checkBox

    checkTxt = QLabel(labelString)
    dialog[key] = checkTxt
    formLayout.addRow(checkTxt, checkBox)


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
        dialog["waterDepth"] = water_depth
        detect_layout.addRow("Water Depth [m]", water_depth)

        search_lines = QTreeWidget(self.window)
        search_lines.setHeaderLabels(["Index", "FileName", "Search Distance [m]"])
        search_lines.setFont(self.app.font())
        search_lines.setColumnWidth(0, 50)
        search_lines.setColumnWidth(1, 200)

        slLayout = self.add_remove_edit_layout(search_lines, "searchLines")
        detect_layout.addRow("Search Lines", slLayout)


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
        erosionWidget = QWidget()
        erosionLayout = QFormLayout(erosionWidget)
        self.tabs.addTab(erosionWidget, "Erosion")

        tErosion = QLineEdit(self.window)
        tErosion.setValidator(validator("positive_real"))
        dialog["tErosion"] = tErosion
        erosionLayout.addRow("Simulation Time [yr]", tErosion)

        addOpenFileRow(erosionLayout, "riverAxis", "River Axis File")

        addOpenFileRow(erosionLayout, "fairway", "Fairway File")

        discharges = QTreeWidget(self.window)
        discharges.setHeaderLabels(["Level", "FileName", "Probability [-]"])
        discharges.setFont(self.app.font())
        discharges.setColumnWidth(0, 50)
        discharges.setColumnWidth(1, 250)
        # c1 = QTreeWidgetItem(discharges, ["0", "test\\filename", "0.5"])

        disLayout = self.add_remove_edit_layout(discharges, "discharges")
        erosionLayout.addRow("Discharges", disLayout)

        refLevel = QLineEdit(self.window)
        refLevel.setValidator(QIntValidator(1, 1))
        dialog["refLevel"] = refLevel
        erosionLayout.addRow("Reference Case", refLevel)

        chainageOutStep = QLineEdit(self.window)
        chainageOutStep.setValidator(validator("positive_real"))
        dialog["chainageOutStep"] = chainageOutStep
        erosionLayout.addRow("Chainage Output Step [km]", chainageOutStep)

        addOpenFileRow(erosionLayout, "outDir", "Output Directory")

        newBankFile = QLineEdit(self.window)
        dialog["newBankFile"] = newBankFile
        erosionLayout.addRow("New Bank File Name", newBankFile)

        newEqBankFile = QLineEdit(self.window)
        dialog["newEqBankFile"] = newEqBankFile
        erosionLayout.addRow("New Eq Bank File Name", newEqBankFile)

        eroVol = QLineEdit(self.window)
        dialog["eroVol"] = eroVol
        erosionLayout.addRow("EroVol File Name", eroVol)

        eroVolEqui = QLineEdit(self.window)
        dialog["eroVolEqui"] = eroVolEqui
        erosionLayout.addRow("EroVolEqui File Name", eroVolEqui)


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
        dialog["strengthPar"] = strengthPar
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
    dialog[key + "Width"] = widthEdit

    useFilter = QCheckBox("")
    useFilter.setChecked(False)
    useFilter.stateChanged.connect(lambda: updateFilter(key))
    gridLayout.addWidget(useFilter, row, 1)
    dialog[key + "Active"] = useFilter

    filterTxt = QLabel(labelString)
    gridLayout.addWidget(filterTxt, row, 0)
    dialog[key + "Txt"] = filterTxt

    updateFilter(key)


def updateFilter(key: str) -> None:
    """
    Implements the dialog setting switching for both general and optional parameters.

    Arguments
    ---------
    key : str
        Short name of the parameter.
    """
    if dialog[key + "Active"].isChecked():
        dialog[key + "Width"].setEnabled(True)
    else:
        dialog[key + "Width"].setEnabled(False)


def bankStrengthSwitch() -> None:
    """Implements the dialog settings depending on the bank strength specification method."""
    type = dialog["strengthPar"].currentText()
    if type == "Bank Type":
        dialog["bankType"].setEnabled(True)
        dialog["bankTypeType"].setEnabled(True)
        typeUpdatePar("bankType")
        dialog["bankShear"].setEnabled(False)
        dialog["bankShearType"].setEnabled(False)
        dialog["bankShearEdit"].setText("")
        dialog["bankShearEdit"].setEnabled(False)
        dialog["bankShearEditFile"].setEnabled(False)
    elif type == "Critical Shear Stress":
        dialog["bankShear"].setEnabled(True)
        dialog["bankShearType"].setEnabled(True)
        dialog["bankShearEdit"].setEnabled(True)
        typeUpdatePar("bankShear")
        dialog["bankType"].setEnabled(False)
        dialog["bankTypeType"].setEnabled(False)
        dialog["bankTypeSelect"].setEnabled(False)
        dialog["bankTypeEdit"].setText("")
        dialog["bankTypeEdit"].setEnabled(False)
        dialog["bankTypeEditFile"].setEnabled(False)


def validator(validstr: str) -> QValidator:
    """
    Wrapper to easily create a validator.

    Arguments
    ---------
    validstr : str
        Identifier for the requested validation method.

    Returns
    -------
    validator : QValidator
        Validator for the requested validation method.
    """
    if validstr == "positive_real":
        validator = QDoubleValidator()
        validator.setBottom(0)
    else:
        raise ValueError(f"Unknown validator type: {validstr}")
    return validator


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
    dialog[key] = Label
    gridLayout.addWidget(Label, row, 0)

    paramTypes = ("Constant", "Variable")
    Type = QComboBox()
    Type.addItems(paramTypes)
    Type.currentIndexChanged.connect(lambda: typeUpdatePar(key))
    dialog[key + "Type"] = Type
    gridLayout.addWidget(Type, row, 1)

    if selectList is None:
        fLayout = openFileLayout(key + "Edit", enabled=False)
        gridLayout.addWidget(fLayout, row, 2)
    else:
        Select = QComboBox()
        Select.addItems(selectList)
        dialog[key + "Select"] = Select
        gridLayout.addWidget(Select, row, 2)

        fLayout = openFileLayout(key + "Edit", enabled=False)
        dialog[key + "Edit"].setEnabled(False)
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
    dialog[key] = Label
    fLayout = openFileLayout(key + "Edit")
    formLayout.addRow(Label, fLayout)


def openFileLayout(key, enabled=True) -> QWidget:
    """
    Create a standard layout with a file or folder edit field and selection button.

    Arguments
    ---------
    key : str
        Short name of the parameter.
    enabled : bool
        Flag indicating whether the file selection button should be enabed by default.

    Returns
    ------
    parent : QWidget
        Parent QtWidget that contains the edit field and selection button.
    """
    parent = QWidget()
    gridly = QGridLayout(parent)
    gridly.setContentsMargins(0, 0, 0, 0)

    myWidget = QLineEdit()
    dialog[key] = myWidget
    gridly.addWidget(myWidget, 0, 0)

    openFile = QPushButton(get_icon(f"{ICONS_DIR}/open.png"), "")
    openFile.clicked.connect(lambda: selectFile(key))
    openFile.setEnabled(enabled)
    dialog[key + "File"] = openFile
    gridly.addWidget(openFile, 0, 2)

    return parent


def updatePlotting() -> None:
    """Update the plotting flags."""

    plotFlag = dialog["makePlotsEdit"].isChecked()
    dialog["savePlots"].setEnabled(plotFlag)
    dialog["savePlotsEdit"].setEnabled(plotFlag)

    saveFlag = dialog["savePlotsEdit"].isChecked() and plotFlag
    dialog["saveZoomPlots"].setEnabled(saveFlag)
    dialog["saveZoomPlotsEdit"].setEnabled(saveFlag)

    saveZoomFlag = dialog["saveZoomPlotsEdit"].isChecked() and saveFlag
    dialog["zoomPlotsRangeTxt"].setEnabled(saveZoomFlag)
    dialog["zoomPlotsRangeEdit"].setEnabled(saveZoomFlag)

    dialog["figureDir"].setEnabled(saveFlag)
    dialog["figureDirEdit"].setEnabled(saveFlag)
    dialog["figureDirEditFile"].setEnabled(saveFlag)

    dialog["closePlots"].setEnabled(plotFlag)
    dialog["closePlotsEdit"].setEnabled(plotFlag)


def addAnItem(key: str) -> None:
    """
    Implements the actions for the add item button.

    Arguments
    ---------
    key : str
        Short name of the parameter.
    """
    nItems = dialog[key].invisibleRootItem().childCount()
    i = nItems + 1
    istr = str(i)
    if key == "searchLines":
        fileName, dist = editASearchLine(key, istr)
        c1 = QTreeWidgetItem(dialog["searchLines"], [istr, fileName, dist])
    elif key == "discharges":
        prob = str(1 / (nItems + 1))
        fileName, prob = editADischarge(key, istr, prob=prob)
        c1 = QTreeWidgetItem(dialog["discharges"], [istr, fileName, prob])
        addTabForLevel(istr)
        dialog["refLevel"].validator().setTop(i)
    dialog[key + "Edit"].setEnabled(True)
    dialog[key + "Remove"].setEnabled(True)


def setDialogSize(editDialog: QDialog, width: int, height: int) -> None:
    """
    Set the width and height of a dialog and position it centered relative to the main window.
    
    Arguments
    ---------
    editDialog : QDialog
        Dialog object to be positioned correctly.
    width : int
        Desired width of the dialog.
    height : int
        Desired height of the dialog.
    """
    parent = dialog["window"]
    x = parent.x()
    y = parent.y()
    pw = parent.width()
    ph = parent.height()
    editDialog.setGeometry(
        x + pw / 2 - width / 2, y + ph / 2 - height / 2, width, height
    )


def editADischarge(key: str, istr: str, fileName: str = "", prob: str = ""):
    """
    Create an edit dialog for simulation file and weighing.

    Arguments
    ---------
    key : str
        Short name of the parameter.
    istr : str
        String representation of the simulation in the list.
    fileName : str
        Name of the simulation file.
    prob : str
        String representation of the weight for this simulation.
    """
    editDialog = QDialog()
    setDialogSize(editDialog, 600, 100)
    editDialog.setWindowFlags(
        Qt.WindowTitleHint | Qt.WindowSystemMenuHint
    )
    editDialog.setWindowTitle("Edit Discharge")
    editLayout = QFormLayout(editDialog)

    label = QLabel(istr)
    editLayout.addRow("Level Nr", label)

    addOpenFileRow(editLayout, "editDischarge", "Simulation File")
    dialog["editDischargeEdit"].setText(fileName)

    probability = QLineEdit()
    probability.setText(prob)
    probability.setValidator(validator("positive_real"))
    editLayout.addRow("Probability [-]", probability)

    done = QPushButton("Done")
    done.clicked.connect(lambda: close_edit(editDialog))
    # edit_SearchDistance.setValidator(validator("positive_real"))
    editLayout.addRow(" ", done)

    editDialog.exec()

    fileName = dialog["editDischargeEdit"].text()
    prob = probability.text()
    return fileName, prob


def close_edit(hDialog: QDialog) -> None:
    """
    Generic close function for edit dialogs.

    Arguments
    ---------
    hDialog : QDialog
        Dialog object to be closed.
    """
    hDialog.close()


def removeAnItem(key: str) -> None:
    """
    Implements the actions for the remove item button.

    Arguments
    ---------
    key : str
        Short name of the parameter.
    """
    selected = dialog[key].selectedItems()
    root = dialog[key].invisibleRootItem()
    if len(selected) > 0:
        istr = selected[0].text(0)
        root.removeChild(selected[0])
        i = int(istr) - 1
        for j in range(i, root.childCount()):
            root.child(j).setText(0, str(j + 1))
    else:
        istr = ""
    if root.childCount() == 0:
        dialog[key + "Edit"].setEnabled(False)
        dialog[key + "Remove"].setEnabled(False)
    if istr == "":
        pass
    elif key == "searchLines":
        pass
    elif key == "discharges":
        tabs = dialog["tabs"]
        dialog["refLevel"].validator().setTop(root.childCount())
        dj = 0
        for j in range(tabs.count()):
            if dj > 0:
                tabs.setTabText(j - 1, "Level " + str(j + dj))
                updateTabKeys(j + dj + 1)
            elif tabs.tabText(j) == "Level " + istr:
                tabs.removeTab(j)
                dj = i - j


def updateTabKeys(i: int) -> None:
    """
    Renumber tab i to tab i-1.

    Arguments
    ---------
    i : str
        Number of the tab to be updated.
    """
    iStart = str(i) + "_"
    newStart = str(i - 1) + "_"
    N = len(iStart)
    keys = [key for key in dialog.keys() if key[:N] == iStart]
    for key in keys:
        obj = dialog.pop(key)
        if key[-4:] == "Type":
            obj.currentIndexChanged.disconnect()
            obj.currentIndexChanged.connect(
                lambda: typeUpdatePar(newStart + key[N:-4])
            )
        elif key[-4:] == "File":
            obj.clicked.disconnect()
            obj.clicked.connect(lambda: selectFile(newStart + key[N:-4]))
        dialog[newStart + key[N:]] = obj


def selectFile(key: str) -> None:
    """
    Select a file or directory via a selection dialog.

    Arguments
    ---------
    key : str
        Short name of the parameter.
    """
    dnm: str
    if not dialog[key + "File"].hasFocus():
        # in the add/edit dialogs, the selectFile is triggered when the user presses enter in one of the lineEdit boxes ...
        # don't trigger the actual selectFile
        fil = ""
    elif key == "simFileEdit":
        fil, fltr = QFileDialog.getOpenFileName(
            caption="Select D-Flow FM Map File", filter="D-Flow FM Map Files (*map.nc)"
        )
        # getOpenFileName returns a tuple van file name and active file filter.
    elif key == "chainFileEdit":
        fil, fltr = QFileDialog.getOpenFileName(
            caption="Select Chainage File", filter="Chainage Files (*.xyc)"
        )
    elif key == "riverAxisEdit":
        fil, fltr = QFileDialog.getOpenFileName(
            caption="Select River Axis File", filter="River Axis Files (*.xyc)"
        )
    elif key == "fairwayEdit":
        fil, fltr = QFileDialog.getOpenFileName(
            caption="Select Fairway File", filter="Fairway Files (*.xyc)"
        )
    elif key == "editSearchLineEdit":
        fil, fltr = QFileDialog.getOpenFileName(
            caption="Select Search Line File", filter="Search Line Files (*.xyc)"
        )
    elif key == "editDischargeEdit":
        fil, fltr = QFileDialog.getOpenFileName(
            caption="Select Simulation File", filter="Simulation File (*map.nc)"
        )
    elif key == "bankDirEdit":
        fil = QFileDialog.getExistingDirectory(
            caption="Select Bank Directory"
        )
    elif key == "figureDirEdit":
        fil = QFileDialog.getExistingDirectory(
            caption="Select Figure Output Directory"
        )
    elif key == "outDirEdit":
        fil = QFileDialog.getExistingDirectory(
            caption="Select Output Directory"
        )
    else:
        if key[-4:] == "Edit":
            rkey = key[:-4]
            nr = ""
            while rkey[0] in "1234567890":
                nr = nr + rkey[0]
                rkey = rkey[1:]
            if rkey[0] == "_":
                rkey = rkey[1:]
            if not nr == "":
                nr = " for Level " + nr
            if rkey == "bankType":
                ftype = "Bank Type"
                ext = ".btp"
                oneFile = False
            elif rkey == "bankShear":
                ftype = "Critical Shear"
                ext = ".btp"
                oneFile = False
            elif rkey == "bankProtect":
                ftype = "Protection Level"
                ext = ".bpl"
                oneFile = False
            elif rkey == "bankSlope":
                ftype = "Bank Slope"
                ext = ".slp"
                oneFile = False
            elif rkey == "bankReed":
                ftype = "Reed Fraction"
                ext = ".rdd"
                oneFile = False
            elif rkey == "shipType":
                ftype = "Ship Type"
                ext = ""
                oneFile = True
            elif rkey == "shipVeloc":
                ftype = "Ship Velocity"
                ext = ""
                oneFile = True
            elif rkey == "nShips":
                ftype = "Number of Ships"
                ext = ""
                oneFile = True
            elif rkey == "shipNWaves":
                ftype = "Number of Ship Waves"
                ext = ""
                oneFile = True
            elif rkey == "shipDraught":
                ftype = "Ship Draught"
                ext = ""
                oneFile = True
            elif rkey == "wavePar0":
                ftype = "Wave0"
                ext = ""
                oneFile = True
            elif rkey == "wavePar1":
                ftype = "Wave1"
                ext = ""
                oneFile = True
            else:
                ftype = "Parameter"
                ext = "*"
            ftype = ftype + " File"
            fil, fltr = QFileDialog.getOpenFileName(
                caption="Select " + ftype + nr, filter=ftype + " (*" + ext + ")"
            )
            if fil != "":
                fil, fext = os.path.splitext(fil)
                if fext == ext:
                    if not oneFile:
                        # file should end on _<nr>
                        nr = ""
                        while fil[-1] in "1234567890":
                             nr = rkey[-1] + nr
                             fil = fil[:-1]
                        if nr == "" or fil[-1] != "_":
                            print("Missing bank number(s) at end of file name. Reference not updated.")
                            fil = ""
                        else:
                            fil = fil[:-1]
                else:
                    if ext == "":
                        print("Unsupported file extension {} while expecting no extension. Reference not updated.".format(fext))
                    else:
                        print("Unsupported file extension {} while expecting {}. Reference not updated.".format(fext,ext))
                    fil = ""
        else:
            print(key)
            fil = ""
    if fil != "":
        dialog[key].setText(fil)


def menu_load_configuration() -> None:
    """
    Select and load a configuration file.

    Arguments
    ---------
    None
    """
    fil = QFileDialog.getOpenFileName(
        caption="Select Configuration File", filter="Config Files (*.cfg)"
    )
    filename = fil[0]
    if filename != "":
        load_configuration(filename)


def load_configuration(config_path: Path) -> None:
    """
    Open a configuration file and update the GUI accordingly.

    This routines opens the specified configuration file and updates the GUI
    to reflect it contents.

    Arguments
    ---------
    config_path : str
        Name of the configuration file to be opened.
    """
    if not config_path.exists():
        if config_path != "dfastbe.cfg":
            show_error(f"The file {config_path} does not exist!")
        return

    config_path_abs = absolute_path(os.getcwd(), config_path)
    rootdir = os.path.dirname(config_path_abs)
    config_file = ConfigFile.read(config_path_abs)

    config_file.path = config_path_abs

    try:
        version = config_file.version
    except KeyError:
        show_error(f"No version information in the file {config_path}!")
        return

    config = config_file.config
    if version == "1.0":
        section = config["General"]
        dialog["chainFileEdit"].setText(section["RiverKM"])
        studyRange = config_file.get_range("General", "Boundaries")
        dialog["startRange"].setText(str(studyRange[0]))
        dialog["endRange"].setText(str(studyRange[1]))
        dialog["bankDirEdit"].setText(section["BankDir"])
        bankFile = config_file.get_str("General", "BankFile", default="bankfile")
        dialog["bankFileName"].setText(bankFile)
        flag = config_file.get_bool("General", "Plotting", default=True)
        dialog["makePlotsEdit"].setChecked(flag)
        flag = config_file.get_bool("General", "SavePlots", default=True)
        dialog["savePlotsEdit"].setChecked(flag)
        flag = config_file.get_bool("General", "SaveZoomPlots", default=False)
        dialog["saveZoomPlotsEdit"].setChecked(flag)
        zoomStepKM = config_file.get_float("General", "ZoomStepKM", default=1.0)
        dialog["zoomPlotsRangeEdit"].setText(str(zoomStepKM))
        figDir = config_file.get_str(
            "General",
            "FigureDir",
            default=absolute_path(rootdir, "figures"),
        )
        dialog["figureDirEdit"].setText(figDir)
        flag = config_file.get_bool("General", "ClosePlots", default=False)
        dialog["closePlotsEdit"].setChecked(flag)
        flag = config_file.get_bool("General", "DebugOutput", default=False)
        dialog["debugOutputEdit"].setChecked(flag)

        section = config["Detect"]
        dialog["simFileEdit"].setText(section["SimFile"])
        waterDepth = config_file.get_float("Detect", "WaterDepth", default=0.0)
        dialog["waterDepth"].setText(str(waterDepth))
        NBank = config_file.get_int("Detect", "NBank", default=0, positive=True)
        DLines = config_file.get_bank_search_distances(NBank)
        dialog["searchLines"].invisibleRootItem().takeChildren()
        for i in range(NBank):
            istr = str(i + 1)
            fileName = config_file.get_str("Detect", "Line" + istr)
            c1 = QTreeWidgetItem(
                dialog["searchLines"], [istr, fileName, str(DLines[i])]
            )
        if NBank > 0:
            dialog["searchLinesEdit"].setEnabled(True)
            dialog["searchLinesRemove"].setEnabled(True)

        section = config["Erosion"]
        dialog["tErosion"].setText(section["TErosion"])
        dialog["riverAxisEdit"].setText(section["RiverAxis"])
        dialog["fairwayEdit"].setText(section["Fairway"])
        dialog["chainageOutStep"].setText(section["OutputInterval"])
        dialog["outDirEdit"].setText(section["OutputDir"])
        bankNew = config_file.get_str("Erosion", "BankNew", default="banknew")
        dialog["newBankFile"].setText(bankNew)
        bankEq = config_file.get_str("Erosion", "BankEq", default="bankeq")
        dialog["newEqBankFile"].setText(bankEq)
        txt = config_file.get_str("Erosion", "EroVol", default="erovol_standard.evo")
        dialog["eroVol"].setText(txt)
        txt = config_file.get_str("Erosion", "EroVolEqui", default="erovol_eq.evo")
        dialog["eroVolEqui"].setText(txt)

        NLevel = config_file.get_int("Erosion", "NLevel", default=0, positive=True)
        dialog["discharges"].invisibleRootItem().takeChildren()
        for i in range(NLevel):
            istr = str(i + 1)
            fileName = config_file.get_str("Erosion", "SimFile" + istr)
            prob = config_file.get_str("Erosion", "PDischarge" + istr)
            c1 = QTreeWidgetItem(dialog["discharges"], [istr, fileName, prob])
        if NLevel > 0:
            dialog["dischargesEdit"].setEnabled(True)
            dialog["dischargesRemove"].setEnabled(True)
        dialog["refLevel"].validator().setTop(NLevel)
        dialog["refLevel"].setText(section["RefLevel"])

        setParam("shipType", config, "Erosion", "ShipType")
        setParam("shipVeloc", config, "Erosion", "VShip")
        setParam("nShips", config, "Erosion", "NShip")
        setParam("shipNWaves", config, "Erosion", "NWave", "5")
        setParam("shipDraught", config, "Erosion", "Draught")
        setParam("wavePar0", config, "Erosion", "Wave0", "200.0")
        wave0 = config_file.get_str("Erosion", "Wave0", "200.0")
        setParam("wavePar1", config_file.config, "Erosion", "Wave1", wave0)

        useBankType = config_file.get_bool("Erosion", "Classes", default=True)
        dialog["bankType"].setEnabled(useBankType)
        dialog["bankTypeType"].setEnabled(useBankType)
        dialog["bankTypeEdit"].setEnabled(useBankType)
        dialog["bankTypeEditFile"].setEnabled(useBankType)
        dialog["bankShear"].setEnabled(not useBankType)
        dialog["bankShearType"].setEnabled(not useBankType)
        dialog["bankShearEdit"].setEnabled(not useBankType)
        dialog["bankShearEditFile"].setEnabled(not useBankType)
        if useBankType:
            dialog["strengthPar"].setCurrentText("Bank Type")
            bankStrengthSwitch()
            setParam("bankType", config_file.config, "Erosion", "BankType")
        else:
            dialog["strengthPar"].setCurrentText("Critical Shear Stress")
            bankStrengthSwitch()
            setParam("bankShear", config, "Erosion", "BankType")
        setParam("bankProtect", config, "Erosion", "ProtectionLevel", "-1000")
        setParam("bankSlope", config, "Erosion", "Slope", "20.0")
        setParam("bankReed", config, "Erosion", "Reed", "0.0")

        setFilter("velFilter", config, "Erosion", "VelFilterDist")
        setFilter("bedFilter", config, "Erosion", "BedFilterDist")

        tabs = dialog["tabs"]
        for i in range(tabs.count() - 1, 4, -1):
            tabs.removeTab(i)

        for i in range(NLevel):
            istr = str(i + 1)
            addTabForLevel(istr)
            setOptParam(istr + "_shipType", config, "Erosion", "ShipType" + istr)
            setOptParam(istr + "_shipVeloc", config, "Erosion", "VShip" + istr)
            setOptParam(istr + "_nShips", config, "Erosion", "NShip" + istr)
            setOptParam(istr + "_shipNWaves", config, "Erosion", "NWave" + istr)
            setOptParam(istr + "_shipDraught", config, "Erosion", "Draught" + istr)
            setOptParam(istr + "_bankSlope", config, "Erosion", "Slope" + istr)
            setOptParam(istr + "_bankReed", config, "Erosion", "Reed" + istr)
            txt = config_file.get_str("Erosion", "EroVol" + istr, default="")
            dialog[istr + "_eroVolEdit"].setText(txt)

    else:
        show_error(f"Unsupported version number {version} in the file {config_path}!")


def addTabForLevel(istr: str) -> None:
    """
    Create the tab for the settings associated with simulation i.

    Arguments
    ---------
    istr : str
        String representation of the simulation number.

    Arguments
    ---------
    None
    """
    newWidget = QWidget()
    newLayout = QGridLayout(newWidget)
    dialog["tabs"].addTab(newWidget, "Level " + istr)

    optionalParLayout(
        newLayout, 0, istr + "_shipType", "Ship Type", selectList=SHIP_TYPES
    )
    optionalParLayout(newLayout, 2, istr + "_shipVeloc", "Velocity [m/s]")
    optionalParLayout(newLayout, 3, istr + "_nShips", "# Ships [1/yr]")
    optionalParLayout(newLayout, 4, istr + "_shipNWaves", "# Waves [1/ship]")
    optionalParLayout(newLayout, 5, istr + "_shipDraught", "Draught [m]")
    optionalParLayout(newLayout, 6, istr + "_bankSlope", "Slope [-]")
    optionalParLayout(newLayout, 7, istr + "_bankReed", "Reed [-]")

    Label = QLabel("EroVol File Name")
    dialog[istr + "_eroVol"] = Label
    newLayout.addWidget(Label, 8, 0)
    Edit = QLineEdit()
    dialog[istr + "_eroVolEdit"] = Edit
    newLayout.addWidget(Edit, 8, 2)

    stretch = QSpacerItem(
        10, 10, QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
    )
    newLayout.addItem(stretch, 9, 0)


def optionalParLayout(
    gridLayout: QGridLayout, row: int, key, labelString, selectList=None
) -> None:
    """
    Add a line of controls for editing an optional parameter.

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
    dialog[key + "Label"] = Label
    gridLayout.addWidget(Label, row, 0)

    paramTypes = ("Use Default", "Constant", "Variable")
    Type = QComboBox()
    Type.addItems(paramTypes)
    Type.currentIndexChanged.connect(lambda: typeUpdatePar(key))
    dialog[key + "Type"] = Type
    gridLayout.addWidget(Type, row, 1)

    if selectList is None:
        fLayout = openFileLayout(key + "Edit", enabled=False)
        dialog[key + "Edit"].setEnabled(False)
        gridLayout.addWidget(fLayout, row, 2)
    else:
        Select = QComboBox()
        Select.addItems(selectList)
        Select.setEnabled(False)
        dialog[key + "Select"] = Select
        gridLayout.addWidget(Select, row, 2)

        fLayout = openFileLayout(key + "Edit", enabled=False)
        dialog[key + "Edit"].setEnabled(False)
        gridLayout.addWidget(fLayout, row + 1, 2)


def typeUpdatePar(key: str) -> None:
    """
    Implements the dialog setting switching for both general and optional parameters.

    Arguments
    ---------
    key : str
        Short name of the parameter.
    """
    type = dialog[key + "Type"].currentText()
    dialog[key + "Edit"].setText("")
    if type == "Use Default":
        dialog[key + "Edit"].setValidator(None)
        dialog[key + "Edit"].setEnabled(False)
        dialog[key + "EditFile"].setEnabled(False)
        if key + "Select" in dialog.keys():
            dialog[key + "Select"].setEnabled(False)
    elif type == "Constant":
        if key + "Select" in dialog.keys():
            dialog[key + "Select"].setEnabled(True)
            dialog[key + "Edit"].setEnabled(False)
        else:
            if key != "bankProtect":
                dialog[key + "Edit"].setValidator(validator("positive_real"))
            dialog[key + "Edit"].setEnabled(True)
        dialog[key + "EditFile"].setEnabled(False)
    elif type == "Variable":
        if key + "Select" in dialog.keys():
            dialog[key + "Select"].setEnabled(False)
        dialog[key + "Edit"].setEnabled(True)
        dialog[key + "Edit"].setValidator(None)
        dialog[key + "EditFile"].setEnabled(True)


def setParam(field: str, config, group: str, key: str, default: str = "??") -> None:
    """
    Update the dialog for a general parameter based on configuration file.

    Arguments
    ---------
    field : str
        Short name of the parameter.
    config : configparser.ConfigParser
        Configuration for the D-FAST Bank Erosion analysis with absolute or relative paths.
    group : str
        Name of the group in the configuration.
    key : str
        Name of the key in the configuration group.
    default : str
        Default string if the group/key pair doesn't exist in the configuration.

    """
    config_file = ConfigFile(config)
    config_value = config_file.get_str(group, key, default)

    try:
        val = float(config_value)
        cast(QComboBox, dialog[field + "Type"]).setCurrentText("Constant")
        if field + "Select" in dialog.keys():
            int_value = int(val)
            if field == "shipType":
                int_value = int_value - 1
            cast(QComboBox, dialog[field + "Select"]).setCurrentIndex(int_value)
        else:
            cast(QLineEdit, dialog[field + "Edit"]).setText(config_value)
    except:
        cast(QComboBox, dialog[field + "Type"]).setCurrentText("Variable")
        cast(QLineEdit, dialog[field + "Edit"]).setText(config_value)


def setFilter(field: str, config, group: str, key: str) -> None:
    """
    Update the dialog for a filter based on configuration file.

    Arguments
    ---------
    field : str
        Short name of the parameter.
    config : configparser.ConfigParser
        Configuration for the D-FAST Bank Erosion analysis with absolute or relative paths.
    group : str
        Name of the group in the configuration.
    key : str
        Name of the key in the configuration group.

    """
    config_file = ConfigFile(config)
    val = config_file.get_float(group, key, 0.0)
    if val > 0.0:
        dialog[field + "Active"].setChecked(True)
        dialog[field + "Width"].setText(str(val))
    else:
        dialog[field + "Active"].setChecked(False)


def setOptParam(field: str, config, group: str, key: str) -> None:
    """
    Update the dialog for an optional parameter based on configuration file.

    Arguments
    ---------
    field : str
        Short name of the parameter.
    config : configparser.ConfigParser
        Configuration for the D-FAST Bank Erosion analysis with absolute or relative paths.
    group : str
        Name of the group in the configuration.
    key : str
        Name of the key in the configuration group.
    """
    config_file = ConfigFile(config)
    str = config_file.get_str(group, key, "")
    if str == "":
        dialog[field + "Type"].setCurrentText("Use Default")
        dialog[field + "Edit"].setText("")
    else:
        try:
            val = float(str)
            dialog[field + "Type"].setCurrentText("Constant")
            if field + "Select" in dialog.keys():
                ival = int(val) - 1  # shipType 1 -> index 0
                dialog[field + "Select"].setCurrentIndex(ival)
            else:
                dialog[field + "Edit"].setText(str)
        except:
            dialog[field + "Type"].setCurrentText("Variable")
            dialog[field + "Edit"].setText(str)


def menu_save_configuration() -> None:
    """
    Ask for a configuration file name and save GUI selection to that file.

    Arguments
    ---------
    None
    """
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


def get_configuration() -> configparser.ConfigParser:
    """Extract a configuration from the GUI.

    Returns
    -------
    config : configparser.ConfigParser
        Configuration for the D-FAST Bank Erosion analysis.
    """
    config = configparser.ConfigParser()
    config.optionxform = str  # case sensitive configuration

    config.add_section("General")
    config["General"]["Version"] = "1.0"
    config["General"]["RiverKM"] = dialog["chainFileEdit"].text()
    config["General"]["Boundaries"] = (
        dialog["startRange"].text() + ":" + dialog["endRange"].text()
    )
    config["General"]["BankDir"] = dialog["bankDirEdit"].text()
    config["General"]["BankFile"] = dialog["bankFileName"].text()
    config["General"]["Plotting"] = str(dialog["makePlotsEdit"].isChecked())
    config["General"]["SavePlots"] = str(dialog["savePlotsEdit"].isChecked())
    config["General"]["SaveZoomPlots"] = str(dialog["saveZoomPlotsEdit"].isChecked())
    config["General"]["ZoomStepKM"] = dialog["zoomPlotsRangeEdit"].text()
    config["General"]["FigureDir"] = dialog["figureDirEdit"].text()
    config["General"]["ClosePlots"] = str(dialog["closePlotsEdit"].isChecked())
    config["General"]["DebugOutput"] = str(dialog["debugOutputEdit"].isChecked())

    config.add_section("Detect")
    config["Detect"]["SimFile"] = dialog["simFileEdit"].text()
    config["Detect"]["WaterDepth"] = dialog["waterDepth"].text()
    nbank = dialog["searchLines"].topLevelItemCount()
    config["Detect"]["NBank"] = str(nbank)
    dlines = "[ "
    for i in range(nbank):
        istr = str(i + 1)
        config["Detect"]["Line" + istr] = dialog["searchLines"].topLevelItem(i).text(1)
        dlines += dialog["searchLines"].topLevelItem(i).text(2) + ", "
    dlines = dlines[:-2] + " ]"
    config["Detect"]["DLines"] = dlines

    config.add_section("Erosion")
    config["Erosion"]["TErosion"] = dialog["tErosion"].text()
    config["Erosion"]["RiverAxis"] = dialog["riverAxisEdit"].text()
    config["Erosion"]["Fairway"] = dialog["fairwayEdit"].text()
    config["Erosion"]["OutputInterval"] = dialog["chainageOutStep"].text()
    config["Erosion"]["OutputDir"] = dialog["outDirEdit"].text()
    config["Erosion"]["BankNew"] = dialog["newBankFile"].text()
    config["Erosion"]["BankEq"] = dialog["newEqBankFile"].text()
    config["Erosion"]["EroVol"] = dialog["eroVol"].text()
    config["Erosion"]["EroVolEqui"] = dialog["eroVolEqui"].text()

    if dialog["shipTypeType"].currentText() == "Constant":
        config["Erosion"]["ShipType"] = str(
            dialog["shipTypeSelect"].currentIndex() + 1
        )  # index 0 -> shipType 1
    else:
        config["Erosion"]["ShipType"] = dialog["shipTypeEdit"].text()
    config["Erosion"]["VShip"] = dialog["shipVelocEdit"].text()
    config["Erosion"]["NShip"] = dialog["nShipsEdit"].text()
    config["Erosion"]["NWaves"] = dialog["shipNWavesEdit"].text()
    config["Erosion"]["Draught"] = dialog["shipDraughtEdit"].text()
    config["Erosion"]["Wave0"] = dialog["wavePar0Edit"].text()
    config["Erosion"]["Wave1"] = dialog["wavePar1Edit"].text()

    if dialog["strengthPar"].currentText() == "Bank Type":
        config["Erosion"]["Classes"] = "true"
        if dialog["bankTypeType"].currentText() == "Constant":
            config["Erosion"]["BankType"] = dialog["bankTypeSelect"].currentIndex()
        else:
            config["Erosion"]["BankType"] = dialog["bankTypeEdit"].text()
    else:
        config["Erosion"]["Classes"] = "false"
        config["Erosion"]["BankType"] = dialog["bankShearEdit"].text()
    config["Erosion"]["ProtectionLevel"] = dialog["bankProtectEdit"].text()
    config["Erosion"]["Slope"] = dialog["bankSlopeEdit"].text()
    config["Erosion"]["Reed"] = dialog["bankReedEdit"].text()

    if dialog["velFilterActive"].isChecked():
        config["Erosion"]["VelFilterDist"] = dialog["velFilterWidth"].text()
    if dialog["bedFilterActive"].isChecked():
        config["Erosion"]["BedFilterDist"] = dialog["bedFilterWidth"].text()

    nlevel = dialog["discharges"].topLevelItemCount()
    config["Erosion"]["NLevel"] = str(nlevel)
    config["Erosion"]["RefLevel"] = dialog["refLevel"].text()
    for i in range(nlevel):
        istr = str(i + 1)
        config["Erosion"]["SimFile" + istr] = (
            dialog["discharges"].topLevelItem(i).text(1)
        )
        config["Erosion"]["PDischarge" + istr] = (
            dialog["discharges"].topLevelItem(i).text(2)
        )
        if dialog[istr + "_shipTypeType"].currentText() != "Use Default":
            if dialog[istr + "_shipTypeType"].currentText() == "Constant":
                config["Erosion"]["ShipType" + istr] = (
                    dialog[istr + "_shipTypeSelect"].currentIndex() + 1
                )  # index 0 -> shipType 1
            else:
                config["Erosion"]["ShipType" + istr] = dialog[
                    istr + "_shipTypeEdit"
                ].text()
        if dialog[istr + "_shipVelocType"].currentText() != "Use Default":
            config["Erosion"]["VShip" + istr] = dialog[istr + "_shipVelocEdit"].text()
        if dialog[istr + "_nShipsType"].currentText() != "Use Default":
            config["Erosion"]["NShip" + istr] = dialog[istr + "_nShipsEdit"].text()
        if dialog[istr + "_shipNWavesType"].currentText() != "Use Default":
            config["Erosion"]["NWaves" + istr] = dialog[istr + "_shipNWavesEdit"].text()
        if dialog[istr + "_shipDraughtType"].currentText() != "Use Default":
            config["Erosion"]["Draught" + istr] = dialog[
                istr + "_shipDraughtEdit"
            ].text()
        if dialog[istr + "_bankSlopeType"].currentText() != "Use Default":
            config["Erosion"]["Slope" + istr] = dialog[istr + "_bankSlopeEdit"].text()
        if dialog[istr + "_bankReedType"].currentText() != "Use Default":
            config["Erosion"]["Reed" + istr] = dialog[istr + "_bankReedEdit"].text()
        if dialog[istr + "_eroVolEdit"].text() != "":
            config["Erosion"]["EroVol" + istr] = dialog[istr + "_eroVolEdit"].text()
    return config


def menu_about_self():
    """
    Show the about dialog for D-FAST Bank Erosion.

    Arguments
    ---------
    None
    """
    msg = QMessageBox()
    msg.setText(f"D-FAST Bank Erosion {__version__}")
    msg.setInformativeText("Copyright (c) 2025 Deltares.")
    msg.setDetailedText(gui_text("license"))
    msg.setWindowTitle(gui_text("about"))
    msg.setStandardButtons(QMessageBox.Ok)
    
    dfast_icon = get_icon(f"{ICONS_DIR}/D-FASTBE.png")
    available_sizes = dfast_icon.availableSizes()
    if available_sizes:
        icon_size = available_sizes[0]
        pixmap = dfast_icon.pixmap(icon_size).scaled(64,64)
        msg.setIconPixmap(pixmap)
    msg.setWindowIcon(dfast_icon)
    msg.exec()


def menu_about_qt():
    """Show the about dialog for Qt."""
    QApplication.aboutQt()


def menu_open_manual():
    """Open the user manual."""
    manual_path = r_dir / USER_MANUAL_FILE_NAME
    filename = str(manual_path)
    if not manual_path.exists():
        show_error(f"User manual not found: {filename}")
        return
    try:
        # bandit complains about os.startfile, but it is the only way to open a file in the default application on Windows.
        # On Linux and MacOS, opening the file might give a security warning.
        os.startfile(filename) # nosec
    except Exception as e:
        show_error(f"Failed to open the user manual: {e}")


def main(config: Optional[Path] = None) -> None:
    """
    Start the user interface using default settings or optional configuration.

    Arguments
    ---------
    config : Optional[str]
        Optional name of configuration file.
    """
    gui = GUI()
    gui.create()
    if not config is None:
        load_configuration(config)

    gui.activate()


class GUI:

    def __init__(self):
        global dialog
        dialog = {}

        self.app = QApplication()
        self.app.setStyle("fusion")
        dialog["application"] = self.app
        self.window, self.layout = self.create_window()
        dialog["window"] = self.window

        self.tabs = QTabWidget(self.window)
        dialog["tabs"] = self.tabs
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
