# -*- coding: utf-8 -*-
"""
Copyright (C) 2020 Stichting Deltares.

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

from typing import Dict, Any, Optional, Tuple, List

from PyQt5 import QtWidgets
from PyQt5 import QtCore
import PyQt5.QtGui
from dfastbe.io import get_text, get_progloc, absolute_path, ConfigFile, config_get_range, \
                        config_get_bool, config_get_float, config_get_int, config_get_bank_search_distances
import pathlib
import sys
import os
import configparser
import matplotlib.pyplot
import subprocess
from functools import partial
from dfastbe import __version__
from dfastbe.bank_lines import banklines_core
from dfastbe.bank_erosion import bankerosion_core
from dfastbe.utils import config_to_relative_paths, config_to_absolute_paths

DialogObject = Dict[str, PyQt5.QtCore.QObject]

dialog: DialogObject


def gui_text(key: str, prefix: str = "gui_", dict: Dict[str, Any] = {}):
    """
    Query the global dictionary of texts for a single string in the GUI.

    This routine concatenates the prefix and the key to query the global
    dictionary of texts. It selects the first line of the text obtained and
    expands and placeholders in the string using the optional dictionary
    provided.

    Arguments
    ---------
    key : str
        The key string used to query the dictionary (extended with prefix).
    prefix : str
        The prefix used in combination with the key (default "gui_").
    dict : Dict[str, Any]
        A dictionary used for placeholder expansions (default empty).

    Returns
    -------
        The first line of the text in the dictionary expanded with the keys.
    """
    cstr = get_text(prefix + key)
    str = cstr[0].format(**dict)
    return str


def create_dialog() -> None:
    """
    Construct the D-FAST Bank Erosion user interface.

    Arguments
    ---------
    None
    """
    global dialog
    dialog = {}

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("fusion")
    dialog["application"] = app

    win = QtWidgets.QMainWindow()
    win.setGeometry(200, 200, 600, 300)
    win.setWindowTitle("D-FAST Bank Erosion")
    dialog["window"] = win

    menubar = win.menuBar()
    createMenus(menubar)

    centralWidget = QtWidgets.QWidget()
    layout = QtWidgets.QBoxLayout(2, centralWidget)
    win.setCentralWidget(centralWidget)

    tabs = QtWidgets.QTabWidget(win)
    dialog["tabs"] = tabs
    layout.addWidget(tabs)

    buttonBar = QtWidgets.QWidget(win)
    buttonBarLayout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight, buttonBar)
    buttonBarLayout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(buttonBar)

    detect = QtWidgets.QPushButton(gui_text("action_detect"), win)
    detect.clicked.connect(run_detection)
    buttonBarLayout.addWidget(detect)

    erode = QtWidgets.QPushButton(gui_text("action_erode"), win)
    erode.clicked.connect(run_erosion)
    buttonBarLayout.addWidget(erode)

    done = QtWidgets.QPushButton(gui_text("action_close"), win)
    done.clicked.connect(close_dialog)
    buttonBarLayout.addWidget(done)

    addGeneralTab(tabs, win)
    addDetectTab(tabs, win, app)
    addErosionTab(tabs, win, app)
    addShippingTab(tabs, win)
    addBankTab(tabs, win)


def createMenus(menubar: PyQt5.QtWidgets.QMenuBar) -> None:
    """
    Add the menus to the menubar.

    Arguments
    ---------
    menubar : PyQt5.QtWidgets.QMenuBar
        Menubar to which menus should be added.
    """
    menu = menubar.addMenu(gui_text("File"))
    item = menu.addAction(gui_text("Load"))
    item.triggered.connect(menu_load_configuration)
    item = menu.addAction(gui_text("Save"))
    item.triggered.connect(menu_save_configuration)
    menu.addSeparator()
    item = menu.addAction(gui_text("Close"))
    item.triggered.connect(close_dialog)

    menu = menubar.addMenu(gui_text("Help"))
    item = menu.addAction(gui_text("Manual"))
    item.triggered.connect(menu_open_manual)
    menu.addSeparator()
    item = menu.addAction(gui_text("Version"))
    item.triggered.connect(menu_about_self)
    item = menu.addAction(gui_text("AboutQt"))
    item.triggered.connect(menu_about_qt)


def addGeneralTab(
    tabs: PyQt5.QtWidgets.QTabWidget, win: PyQt5.QtWidgets.QMainWindow
) -> None:
    """
    Create the tab for the general settings.

    These settings are used by both the bank line detection and the bank
    erosion analysis.

    Arguments
    ---------
    tabs : PyQt5.QtWidgets.QTabWidget
        Tabs object to which the tab should be added.
    win : PyQt5.QtWidgets.QMainWindow
        Windows in which the tab item is located.
    """
    generalWidget = QtWidgets.QWidget()
    generalLayout = QtWidgets.QFormLayout(generalWidget)
    tabs.addTab(generalWidget, "General")

    addOpenFileRow(generalLayout, "chainFile", "Chain File")

    chainRange = QtWidgets.QWidget()
    gridly = QtWidgets.QGridLayout(chainRange)
    gridly.setContentsMargins(0, 0, 0, 0)

    gridly.addWidget(QtWidgets.QLabel("From [km]", win), 0, 0)
    startRange = QtWidgets.QLineEdit(win)
    dialog["startRange"] = startRange
    gridly.addWidget(startRange, 0, 1)
    gridly.addWidget(QtWidgets.QLabel("To [km]", win), 0, 2)
    endRange = QtWidgets.QLineEdit(win)
    dialog["endRange"] = endRange
    gridly.addWidget(endRange, 0, 3)

    generalLayout.addRow("Study Range", chainRange)

    addOpenFileRow(generalLayout, "bankDir", "Bank Directory")

    bankFileName = QtWidgets.QLineEdit(win)
    dialog["bankFileName"] = bankFileName
    generalLayout.addRow("Bank File Name", bankFileName)

    addCheckBox(generalLayout, "makePlots", "Create Figures", True)
    dialog["makePlotsEdit"].stateChanged.connect(updatePlotting)

    addCheckBox(generalLayout, "savePlots", "Save Figures", True)
    dialog["savePlotsEdit"].stateChanged.connect(updatePlotting)

    zoomPlots = QtWidgets.QWidget()
    gridly = QtWidgets.QGridLayout(zoomPlots)
    gridly.setContentsMargins(0, 0, 0, 0)

    saveZoomPlotsEdit = QtWidgets.QCheckBox("", win)
    saveZoomPlotsEdit.stateChanged.connect(updatePlotting)
    saveZoomPlotsEdit.setChecked(False)
    gridly.addWidget(saveZoomPlotsEdit, 0, 0)
    dialog["saveZoomPlotsEdit"] = saveZoomPlotsEdit

    zoomPlotsRangeTxt = QtWidgets.QLabel("Zoom Range [km]", win)
    zoomPlotsRangeTxt.setEnabled(False)
    gridly.addWidget(zoomPlotsRangeTxt, 0, 1)
    dialog["zoomPlotsRangeTxt"] = zoomPlotsRangeTxt

    zoomPlotsRangeEdit = QtWidgets.QLineEdit("1.0",win)
    zoomPlotsRangeEdit.setValidator(validator("positive_real"))
    zoomPlotsRangeEdit.setEnabled(False)
    gridly.addWidget(zoomPlotsRangeEdit, 0, 2)
    dialog["zoomPlotsRangeEdit"] = zoomPlotsRangeEdit

    saveZoomPlots = QtWidgets.QLabel("Save Zoomed Figures", win)
    generalLayout.addRow(saveZoomPlots, zoomPlots)
    dialog["saveZoomPlots"] = saveZoomPlots

    addOpenFileRow(generalLayout, "figureDir", "Figure Directory")
    addCheckBox(generalLayout, "closePlots", "Close Figures")
    addCheckBox(generalLayout, "debugOutput", "Debug Output")


def addCheckBox(
    formLayout: PyQt5.QtWidgets.QFormLayout,
    key: str,
    labelString: str,
    isChecked: bool = False,
) -> None:
    """
    Add a line of with checkbox control to a form layout.

    Arguments
    ---------
    formLayout : PyQt5.QtWidgets.QFormLayout
        Form layout object in which to position the edit controls.
    key : str
        Short name of the parameter.
    labelString : str
        String describing the parameter to be displayed as label.
    isChecked : bool
        Initial state of the check box.
    """
    checkBox = QtWidgets.QCheckBox("")
    checkBox.setChecked(isChecked)
    dialog[key + "Edit"] = checkBox

    checkTxt = QtWidgets.QLabel(labelString)
    dialog[key] = checkTxt
    formLayout.addRow(checkTxt, checkBox)


def addDetectTab(
    tabs: PyQt5.QtWidgets.QTabWidget,
    win: PyQt5.QtWidgets.QMainWindow,
    app: PyQt5.QtWidgets.QApplication,
) -> None:
    """
    Create the tab for the bank line detection settings.

    Arguments
    ---------
    tabs : PyQt5.QtWidgets.QTabWidget
        Tabs object to which the tab should be added.
    win : PyQt5.QtWidgets.QMainWindow
        The window object in which the tab item is located.
    app : PyQt5.QtWidgets.QApplication
        The application object to which the window belongs, needed for font information.
    """
    detectWidget = QtWidgets.QWidget()
    detectLayout = QtWidgets.QFormLayout(detectWidget)
    tabs.addTab(detectWidget, "Detection")

    addOpenFileRow(detectLayout, "simFile", "Simulation File")

    waterDepth = QtWidgets.QLineEdit(win)
    waterDepth.setValidator(validator("positive_real"))
    dialog["waterDepth"] = waterDepth
    detectLayout.addRow("Water Depth [m]", waterDepth)

    searchLines = QtWidgets.QTreeWidget(win)
    searchLines.setHeaderLabels(["Index", "FileName", "Search Distance [m]"])
    searchLines.setFont(app.font())
    searchLines.setColumnWidth(0, 50)
    searchLines.setColumnWidth(1, 200)
    # c1 = QtWidgets.QTreeWidgetItem(searchLines, ["0", "test\\filename", "50"])

    slLayout = addRemoveEditLayout(searchLines, "searchLines")
    detectLayout.addRow("Search Lines", slLayout)


def addErosionTab(
    tabs: PyQt5.QtWidgets.QTabWidget,
    win: PyQt5.QtWidgets.QMainWindow,
    app: PyQt5.QtWidgets.QApplication,
) -> None:
    """
    Create the tab for the main bank erosion settings.

    Arguments
    ---------
    tabs : PyQt5.QtWidgets.QTabWidget
        Tabs object to which the tab should be added.
    win : PyQt5.QtWidgets.QMainWindow
        The window object in which the tab item is located.
    app : PyQt5.QtWidgets.QApplication
        The application object to which the window belongs, needed for font information.
    """
    erosionWidget = QtWidgets.QWidget()
    erosionLayout = QtWidgets.QFormLayout(erosionWidget)
    tabs.addTab(erosionWidget, "Erosion")

    tErosion = QtWidgets.QLineEdit(win)
    tErosion.setValidator(validator("positive_real"))
    dialog["tErosion"] = tErosion
    erosionLayout.addRow("Simulation Time [yr]", tErosion)

    addOpenFileRow(erosionLayout, "riverAxis", "River Axis File")

    addOpenFileRow(erosionLayout, "fairway", "Fairway File")

    discharges = QtWidgets.QTreeWidget(win)
    discharges.setHeaderLabels(["Level", "FileName", "Probability [-]"])
    discharges.setFont(app.font())
    discharges.setColumnWidth(0, 50)
    discharges.setColumnWidth(1, 250)
    # c1 = QtWidgets.QTreeWidgetItem(discharges, ["0", "test\\filename", "0.5"])

    disLayout = addRemoveEditLayout(discharges, "discharges")
    erosionLayout.addRow("Discharges", disLayout)

    refLevel = QtWidgets.QLineEdit(win)
    refLevel.setValidator(PyQt5.QtGui.QIntValidator(1, 1))
    dialog["refLevel"] = refLevel
    erosionLayout.addRow("Reference Case", refLevel)

    chainageOutStep = QtWidgets.QLineEdit(win)
    chainageOutStep.setValidator(validator("positive_real"))
    dialog["chainageOutStep"] = chainageOutStep
    erosionLayout.addRow("Chainage Output Step [km]", chainageOutStep)

    addOpenFileRow(erosionLayout, "outDir", "Output Directory")

    newBankFile = QtWidgets.QLineEdit(win)
    dialog["newBankFile"] = newBankFile
    erosionLayout.addRow("New Bank File Name", newBankFile)

    newEqBankFile = QtWidgets.QLineEdit(win)
    dialog["newEqBankFile"] = newEqBankFile
    erosionLayout.addRow("New Eq Bank File Name", newEqBankFile)

    eroVol = QtWidgets.QLineEdit(win)
    dialog["eroVol"] = eroVol
    erosionLayout.addRow("EroVol File Name", eroVol)

    eroVolEqui = QtWidgets.QLineEdit(win)
    dialog["eroVolEqui"] = eroVolEqui
    erosionLayout.addRow("EroVolEqui File Name", eroVolEqui)


def addShippingTab(
    tabs: PyQt5.QtWidgets.QTabWidget, win: PyQt5.QtWidgets.QMainWindow
) -> None:
    """
    Create the tab for the general shipping settings.

    Arguments
    ---------
    tabs : PyQt5.QtWidgets.QTabWidget
        Tabs object to which the tab should be added.
    win : PyQt5.QtWidgets.QMainWindow
        The window object in which the tab item is located.
    """
    eParamsWidget = QtWidgets.QWidget()
    eParamsLayout = QtWidgets.QGridLayout(eParamsWidget)
    tabs.addTab(eParamsWidget, "Shipping Parameters")

    generalParLayout(eParamsLayout, 0, "shipType", "Ship Type", selectList=shipTypes())
    generalParLayout(eParamsLayout, 2, "shipVeloc", "Velocity [m/s]")
    generalParLayout(eParamsLayout, 3, "nShips", "# Ships [1/yr]")
    generalParLayout(eParamsLayout, 4, "shipNWaves", "# Waves [1/ship]")
    generalParLayout(eParamsLayout, 5, "shipDraught", "Draught [m]")
    generalParLayout(eParamsLayout, 6, "wavePar0", "Wave0 [m]")
    generalParLayout(eParamsLayout, 7, "wavePar1", "Wave1 [m]")

    stretch = QtWidgets.QSpacerItem(10, 10, 13, 7)
    eParamsLayout.addItem(stretch, 8, 0)


def shipTypes() -> List[str]:
    """
    Return the tuple of ship types.

    Arguments
    ---------
    None

    Returns
    -------
    types : List[str]
        List of three ship types.
    """
    return ["1 (multiple barge convoy set)", "2 (RHK ship / motorship)", "3 (towboat)"]


def addBankTab(
    tabs: PyQt5.QtWidgets.QTabWidget, win: PyQt5.QtWidgets.QMainWindow
) -> None:
    """
    Create the tab for the general bank properties.

    Arguments
    ---------
    tabs : PyQt5.QtWidgets.QTabWidget
        Tabs object to which the tab should be added.
    win : PyQt5.QtWidgets.QMainWindow
        The window object in which the tab item is located.
    """
    eParamsWidget = QtWidgets.QWidget()
    eParamsLayout = QtWidgets.QGridLayout(eParamsWidget)
    tabs.addTab(eParamsWidget, "Bank Parameters")

    strength = QtWidgets.QLabel("Strength Parameter")
    eParamsLayout.addWidget(strength, 0, 0)
    strengthPar = QtWidgets.QComboBox()
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

    stretch = QtWidgets.QSpacerItem(10, 10, 13, 7)
    eParamsLayout.addItem(stretch, 9, 0)


def addFilter(
    gridLayout: PyQt5.QtWidgets.QGridLayout, row: int, key: str, labelString: str
) -> None:
    """
    Add a line of controls for a filter

    Arguments
    ---------
    gridLayout : PyQt5.QtWidgets.QGridLayout
        Grid layout object in which to position the edit controls.
    row : int
        Grid row number to be used for this parameter.
    key : str
        Short name of the parameter.
    labelString : str
        String describing the parameter to be displayed as label.
    """

    widthEdit = QtWidgets.QLineEdit("0.3")
    widthEdit.setValidator(validator("positive_real"))
    gridLayout.addWidget(widthEdit, row, 2)
    dialog[key + "Width"] = widthEdit

    useFilter = QtWidgets.QCheckBox("")
    useFilter.setChecked(False)
    useFilter.stateChanged.connect(partial(updateFilter, key))
    gridLayout.addWidget(useFilter, row, 1)
    dialog[key + "Active"] = useFilter

    filterTxt = QtWidgets.QLabel(labelString)
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
    """
    Implements the dialog settings depending on the bank strength specification method.

    Arguments
    ---------
    None
    """
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


def validator(validstr: str) -> PyQt5.QtGui.QValidator:
    """
    Wrapper to easily create a validator.

    Arguments
    ---------
    validstr : str
        Identifier for the requested validation method.

    Returns
    -------
    validator : PyQt5.QtGui.QValidator
        Validator for the requested validation method.
    """
    if validstr == "positive_real":
        validator = PyQt5.QtGui.QDoubleValidator()
        validator.setBottom(0)
    else:
        raise Exception("Unknown validator type: {}".format(validstr))
    return validator


def activate_dialog() -> None:
    """
    Activate the user interface and run the program.

    Arguments
    ---------
    None
    """
    app = dialog["application"]
    win = dialog["window"]
    win.show()
    sys.exit(app.exec_())


def generalParLayout(
    gridLayout: PyQt5.QtWidgets.QGridLayout,
    row: int,
    key: str,
    labelString: str,
    selectList: Optional[List[str]] = None,
) -> None:
    """
    Add a line of controls for editing a general parameter.

    Arguments
    ---------
    gridLayout : PyQt5.QtWidgets.QGridLayout
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
    Label = QtWidgets.QLabel(labelString)
    dialog[key] = Label
    gridLayout.addWidget(Label, row, 0)

    paramTypes = ("Constant", "Variable")
    Type = QtWidgets.QComboBox()
    Type.addItems(paramTypes)
    Type.currentIndexChanged.connect(partial(typeUpdatePar, key))
    dialog[key + "Type"] = Type
    gridLayout.addWidget(Type, row, 1)

    if selectList is None:
        fLayout = openFileLayout(key + "Edit", enabled=False)
        gridLayout.addWidget(fLayout, row, 2)
    else:
        Select = QtWidgets.QComboBox()
        Select.addItems(selectList)
        dialog[key + "Select"] = Select
        gridLayout.addWidget(Select, row, 2)

        fLayout = openFileLayout(key + "Edit", enabled=False)
        dialog[key + "Edit"].setEnabled(False)
        gridLayout.addWidget(fLayout, row + 1, 2)

    typeUpdatePar(key)


def addOpenFileRow(
    formLayout: PyQt5.QtWidgets.QFormLayout, key: str, labelString: str
) -> None:
    """
    Add a line of controls for selecting a file or folder in a form layout.

    Arguments
    ---------
    formLayout : PyQt5.QtWidgets.QFormLayout
        Form layout object in which to position the edit controls.
    key : str
        Short name of the parameter.
    labelString : str
        String describing the parameter to be displayed as label.
    """
    Label = QtWidgets.QLabel(labelString)
    dialog[key] = Label
    fLayout = openFileLayout(key + "Edit")
    formLayout.addRow(Label, fLayout)


def getIcon(filename: str) -> PyQt5.QtGui.QIcon:
    """
    Opens the icon file relative to the location of the program.

    Arguments
    ---------
    filename : str
        Name of the icon file.
    """
    progloc = str(pathlib.Path(__file__).parent.absolute())
    return PyQt5.QtGui.QIcon(progloc + os.path.sep + filename)


def openFileLayout(key, enabled=True) -> PyQt5.QtWidgets.QWidget:
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
    parent : PyQt5.QtWidgets.QWidget
        Parent QtWidget that contains the edit field and selection button.
    """
    parent = QtWidgets.QWidget()
    gridly = QtWidgets.QGridLayout(parent)
    gridly.setContentsMargins(0, 0, 0, 0)

    myWidget = QtWidgets.QLineEdit()
    dialog[key] = myWidget
    gridly.addWidget(myWidget, 0, 0)

    openFile = QtWidgets.QPushButton(getIcon("open.png"), "")
    openFile.clicked.connect(partial(selectFile, key))
    openFile.setEnabled(enabled)
    dialog[key + "File"] = openFile
    gridly.addWidget(openFile, 0, 2)

    return parent


def addRemoveEditLayout(
    mainWidget: PyQt5.QtWidgets.QWidget, key: str
) -> PyQt5.QtWidgets.QWidget:
    """
    Create a standard layout with list control and add, edit and remove buttons.

    Arguments
    ---------
    mainWidget : PyQt5.QtWidgets.QWidget
        Main object on which the add, edit and remove buttons should operate.
    key : str
        Short name of the parameter.

    Returns
    -------
    parent : PyQt5.QtWidgets.QWidget
        Parent QtWidget that contains the add, edit and remove buttons.
    """
    parent = QtWidgets.QWidget()
    gridly = QtWidgets.QGridLayout(parent)
    gridly.setContentsMargins(0, 0, 0, 0)

    dialog[key] = mainWidget
    gridly.addWidget(mainWidget, 0, 0)

    buttonBar = QtWidgets.QWidget()
    buttonBarLayout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom, buttonBar)
    buttonBarLayout.setContentsMargins(0, 0, 0, 0)
    gridly.addWidget(buttonBar, 0, 1)

    addBtn = QtWidgets.QPushButton(getIcon("add.png"), "")
    addBtn.clicked.connect(partial(addAnItem, key))
    dialog[key + "Add"] = addBtn
    buttonBarLayout.addWidget(addBtn)

    editBtn = QtWidgets.QPushButton(getIcon("edit.png"), "")
    editBtn.clicked.connect(partial(editAnItem, key))
    editBtn.setEnabled(False)
    dialog[key + "Edit"] = editBtn
    buttonBarLayout.addWidget(editBtn)

    delBtn = QtWidgets.QPushButton(getIcon("remove.png"), "")
    delBtn.clicked.connect(partial(removeAnItem, key))
    delBtn.setEnabled(False)
    dialog[key + "Remove"] = delBtn
    buttonBarLayout.addWidget(delBtn)

    stretch = QtWidgets.QSpacerItem(10, 10, 13, 7)
    buttonBarLayout.addItem(stretch)

    return parent


def updatePlotting() -> None:
    """
    Update the plotting flags.
    
    Arguments
    ---------
    None
    """
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
        c1 = QtWidgets.QTreeWidgetItem(dialog["searchLines"], [istr, fileName, dist])
    elif key == "discharges":
        prob = str(1 / (nItems + 1))
        fileName, prob = editADischarge(key, istr, prob=prob)
        c1 = QtWidgets.QTreeWidgetItem(dialog["discharges"], [istr, fileName, prob])
        addTabForLevel(istr)
        dialog["refLevel"].validator().setTop(i)
    dialog[key + "Edit"].setEnabled(True)
    dialog[key + "Remove"].setEnabled(True)


def setDialogSize(editDialog: PyQt5.QtWidgets.QDialog, width: int, height: int) -> None:
    """
    Set the width and height of a dialog and position it centered relative to the main window.
    
    Arguments
    ---------
    editDialog : QtWidgets.QDialog
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
    editDialog = QtWidgets.QDialog()
    setDialogSize(editDialog, 600, 100)
    editDialog.setWindowFlags(
        PyQt5.QtCore.Qt.WindowTitleHint | PyQt5.QtCore.Qt.WindowSystemMenuHint
    )
    editDialog.setWindowTitle("Edit Search Line")
    editLayout = QtWidgets.QFormLayout(editDialog)

    label = QtWidgets.QLabel(istr)
    editLayout.addRow("Search Line Nr", label)

    addOpenFileRow(editLayout, "editSearchLine", "Search Line File")
    dialog["editSearchLineEdit"].setText(fileName)

    searchDistance = QtWidgets.QLineEdit()
    searchDistance.setText(dist)
    searchDistance.setValidator(validator("positive_real"))
    editLayout.addRow("Search Distance [m]", searchDistance)

    done = QtWidgets.QPushButton("Done")
    done.clicked.connect(partial(close_edit, editDialog))
    # edit_SearchDistance.setValidator(validator("positive_real"))
    editLayout.addRow(" ", done)

    editDialog.exec()

    fileName = dialog["editSearchLineEdit"].text()
    dist = searchDistance.text()
    return fileName, dist


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
    editDialog = QtWidgets.QDialog()
    setDialogSize(editDialog, 600, 100)
    editDialog.setWindowFlags(
        PyQt5.QtCore.Qt.WindowTitleHint | PyQt5.QtCore.Qt.WindowSystemMenuHint
    )
    editDialog.setWindowTitle("Edit Discharge")
    editLayout = QtWidgets.QFormLayout(editDialog)

    label = QtWidgets.QLabel(istr)
    editLayout.addRow("Level Nr", label)

    addOpenFileRow(editLayout, "editDischarge", "Simulation File")
    dialog["editDischargeEdit"].setText(fileName)

    probability = QtWidgets.QLineEdit()
    probability.setText(prob)
    probability.setValidator(validator("positive_real"))
    editLayout.addRow("Probability [-]", probability)

    done = QtWidgets.QPushButton("Done")
    done.clicked.connect(partial(close_edit, editDialog))
    # edit_SearchDistance.setValidator(validator("positive_real"))
    editLayout.addRow(" ", done)

    editDialog.exec()

    fileName = dialog["editDischargeEdit"].text()
    prob = probability.text()
    return fileName, prob


def close_edit(hDialog: PyQt5.QtWidgets.QDialog) -> None:
    """
    Generic close function for edit dialogs.

    Arguments
    ---------
    hDialog : PyQt5.QtWidgets.QDialog
        Dialog object to be closed.
    """
    hDialog.close()


def editAnItem(key: str) -> None:
    """
    Implements the actions for the edit item button.

    Dialog implemented in separate routines.

    Arguments
    ---------
    key : str
        Short name of the parameter.
    """
    selected = dialog[key].selectedItems()
    root = dialog[key].invisibleRootItem()
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
                partial(typeUpdatePar, newStart + key[N:-4])
            )
        elif key[-4:] == "File":
            obj.clicked.disconnect()
            obj.clicked.connect(partial(selectFile, newStart + key[N:-4]))
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
        fil, fltr = QtWidgets.QFileDialog.getOpenFileName(
            caption="Select D-Flow FM Map File", filter="D-Flow FM Map Files (*map.nc)"
        )
        # getOpenFileName returns a tuple van file name and active file filter.
    elif key == "chainFileEdit":
        fil, fltr = QtWidgets.QFileDialog.getOpenFileName(
            caption="Select Chainage File", filter="Chainage Files (*.xyc)"
        )
    elif key == "riverAxisEdit":
        fil, fltr = QtWidgets.QFileDialog.getOpenFileName(
            caption="Select River Axis File", filter="River Axis Files (*.xyc)"
        )
    elif key == "fairwayEdit":
        fil, fltr = QtWidgets.QFileDialog.getOpenFileName(
            caption="Select Fairway File", filter="Fairway Files (*.xyc)"
        )
    elif key == "editSearchLineEdit":
        fil, fltr = QtWidgets.QFileDialog.getOpenFileName(
            caption="Select Search Line File", filter="Search Line Files (*.xyc)"
        )
    elif key == "editDischargeEdit":
        fil, fltr = QtWidgets.QFileDialog.getOpenFileName(
            caption="Select Simulation File", filter="Simulation File (*map.nc)"
        )
    elif key == "bankDirEdit":
        fil = QtWidgets.QFileDialog.getExistingDirectory(
            caption="Select Bank Directory"
        )
    elif key == "figureDirEdit":
        fil = QtWidgets.QFileDialog.getExistingDirectory(
            caption="Select Figure Output Directory"
        )
    elif key == "outDirEdit":
        fil = QtWidgets.QFileDialog.getExistingDirectory(
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
            fil, fltr = QtWidgets.QFileDialog.getOpenFileName(
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


def run_detection() -> None:
    """
    Run the bank line detection based on settings in the GUI.
    
    Use a dummy configuration name in the current work directory to create
    relative paths.

    Arguments
    ---------
    None
    """
    config = get_configuration()
    rootdir = os.getcwd()
    config = config_to_relative_paths(rootdir, config)
    dialog["application"].setOverrideCursor(QtCore.Qt.WaitCursor)
    matplotlib.pyplot.close("all")
    # should maybe use a separate thread for this ...
    msg = ""
    try:
        banklines_core(config, rootdir, True)
    except Exception as Ex:
        msg = str(Ex)
    dialog["application"].restoreOverrideCursor()
    if msg != "":
        print(msg)
        showError(msg)


def run_erosion() -> None:
    """
    Run the D-FAST Bank Erosion analysis based on settings in the GUI.

    Use a dummy configuration name in the current work directory to create
    relative paths.

    Arguments
    ---------
    None
    """
    config = get_configuration()
    rootdir = os.getcwd()
    config = config_to_relative_paths(rootdir, config)
    dialog["application"].setOverrideCursor(QtCore.Qt.WaitCursor)
    matplotlib.pyplot.close("all")
    # should maybe use a separate thread for this ...
    msg = ""
    # try:
    bankerosion_core(config, rootdir, True)
    # except Exception as Ex:
    #    msg = str(Ex)
    dialog["application"].restoreOverrideCursor()
    # if msg != "":
    #    print(msg)
    #    showError(msg)


def close_dialog() -> None:
    """
    Close the dialog and program.

    Arguments
    ---------
    None
    """
    matplotlib.pyplot.close("all")
    dialog["window"].close()


def menu_load_configuration() -> None:
    """
    Select and load a configuration file.

    Arguments
    ---------
    None
    """
    fil = QtWidgets.QFileDialog.getOpenFileName(
        caption="Select Configuration File", filter="Config Files (*.cfg)"
    )
    filename = fil[0]
    if filename != "":
        load_configuration(filename)


def load_configuration(filename: str) -> None:
    """
    Open a configuration file and update the GUI accordingly.

    This routines opens the specified configuration file and updates the GUI
    to reflect it contents.

    Arguments
    ---------
    filename : str
        Name of the configuration file to be opened.
    """
    if not os.path.exists(filename):
        if filename != "dfastbe.cfg":
            showError("The file '{}' does not exist!".format(filename))
        return
    absfilename = absolute_path(os.getcwd(), filename)
    rootdir = os.path.dirname(absfilename)
    config = ConfigFile.read(absfilename).config
    config = config_to_absolute_paths(rootdir, config)
    config_file = ConfigFile(config, path=absfilename)
    try:
        version = config["General"]["Version"]
    except:
        showError("No version information in the file!")
        return
    if version == "1.0":
        section = config["General"]
        dialog["chainFileEdit"].setText(section["RiverKM"])
        studyRange = config_get_range(config, "General", "Boundaries")
        dialog["startRange"].setText(str(studyRange[0]))
        dialog["endRange"].setText(str(studyRange[1]))
        dialog["bankDirEdit"].setText(section["BankDir"])
        bankFile = config_file.get_str("General", "BankFile", default="bankfile")
        dialog["bankFileName"].setText(bankFile)
        flag = config_get_bool(config, "General", "Plotting", default=True)
        dialog["makePlotsEdit"].setChecked(flag)
        flag = config_get_bool(config, "General", "SavePlots", default=True)
        dialog["savePlotsEdit"].setChecked(flag)
        flag = config_get_bool(config, "General", "SaveZoomPlots", default=False)
        dialog["saveZoomPlotsEdit"].setChecked(flag)
        zoomStepKM = config_get_float(config, "General", "ZoomStepKM", default=1.0)
        dialog["zoomPlotsRangeEdit"].setText(str(zoomStepKM))
        figDir = config_file.get_str(
            "General",
            "FigureDir",
            default=absolute_path(rootdir, "figures"),
        )
        dialog["figureDirEdit"].setText(figDir)
        flag = config_get_bool(
            config, "General", "ClosePlots", default=False
        )
        dialog["closePlotsEdit"].setChecked(flag)
        flag = config_get_bool(
            config, "General", "DebugOutput", default=False
        )
        dialog["debugOutputEdit"].setChecked(flag)

        section = config["Detect"]
        dialog["simFileEdit"].setText(section["SimFile"])
        waterDepth = config_get_float(
            config, "Detect", "WaterDepth", default=0.0,
        )
        dialog["waterDepth"].setText(str(waterDepth))
        NBank = config_get_int(
            config, "Detect", "NBank", default=0, positive=True
        )
        DLines = config_get_bank_search_distances(config, NBank)
        dialog["searchLines"].invisibleRootItem().takeChildren()
        for i in range(NBank):
            istr = str(i + 1)
            fileName = config_file.get_str("Detect", "Line" + istr)
            c1 = QtWidgets.QTreeWidgetItem(
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

        NLevel = config_get_int(
            config, "Erosion", "NLevel", default=0, positive=True
        )
        dialog["discharges"].invisibleRootItem().takeChildren()
        for i in range(NLevel):
            istr = str(i + 1)
            fileName = config_file.get_str("Erosion", "SimFile" + istr)
            prob = config_file.get_str("Erosion", "PDischarge" + istr)
            c1 = QtWidgets.QTreeWidgetItem(dialog["discharges"], [istr, fileName, prob])
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
        setParam("wavePar1", config, "Erosion", "Wave1", wave0)

        useBankType = config_get_bool(
            config, "Erosion", "Classes", default=True
        )
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
            setParam("bankType", config, "Erosion", "BankType")
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
            # tab = tabs.widget(i)
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
        showError("Unsupported version number {} in the file!".format(version))


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
    newWidget = QtWidgets.QWidget()
    newLayout = QtWidgets.QGridLayout(newWidget)
    dialog["tabs"].addTab(newWidget, "Level " + istr)

    optionalParLayout(
        newLayout, 0, istr + "_shipType", "Ship Type", selectList=shipTypes()
    )
    optionalParLayout(newLayout, 2, istr + "_shipVeloc", "Velocity [m/s]")
    optionalParLayout(newLayout, 3, istr + "_nShips", "# Ships [1/yr]")
    optionalParLayout(newLayout, 4, istr + "_shipNWaves", "# Waves [1/ship]")
    optionalParLayout(newLayout, 5, istr + "_shipDraught", "Draught [m]")
    optionalParLayout(newLayout, 6, istr + "_bankSlope", "Slope [-]")
    optionalParLayout(newLayout, 7, istr + "_bankReed", "Reed [-]")

    Label = QtWidgets.QLabel("EroVol File Name")
    dialog[istr + "_eroVol"] = Label
    newLayout.addWidget(Label, 8, 0)
    Edit = QtWidgets.QLineEdit()
    dialog[istr + "_eroVolEdit"] = Edit
    newLayout.addWidget(Edit, 8, 2)

    stretch = QtWidgets.QSpacerItem(10, 10, 13, 7)
    newLayout.addItem(stretch, 9, 0)


def optionalParLayout(
    gridLayout: PyQt5.QtWidgets.QGridLayout, row: int, key, labelString, selectList=None
) -> None:
    """
    Add a line of controls for editing an optional parameter.

    Arguments
    ---------
    gridLayout : PyQt5.QtWidgets.QGridLayout
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
    Label = QtWidgets.QLabel(labelString)
    dialog[key + "Label"] = Label
    gridLayout.addWidget(Label, row, 0)

    paramTypes = ("Use Default", "Constant", "Variable")
    Type = QtWidgets.QComboBox()
    Type.addItems(paramTypes)
    Type.currentIndexChanged.connect(partial(typeUpdatePar, key))
    dialog[key + "Type"] = Type
    gridLayout.addWidget(Type, row, 1)

    if selectList is None:
        fLayout = openFileLayout(key + "Edit", enabled=False)
        dialog[key + "Edit"].setEnabled(False)
        gridLayout.addWidget(fLayout, row, 2)
    else:
        Select = QtWidgets.QComboBox()
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
    str = config_file.get_str(group, key, default)

    try:
        val = float(str)
        dialog[field + "Type"].setCurrentText("Constant")
        if field + "Select" in dialog.keys():
            ival = int(val)
            if field == "shipType":
                ival = ival - 1
            dialog[field + "Select"].setCurrentIndex(ival)
        else:
            dialog[field + "Edit"].setText(str)
    except:
        dialog[field + "Type"].setCurrentText("Variable")
        dialog[field + "Edit"].setText(str)


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
    val = config_get_float(config, group, key, 0.0)
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
    fil = QtWidgets.QFileDialog.getSaveFileName(
        caption="Save Configuration As", filter="Config Files (*.cfg)"
    )
    filename = fil[0]
    if filename != "":
        config = get_configuration()
        rootdir = os.path.dirname(filename)
        config = config_to_relative_paths(rootdir, config)
        config = ConfigFile(config)
        config.write(filename)


def get_configuration() -> configparser.ConfigParser:
    """
    Extract a configuration from the GUI.

    Arguments
    ---------
    None

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


def showError(message):
    """
    Display an error message box with specified string.

    Arguments
    ---------
    message : str
        Text to be displayed in the message box.
    """
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Critical)
    msg.setText(message)
    msg.setWindowTitle("Error")
    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
    msg.exec_()


def menu_about_self():
    """
    Show the about dialog for D-FAST Bank Erosion.

    Arguments
    ---------
    None
    """
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Information)
    msg.setText(f"D-FAST Bank Erosion {__version__}")
    msg.setInformativeText("Copyright (c) 2020 Deltares.")
    msg.setDetailedText(gui_text("license"))
    msg.setWindowTitle(gui_text("about"))
    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
    msg.setStyleSheet("QDialogButtonBox{min-width: 400px;}")
    msg.exec_()


def menu_about_qt():
    """
    Show the about dialog for Qt.

    Arguments
    ---------
    None
    """
    QtWidgets.QApplication.aboutQt()


def menu_open_manual():
    """
    Open the user manual.

    Arguments
    ---------
    None
    """
    progloc = get_progloc()
    filename = progloc + os.path.sep + "dfastbe_usermanual.pdf"
    subprocess.Popen(filename, shell=True)


def main(config: Optional[str] = None) -> None:
    """
    Start the user interface using default settings or optional configuration.

    Arguments
    ---------
    config : Optional[str]
        Optional name of configuration file.
    """
    create_dialog()
    if not config is None:
        load_configuration(config)

    activate_dialog()
