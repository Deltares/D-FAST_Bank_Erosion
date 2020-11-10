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

from typing import Dict, Any, Optional, Tuple

from PyQt5 import QtWidgets
from PyQt5 import QtCore
import PyQt5.QtGui
import dfastbe.cli
import dfastbe.io
import dfastbe.kernel
import pathlib
import sys
import os
import configparser
from functools import partial

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
    cstr = dfastbe.io.program_texts(prefix + key)
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

    bankLines = QtWidgets.QTreeWidget(win)
    bankLines.setHeaderLabels(["Index", "FileName", "Search Distance [m]"])
    bankLines.setFont(app.font())
    bankLines.setColumnWidth(0, 50)
    bankLines.setColumnWidth(1, 200)
    # c1 = QtWidgets.QTreeWidgetItem(bankLines, ["0", "test\\filename", "50"])

    blLayout = addRemoveEditLayout(bankLines, "bankLines")
    detectLayout.addRow("Bank Lines", blLayout)


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


def shipTypes() -> Tuple[str, str, str]:
    """
    Return the tuple of ship types.

    Arguments
    ---------
    None

    Returns
    -------
    types : Tuple[str, str, str]
        Tuple of three ship types.
    """
    return ("1 (multiple barge convoy set)", "2 (RHK ship / motorship)", "3 (towboat)")


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
        selectList=(
            "0 (Beschermde oeverlijn)",
            "1 (Begroeide oeverlijn)",
            "2 (Goede klei)",
            "3 (Matig / slechte klei)",
            "4 (Zand)",
        ),
    )
    generalParLayout(eParamsLayout, 3, "bankShear", "Critical Shear Stress [N/m2]")
    bankStrengthSwitch()
    generalParLayout(eParamsLayout, 4, "bankProtect", "Protection [m]")
    generalParLayout(eParamsLayout, 5, "bankSlope", "Slope [-]")
    generalParLayout(eParamsLayout, 6, "bankReed", "Reed [-]")

    stretch = QtWidgets.QSpacerItem(10, 10, 13, 7)
    eParamsLayout.addItem(stretch, 7, 0)


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
    gridLayout: PyQt5.QtWidgets.QGridLayout, row: int, key, labelString, selectList=None
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
    formLayout: PyQt5.QtWidgets.QFormLayout, key: str, label: str
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
    fLayout = openFileLayout(key)
    formLayout.addRow(label, fLayout)


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
    if key == "bankLines":
        fileName, dist = editABankLine(key, istr)
        c1 = QtWidgets.QTreeWidgetItem(dialog["bankLines"], [istr, fileName, dist])
    elif key == "discharges":
        prob = str(1 / (nItems + 1))
        fileName, prob = editADischarge(key, istr, prob=prob)
        c1 = QtWidgets.QTreeWidgetItem(dialog["discharges"], [istr, fileName, prob])
        addTabForLevel(istr)
        dialog["refLevel"].validator().setTop(i)
    dialog[key + "Edit"].setEnabled(True)
    dialog[key + "Remove"].setEnabled(True)


def editABankLine(
    key: str, istr: str, fileName: str = "", dist: str = "50"
) -> Tuple[str, str]:
    """
    Create an edit dialog for the bank lines list.

    Arguments
    ---------
    key : str
        Short name of the parameter.
    istr : str
        String representation of the bank line in the list.
    fileName : str
        Name of the bank line file.
    dist : str
        String representation of the search distance.

    Returns
    -------
    fileName1 : str
        Updated name of the bank line file.
    dist1 : str
        Updated string representation of the search distance.
    """
    editDialog = QtWidgets.QDialog()
    editDialog.setWindowFlags(
        PyQt5.QtCore.Qt.WindowTitleHint | PyQt5.QtCore.Qt.WindowSystemMenuHint
    )
    editDialog.setWindowTitle("Edit Bank Line")
    editLayout = QtWidgets.QFormLayout(editDialog)

    label = QtWidgets.QLabel(istr)
    editLayout.addRow("Bank Line Nr", label)

    addOpenFileRow(editLayout, "editBankLine", "Bank Line File")
    dialog["editBankLine"].setText(fileName)

    searchDistance = QtWidgets.QLineEdit()
    searchDistance.setText(dist)
    searchDistance.setValidator(validator("positive_real"))
    editLayout.addRow("Search Distance [m]", searchDistance)

    done = QtWidgets.QPushButton("Done")
    done.clicked.connect(partial(close_edit, editDialog))
    # edit_SearchDistance.setValidator(validator("positive_real"))
    editLayout.addRow(" ", done)

    editDialog.exec()

    fileName = dialog["editBankLine"].text()
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
    editDialog.setWindowFlags(
        PyQt5.QtCore.Qt.WindowTitleHint | PyQt5.QtCore.Qt.WindowSystemMenuHint
    )
    editDialog.setWindowTitle("Edit Discharge")
    editLayout = QtWidgets.QFormLayout(editDialog)

    label = QtWidgets.QLabel(istr)
    editLayout.addRow("Level Nr", label)

    addOpenFileRow(editLayout, "editDischarge", "Simulation File")
    dialog["editDischarge"].setText(fileName)

    probability = QtWidgets.QLineEdit()
    probability.setText(prob)
    probability.setValidator(validator("positive_real"))
    editLayout.addRow("Probability [-]", probability)

    done = QtWidgets.QPushButton("Done")
    done.clicked.connect(partial(close_edit, editDialog))
    # edit_SearchDistance.setValidator(validator("positive_real"))
    editLayout.addRow(" ", done)

    editDialog.exec()

    fileName = dialog["editDischarge"].text()
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
        if key == "bankLines":
            fileName = selected[0].text(1)
            dist = selected[0].text(2)
            fileName, dist = editABankLine(key, istr, fileName=fileName, dist=dist)
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
    elif key == "bankLines":
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
    Select a file via a file selection dialog.

    Arguments
    ---------
    key : str
        Short name of the parameter.
    """
    if not dialog[key + "File"].hasFocus():
        # in the add/edit dialogs, the selectFile is triggered when the user presses enter in one of the lineEdit boxes ...
        # don't trigger the actual selectFile
        fil = [""]
    elif key == "simFile":
        fil = QtWidgets.QFileDialog.getOpenFileName(
            caption="Select D-Flow FM Map File", filter="D-Flow FM Map Files (*map.nc)"
        )
        # getOpenFileName returns a tuple van file name and active file filter.
    elif key == "chainFile":
        fil = QtWidgets.QFileDialog.getOpenFileName(
            caption="Select Chainage File", filter="Chainage Files (*.xyc)"
        )
    elif key == "riverAxis":
        fil = QtWidgets.QFileDialog.getOpenFileName(
            caption="Select River Axis File", filter="River Axis Files (*.xyc)"
        )
    elif key == "fairway":
        fil = QtWidgets.QFileDialog.getOpenFileName(
            caption="Select Fairway File", filter="Fairway Files (*.xyc)"
        )
    elif key == "editBankLine":
        fil = QtWidgets.QFileDialog.getOpenFileName(
            caption="Select Bank Line File", filter="Bank Line Files (*.xyc)"
        )
    elif key == "editDischarge":
        fil = QtWidgets.QFileDialog.getOpenFileName(
            caption="Select Simulation File", filter="Simulation File (*map.nc)"
        )
    elif key == "bankDir":
        fil = QtWidgets.QFileDialog.getExistingDirectory(
            caption="Select Bank Directory"
        )
        # getExistingDirectory just returns the folder name; make consistent with getOpenFileName.
        fil = fil
    elif key == "outDir":
        fil = QtWidgets.QFileDialog.getExistingDirectory(
            caption="Select Output Directory"
        )
        fil = fil
    else:
        if key[-4:] == "Edit":
            rkey = key[:-4]
            nr = ""
            while rkey[0] in "1234567890":
                nr = nr + rkey[0]
                rkey = rkey[1:]
            if not nr == "":
                nr = " for Level " + nr
            fil = QtWidgets.QFileDialog.getOpenFileName(
                caption="Select Parameter File" + nr, filter="Parameter File (*.)"
            )
        else:
            print(key)
            fil = [""]
    if fil[0] != "":
        dialog[key].setText(fil[0])


def selectFolder(key: str) -> None:
    """
    Select a folder via a folder selection dialog.

    Arguments
    ---------
    key : str
        Short name of the parameter.
    """
    if not dialog[key + "Folder"].hasFocus():
        # in the add/edit dialogs, the selectFolder is triggered when the user presses enter in one of the lineEdit boxes ...
        # don't trigger the actual selectFile
        folder = [""]
    else:
        folder = QtWidgets.QFileDialog.getExistingDirectory(caption="Select Folder")
    if folder[0] != "":
        dialog[key].setText(folder[0])


def run_detection() -> None:
    """
    Run the bank line detection based on settings in the GUI.

    Arguments
    ---------
    None
    """
    config = get_configuration()
    dfastbe.cli.banklines_core(config)


def run_erosion() -> None:
    """
    Run the D-FAST Bank Erosion analysis based on settings in the GUI.

    Arguments
    ---------
    None
    """
    config = get_configuration()
    dfastbe.cli.bankerosion_core(config)


def close_dialog() -> None:
    """
    Close the dialog and program.

    Arguments
    ---------
    None
    """
    dialog["window"].close()


def menu_load_configuration() -> None:
    """
    Select and load a configuration file.

    Arguments
    ---------
    None
    """
    fil = QtWidgets.QFileDialog.getOpenFileName(
        caption="Select Configuration File", filter="Config Files (*.ini)"
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
    config = dfastbe.io.read_config(filename)
    config = config_to_absolute_paths(filename, config)
    try:
        version = config["General"]["Version"]
    except:
        showError("No version information in the file!")
        return
    if version == "1.0":
        section = config["General"]
        dialog["chainFile"].setText(section["RiverKM"])
        studyRange = dfastbe.io.config_get_range(config, "General", "Boundaries")
        dialog["startRange"].setText(str(studyRange[0]))
        dialog["endRange"].setText(str(studyRange[1]))
        dialog["bankDir"].setText(section["BankDir"])
        dialog["bankFileName"].setText(section["BankFile"])

        section = config["Detect"]
        dialog["simFile"].setText(section["SimFile"])
        dialog["waterDepth"].setText(section["WaterDepth"])
        NBank = dfastbe.io.config_get_int(
            config, "Detect", "NBank", default=0, positive=True
        )
        DLines = dfastbe.io.config_get_bank_search_distances(config, NBank)
        dialog["bankLines"].invisibleRootItem().takeChildren()
        for i in range(NBank):
            istr = str(i + 1)
            fileName = dfastbe.io.config_get_str(config, "Detect", "Line" + istr)
            c1 = QtWidgets.QTreeWidgetItem(
                dialog["bankLines"], [istr, fileName, str(DLines[i])]
            )

        section = config["Erosion"]
        dialog["tErosion"].setText(section["TErosion"])
        dialog["riverAxis"].setText(section["RiverAxis"])
        dialog["fairway"].setText(section["Fairway"])
        dialog["chainageOutStep"].setText(section["OutputInterval"])
        dialog["outDir"].setText(section["OutputDir"])
        dialog["newBankFile"].setText(section["BankNew"])
        dialog["newEqBankFile"].setText(section["BankEq"])
        txt = dfastbe.io.config_get_str(
            config, "Erosion", "EroVol", default="erovol_standard.evo"
        )
        dialog["eroVol"].setText(txt)
        txt = dfastbe.io.config_get_str(
            config, "Erosion", "EroVolEqui", default="erovol_eq.evo"
        )
        dialog["eroVolEqui"].setText(txt)

        NLevel = dfastbe.io.config_get_int(
            config, "Erosion", "NLevel", default=0, positive=True
        )
        dialog["discharges"].invisibleRootItem().takeChildren()
        for i in range(NLevel):
            istr = str(i + 1)
            fileName = dfastbe.io.config_get_str(config, "Erosion", "SimFile" + istr)
            prob = dfastbe.io.config_get_str(config, "Erosion", "PDischarge" + istr)
            c1 = QtWidgets.QTreeWidgetItem(dialog["discharges"], [istr, fileName, prob])
        dialog["refLevel"].validator().setTop(NLevel)
        dialog["refLevel"].setText(section["RefLevel"])

        setParam("shipType", config, "Erosion", "ShipType")
        setParam("shipVeloc", config, "Erosion", "VShip")
        setParam("nShips", config, "Erosion", "NShip")
        setParam("shipNWaves", config, "Erosion", "NWave", "5")
        setParam("shipDraught", config, "Erosion", "Draught")
        setParam("wavePar0", config, "Erosion", "Wave0", "200.0")
        wave0 = dfastbe.io.config_get_str(config, "Erosion", "Wave0", "200.0")
        setParam("wavePar1", config, "Erosion", "Wave1", wave0)

        useBankType = dfastbe.io.config_get_bool(
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
        setParam("bankProtect", config, "Erosion", "ProtectLevel", "-1000")
        setParam("bankSlope", config, "Erosion", "Slope", "20.0")
        setParam("bankReed", config, "Erosion", "Reed", "0.0")

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
            txt = dfastbe.io.config_get_str(
                config, "Erosion", "EroVol" + istr, default=""
            )
            dialog[istr + "_eroVolEdit"].setText(txt)

    else:
        showError("Unsupported version number {} in the file!".format(version))


def config_to_absolute_paths(
    filename: str, config: configparser.ConfigParser
) -> configparser.ConfigParser:
    """
    Convert a configuration object to contain absolute paths (for editing).

    Arguments
    ---------
    filename : str
        The name of the file: all relative paths in the configuration will be assumed relative to this.
    config : configparser.ConfigParser
        Configuration for the D-FAST Bank Erosion analysis with absolute or relative paths.

    Returns
    -------
    config1 : configparser.ConfigParser
        Configuration for the D-FAST Bank Erosion analysis with only absolute paths.
    """
    rootdir = os.path.dirname(filename)

    if "General" in config:
        config = parameter_absolute_path(config, "General", "RiverKM", rootdir)
        config = parameter_absolute_path(config, "General", "BankDir", rootdir)

    if "Detect" in config:
        config = parameter_absolute_path(config, "Detect", "SimFile", rootdir)
        i = 0
        while True:
            i = i + 1
            Line = "Line" + str(i)
            if Line in config["Detect"]:
                config = parameter_absolute_path(config, "Detect", Line, rootdir)
            else:
                break

    if "Erosion" in config:
        config = parameter_absolute_path(config, "Erosion", "RiverAxis", rootdir)
        config = parameter_absolute_path(config, "Erosion", "Fairway", rootdir)
        config = parameter_absolute_path(config, "Erosion", "OutputDir", rootdir)
        config = parameter_absolute_path(config, "Erosion", "ShipType", rootdir)
        config = parameter_absolute_path(config, "Erosion", "VShip", rootdir)
        config = parameter_absolute_path(config, "Erosion", "NWave", rootdir)
        config = parameter_absolute_path(config, "Erosion", "Draught", rootdir)
        config = parameter_absolute_path(config, "Erosion", "Wave0", rootdir)
        config = parameter_absolute_path(config, "Erosion", "Wave1", rootdir)
        NLevel = dfastbe.io.config_get_int(config, "Erosion", "NLevel", default=0)
        for i in range(NLevel):
            istr = str(i + 1)
            config = parameter_absolute_path(
                config, "Erosion", "ShipType" + istr, rootdir
            )
            config = parameter_absolute_path(config, "Erosion", "VShip" + istr, rootdir)
            config = parameter_absolute_path(config, "Erosion", "NShip" + istr, rootdir)
            config = parameter_absolute_path(config, "Erosion", "NWave" + istr, rootdir)
            config = parameter_absolute_path(
                config, "Erosion", "Draught" + istr, rootdir
            )
            config = parameter_absolute_path(config, "Erosion", "Slope" + istr, rootdir)
            config = parameter_absolute_path(config, "Erosion", "Reed" + istr, rootdir)

    return config


def parameter_absolute_path(
    config: configparser.ConfigParser, group: str, key: str, rootdir: str
) -> configparser.ConfigParser:
    """
    Convert a parameter value to contain an absolute path.

    Determine whether the string represents a number.
    If not, try to convert to an absolute path.

    Arguments
    ---------
    config : configparser.ConfigParser
        Configuration for the D-FAST Bank Erosion analysis.
    group : str
        Name of the group in the configuration.
    key : str
        Name of the key in the configuration.
    rootdir : str
        The path to be used as base for the absolute paths.

    Returns
    -------
    config1 : configparser.ConfigParser
        Updated configuration for the D-FAST Bank Erosion analysis.
    """
    if key in config[group]:
        valstr = config[group][key]
        try:
            val = float(valstr)
        except:
            config[group][key] = dfastbe.io.absolute_path(rootdir, valstr)
    return config


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
    str = dfastbe.io.config_get_str(config, group, key, default)
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
    str = dfastbe.io.config_get_str(config, group, key, "")
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
        caption="Save Configuration As", filter="Config Files (*.ini)"
    )
    filename = fil[0]
    if filename != "":
        config = get_configuration()
        config = config_to_relative_paths(filename, config)
        dfastbe.io.write_config(filename, config)


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
    config["General"]["RiverKM"] = dialog["chainFile"].text()
    config["General"]["Boundaries"] = (
        dialog["startRange"].text() + ":" + dialog["endRange"].text()
    )
    config["General"]["BankDir"] = dialog["bankDir"].text()
    config["General"]["BankFile"] = dialog["bankFileName"].text()

    config.add_section("Detect")
    config["Detect"]["SimFile"] = dialog["simFile"].text()
    config["Detect"]["WaterDepth"] = dialog["waterDepth"].text()
    nbank = dialog["bankLines"].topLevelItemCount()
    config["Detect"]["NBank"] = str(nbank)
    dlines = "[ "
    for i in range(nbank):
        istr = str(i + 1)
        config["Detect"]["Line" + istr] = dialog["bankLines"].topLevelItem(i).text(1)
        dlines += dialog["bankLines"].topLevelItem(i).text(2) + ", "
    dlines = dlines[:-2] + " ]"
    config["Detect"]["DLines"] = dlines

    config.add_section("Erosion")
    config["Erosion"]["TErosion"] = dialog["tErosion"].text()
    config["Erosion"]["RiverAxis"] = dialog["riverAxis"].text()
    config["Erosion"]["Fairway"] = dialog["fairway"].text()
    config["Erosion"]["OutputInterval"] = dialog["chainageOutStep"].text()
    config["Erosion"]["OutputDir"] = dialog["outDir"].text()
    config["Erosion"]["BankNew"] = dialog["newBankFile"].text()
    config["Erosion"]["BankEq"] = dialog["newEqBankFile"].text()
    config["Erosion"]["EroVol"] = dialog["eroVol"].text()
    config["Erosion"]["EroVolEqui"] = dialog["eroVolEqui"].text()

    if dialog["shipTypeType"].currentText() == "Constant":
        config["Erosion"]["ShipType"] = (
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

    if dialog["strengthPar"].currentText == "Bank Type":
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


def config_to_relative_paths(
    filename: str, config: configparser.ConfigParser
) -> configparser.ConfigParser:
    """
    Convert a configuration object to contain relative paths (for saving).

    Arguments
    ---------
    filename : str
        The name of the file: all paths will be defined relative to this.
    config : configparser.ConfigParser
        Configuration for the D-FAST Bank Erosion analysis with only absolute paths.

    Returns
    -------
    config1 : configparser.ConfigParser
        Updated configuration for D-FAST Bank Erosion analysis with as much as possible relative paths.
    """
    rootdir = os.path.dirname(filename)

    if "General" in config:
        config = parameter_relative_path(config, "General", "RiverKM", rootdir)
        config = parameter_relative_path(config, "General", "BankDir", rootdir)

    if "Detect" in config:
        config = parameter_relative_path(config, "Detect", "SimFile", rootdir)
        i = 0
        while True:
            i = i + 1
            Line = "Line" + str(i)
            if Line in config["Detect"]:
                config = parameter_relative_path(config, "Detect", Line, rootdir)
            else:
                break

    if "Erosion" in config:
        config = parameter_relative_path(config, "Erosion", "RiverAxis", rootdir)
        config = parameter_relative_path(config, "Erosion", "Fairway", rootdir)
        config = parameter_relative_path(config, "Erosion", "OutputDir", rootdir)
        config = parameter_relative_path(config, "Erosion", "ShipType", rootdir)
        config = parameter_relative_path(config, "Erosion", "VShip", rootdir)
        config = parameter_relative_path(config, "Erosion", "NWave", rootdir)
        config = parameter_relative_path(config, "Erosion", "Draught", rootdir)
        config = parameter_relative_path(config, "Erosion", "Wave0", rootdir)
        config = parameter_relative_path(config, "Erosion", "Wave1", rootdir)
        NLevel = dfastbe.io.config_get_int(config, "Erosion", "NLevel", default=0)
        for i in range(NLevel):
            istr = str(i + 1)
            config = parameter_relative_path(
                config, "Erosion", "ShipType" + istr, rootdir
            )
            config = parameter_relative_path(config, "Erosion", "VShip" + istr, rootdir)
            config = parameter_relative_path(config, "Erosion", "NShip" + istr, rootdir)
            config = parameter_relative_path(config, "Erosion", "NWave" + istr, rootdir)
            config = parameter_relative_path(
                config, "Erosion", "Draught" + istr, rootdir
            )
            config = parameter_relative_path(config, "Erosion", "Slope" + istr, rootdir)
            config = parameter_relative_path(config, "Erosion", "Reed" + istr, rootdir)

    return config


def parameter_relative_path(
    config: configparser.ConfigParser, group: str, key: str, rootdir: str
) -> configparser.ConfigParser:
    """
    Convert a parameter value to contain a relative path.

    Determine whether the string represents a number.
    If not, try to convert to a relative path.

    Arguments
    ---------
    config : configparser.ConfigParser
        Configuration for the D-FAST Bank Erosion analysis.
    group : str
        Name of the group in the configuration.
    key : str
        Name of the key in the configuration.
    rootdir : str
        The path to be used as base for the relative paths.

    Returns
    -------
    config1 : configparser.ConfigParser
        Updated configuration for the D-FAST Bank Erosion analysis.
    """
    if key in config[group]:
        valstr = config[group][key]
        try:
            val = float(valstr)
        except:
            config[group][key] = dfastbe.io.relative_path(rootdir, valstr)
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
    msg.setText("D-FAST Bank Erosion " + dfastbe.__version__)
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
