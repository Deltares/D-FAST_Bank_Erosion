# coding: utf-8
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

from PyQt5 import QtWidgets
from PyQt5 import QtCore
import PyQt5.QtGui
import dfastbe_io
import dfastbe_kernel
import pathlib
import sys
import configparser
from functools import partial


def create_dialog():
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

    detect = QtWidgets.QPushButton("Detect", win)
    detect.clicked.connect(run_detection)
    buttonBarLayout.addWidget(detect)

    erode = QtWidgets.QPushButton("Erode", win)
    erode.clicked.connect(run_erosion)
    buttonBarLayout.addWidget(erode)

    done = QtWidgets.QPushButton("Close", win)
    done.clicked.connect(close_dialog)
    buttonBarLayout.addWidget(done)

    addGeneralTab(tabs, win)
    addDetectTab(tabs, win, app)
    addErosionTab(tabs, win, app)
    addShippingTab(tabs, win)
    addBankTab(tabs, win)


def createMenus(menubar):
    menu = menubar.addMenu("&File")
    item = menu.addAction("&Open...")
    item.triggered.connect(load_configuration)
    item = menu.addAction("&Save As...")
    item.triggered.connect(save_configuration)
    menu.addSeparator()
    item = menu.addAction("&Close")
    item.triggered.connect(close_dialog)

    menu = menubar.addMenu("&Help")
    item = menu.addAction("Version")
    item.triggered.connect(menu_about_self)
    item = menu.addAction("About Qt")
    item.triggered.connect(menu_about_qt)


def addGeneralTab(tabs, win):
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

    bankDir = QtWidgets.QLineEdit(win)
    dialog["bankDir"] = bankDir
    generalLayout.addRow("Bank Directory", bankDir)

    bankFileName = QtWidgets.QLineEdit(win)
    dialog["bankFileName"] = bankFileName
    generalLayout.addRow("Bank File Name", bankFileName)


def addDetectTab(tabs, win, app):
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
    bankLines.setColumnWidth(0,50)
    bankLines.setColumnWidth(1,200)
    #c1 = QtWidgets.QTreeWidgetItem(bankLines, ["0", "test\\filename", "50"])

    blLayout = addRemoveEditLayout(bankLines, "bankLines")
    detectLayout.addRow("Bank Lines", blLayout)


def addErosionTab(tabs, win, app):
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
    discharges.setColumnWidth(0,50)
    discharges.setColumnWidth(1,250)
    #c1 = QtWidgets.QTreeWidgetItem(discharges, ["0", "test\\filename", "0.5"])

    disLayout = addRemoveEditLayout(discharges, "discharges")
    erosionLayout.addRow("Discharges", disLayout)

    refLevel = QtWidgets.QLineEdit(win)
    refLevel.setValidator(PyQt5.QtGui.QIntValidator(1,1))
    dialog["refLevel"] = refLevel
    erosionLayout.addRow("Reference Level", refLevel)
    
    chainageOutStep = QtWidgets.QLineEdit(win)
    chainageOutStep.setValidator(validator("positive_real"))
    dialog["chainageOutStep"] = chainageOutStep
    erosionLayout.addRow("Chainage Output Step [km]", chainageOutStep)

    outDir = QtWidgets.QLineEdit(win)
    dialog["outDir"] = outDir
    erosionLayout.addRow("Output Directory", outDir)

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


def addShippingTab(tabs, win):
    eParamsWidget = QtWidgets.QWidget()
    eParamsLayout = QtWidgets.QGridLayout(eParamsWidget)
    tabs.addTab(eParamsWidget, "Shipping Parameters")

    generalParLayout(eParamsLayout, 0, "shipType", "Ship Type", selectList = shipTypes())
    generalParLayout(eParamsLayout, 2, "shipVeloc", "Velocity [m/s]")
    generalParLayout(eParamsLayout, 3, "nShips", "# Ships [1/yr]")
    generalParLayout(eParamsLayout, 4, "shipNWaves", "# Waves [1/ship]")
    generalParLayout(eParamsLayout, 5, "shipDraught", "Draught [m]")
    generalParLayout(eParamsLayout, 6, "wavePar0", "Wave0 [m]")
    generalParLayout(eParamsLayout, 7, "wavePar1", "Wave1 [m]")

    stretch = QtWidgets.QSpacerItem(10, 10, 13, 7)
    eParamsLayout.addItem(stretch, 8, 0)


def shipTypes():
    return ("1 (multiple barge convoy set)","2 (RHK ship / motorship)","3 (towboat)")


def addBankTab(tabs, win):
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

    generalParLayout(eParamsLayout, 1, "bankType", "Bank Type", selectList = ("0 (Beschermde oeverlijn)", "1 (Begroeide oeverlijn)", "2 (Goede klei)", "3 (Matig / slechte klei)", "4 (Zand)"))
    generalParLayout(eParamsLayout, 3, "bankShear", "Critical Shear Stress [N/m2]")
    bankStrengthSwitch()
    generalParLayout(eParamsLayout, 4, "bankProtect", "Protection [m]")
    generalParLayout(eParamsLayout, 5, "bankSlope", "Slope [-]")
    generalParLayout(eParamsLayout, 6, "bankReed", "Reed [-]")

    stretch = QtWidgets.QSpacerItem(10, 10, 13, 7)
    eParamsLayout.addItem(stretch, 7, 0)


def bankStrengthSwitch():
    type = dialog["strengthPar"].currentText()
    if type == "Bank Type":
        dialog["bankType"].setEnabled(True)
        dialog["bankTypeType"].setEnabled(True)
        typeUpdateGenPar("bankType")
        dialog["bankShear"].setEnabled(False)
        dialog["bankShearType"].setEnabled(False)
        dialog["bankShearEdit"].setText("")
        dialog["bankShearEdit"].setEnabled(False)
        dialog["bankShearEditFile"].setEnabled(False)
    elif type == "Critical Shear Stress":
        dialog["bankShear"].setEnabled(True)
        dialog["bankShearType"].setEnabled(True)
        dialog["bankShearEdit"].setEnabled(True)
        typeUpdateGenPar("bankShear")
        dialog["bankType"].setEnabled(False)
        dialog["bankTypeType"].setEnabled(False)
        dialog["bankTypeSelect"].setEnabled(False)
        dialog["bankTypeEdit"].setText("")
        dialog["bankTypeEdit"].setEnabled(False)
        dialog["bankTypeEditFile"].setEnabled(False)


def validator(str):
    if str == "positive_real":
        validator = PyQt5.QtGui.QDoubleValidator()
        validator.setBottom(0)
    return validator


def activate_dialog():
    app = dialog["application"]
    win = dialog["window"]
    win.show()
    sys.exit(app.exec_())


def generalParLayout(gridLayout, row, key, labelString, selectList = None):
    Label = QtWidgets.QLabel(labelString)
    dialog[key] = Label
    gridLayout.addWidget(Label, row, 0)

    paramTypes = ("Constant", "Variable")
    Type = QtWidgets.QComboBox()
    Type.addItems(paramTypes)
    Type.currentIndexChanged.connect(partial(typeUpdateGenPar,key))
    dialog[key + "Type"] = Type
    gridLayout.addWidget(Type, row, 1)

    if selectList is None:
        fLayout = openFileLayout(key + "Edit", enabled = False)
        gridLayout.addWidget(fLayout, row, 2)
    else:
        Select = QtWidgets.QComboBox()
        Select.addItems(selectList)
        dialog[key + "Select"] = Select
        gridLayout.addWidget(Select, row, 2)

        fLayout = openFileLayout(key + "Edit", enabled = False)
        dialog[key + "Edit"].setEnabled(False)
        gridLayout.addWidget(fLayout, row + 1, 2)

    typeUpdateGenPar(key)


def typeUpdateGenPar(key):
    type = dialog[key + "Type"].currentText()
    if type == "Constant":
        dialog[key + "Edit"].setText("")
        dialog[key + "EditFile"].setEnabled(False)
        if key + "Select" in dialog.keys():
            dialog[key + "Select"].setEnabled(True)
            dialog[key + "Edit"].setEnabled(False)
        else:
            dialog[key + "Edit"].setValidator(validator("positive_real"))
    elif type == "Variable":
        if key + "Select" in dialog.keys():
            dialog[key + "Select"].setEnabled(False)
            dialog[key + "Edit"].setEnabled(True)
        dialog[key + "EditFile"].setEnabled(True)
        dialog[key + "Edit"].setText("")
        dialog[key + "Edit"].setValidator(None)


def addOpenFileRow(formLayout, key, label):
    fLayout = openFileLayout(key)
    formLayout.addRow(label, fLayout)


def openFileLayout(key, enabled = True):
    parent = QtWidgets.QWidget()
    gridly = QtWidgets.QGridLayout(parent)
    gridly.setContentsMargins(0, 0, 0, 0)

    myWidget = QtWidgets.QLineEdit()
    dialog[key] = myWidget
    gridly.addWidget(myWidget, 0, 0)

    openFile = QtWidgets.QPushButton(PyQt5.QtGui.QIcon("open.png"), "")
    openFile.clicked.connect(partial(selectFile, key))
    openFile.setEnabled(enabled)
    dialog[key+"File"] = openFile
    gridly.addWidget(openFile, 0, 2)

    return parent


def addRemoveEditLayout(mainWidget, key):
    parent = QtWidgets.QWidget()
    gridly = QtWidgets.QGridLayout(parent)
    gridly.setContentsMargins(0, 0, 0, 0)

    dialog[key] = mainWidget
    gridly.addWidget(mainWidget,0,0)

    buttonBar = QtWidgets.QWidget()
    buttonBarLayout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom, buttonBar)
    buttonBarLayout.setContentsMargins(0, 0, 0, 0)
    gridly.addWidget(buttonBar,0,1)

    addBtn = QtWidgets.QPushButton(PyQt5.QtGui.QIcon("add.png"), "")
    addBtn.clicked.connect(partial(addAnItem, key))
    dialog[key + "Add"] = addBtn
    buttonBarLayout.addWidget(addBtn)
    
    editBtn = QtWidgets.QPushButton(PyQt5.QtGui.QIcon("edit.png"), "")
    editBtn.clicked.connect(partial(editAnItem, key))
    editBtn.setEnabled(False)
    dialog[key + "Edit"] = editBtn
    buttonBarLayout.addWidget(editBtn)
    
    delBtn = QtWidgets.QPushButton(PyQt5.QtGui.QIcon("remove.png"), "")
    delBtn.clicked.connect(partial(removeAnItem, key))
    delBtn.setEnabled(False)
    dialog[key + "Remove"] = delBtn
    buttonBarLayout.addWidget(delBtn)
    
    stretch = QtWidgets.QSpacerItem(10, 10, 13, 7)
    buttonBarLayout.addItem(stretch)

    return parent


def addAnItem(key):
    nItems = dialog[key].invisibleRootItem().childCount()
    i = nItems + 1
    istr = str(i)
    if key == "bankLines":
        fileName, dist = editABankLine(key, istr)
        c1 = QtWidgets.QTreeWidgetItem(dialog["bankLines"],
            [istr, fileName, dist]
        )
    elif key == "discharges":
        prob = str(1/(nItems + 1))
        fileName, prob = editADischarge(key, istr, prob = prob)
        c1 = QtWidgets.QTreeWidgetItem(dialog["discharges"],
            [istr, fileName, prob]
        )
        addTabForLevel(istr)
        dialog["refLevel"].validator().setTop(i)
    dialog[key + "Edit"].setEnabled(True)
    dialog[key + "Remove"].setEnabled(True)


def editABankLine(key, istr, fileName = "", dist = "50"):
    editDialog = QtWidgets.QDialog()
    editDialog.setWindowFlags(PyQt5.QtCore.Qt.WindowTitleHint | PyQt5.QtCore.Qt.WindowSystemMenuHint)
    editDialog.setWindowTitle("Edit Bank Line")
    editLayout = QtWidgets.QFormLayout(editDialog)
    
    label= QtWidgets.QLabel(istr)
    editLayout.addRow("Bank Line Nr", label)

    addOpenFileRow(editLayout, "editBankLine", "Bank Line File")
    dialog["editBankLine"].setText(fileName)

    searchDistance = QtWidgets.QLineEdit()
    searchDistance.setText(dist)
    searchDistance.setValidator(validator("positive_real"))
    editLayout.addRow("Search Distance [m]", searchDistance)

    done = QtWidgets.QPushButton("Done")
    done.clicked.connect(partial(close_edit, editDialog))
    #edit_SearchDistance.setValidator(validator("positive_real"))
    editLayout.addRow(" ", done)

    editDialog.exec()
    
    fileName = dialog["editBankLine"].text()
    dist = searchDistance.text()
    return fileName, dist


def editADischarge(key, istr, fileName = "", prob = ""):
    editDialog = QtWidgets.QDialog()
    editDialog.setWindowFlags(PyQt5.QtCore.Qt.WindowTitleHint | PyQt5.QtCore.Qt.WindowSystemMenuHint)
    editDialog.setWindowTitle("Edit Discharge")
    editLayout = QtWidgets.QFormLayout(editDialog)
    
    label= QtWidgets.QLabel(istr)
    editLayout.addRow("Level Nr", label)

    addOpenFileRow(editLayout, "editDischarge", "Simulation File")
    dialog["editDischarge"].setText(fileName)

    probability = QtWidgets.QLineEdit()
    probability.setText(prob)
    probability.setValidator(validator("positive_real"))
    editLayout.addRow("Probability [-]", probability)

    done = QtWidgets.QPushButton("Done")
    done.clicked.connect(partial(close_edit, editDialog))
    #edit_SearchDistance.setValidator(validator("positive_real"))
    editLayout.addRow(" ", done)

    editDialog.exec()
    
    fileName = dialog["editDischarge"].text()
    prob = probability.text()
    return fileName, prob


def close_edit(hDialog):
    hDialog.close()


def editAnItem(key):
    selected = dialog[key].selectedItems()
    root = dialog[key].invisibleRootItem()
    if len(selected)>0:
        istr = selected[0].text(0)
        if key == "bankLines":
            fileName = selected[0].text(1)
            dist = selected[0].text(2)
            fileName, dist = editABankLine(key, istr, fileName = fileName, dist = dist)
            selected[0].setText(1, fileName)
            selected[0].setText(2, dist)
        elif key == "discharges":
            fileName = selected[0].text(1)
            prob = selected[0].text(2)
            fileName, prob = editADischarge(key, istr, fileName = fileName, prob = prob)
            selected[0].setText(1, fileName)
            selected[0].setText(2, prob)


def removeAnItem(key):
    selected = dialog[key].selectedItems()
    root = dialog[key].invisibleRootItem()
    if len(selected)>0:
        istr = selected[0].text(0)
        root.removeChild(selected[0])
        i = int(istr) - 1
        for j in range(i, root.childCount()):
           root.child(j).setText(0, str(j+1))
    else:
        istr = ""
    if root.childCount()==0:
        dialog[key + "Edit"].setEnabled(False)
        dialog[key + "Remove"].setEnabled(False)
    if istr == "":
        pass
    elif key == "bankLines":
        pass
    elif key == "discharges":
        tabs = dialog["tabs"]
        renumber = False
        dialog["refLevel"].validator().setTop(root.childCount())
        for j in range(tabs.count()):
            if renumber:
                tabs.setTabText(j - 1, "Level " + str(j + dj))
                updateTabKeys(j + dj +1)
            elif tabs.tabText(j) == "Level " + istr:
                tabs.removeTab(j)
                renumber = True
                dj = i - j


def updateTabKeys(i):
    iStart = str(i) + "_"
    newStart = str(i-1) + "_"
    N = len(iStart)
    keys = [key for key in dialog.keys() if key[:N] == iStart]
    for key in keys:
        obj = dialog.pop(key)
        if key[-4:] == "Type":
            obj.currentIndexChanged.disconnect()
            obj.currentIndexChanged.connect(partial(typeUpdateOptPar, newStart + key[N:-4]))
        elif key[-4:] == "File":
            obj.clicked.disconnect()
            obj.clicked.connect(partial(selectFile, newStart + key[N:-4]))
        dialog[newStart + key[N:]] = obj


def selectFile(key):
    if not dialog[key + "File"].hasFocus():
        # in the add/edit dialogs, the selectFile is triggered when the user presses enter in one of the lineEdit boxes ...
        # don't trigger the actual selectFile
        fil = [""]
    elif key == "simFile":
        fil = QtWidgets.QFileDialog.getOpenFileName(
            caption="Select D-Flow FM Map File", filter="D-Flow FM Map Files (*map.nc)"
        )
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


def run_detection():
    showError("Not yet implemented!")


def run_erosion():
    showError("Not yet implemented!")


def close_dialog():
    dialog["window"].close()


def load_configuration():
    fil = QtWidgets.QFileDialog.getOpenFileName(
        caption="Select Configuration File", filter="Config Files (*.ini)"
    )
    filename = fil[0]
    if filename == "":
        return

    config = dfastbe_io.read_config(filename)
    try:
        version = config["General"]["Version"]
    except:
        showError("No version information in the file!")
        return
    if version == "1.0":
        section = config["General"]
        dialog["chainFile"].setText(section["RiverKM"])
        studyRange = dfastbe_io.config_get_range(config, "General", "Boundaries")
        dialog["startRange"].setText(str(studyRange[0]))
        dialog["endRange"].setText(str(studyRange[1]))
        dialog["bankDir"].setText(section["BankDir"])
        dialog["bankFileName"].setText(section["BankFile"])

        section = config["Detect"]
        dialog["simFile"].setText(section["SimFile"])
        dialog["waterDepth"].setText(section["WaterDepth"])
        NBank = dfastbe_io.config_get_int(config, "Detect", "NBank", default = 0, positive = True)
        DLines = dfastbe_io.config_get_bank_search_distances(config, NBank)
        dialog["bankLines"].invisibleRootItem().takeChildren()
        for i in range(NBank):
            istr =str(i+1)
            fileName = dfastbe_io.config_get_str(config, "Detect", "Line" + istr)
            c1 = QtWidgets.QTreeWidgetItem(dialog["bankLines"],
                [istr, fileName, str(DLines[i])]
            )

        section = config["Erosion"]
        dialog["tErosion"].setText(section["TErosion"])
        dialog["riverAxis"].setText(section["RiverAxis"])
        dialog["fairway"].setText(section["Fairway"])
        dialog["chainageOutStep"].setText(section["OutputInterval"])
        dialog["outDir"].setText(section["OutputDir"])
        dialog["newBankFile"].setText(section["BankNew"])
        dialog["newEqBankFile"].setText(section["BankEq"])
        txt = dfastbe_io.config_get_str(config, "Erosion", "EroVol", default = "erovol_standard.evo")
        dialog["eroVol"].setText(txt)
        txt = dfastbe_io.config_get_str(config, "Erosion", "EroVolEqui", default = "erovol_eq.evo")
        dialog["eroVolEqui"].setText(txt)

        NLevel = dfastbe_io.config_get_int(config, "Erosion", "NLevel", default = 0, positive = True)
        dialog["discharges"].invisibleRootItem().takeChildren()
        for i in range(NLevel):
            istr = str(i+1)
            fileName = dfastbe_io.config_get_str(config, "Erosion", "SimFile" + istr)
            prob = dfastbe_io.config_get_str(config, "Erosion", "PDischarge" + istr)
            c1 = QtWidgets.QTreeWidgetItem(dialog["discharges"],
                [istr, fileName, prob]
            )
        dialog["refLevel"].validator().setTop(NLevel)
        dialog["refLevel"].setText(section["RefLevel"])

        setParam("shipType", config, "Erosion", "ShipType")
        setParam("shipVeloc", config, "Erosion", "VShip")
        setParam("nShips", config, "Erosion", "NShip")
        setParam("shipNWaves", config, "Erosion", "NWave", "5")
        setParam("shipDraught", config, "Erosion", "Draught")
        setParam("wavePar0", config, "Erosion", "Wave0", "200.0")
        wave0 = dfastbe_io.config_get_str(config, "Erosion", "Wave0", "200.0")
        setParam("wavePar1", config, "Erosion", "Wave1", wave0)

        useBankType = dfastbe_io.config_get_bool(config, "Erosion", "Classes", default = True)
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
        for i in range(tabs.count()-1,4,-1):
            #tab = tabs.widget(i)
            tabs.removeTab(i)

        for i in range(NLevel):
            istr = str(i+1)
            addTabForLevel(istr)
            setOptParam(istr + "_shipType", config, "Erosion", "ShipType" + istr)
            setOptParam(istr + "_shipVeloc", config, "Erosion", "VShip" + istr)
            setOptParam(istr + "_nShips", config, "Erosion", "NShip" + istr)
            setOptParam(istr + "_shipNWaves", config, "Erosion", "NWave" + istr)
            setOptParam(istr + "_shipDraught", config, "Erosion", "Draught" + istr)
            setOptParam(istr + "_bankSlope", config, "Erosion", "Slope" + istr)
            setOptParam(istr + "_bankReed", config, "Erosion", "Reed" + istr)
            txt = dfastbe_io.config_get_str(config, "Erosion", "EroVol" + istr, default = "")
            dialog[istr + "_eroVolEdit"].setText(txt)

    else:
        showError("Unsupported version number {} in the file!".format(version))


def addTabForLevel(istr):
    newWidget = QtWidgets.QWidget()
    newLayout = QtWidgets.QGridLayout(newWidget)
    dialog["tabs"].addTab(newWidget, "Level " + istr)

    optionalParLayout(newLayout, 0, istr + "_shipType", "Ship Type", selectList = shipTypes())
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


def optionalParLayout(gridLayout, row, key, labelString, selectList = None):
    Label = QtWidgets.QLabel(labelString)
    dialog[key + "Label"] = Label
    gridLayout.addWidget(Label, row, 0)

    paramTypes = ("Use Default", "Constant", "Variable")
    Type = QtWidgets.QComboBox()
    Type.addItems(paramTypes)
    Type.currentIndexChanged.connect(partial(typeUpdateOptPar, key))
    dialog[key + "Type"] = Type
    gridLayout.addWidget(Type, row, 1)

    if selectList is None:
        fLayout = openFileLayout(key + "Edit", enabled = False)
        dialog[key + "Edit"].setEnabled(False)
        gridLayout.addWidget(fLayout, row, 2)
    else:
        Select = QtWidgets.QComboBox()
        Select.addItems(selectList)
        Select.setEnabled(False)
        dialog[key + "Select"] = Select
        gridLayout.addWidget(Select, row, 2)

        fLayout = openFileLayout(key + "Edit", enabled = False)
        dialog[key + "Edit"].setEnabled(False)
        gridLayout.addWidget(fLayout, row + 1, 2)


def typeUpdateOptPar(key):
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
        

def setParam(field, config, group, key, default = "??"):
    str = dfastbe_io.config_get_str(config, group, key, default)
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


def setOptParam(field, config, group, key):
    str = dfastbe_io.config_get_str(config, group, key, "")
    if str == "":
        dialog[field + "Type"].setCurrentText("Use Default")
        dialog[field + "Edit"].setText("")
    else:
        try:
            val = float(str)
            dialog[field + "Type"].setCurrentText("Constant")
            if field + "Select" in dialog.keys():
                ival = int(val) - 1 # shipType 1 -> index 0
                dialog[field + "Select"].setCurrentIndex(ival)
            else:
                dialog[field + "Edit"].setText(str)
        except:
            dialog[field + "Type"].setCurrentText("Variable")
            dialog[field + "Edit"].setText(str)


def save_configuration():
    fil = QtWidgets.QFileDialog.getSaveFileName(
        caption="Save Configuration As", filter="Config Files (*.ini)"
    )
    filename = fil[0]
    if filename == "":
        return

    config = configparser.ConfigParser()
    config.optionxform = str

    config.add_section("General")
    config["General"]["Version"] = "1.0"
    config["General"]["RiverKM"] = dialog["chainFile"].text()
    config["General"]["Boundaries"] = dialog["startRange"].text() + ":" + dialog["endRange"].text()
    config["General"]["BankDir"] = dialog["bankDir"].text()
    config["General"]["BankFile"] = dialog["bankFileName"].text()

    config.add_section("Detect")
    config["Detect"]["SimFile"] = dialog["simFile"].text()
    config["Detect"]["WaterDepth"] = dialog["waterDepth"].text()
    nbank = dialog["bankLines"].topLevelItemCount()
    config["Detect"]["NBank"] = str(nbank)
    dlines = "[ "
    for i in range(nbank):
       istr = str(i+1)
       config["Detect"]["Line" + istr] = dialog["bankLines"].topLevelItem(i).text(1)
       dlines += dialog["bankLines"].topLevelItem(i).text(2)+", "
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
        config["Erosion"]["ShipType"] = dialog["shipTypeSelect"].currentIndex() + 1 # index 0 -> shipType 1
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
        istr = str(i+1)
        config["Erosion"]["SimFile" + istr] = dialog["discharges"].topLevelItem(i).text(1)
        config["Erosion"]["PDischarge" + istr] = dialog["discharges"].topLevelItem(i).text(2)
        if dialog[istr + "_shipTypeType"].currentText() != "Use Default":
            if dialog[istr + "_shipTypeType"].currentText() == "Constant":
                config["Erosion"]["ShipType" + istr] = dialog[istr + "_shipTypeSelect"].currentIndex() + 1 # index 0 -> shipType 1
            else:
                config["Erosion"]["ShipType" + istr] = dialog[istr + "_shipTypeEdit"].text()
        if dialog[istr + "_shipVelocType"].currentText() != "Use Default":
            config["Erosion"]["VShip" + istr] = dialog[istr + "_shipVelocEdit"].text()
        if dialog[istr + "_nShipsType"].currentText() != "Use Default":
            config["Erosion"]["NShip" + istr] = dialog[istr + "_nShipsEdit"].text()
        if dialog[istr + "_shipNWavesType"].currentText() != "Use Default":
            config["Erosion"]["NWaves" + istr] = dialog[istr + "_shipNWavesEdit"].text()
        if dialog[istr + "_shipDraughtType"].currentText() != "Use Default":
            config["Erosion"]["Draught" + istr] = dialog[istr + "_shipDraughtEdit"].text()
        if dialog[istr + "_bankSlopeType"].currentText() != "Use Default":
            config["Erosion"]["Slope" + istr] = dialog[istr + "_bankSlopeEdit"].text()
        if dialog[istr + "_bankReedType"].currentText() != "Use Default":
            config["Erosion"]["Reed" + istr] = dialog[istr + "_bankReedEdit"].text()
        if dialog[istr + "_eroVolEdit"].text() != "":
            config["Erosion"]["EroVol" + istr] = dialog[istr + "_eroVolEdit"].text()

    dfastbe_io.write_config(filename, config)


def showError(message):
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Critical)
    msg.setText(message)
    msg.setWindowTitle("Error")
    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
    msg.exec_()


def menu_about_self():
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Information)
    msg.setText("D-FAST Bank Erosion " + dfastbe_kernel.program_version())
    msg.setInformativeText("Copyright (c) 2020 Deltares.")
    msg.setDetailedText(
        "This program is distributed under the terms of the\nGNU Lesser General Public License Version 2.1; see\nthe LICENSE.md file for details."
    )
    msg.setWindowTitle("About ...")
    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
    msg.setStyleSheet("QDialogButtonBox{min-width: 400px;}");
    msg.exec_()


def menu_about_qt():
    QtWidgets.QApplication.aboutQt()


if __name__ == "__main__":
    global dialog
    dialog = {}
    progloc = str(pathlib.Path(__file__).parent.absolute())
    create_dialog()
    #dialog["branch"].addItems(rivers["branches"])
    #dialog["reach"].addItems(rivers["reaches"][0])
    activate_dialog()
