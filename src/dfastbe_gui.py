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
    dialog = {}

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("fusion")
    dialog["application"] = app

    win = QtWidgets.QMainWindow()
    win.setGeometry(200, 200, 600, 300)
    win.setWindowTitle("D-FAST Bank Erosion")
    dialog["window"] = win

    menubar = win.menuBar()

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

    #------

    generalWidget = QtWidgets.QWidget()
    generalLayout = QtWidgets.QFormLayout(generalWidget)
    tabs.addTab(generalWidget, "General")

    chainFile = QtWidgets.QLineEdit(win)
    dialog["chainFile"] = chainFile
    dialog = addOpenFileRow(generalLayout, dialog, "chainFile", "Chain File")

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

    #model = QtWidgets.QFileSystemModel()
    #model.setRootPath("d:\\")
    #tree = QtWidgets.QTreeView()
    #tree.setModel(model)
    #generalLayout.addRow("FileTree", tree)

    #------

    detectWidget = QtWidgets.QWidget()
    detectLayout = QtWidgets.QFormLayout(detectWidget)
    tabs.addTab(detectWidget, "Detection")

    simFile = QtWidgets.QLineEdit(win)
    dialog["simFile"] = simFile
    dialog = addOpenFileRow(detectLayout, dialog, "simFile", "Simulation File")

    waterDepth = QtWidgets.QLineEdit(win)
    dialog["waterDepth"] = waterDepth
    detectLayout.addRow("Water Depth [m]", waterDepth)

    bankLines = QtWidgets.QTreeWidget(win)
    bankLines.setHeaderLabels(["Index", "FileName", "Search Distance [m]"])
    bankLines.setFont(app.font())
    bankLines.setColumnWidth(0,50)
    bankLines.setColumnWidth(1,250)
    #c1 = QtWidgets.QTreeWidgetItem(bankLines, ["0", "test\\filename", "50"])
    dialog["bankLines"] = bankLines
    detectLayout.addRow("Bank Lines", bankLines)

    #------

    erosionWidget = QtWidgets.QWidget()
    erosionLayout = QtWidgets.QFormLayout(erosionWidget)
    tabs.addTab(erosionWidget, "Erosion")

    tErosion = QtWidgets.QLineEdit(win)
    dialog["tErosion"] = tErosion
    erosionLayout.addRow("Simulation Time [yr]", tErosion)

    riverAxis = QtWidgets.QLineEdit(win)
    dialog["riverAxis"] = riverAxis
    dialog = addOpenFileRow(erosionLayout, dialog, "riverAxis", "River Axis File")

    fairway = QtWidgets.QLineEdit(win)
    dialog["fairway"] = fairway
    dialog = addOpenFileRow(erosionLayout, dialog, "fairway", "Fairway File")

    discharges = QtWidgets.QTreeWidget(win)
    discharges.setHeaderLabels(["Level", "FileName", "Probabily [-]"])
    discharges.setFont(app.font())
    discharges.setColumnWidth(0,50)
    discharges.setColumnWidth(1,250)
    #c1 = QtWidgets.QTreeWidgetItem(discharges, ["0", "test\\filename", "0.5"])
    dialog["discharges"] = discharges
    erosionLayout.addRow("Discharges", discharges)

    refLevel = QtWidgets.QLineEdit(win)
    dialog["refLevel"] = refLevel
    erosionLayout.addRow("Reference Level", refLevel)
    
    chainageOutStep = QtWidgets.QLineEdit(win)
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

    #------

    eParamsWidget = QtWidgets.QWidget()
    eParamsLayout = QtWidgets.QGridLayout(eParamsWidget)
    tabs.addTab(eParamsWidget, "Shipping Parameters")

    dialog = generalParLayout(eParamsLayout, 0, "shipType", "Ship Type", dialog)
    dialog = generalParLayout(eParamsLayout, 1, "shipVeloc", "Velocity [m/s]", dialog)
    dialog = generalParLayout(eParamsLayout, 2, "nShips", "# Ships [1/yr]", dialog)
    dialog = generalParLayout(eParamsLayout, 3, "shipNWaves", "# Waves [1/ship]", dialog)
    dialog = generalParLayout(eParamsLayout, 4, "shipDraught", "Draught [m]", dialog)
    dialog = generalParLayout(eParamsLayout, 5, "wavePar0", "Wave0 [m]", dialog)
    dialog = generalParLayout(eParamsLayout, 6, "wavePar1", "Wave1 [m]", dialog)

    stretch = QtWidgets.QSpacerItem(10, 10, 13, 7)
    eParamsLayout.addItem(stretch, 7, 0)

    #------

    eParamsWidget = QtWidgets.QWidget()
    eParamsLayout = QtWidgets.QGridLayout(eParamsWidget)
    tabs.addTab(eParamsWidget, "Bank Parameters")

    strength = QtWidgets.QLabel("Strength Parameter")
    eParamsLayout.addWidget(strength, 0, 0)
    strengthPar = QtWidgets.QComboBox()
    strengthPar.addItems(("Bank Type", "Critical Shear Stress"))
    dialog["strengthPar"] = strengthPar
    eParamsLayout.addWidget(strengthPar, 0, 2)

    dialog = generalParLayout(eParamsLayout, 1, "bankType", "Bank Type", dialog)
    dialog = generalParLayout(eParamsLayout, 2, "bankShear", "Critical Shear Stress [N/m2]", dialog)
    dialog["bankShear"].setEnabled(False)
    dialog["bankShearType"].setEnabled(False)
    dialog["bankShearEdit"].setEnabled(False)
    dialog["bankShearEditFile"].setEnabled(False)
    dialog = generalParLayout(eParamsLayout, 3, "bankProtect", "Protection [m]", dialog)
    dialog = generalParLayout(eParamsLayout, 4, "bankSlope", "Slope [-]", dialog)
    dialog = generalParLayout(eParamsLayout, 5, "bankReed", "Reed [-]", dialog)

    stretch = QtWidgets.QSpacerItem(10, 10, 13, 7)
    eParamsLayout.addItem(stretch, 6, 0)

    return dialog


def activate_dialog(dialog):
    app = dialog["application"]
    win = dialog["window"]
    win.show()
    sys.exit(app.exec_())


def generalParLayout(gridLayout, row, key, labelString, dialog):
    Label = QtWidgets.QLabel(labelString)
    dialog[key] = Label
    gridLayout.addWidget(Label, row, 0)

    paramTypes = ("Constant", "Variable")
    Type = QtWidgets.QComboBox()
    Type.addItems(paramTypes)
    dialog[key+"Type"] = Type
    gridLayout.addWidget(Type, row, 1)

    Edit = QtWidgets.QLineEdit()
    dialog[key+"Edit"] = Edit
    dialog, fLayout = openFileLayout(dialog, Edit, key+"Edit", enabled = False)
    gridLayout.addWidget(fLayout, row, 2)

    return dialog


def addOpenFileRow(formLayout, dialog, key, label):
    dialog, fLayout = openFileLayout(dialog, dialog[key], key)
    formLayout.addRow(label, fLayout)
    return dialog


def openFileLayout(dialog, myWidget, key, enabled = True):
    parent = QtWidgets.QWidget()
    gridly = QtWidgets.QGridLayout(parent)
    gridly.setContentsMargins(0, 0, 0, 0)
    gridly.addWidget(myWidget, 0, 0)

    openFile = QtWidgets.QPushButton(PyQt5.QtGui.QIcon("open.png"), "")
    openFile.clicked.connect(partial(selectFile, key))
    openFile.setEnabled(enabled)
    dialog[key+"File"] = openFile
    gridly.addWidget(openFile, 0, 2)

    return dialog, parent


def selectFile(key):
    if key == "simFile":
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
        dialog["bankLines"].invisibleRootItem().takeChildren() # sufficient to destroy items?
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
        dialog["eroVol"].setText(section["EroVol"])

        NLevel = dfastbe_io.config_get_int(config, "Erosion", "NLevel", default = 0, positive = True)
        dialog["discharges"].invisibleRootItem().takeChildren() # sufficient to destroy items?
        for i in range(NLevel):
            istr = str(i+1)
            fileName = dfastbe_io.config_get_str(config, "Erosion", "SimFile" + istr)
            prob = dfastbe_io.config_get_str(config, "Erosion", "PDischarge" + istr)
            c1 = QtWidgets.QTreeWidgetItem(dialog["discharges"],
                [istr, fileName, prob]
            )
        dialog["refLevel"].setText(section["RefLevel"])

        setParam(dialog, "shipType", config, "Erosion", "ShipType")
        setParam(dialog, "shipVeloc", config, "Erosion", "VShip")
        setParam(dialog, "nShips", config, "Erosion", "NShip")
        setParam(dialog, "shipNWaves", config, "Erosion", "NWave", "5")
        setParam(dialog, "shipDraught", config, "Erosion", "Draught")
        setParam(dialog, "wavePar0", config, "Erosion", "Wave0", "200.0")
        wave0 = dfastbe_io.config_get_str(config, "Erosion", "Wave0", "200.0")
        setParam(dialog, "wavePar1", config, "Erosion", "Wave1", wave0)

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
            setParam(dialog, "bankType", config, "Erosion", "BankType")
        else:
            dialog["strengthPar"].setCurrentText("Critical Shear Stress")
            setParam(dialog, "bankShear", config, "Erosion", "BankType")
        setParam(dialog, "bankProtect", config, "Erosion", "ProtectLevel", "-1000")
        setParam(dialog, "bankSlope", config, "Erosion", "Slope", "20.0")
        setParam(dialog, "bankReed", config, "Erosion", "Reed", "0.0")

        tabs = dialog["tabs"]
        for i in range(tabs.count()-1,4,-1):
            tabs.removeTab(i) # Is this enough to destroy the objects?
            # at leat old dialog entries persist

        for i in range(NLevel):
            istr = str(i+1)
            addTabForLevel(dialog, istr)
            setOptParam(dialog, "shipType" + istr, config, "Erosion", "ShipType" + istr)
            setOptParam(dialog, "shipVeloc" + istr, config, "Erosion", "VShip" + istr)
            setOptParam(dialog, "nShips" + istr, config, "Erosion", "NShip" + istr)
            setOptParam(dialog, "shipNWaves" + istr, config, "Erosion", "NWave" + istr)
            setOptParam(dialog, "shipDraught" + istr, config, "Erosion", "Draught" + istr)
            setOptParam(dialog, "bankSlope" + istr, config, "Erosion", "Slope" + istr)
            setOptParam(dialog, "bankReed" + istr, config, "Erosion", "Reed" + istr)

    else:
        showError("Unsupported version number {} in the file!".format(version))


def addTabForLevel(dialog, istr):
    newWidget = QtWidgets.QWidget()
    newLayout = QtWidgets.QGridLayout(newWidget)
    dialog["tabs"].addTab(newWidget, "Level" + istr)

    dialog = optionalParLayout(newLayout, 0, "shipType" + istr, "Ship Type", dialog)
    dialog = optionalParLayout(newLayout, 1, "shipVeloc" + istr, "Velocity [m/s]", dialog)
    dialog = optionalParLayout(newLayout, 2, "nShips" + istr, "# Ships [1/yr]", dialog)
    dialog = optionalParLayout(newLayout, 3, "shipNWaves" + istr, "# Waves [1/ship]", dialog)
    dialog = optionalParLayout(newLayout, 4, "shipDraught" + istr, "Draught [m]", dialog)
    dialog = optionalParLayout(newLayout, 5, "bankSlope" + istr, "Slope [-]", dialog)
    dialog = optionalParLayout(newLayout, 6, "bankReed" + istr, "Reed [-]", dialog)

    stretch = QtWidgets.QSpacerItem(10, 10, 13, 7)
    newLayout.addItem(stretch, 7, 0)


def optionalParLayout(gridLayout, row, key, labelString, dialog):
    Label = QtWidgets.QLabel(labelString)
    dialog[key] = Label
    gridLayout.addWidget(Label, row, 0)

    paramTypes = ("Use Default", "Constant", "Variable")
    Type = QtWidgets.QComboBox()
    Type.addItems(paramTypes)
    dialog[key+"Type"] = Type
    gridLayout.addWidget(Type, row, 1)

    Edit = QtWidgets.QLineEdit()
    Edit.setEnabled(False)
    dialog[key+"Edit"] = Edit
    dialog, fLayout = openFileLayout(dialog, Edit, key+"Edit", enabled = False)
    gridLayout.addWidget(fLayout, row, 2)

    return dialog


def setParam(dialog, field, config, group, key, default = "??"):
    str = dfastbe_io.config_get_str(config, group, key, default)
    try:
        val = float(str)
        dialog[field + "Type"].setCurrentText("Constant")
        dialog[field + "Edit"].setText(str)
        dialog[field + "EditFile"].setEnabled(False)
    except:
        dialog[field + "Type"].setCurrentText("Variable")
        dialog[field + "Edit"].setText(str)
        dialog[field + "EditFile"].setEnabled(True)


def setOptParam(dialog, field, config, group, key):
    str = dfastbe_io.config_get_str(config, group, key, "")
    if str == "":
        dialog[field + "Type"].setCurrentText("Use Default")
        dialog[field + "Edit"].setText("")
        dialog[field + "Edit"].setEnabled(False)
        dialog[field + "EditFile"].setEnabled(False)
    else:
        try:
            val = float(str)
            dialog[field + "Type"].setCurrentText("Constant")
            dialog[field + "Edit"].setText(str)
            dialog[field + "Edit"].setEnabled(True)
            dialog[field + "EditFile"].setEnabled(False)
        except:
            dialog[field + "Type"].setCurrentText("Variable")
            dialog[field + "Edit"].setText(str)
            dialog[field + "Edit"].setEnabled(True)
            dialog[field + "EditFile"].setEnabled(True)


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

    config["Erosion"]["ShipType"] = dialog["shipTypeEdit"].text()
    config["Erosion"]["VShip"] = dialog["shipVelocEdit"].text()
    config["Erosion"]["NShip"] = dialog["nShipsEdit"].text()
    config["Erosion"]["NWaves"] = dialog["shipNWavesEdit"].text()
    config["Erosion"]["Draught"] = dialog["shipDraughtEdit"].text()
    config["Erosion"]["Wave0"] = dialog["wavePar0Edit"].text()
    config["Erosion"]["Wave1"] = dialog["wavePar1Edit"].text()

    if dialog["strengthPar"].currentText == "Bank Type":
        config["Erosion"]["Classes"] = "true"
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
        if dialog["shipType" + istr + "Type"].currentText() != "Use Default":
            config["Erosion"]["ShipType" + istr] = dialog["shipType" + istr + "Edit"].text()
        if dialog["shipVeloc" + istr + "Type"].currentText() != "Use Default":
            config["Erosion"]["VShip" + istr] = dialog["shipVeloc" + istr + "Edit"].text()
        if dialog["nShips" + istr + "Type"].currentText() != "Use Default":
            config["Erosion"]["NShip" + istr] = dialog["nShips" + istr + "Edit"].text()
        if dialog["shipNWaves" + istr + "Type"].currentText() != "Use Default":
            config["Erosion"]["NWaves" + istr] = dialog["shipNWaves" + istr + "Edit"].text()
        if dialog["shipDraught" + istr + "Type"].currentText() != "Use Default":
            config["Erosion"]["Draught" + istr] = dialog["shipDraught" + istr + "Edit"].text()
        if dialog["bankSlope" + istr + "Type"].currentText() != "Use Default":
            config["Erosion"]["Slope" + istr] = dialog["bankSlope" + istr + "Edit"].text()
        if dialog["bankReed" + istr + "Type"].currentText() != "Use Default":
            config["Erosion"]["Reed" + istr] = dialog["bankReed" + istr + "Edit"].text()

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
    msg.exec_()


def menu_about_qt():
    QtWidgets.QApplication.aboutQt()


if __name__ == "__main__":
    global dialog
    progloc = str(pathlib.Path(__file__).parent.absolute())
    dialog = create_dialog()
    #dialog["branch"].addItems(rivers["branches"])
    #dialog["reach"].addItems(rivers["reaches"][0])
    activate_dialog(dialog)
