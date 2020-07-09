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
import PyQt5.QtGui
import dfastbe_kernel
import pathlib
import sys


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
    item = menu.addAction("&Load")
    #item.triggered.connect(load_configuration)
    item = menu.addAction("&Save")
    #item.triggered.connect(save_configuration)
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

    topLayout = QtWidgets.QFormLayout()
    layout.addLayout(topLayout)

    tabs = QtWidgets.QTabWidget(win)
    layout.addWidget(tabs)

    chainFile = QtWidgets.QLineEdit(win)
    topLayout.addRow("Chainage File", chainFile)

    chainRange = QtWidgets.QLineEdit(win)
    topLayout.addRow("Analysis Range", chainRange)

    #------

    detectWidget = QtWidgets.QWidget()
    detectLayout = QtWidgets.QFormLayout(detectWidget)
    tabs.addTab(detectWidget, "Detection")

    simFile = QtWidgets.QLineEdit(win)
    detectLayout.addRow("SimFile", simFile)

    bankLinesFile = QtWidgets.QLineEdit(win)
    detectLayout.addRow("Bank Lines", bankLinesFile)

    bankLines = QtWidgets.QLineEdit(win)
    detectLayout.addRow("Bank Lines", bankLines)

    bankDir = QtWidgets.QLineEdit(win)
    detectLayout.addRow("Bank Directory", bankDir)

    bankFileName = QtWidgets.QLineEdit(win)
    detectLayout.addRow("Bank File Name", bankFileName)

    waterDepth = QtWidgets.QLineEdit(win)
    detectLayout.addRow("Water Depth", waterDepth)

    #------

    erosionWidget = QtWidgets.QWidget()
    erosionLayout = QtWidgets.QFormLayout(erosionWidget)
    tabs.addTab(erosionWidget, "Erosion")

    tErosion = QtWidgets.QLineEdit(win)
    erosionLayout.addRow("Terosion", tErosion)

    riverAxis = QtWidgets.QLineEdit(win)
    erosionLayout.addRow("River Axis", riverAxis)

    fairway = QtWidgets.QLineEdit(win)
    erosionLayout.addRow("Fairway", fairway)

    discharges = QtWidgets.QLineEdit(win)
    erosionLayout.addRow("Discharges", discharges)

    chainageOutStep = QtWidgets.QLineEdit(win)
    erosionLayout.addRow("Chainage Output Step", chainageOutStep)

    outDir = QtWidgets.QLineEdit(win)
    erosionLayout.addRow("Output Directory", outDir)

    newBankFile = QtWidgets.QLineEdit(win)
    erosionLayout.addRow("New Bank File Name", newBankFile)

    newEqBankFile = QtWidgets.QLineEdit(win)
    erosionLayout.addRow("New Eq Bank File Name", newEqBankFile)

    eroVol = QtWidgets.QLineEdit(win)
    erosionLayout.addRow("EroVol File Name", eroVol)

    #------

    eParamsWidget = QtWidgets.QWidget()
    eParamsLayout = QtWidgets.QGridLayout(eParamsWidget)
    tabs.addTab(eParamsWidget, "Erosion Parameters")

    strShipping = QtWidgets.QLabel("Shipping")
    eParamsLayout.addWidget(strShipping, 0, 0)

    paramTypes = ("Constant", "Variable")

    strShipType = QtWidgets.QLabel("Type")
    eParamsLayout.addWidget(strShipType, 1, 0)
    varShipType = QtWidgets.QComboBox()
    varShipType.addItems(paramTypes)
    eParamsLayout.addWidget(varShipType, 1, 1)
    edtShipType = QtWidgets.QLineEdit()
    eParamsLayout.addWidget(edtShipType, 1, 2)

    strShipVel = QtWidgets.QLabel("Velocity")
    eParamsLayout.addWidget(strShipVel, 2, 0)
    varShipVel = QtWidgets.QComboBox()
    varShipVel.addItems(paramTypes)
    eParamsLayout.addWidget(varShipVel, 2, 1)
    edtShipVel = QtWidgets.QLineEdit()
    eParamsLayout.addWidget(edtShipVel, 2, 2)

    strShipNum = QtWidgets.QLabel("Number per year")
    eParamsLayout.addWidget(strShipNum, 3, 0)

    strShipWaves = QtWidgets.QLabel("Waves per ship")
    eParamsLayout.addWidget(strShipWaves, 4, 0)

    strShipDrgt = QtWidgets.QLabel("Draught")
    eParamsLayout.addWidget(strShipDrgt, 5, 0)

    strShipWave0 = QtWidgets.QLabel("Wave0")
    eParamsLayout.addWidget(strShipWave0, 6, 0)

    strShipWave1 = QtWidgets.QLabel("Wave1")
    eParamsLayout.addWidget(strShipWave1, 7, 0)

    stretch1 = QtWidgets.QSpacerItem(10, 10, 13, 7)
    eParamsLayout.addItem(stretch1, 8, 0)

    #---

    strBanks = QtWidgets.QLabel("Banks")
    eParamsLayout.addWidget(strBanks, 9, 0)

    strBankType = QtWidgets.QLabel("Type")
    eParamsLayout.addWidget(strBankType, 10, 0)

    strBankShear = QtWidgets.QLabel("Shear Stress")
    eParamsLayout.addWidget(strBankShear, 11, 0)

    strBankProt = QtWidgets.QLabel("Protection")
    eParamsLayout.addWidget(strBankProt, 12, 0)

    strBankSlope = QtWidgets.QLabel("Slope")
    eParamsLayout.addWidget(strBankSlope, 13, 0)

    strBankReed = QtWidgets.QLabel("Reed")
    eParamsLayout.addWidget(strBankReed, 14, 0)

    stretch2 = QtWidgets.QSpacerItem(10, 10, 13, 7)
    eParamsLayout.addItem(stretch2, 15, 0)

    return dialog


def activate_dialog(dialog):
    app = dialog["application"]
    win = dialog["window"]
    win.show()
    sys.exit(app.exec_())


def close_dialog():
    dialog["window"].close()


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
