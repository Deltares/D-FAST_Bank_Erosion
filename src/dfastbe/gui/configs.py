import os
from typing import cast
from pathlib import Path
from configparser import ConfigParser
from PySide6.QtWidgets import (
    QTreeWidgetItem,
    QComboBox,
    QLineEdit,
    QLabel,
    QGridLayout,
    QWidget,
    QSpacerItem,
    QSizePolicy,
    QPushButton,
    QFileDialog
)

from dfastbe.io.config import ConfigFile
from dfastbe.gui.utils import show_error, SHIP_TYPES, validator, get_icon, ICONS_DIR
from dfastbe.io.file_utils import absolute_path

__all__ = [
    "get_configuration",
    "load_configuration",
    "bankStrengthSwitch",
    "typeUpdatePar",
    "openFileLayout",
    "addTabForLevel",
    "selectFile",
]


def load_configuration(config_path: Path) -> None:
    """Open a configuration file and update the GUI accordingly.

    This routines opens the specified configuration file and updates the GUI
    to reflect it contents.

    Args:
        config_path : str
            Name of the configuration file to be opened.
    """
    from dfastbe.gui.gui import dialog
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
        study_range = config_file.get_range("General", "Boundaries")
        dialog["startRange"].setText(str(study_range[0]))
        dialog["endRange"].setText(str(study_range[1]))
        dialog["bankDirEdit"].setText(section["BankDir"])
        bank_file = config_file.get_str("General", "BankFile", default="bankfile")
        dialog["bankFileName"].setText(bank_file)
        flag = config_file.get_bool("General", "Plotting", default=True)
        dialog["makePlotsEdit"].setChecked(flag)
        flag = config_file.get_bool("General", "SavePlots", default=True)
        dialog["savePlotsEdit"].setChecked(flag)
        flag = config_file.get_bool("General", "SaveZoomPlots", default=False)
        dialog["saveZoomPlotsEdit"].setChecked(flag)
        zoom_step_km = config_file.get_float("General", "ZoomStepKM", default=1.0)
        dialog["zoomPlotsRangeEdit"].setText(str(zoom_step_km))
        fig_dir = config_file.get_str(
            "General",
            "FigureDir",
            default=absolute_path(rootdir, "figures"),
        )
        dialog["figureDirEdit"].setText(fig_dir)
        flag = config_file.get_bool("General", "ClosePlots", default=False)
        dialog["closePlotsEdit"].setChecked(flag)
        flag = config_file.get_bool("General", "DebugOutput", default=False)
        dialog["debugOutputEdit"].setChecked(flag)

        section = config["Detect"]
        dialog["simFileEdit"].setText(section["SimFile"])
        water_depth = config_file.get_float("Detect", "WaterDepth", default=0.0)
        dialog["waterDepth"].setText(str(water_depth))
        n_bank = config_file.get_int("Detect", "NBank", default=0, positive=True)
        d_lines = config_file.get_bank_search_distances(n_bank)
        dialog["searchLines"].invisibleRootItem().takeChildren()
        for i in range(n_bank):
            istr = str(i + 1)
            file_name = config_file.get_str("Detect", "Line" + istr)
            c1 = QTreeWidgetItem(
                dialog["searchLines"], [istr, file_name, str(d_lines[i])]
            )
        if n_bank > 0:
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
            file_name = config_file.get_str("Erosion", "SimFile" + istr)
            prob = config_file.get_str("Erosion", "PDischarge" + istr)
            c1 = QTreeWidgetItem(dialog["discharges"], [istr, file_name, prob])
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


def get_configuration() -> ConfigParser:
    """Extract a configuration from the GUI.

    Returns
    -------
    config : ConfigParser
        Configuration for the D-FAST Bank Erosion analysis.
    """
    from dfastbe.gui.gui import dialog
    config = ConfigParser()
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
    from dfastbe.gui.gui import dialog
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


def setOptParam(field: str, config, group: str, key: str) -> None:
    """Update the dialog for an optional parameter based on configuration file.

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
    from dfastbe.gui.gui import dialog
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


def bankStrengthSwitch() -> None:
    """Implements the dialog settings depending on the bank strength specification method."""
    from dfastbe.gui.gui import dialog
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


def addTabForLevel(istr: str) -> None:
    """Create the tab for the settings associated with simulation i.

    Args:
        istr : str
            String representation of the simulation number.
    """
    from dfastbe.gui.gui import dialog
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


def typeUpdatePar(key: str) -> None:
    """
    Implements the dialog setting switching for both general and optional parameters.

    Arguments
    ---------
    key : str
        Short name of the parameter.
    """
    from dfastbe.gui.gui import dialog
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


def optionalParLayout(
    gridLayout: QGridLayout, row: int, key, labelString, selectList=None
) -> None:
    """Add a line of controls for editing an optional parameter.

    Args:
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
    from dfastbe.gui.gui import dialog
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
    from dfastbe.gui.gui import dialog
    config_file = ConfigFile(config)
    val = config_file.get_float(group, key, 0.0)
    if val > 0.0:
        dialog[field + "Active"].setChecked(True)
        dialog[field + "Width"].setText(str(val))
    else:
        dialog[field + "Active"].setChecked(False)



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
    from dfastbe.gui.gui import dialog
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


def selectFile(key: str) -> None:
    """Select a file or directory via a selection dialog.

    Args:
        key : str
            Short name of the parameter.
    """
    from dfastbe.gui.gui import dialog
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