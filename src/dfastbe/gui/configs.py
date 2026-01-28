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
from dfastbe.gui.state_management import StateStore

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
    StateManagement = StateStore.instance()
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
        StateManagement["chainFileEdit"].setText(section["RiverKM"])
        study_range = config_file.get_range("General", "Boundaries")
        StateManagement["startRange"].setText(str(study_range[0]))
        StateManagement["endRange"].setText(str(study_range[1]))
        StateManagement["bankDirEdit"].setText(section["BankDir"])
        bank_file = config_file.get_str("General", "BankFile", default="bankfile")
        StateManagement["bankFileName"].setText(bank_file)
        flag = config_file.get_bool("General", "Plotting", default=True)
        StateManagement["makePlotsEdit"].setChecked(flag)
        flag = config_file.get_bool("General", "SavePlots", default=True)
        StateManagement["savePlotsEdit"].setChecked(flag)
        flag = config_file.get_bool("General", "SaveZoomPlots", default=False)
        StateManagement["saveZoomPlotsEdit"].setChecked(flag)
        zoom_step_km = config_file.get_float("General", "ZoomStepKM", default=1.0)
        StateManagement["zoomPlotsRangeEdit"].setText(str(zoom_step_km))
        fig_dir = config_file.get_str(
            "General",
            "FigureDir",
            default=absolute_path(rootdir, "figures"),
        )
        StateManagement["figureDirEdit"].setText(fig_dir)
        flag = config_file.get_bool("General", "ClosePlots", default=False)
        StateManagement["closePlotsEdit"].setChecked(flag)
        flag = config_file.get_bool("General", "DebugOutput", default=False)
        StateManagement["debugOutputEdit"].setChecked(flag)

        section = config["Detect"]
        StateManagement["simFileEdit"].setText(section["SimFile"])
        water_depth = config_file.get_float("Detect", "WaterDepth", default=0.0)
        StateManagement["waterDepth"].setText(str(water_depth))
        n_bank = config_file.get_int("Detect", "NBank", default=0, positive=True)
        d_lines = config_file.get_bank_search_distances(n_bank)
        StateManagement["searchLines"].invisibleRootItem().takeChildren()
        for i in range(n_bank):
            istr = str(i + 1)
            file_name = config_file.get_str("Detect", "Line" + istr)
            c1 = QTreeWidgetItem(
                StateManagement["searchLines"], [istr, file_name, str(d_lines[i])]
            )
        if n_bank > 0:
            StateManagement["searchLinesEdit"].setEnabled(True)
            StateManagement["searchLinesRemove"].setEnabled(True)

        section = config["Erosion"]
        StateManagement["tErosion"].setText(section["TErosion"])
        StateManagement["riverAxisEdit"].setText(section["RiverAxis"])
        StateManagement["fairwayEdit"].setText(section["Fairway"])
        StateManagement["chainageOutStep"].setText(section["OutputInterval"])
        StateManagement["outDirEdit"].setText(section["OutputDir"])
        bankNew = config_file.get_str("Erosion", "BankNew", default="banknew")
        StateManagement["newBankFile"].setText(bankNew)
        bankEq = config_file.get_str("Erosion", "BankEq", default="bankeq")
        StateManagement["newEqBankFile"].setText(bankEq)
        txt = config_file.get_str("Erosion", "EroVol", default="erovol_standard.evo")
        StateManagement["eroVol"].setText(txt)
        txt = config_file.get_str("Erosion", "EroVolEqui", default="erovol_eq.evo")
        StateManagement["eroVolEqui"].setText(txt)

        NLevel = config_file.get_int("Erosion", "NLevel", default=0, positive=True)
        StateManagement["discharges"].invisibleRootItem().takeChildren()
        for i in range(NLevel):
            istr = str(i + 1)
            file_name = config_file.get_str("Erosion", "SimFile" + istr)
            prob = config_file.get_str("Erosion", "PDischarge" + istr)
            c1 = QTreeWidgetItem(StateManagement["discharges"], [istr, file_name, prob])
        if NLevel > 0:
            StateManagement["dischargesEdit"].setEnabled(True)
            StateManagement["dischargesRemove"].setEnabled(True)
        StateManagement["refLevel"].validator().setTop(NLevel)
        StateManagement["refLevel"].setText(section["RefLevel"])

        setParam("shipType", config, "Erosion", "ShipType")
        setParam("shipVeloc", config, "Erosion", "VShip")
        setParam("nShips", config, "Erosion", "NShip")
        setParam("shipNWaves", config, "Erosion", "NWave", "5")
        setParam("shipDraught", config, "Erosion", "Draught")
        setParam("wavePar0", config, "Erosion", "Wave0", "200.0")
        wave0 = config_file.get_str("Erosion", "Wave0", "200.0")
        setParam("wavePar1", config_file.config, "Erosion", "Wave1", wave0)

        useBankType = config_file.get_bool("Erosion", "Classes", default=True)
        StateManagement["bankType"].setEnabled(useBankType)
        StateManagement["bankTypeType"].setEnabled(useBankType)
        StateManagement["bankTypeEdit"].setEnabled(useBankType)
        StateManagement["bankTypeEditFile"].setEnabled(useBankType)
        StateManagement["bankShear"].setEnabled(not useBankType)
        StateManagement["bankShearType"].setEnabled(not useBankType)
        StateManagement["bankShearEdit"].setEnabled(not useBankType)
        StateManagement["bankShearEditFile"].setEnabled(not useBankType)
        if useBankType:
            StateManagement["strengthPar"].setCurrentText("Bank Type")
            bankStrengthSwitch()
            setParam("bankType", config_file.config, "Erosion", "BankType")
        else:
            StateManagement["strengthPar"].setCurrentText("Critical Shear Stress")
            bankStrengthSwitch()
            setParam("bankShear", config, "Erosion", "BankType")
        setParam("bankProtect", config, "Erosion", "ProtectionLevel", "-1000")
        setParam("bankSlope", config, "Erosion", "Slope", "20.0")
        setParam("bankReed", config, "Erosion", "Reed", "0.0")

        setFilter("velFilter", config, "Erosion", "VelFilterDist")
        setFilter("bedFilter", config, "Erosion", "BedFilterDist")

        tabs = StateManagement["tabs"]
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
            StateManagement[istr + "_eroVolEdit"].setText(txt)

    else:
        show_error(f"Unsupported version number {version} in the file {config_path}!")


def get_configuration() -> ConfigParser:
    """Extract a configuration from the GUI.

    Returns
    -------
    config : ConfigParser
        Configuration for the D-FAST Bank Erosion analysis.
    """
    StateManagement = StateStore.instance()
    config = ConfigParser()
    config.optionxform = str  # case sensitive configuration

    config.add_section("General")
    config["General"]["Version"] = "1.0"
    config["General"]["RiverKM"] = StateManagement["chainFileEdit"].text()
    config["General"]["Boundaries"] = (
            StateManagement["startRange"].text() + ":" + StateManagement["endRange"].text()
    )
    config["General"]["BankDir"] = StateManagement["bankDirEdit"].text()
    config["General"]["BankFile"] = StateManagement["bankFileName"].text()
    config["General"]["Plotting"] = str(StateManagement["makePlotsEdit"].isChecked())
    config["General"]["SavePlots"] = str(StateManagement["savePlotsEdit"].isChecked())
    config["General"]["SaveZoomPlots"] = str(StateManagement["saveZoomPlotsEdit"].isChecked())
    config["General"]["ZoomStepKM"] = StateManagement["zoomPlotsRangeEdit"].text()
    config["General"]["FigureDir"] = StateManagement["figureDirEdit"].text()
    config["General"]["ClosePlots"] = str(StateManagement["closePlotsEdit"].isChecked())
    config["General"]["DebugOutput"] = str(StateManagement["debugOutputEdit"].isChecked())

    config.add_section("Detect")
    config["Detect"]["SimFile"] = StateManagement["simFileEdit"].text()
    config["Detect"]["WaterDepth"] = StateManagement["waterDepth"].text()
    nbank = StateManagement["searchLines"].topLevelItemCount()
    config["Detect"]["NBank"] = str(nbank)
    dlines = "[ "
    for i in range(nbank):
        istr = str(i + 1)
        config["Detect"]["Line" + istr] = StateManagement["searchLines"].topLevelItem(i).text(1)
        dlines += StateManagement["searchLines"].topLevelItem(i).text(2) + ", "
    dlines = dlines[:-2] + " ]"
    config["Detect"]["DLines"] = dlines

    config.add_section("Erosion")
    config["Erosion"]["TErosion"] = StateManagement["tErosion"].text()
    config["Erosion"]["RiverAxis"] = StateManagement["riverAxisEdit"].text()
    config["Erosion"]["Fairway"] = StateManagement["fairwayEdit"].text()
    config["Erosion"]["OutputInterval"] = StateManagement["chainageOutStep"].text()
    config["Erosion"]["OutputDir"] = StateManagement["outDirEdit"].text()
    config["Erosion"]["BankNew"] = StateManagement["newBankFile"].text()
    config["Erosion"]["BankEq"] = StateManagement["newEqBankFile"].text()
    config["Erosion"]["EroVol"] = StateManagement["eroVol"].text()
    config["Erosion"]["EroVolEqui"] = StateManagement["eroVolEqui"].text()

    if StateManagement["shipTypeType"].currentText() == "Constant":
        config["Erosion"]["ShipType"] = str(
            StateManagement["shipTypeSelect"].currentIndex() + 1
        )  # index 0 -> shipType 1
    else:
        config["Erosion"]["ShipType"] = StateManagement["shipTypeEdit"].text()
    config["Erosion"]["VShip"] = StateManagement["shipVelocEdit"].text()
    config["Erosion"]["NShip"] = StateManagement["nShipsEdit"].text()
    config["Erosion"]["NWaves"] = StateManagement["shipNWavesEdit"].text()
    config["Erosion"]["Draught"] = StateManagement["shipDraughtEdit"].text()
    config["Erosion"]["Wave0"] = StateManagement["wavePar0Edit"].text()
    config["Erosion"]["Wave1"] = StateManagement["wavePar1Edit"].text()

    if StateManagement["strengthPar"].currentText() == "Bank Type":
        config["Erosion"]["Classes"] = "true"
        if StateManagement["bankTypeType"].currentText() == "Constant":
            config["Erosion"]["BankType"] = StateManagement["bankTypeSelect"].currentIndex()
        else:
            config["Erosion"]["BankType"] = StateManagement["bankTypeEdit"].text()
    else:
        config["Erosion"]["Classes"] = "false"
        config["Erosion"]["BankType"] = StateManagement["bankShearEdit"].text()
    config["Erosion"]["ProtectionLevel"] = StateManagement["bankProtectEdit"].text()
    config["Erosion"]["Slope"] = StateManagement["bankSlopeEdit"].text()
    config["Erosion"]["Reed"] = StateManagement["bankReedEdit"].text()

    if StateManagement["velFilterActive"].isChecked():
        config["Erosion"]["VelFilterDist"] = StateManagement["velFilterWidth"].text()
    if StateManagement["bedFilterActive"].isChecked():
        config["Erosion"]["BedFilterDist"] = StateManagement["bedFilterWidth"].text()

    nlevel = StateManagement["discharges"].topLevelItemCount()
    config["Erosion"]["NLevel"] = str(nlevel)
    config["Erosion"]["RefLevel"] = StateManagement["refLevel"].text()
    for i in range(nlevel):
        istr = str(i + 1)
        config["Erosion"]["SimFile" + istr] = (
            StateManagement["discharges"].topLevelItem(i).text(1)
        )
        config["Erosion"]["PDischarge" + istr] = (
            StateManagement["discharges"].topLevelItem(i).text(2)
        )
        if StateManagement[istr + "_shipTypeType"].currentText() != "Use Default":
            if StateManagement[istr + "_shipTypeType"].currentText() == "Constant":
                config["Erosion"]["ShipType" + istr] = (
                        StateManagement[istr + "_shipTypeSelect"].currentIndex() + 1
                )  # index 0 -> shipType 1
            else:
                config["Erosion"]["ShipType" + istr] = StateManagement[
                    istr + "_shipTypeEdit"
                    ].text()
        if StateManagement[istr + "_shipVelocType"].currentText() != "Use Default":
            config["Erosion"]["VShip" + istr] = StateManagement[istr + "_shipVelocEdit"].text()
        if StateManagement[istr + "_nShipsType"].currentText() != "Use Default":
            config["Erosion"]["NShip" + istr] = StateManagement[istr + "_nShipsEdit"].text()
        if StateManagement[istr + "_shipNWavesType"].currentText() != "Use Default":
            config["Erosion"]["NWaves" + istr] = StateManagement[istr + "_shipNWavesEdit"].text()
        if StateManagement[istr + "_shipDraughtType"].currentText() != "Use Default":
            config["Erosion"]["Draught" + istr] = StateManagement[
                istr + "_shipDraughtEdit"
                ].text()
        if StateManagement[istr + "_bankSlopeType"].currentText() != "Use Default":
            config["Erosion"]["Slope" + istr] = StateManagement[istr + "_bankSlopeEdit"].text()
        if StateManagement[istr + "_bankReedType"].currentText() != "Use Default":
            config["Erosion"]["Reed" + istr] = StateManagement[istr + "_bankReedEdit"].text()
        if StateManagement[istr + "_eroVolEdit"].text() != "":
            config["Erosion"]["EroVol" + istr] = StateManagement[istr + "_eroVolEdit"].text()
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
    StateManagement = StateStore.instance()
    config_file = ConfigFile(config)
    config_value = config_file.get_str(group, key, default)

    try:
        val = float(config_value)
        cast(QComboBox, StateManagement[field + "Type"]).setCurrentText("Constant")
        if field + "Select" in StateManagement.keys():
            int_value = int(val)
            if field == "shipType":
                int_value = int_value - 1
            cast(QComboBox, StateManagement[field + "Select"]).setCurrentIndex(int_value)
        else:
            cast(QLineEdit, StateManagement[field + "Edit"]).setText(config_value)
    except:
        cast(QComboBox, StateManagement[field + "Type"]).setCurrentText("Variable")
        cast(QLineEdit, StateManagement[field + "Edit"]).setText(config_value)


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
    StateManagement = StateStore.instance()
    config_file = ConfigFile(config)
    str = config_file.get_str(group, key, "")
    if str == "":
        StateManagement[field + "Type"].setCurrentText("Use Default")
        StateManagement[field + "Edit"].setText("")
    else:
        try:
            val = float(str)
            StateManagement[field + "Type"].setCurrentText("Constant")
            if field + "Select" in StateManagement.keys():
                ival = int(val) - 1  # shipType 1 -> index 0
                StateManagement[field + "Select"].setCurrentIndex(ival)
            else:
                StateManagement[field + "Edit"].setText(str)
        except:
            StateManagement[field + "Type"].setCurrentText("Variable")
            StateManagement[field + "Edit"].setText(str)


def bankStrengthSwitch() -> None:
    """Implements the dialog settings depending on the bank strength specification method."""
    StateManagement = StateStore.instance()
    type = StateManagement["strengthPar"].currentText()
    if type == "Bank Type":
        StateManagement["bankType"].setEnabled(True)
        StateManagement["bankTypeType"].setEnabled(True)
        typeUpdatePar("bankType")
        StateManagement["bankShear"].setEnabled(False)
        StateManagement["bankShearType"].setEnabled(False)
        StateManagement["bankShearEdit"].setText("")
        StateManagement["bankShearEdit"].setEnabled(False)
        StateManagement["bankShearEditFile"].setEnabled(False)
    elif type == "Critical Shear Stress":
        StateManagement["bankShear"].setEnabled(True)
        StateManagement["bankShearType"].setEnabled(True)
        StateManagement["bankShearEdit"].setEnabled(True)
        typeUpdatePar("bankShear")
        StateManagement["bankType"].setEnabled(False)
        StateManagement["bankTypeType"].setEnabled(False)
        StateManagement["bankTypeSelect"].setEnabled(False)
        StateManagement["bankTypeEdit"].setText("")
        StateManagement["bankTypeEdit"].setEnabled(False)
        StateManagement["bankTypeEditFile"].setEnabled(False)


def addTabForLevel(istr: str) -> None:
    """Create the tab for the settings associated with simulation i.

    Args:
        istr : str
            String representation of the simulation number.
    """
    StateManagement = StateStore.instance()
    newWidget = QWidget()
    newLayout = QGridLayout(newWidget)
    StateManagement["tabs"].addTab(newWidget, "Level " + istr)

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
    StateManagement[istr + "_eroVol"] = Label
    newLayout.addWidget(Label, 8, 0)
    Edit = QLineEdit()
    StateManagement[istr + "_eroVolEdit"] = Edit
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
    StateManagement = StateStore.instance()
    type = StateManagement[key + "Type"].currentText()
    StateManagement[key + "Edit"].setText("")
    if type == "Use Default":
        StateManagement[key + "Edit"].setValidator(None)
        StateManagement[key + "Edit"].setEnabled(False)
        StateManagement[key + "EditFile"].setEnabled(False)
        if key + "Select" in StateManagement.keys():
            StateManagement[key + "Select"].setEnabled(False)
    elif type == "Constant":
        if key + "Select" in StateManagement.keys():
            StateManagement[key + "Select"].setEnabled(True)
            StateManagement[key + "Edit"].setEnabled(False)
        else:
            if key != "bankProtect":
                StateManagement[key + "Edit"].setValidator(validator("positive_real"))
            StateManagement[key + "Edit"].setEnabled(True)
        StateManagement[key + "EditFile"].setEnabled(False)
    elif type == "Variable":
        if key + "Select" in StateManagement.keys():
            StateManagement[key + "Select"].setEnabled(False)
        StateManagement[key + "Edit"].setEnabled(True)
        StateManagement[key + "Edit"].setValidator(None)
        StateManagement[key + "EditFile"].setEnabled(True)


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
    StateManagement = StateStore.instance()
    Label = QLabel(labelString)
    StateManagement[key + "Label"] = Label
    gridLayout.addWidget(Label, row, 0)

    paramTypes = ("Use Default", "Constant", "Variable")
    Type = QComboBox()
    Type.addItems(paramTypes)
    Type.currentIndexChanged.connect(lambda: typeUpdatePar(key))
    StateManagement[key + "Type"] = Type
    gridLayout.addWidget(Type, row, 1)

    if selectList is None:
        fLayout = openFileLayout(key + "Edit", enabled=False)
        StateManagement[key + "Edit"].setEnabled(False)
        gridLayout.addWidget(fLayout, row, 2)
    else:
        Select = QComboBox()
        Select.addItems(selectList)
        Select.setEnabled(False)
        StateManagement[key + "Select"] = Select
        gridLayout.addWidget(Select, row, 2)

        fLayout = openFileLayout(key + "Edit", enabled=False)
        StateManagement[key + "Edit"].setEnabled(False)
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
    StateManagement = StateStore.instance()
    config_file = ConfigFile(config)
    val = config_file.get_float(group, key, 0.0)
    if val > 0.0:
        StateManagement[field + "Active"].setChecked(True)
        StateManagement[field + "Width"].setText(str(val))
    else:
        StateManagement[field + "Active"].setChecked(False)



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
    StateManagement = StateStore.instance()
    parent = QWidget()
    gridly = QGridLayout(parent)
    gridly.setContentsMargins(0, 0, 0, 0)

    myWidget = QLineEdit()
    StateManagement[key] = myWidget
    gridly.addWidget(myWidget, 0, 0)

    openFile = QPushButton(get_icon(f"{ICONS_DIR}/open.png"), "")
    openFile.clicked.connect(lambda: selectFile(key))
    openFile.setEnabled(enabled)
    StateManagement[key + "File"] = openFile
    gridly.addWidget(openFile, 0, 2)

    return parent


def selectFile(key: str) -> None:
    """Select a file or directory via a selection dialog.

    Args:
        key : str
            Short name of the parameter.
    """
    StateManagement = StateStore.instance()
    dnm: str
    if not StateManagement[key + "File"].hasFocus():
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
        StateManagement[key].setText(fil)