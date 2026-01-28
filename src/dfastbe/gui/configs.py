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
    state_management = StateStore.instance()
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
        state_management["chainFileEdit"].setText(section["RiverKM"])
        study_range = config_file.get_range("General", "Boundaries")
        state_management["startRange"].setText(str(study_range[0]))
        state_management["endRange"].setText(str(study_range[1]))
        state_management["bankDirEdit"].setText(section["BankDir"])
        bank_file = config_file.get_str("General", "BankFile", default="bankfile")
        state_management["bankFileName"].setText(bank_file)
        flag = config_file.get_bool("General", "Plotting", default=True)
        state_management["makePlotsEdit"].setChecked(flag)
        flag = config_file.get_bool("General", "SavePlots", default=True)
        state_management["savePlotsEdit"].setChecked(flag)
        flag = config_file.get_bool("General", "SaveZoomPlots", default=False)
        state_management["saveZoomPlotsEdit"].setChecked(flag)
        zoom_step_km = config_file.get_float("General", "ZoomStepKM", default=1.0)
        state_management["zoomPlotsRangeEdit"].setText(str(zoom_step_km))
        fig_dir = config_file.get_str(
            "General",
            "FigureDir",
            default=absolute_path(rootdir, "figures"),
        )
        state_management["figureDirEdit"].setText(fig_dir)
        flag = config_file.get_bool("General", "ClosePlots", default=False)
        state_management["closePlotsEdit"].setChecked(flag)
        flag = config_file.get_bool("General", "DebugOutput", default=False)
        state_management["debugOutputEdit"].setChecked(flag)

        section = config["Detect"]
        state_management["simFileEdit"].setText(section["SimFile"])
        water_depth = config_file.get_float("Detect", "WaterDepth", default=0.0)
        state_management["waterDepth"].setText(str(water_depth))
        n_bank = config_file.get_int("Detect", "NBank", default=0, positive=True)
        d_lines = config_file.get_bank_search_distances(n_bank)
        state_management["searchLines"].invisibleRootItem().takeChildren()
        for i in range(n_bank):
            istr = str(i + 1)
            file_name = config_file.get_str("Detect", "Line" + istr)
            c1 = QTreeWidgetItem(
                state_management["searchLines"], [istr, file_name, str(d_lines[i])]
            )
        if n_bank > 0:
            state_management["searchLinesEdit"].setEnabled(True)
            state_management["searchLinesRemove"].setEnabled(True)

        section = config["Erosion"]
        state_management["tErosion"].setText(section["TErosion"])
        state_management["riverAxisEdit"].setText(section["RiverAxis"])
        state_management["fairwayEdit"].setText(section["Fairway"])
        state_management["chainageOutStep"].setText(section["OutputInterval"])
        state_management["outDirEdit"].setText(section["OutputDir"])
        bank_new = config_file.get_str("Erosion", "BankNew", default="banknew")
        state_management["newBankFile"].setText(bank_new)
        bank_eq = config_file.get_str("Erosion", "BankEq", default="bankeq")
        state_management["newEqBankFile"].setText(bank_eq)
        txt = config_file.get_str("Erosion", "EroVol", default="erovol_standard.evo")
        state_management["eroVol"].setText(txt)
        txt = config_file.get_str("Erosion", "EroVolEqui", default="erovol_eq.evo")
        state_management["eroVolEqui"].setText(txt)

        n_level = config_file.get_int("Erosion", "NLevel", default=0, positive=True)
        state_management["discharges"].invisibleRootItem().takeChildren()
        for i in range(n_level):
            istr = str(i + 1)
            file_name = config_file.get_str("Erosion", "SimFile" + istr)
            prob = config_file.get_str("Erosion", "PDischarge" + istr)
            c1 = QTreeWidgetItem(state_management["discharges"], [istr, file_name, prob])
        if n_level > 0:
            state_management["dischargesEdit"].setEnabled(True)
            state_management["dischargesRemove"].setEnabled(True)

        state_management["refLevel"].validator().setTop(n_level)
        state_management["refLevel"].setText(section["RefLevel"])

        setParam("shipType", config, "Erosion", "ShipType")
        setParam("shipVeloc", config, "Erosion", "VShip")
        setParam("nShips", config, "Erosion", "NShip")
        setParam("shipNWaves", config, "Erosion", "NWave", "5")
        setParam("shipDraught", config, "Erosion", "Draught")
        setParam("wavePar0", config, "Erosion", "Wave0", "200.0")
        wave0 = config_file.get_str("Erosion", "Wave0", "200.0")
        setParam("wavePar1", config_file.config, "Erosion", "Wave1", wave0)

        use_bank_type = config_file.get_bool("Erosion", "Classes", default=True)
        state_management["bankType"].setEnabled(use_bank_type)
        state_management["bankTypeType"].setEnabled(use_bank_type)
        state_management["bankTypeEdit"].setEnabled(use_bank_type)
        state_management["bankTypeEditFile"].setEnabled(use_bank_type)
        state_management["bankShear"].setEnabled(not use_bank_type)
        state_management["bankShearType"].setEnabled(not use_bank_type)
        state_management["bankShearEdit"].setEnabled(not use_bank_type)
        state_management["bankShearEditFile"].setEnabled(not use_bank_type)

        if use_bank_type:
            state_management["strengthPar"].setCurrentText("Bank Type")
            bankStrengthSwitch()
            setParam("bankType", config_file.config, "Erosion", "BankType")
        else:
            state_management["strengthPar"].setCurrentText("Critical Shear Stress")
            bankStrengthSwitch()
            setParam("bankShear", config, "Erosion", "BankType")
        setParam("bankProtect", config, "Erosion", "ProtectionLevel", "-1000")
        setParam("bankSlope", config, "Erosion", "Slope", "20.0")
        setParam("bankReed", config, "Erosion", "Reed", "0.0")

        setFilter("velFilter", config, "Erosion", "VelFilterDist")
        setFilter("bedFilter", config, "Erosion", "BedFilterDist")

        tabs = state_management["tabs"]
        for i in range(tabs.count() - 1, 4, -1):
            tabs.removeTab(i)

        for i in range(n_level):
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
            state_management[istr + "_eroVolEdit"].setText(txt)

    else:
        show_error(f"Unsupported version number {version} in the file {config_path}!")


def get_configuration() -> ConfigParser:
    """Extract a configuration from the GUI.

    Returns
    -------
    config : ConfigParser
        Configuration for the D-FAST Bank Erosion analysis.
    """
    state_management = StateStore.instance()
    config = ConfigParser()
    config.optionxform = str  # case sensitive configuration

    config.add_section("General")
    config["General"]["Version"] = "1.0"
    config["General"]["RiverKM"] = state_management["chainFileEdit"].text()
    config["General"]["Boundaries"] = (
            state_management["startRange"].text() + ":" + state_management["endRange"].text()
    )
    config["General"]["BankDir"] = state_management["bankDirEdit"].text()
    config["General"]["BankFile"] = state_management["bankFileName"].text()
    config["General"]["Plotting"] = str(state_management["makePlotsEdit"].isChecked())
    config["General"]["SavePlots"] = str(state_management["savePlotsEdit"].isChecked())
    config["General"]["SaveZoomPlots"] = str(state_management["saveZoomPlotsEdit"].isChecked())
    config["General"]["ZoomStepKM"] = state_management["zoomPlotsRangeEdit"].text()
    config["General"]["FigureDir"] = state_management["figureDirEdit"].text()
    config["General"]["ClosePlots"] = str(state_management["closePlotsEdit"].isChecked())
    config["General"]["DebugOutput"] = str(state_management["debugOutputEdit"].isChecked())

    config.add_section("Detect")
    config["Detect"]["SimFile"] = state_management["simFileEdit"].text()
    config["Detect"]["WaterDepth"] = state_management["waterDepth"].text()
    nbank = state_management["searchLines"].topLevelItemCount()
    config["Detect"]["NBank"] = str(nbank)
    dlines = "[ "
    for i in range(nbank):
        istr = str(i + 1)
        config["Detect"]["Line" + istr] = state_management["searchLines"].topLevelItem(i).text(1)
        dlines += state_management["searchLines"].topLevelItem(i).text(2) + ", "
    dlines = dlines[:-2] + " ]"
    config["Detect"]["DLines"] = dlines

    config.add_section("Erosion")
    config["Erosion"]["TErosion"] = state_management["tErosion"].text()
    config["Erosion"]["RiverAxis"] = state_management["riverAxisEdit"].text()
    config["Erosion"]["Fairway"] = state_management["fairwayEdit"].text()
    config["Erosion"]["OutputInterval"] = state_management["chainageOutStep"].text()
    config["Erosion"]["OutputDir"] = state_management["outDirEdit"].text()
    config["Erosion"]["BankNew"] = state_management["newBankFile"].text()
    config["Erosion"]["BankEq"] = state_management["newEqBankFile"].text()
    config["Erosion"]["EroVol"] = state_management["eroVol"].text()
    config["Erosion"]["EroVolEqui"] = state_management["eroVolEqui"].text()

    if state_management["shipTypeType"].currentText() == "Constant":
        config["Erosion"]["ShipType"] = str(
            state_management["shipTypeSelect"].currentIndex() + 1
        )  # index 0 -> shipType 1
    else:
        config["Erosion"]["ShipType"] = state_management["shipTypeEdit"].text()
    config["Erosion"]["VShip"] = state_management["shipVelocEdit"].text()
    config["Erosion"]["NShip"] = state_management["nShipsEdit"].text()
    config["Erosion"]["NWaves"] = state_management["shipNWavesEdit"].text()
    config["Erosion"]["Draught"] = state_management["shipDraughtEdit"].text()
    config["Erosion"]["Wave0"] = state_management["wavePar0Edit"].text()
    config["Erosion"]["Wave1"] = state_management["wavePar1Edit"].text()

    if state_management["strengthPar"].currentText() == "Bank Type":
        config["Erosion"]["Classes"] = "true"
        if state_management["bankTypeType"].currentText() == "Constant":
            config["Erosion"]["BankType"] = state_management["bankTypeSelect"].currentIndex()
        else:
            config["Erosion"]["BankType"] = state_management["bankTypeEdit"].text()
    else:
        config["Erosion"]["Classes"] = "false"
        config["Erosion"]["BankType"] = state_management["bankShearEdit"].text()
    config["Erosion"]["ProtectionLevel"] = state_management["bankProtectEdit"].text()
    config["Erosion"]["Slope"] = state_management["bankSlopeEdit"].text()
    config["Erosion"]["Reed"] = state_management["bankReedEdit"].text()

    if state_management["velFilterActive"].isChecked():
        config["Erosion"]["VelFilterDist"] = state_management["velFilterWidth"].text()
    if state_management["bedFilterActive"].isChecked():
        config["Erosion"]["BedFilterDist"] = state_management["bedFilterWidth"].text()

    nlevel = state_management["discharges"].topLevelItemCount()
    config["Erosion"]["NLevel"] = str(nlevel)
    config["Erosion"]["RefLevel"] = state_management["refLevel"].text()
    for i in range(nlevel):
        istr = str(i + 1)
        config["Erosion"]["SimFile" + istr] = (
            state_management["discharges"].topLevelItem(i).text(1)
        )
        config["Erosion"]["PDischarge" + istr] = (
            state_management["discharges"].topLevelItem(i).text(2)
        )
        if state_management[istr + "_shipTypeType"].currentText() != "Use Default":
            if state_management[istr + "_shipTypeType"].currentText() == "Constant":
                config["Erosion"]["ShipType" + istr] = (
                        state_management[istr + "_shipTypeSelect"].currentIndex() + 1
                )  # index 0 -> shipType 1
            else:
                config["Erosion"]["ShipType" + istr] = state_management[
                    istr + "_shipTypeEdit"
                    ].text()
        if state_management[istr + "_shipVelocType"].currentText() != "Use Default":
            config["Erosion"]["VShip" + istr] = state_management[istr + "_shipVelocEdit"].text()
        if state_management[istr + "_nShipsType"].currentText() != "Use Default":
            config["Erosion"]["NShip" + istr] = state_management[istr + "_nShipsEdit"].text()
        if state_management[istr + "_shipNWavesType"].currentText() != "Use Default":
            config["Erosion"]["NWaves" + istr] = state_management[istr + "_shipNWavesEdit"].text()
        if state_management[istr + "_shipDraughtType"].currentText() != "Use Default":
            config["Erosion"]["Draught" + istr] = state_management[
                istr + "_shipDraughtEdit"
                ].text()
        if state_management[istr + "_bankSlopeType"].currentText() != "Use Default":
            config["Erosion"]["Slope" + istr] = state_management[istr + "_bankSlopeEdit"].text()
        if state_management[istr + "_bankReedType"].currentText() != "Use Default":
            config["Erosion"]["Reed" + istr] = state_management[istr + "_bankReedEdit"].text()
        if state_management[istr + "_eroVolEdit"].text() != "":
            config["Erosion"]["EroVol" + istr] = state_management[istr + "_eroVolEdit"].text()
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
    state_management = StateStore.instance()
    config_file = ConfigFile(config)
    config_value = config_file.get_str(group, key, default)

    try:
        val = float(config_value)
        cast(QComboBox, state_management[field + "Type"]).setCurrentText("Constant")
        if field + "Select" in state_management.keys():
            int_value = int(val)
            if field == "shipType":
                int_value = int_value - 1
            cast(QComboBox, state_management[field + "Select"]).setCurrentIndex(int_value)
        else:
            cast(QLineEdit, state_management[field + "Edit"]).setText(config_value)
    except:
        cast(QComboBox, state_management[field + "Type"]).setCurrentText("Variable")
        cast(QLineEdit, state_management[field + "Edit"]).setText(config_value)


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
    state_management = StateStore.instance()
    config_file = ConfigFile(config)
    str = config_file.get_str(group, key, "")
    if str == "":
        state_management[field + "Type"].setCurrentText("Use Default")
        state_management[field + "Edit"].setText("")
    else:
        try:
            val = float(str)
            state_management[field + "Type"].setCurrentText("Constant")
            if field + "Select" in state_management.keys():
                ival = int(val) - 1  # shipType 1 -> index 0
                state_management[field + "Select"].setCurrentIndex(ival)
            else:
                state_management[field + "Edit"].setText(str)
        except:
            state_management[field + "Type"].setCurrentText("Variable")
            state_management[field + "Edit"].setText(str)


def bankStrengthSwitch() -> None:
    """Implements the dialog settings depending on the bank strength specification method."""
    state_management = StateStore.instance()
    type = state_management["strengthPar"].currentText()
    if type == "Bank Type":
        state_management["bankType"].setEnabled(True)
        state_management["bankTypeType"].setEnabled(True)
        typeUpdatePar("bankType")
        state_management["bankShear"].setEnabled(False)
        state_management["bankShearType"].setEnabled(False)
        state_management["bankShearEdit"].setText("")
        state_management["bankShearEdit"].setEnabled(False)
        state_management["bankShearEditFile"].setEnabled(False)
    elif type == "Critical Shear Stress":
        state_management["bankShear"].setEnabled(True)
        state_management["bankShearType"].setEnabled(True)
        state_management["bankShearEdit"].setEnabled(True)
        typeUpdatePar("bankShear")
        state_management["bankType"].setEnabled(False)
        state_management["bankTypeType"].setEnabled(False)
        state_management["bankTypeSelect"].setEnabled(False)
        state_management["bankTypeEdit"].setText("")
        state_management["bankTypeEdit"].setEnabled(False)
        state_management["bankTypeEditFile"].setEnabled(False)


def addTabForLevel(istr: str) -> None:
    """Create the tab for the settings associated with simulation i.

    Args:
        istr : str
            String representation of the simulation number.
    """
    state_management = StateStore.instance()
    newWidget = QWidget()
    newLayout = QGridLayout(newWidget)
    state_management["tabs"].addTab(newWidget, "Level " + istr)

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
    state_management[istr + "_eroVol"] = Label
    newLayout.addWidget(Label, 8, 0)
    Edit = QLineEdit()
    state_management[istr + "_eroVolEdit"] = Edit
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
    state_management = StateStore.instance()
    type = state_management[key + "Type"].currentText()
    state_management[key + "Edit"].setText("")
    if type == "Use Default":
        state_management[key + "Edit"].setValidator(None)
        state_management[key + "Edit"].setEnabled(False)
        state_management[key + "EditFile"].setEnabled(False)
        if key + "Select" in state_management.keys():
            state_management[key + "Select"].setEnabled(False)
    elif type == "Constant":
        if key + "Select" in state_management.keys():
            state_management[key + "Select"].setEnabled(True)
            state_management[key + "Edit"].setEnabled(False)
        else:
            if key != "bankProtect":
                state_management[key + "Edit"].setValidator(validator("positive_real"))
            state_management[key + "Edit"].setEnabled(True)
        state_management[key + "EditFile"].setEnabled(False)
    elif type == "Variable":
        if key + "Select" in state_management.keys():
            state_management[key + "Select"].setEnabled(False)
        state_management[key + "Edit"].setEnabled(True)
        state_management[key + "Edit"].setValidator(None)
        state_management[key + "EditFile"].setEnabled(True)


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
    state_management = StateStore.instance()
    Label = QLabel(labelString)
    state_management[key + "Label"] = Label
    gridLayout.addWidget(Label, row, 0)

    paramTypes = ("Use Default", "Constant", "Variable")
    Type = QComboBox()
    Type.addItems(paramTypes)
    Type.currentIndexChanged.connect(lambda: typeUpdatePar(key))
    state_management[key + "Type"] = Type
    gridLayout.addWidget(Type, row, 1)

    if selectList is None:
        fLayout = openFileLayout(key + "Edit", enabled=False)
        state_management[key + "Edit"].setEnabled(False)
        gridLayout.addWidget(fLayout, row, 2)
    else:
        Select = QComboBox()
        Select.addItems(selectList)
        Select.setEnabled(False)
        state_management[key + "Select"] = Select
        gridLayout.addWidget(Select, row, 2)

        fLayout = openFileLayout(key + "Edit", enabled=False)
        state_management[key + "Edit"].setEnabled(False)
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
    state_management = StateStore.instance()
    config_file = ConfigFile(config)
    val = config_file.get_float(group, key, 0.0)
    if val > 0.0:
        state_management[field + "Active"].setChecked(True)
        state_management[field + "Width"].setText(str(val))
    else:
        state_management[field + "Active"].setChecked(False)



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
    state_management = StateStore.instance()
    parent = QWidget()
    gridly = QGridLayout(parent)
    gridly.setContentsMargins(0, 0, 0, 0)

    myWidget = QLineEdit()
    state_management[key] = myWidget
    gridly.addWidget(myWidget, 0, 0)

    openFile = QPushButton(get_icon(f"{ICONS_DIR}/open.png"), "")
    openFile.clicked.connect(lambda: selectFile(key))
    openFile.setEnabled(enabled)
    state_management[key + "File"] = openFile
    gridly.addWidget(openFile, 0, 2)

    return parent


def selectFile(key: str) -> None:
    """Select a file or directory via a selection dialog.

    Args:
        key : str
            Short name of the parameter.
    """
    state_management = StateStore.instance()
    dnm: str
    if not state_management[key + "File"].hasFocus():
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
        state_management[key].setText(fil)