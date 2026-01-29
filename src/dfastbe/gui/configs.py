import os
from typing import cast
from pathlib import Path
from configparser import ConfigParser
from PySide6.QtWidgets import (
    QTreeWidgetItem,
    QComboBox,
    QLineEdit,
)

from dfastbe.io.config import ConfigFile
from dfastbe.gui.utils import show_error, typeUpdatePar, addTabForLevel
from dfastbe.io.file_utils import absolute_path
from dfastbe.gui.state_management import StateStore

__all__ = [
    "get_configuration",
    "load_configuration",
    "bankStrengthSwitch",
]


def load_configuration(config_path: Path) -> None:
    """Open a configuration file and update the GUI accordingly.

    This routines opens the specified configuration file and updates the GUI
    to reflect it contents.

    Args:
        config_path (Path):
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