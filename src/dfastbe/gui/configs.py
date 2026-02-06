from __future__ import annotations
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
    "ConfigurationExporter",
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


class ConfigurationExporter:
    """Exports GUI state to a ConfigParser configuration.

    This class encapsulates the logic for building a configuration from the
    StateStore, organizing it into logical sections with single-responsibility methods.
    """

    def __init__(self, state_store: StateStore):
        """Initialize the exporter with a state store.

        Args:
            state_store: The StateStore instance containing GUI state.
        """
        self.state = state_store
        self.config = ConfigParser()
        self.config.optionxform = str  # case sensitive configuration

    def build(self) -> ConfigParser:
        """Build and return the complete configuration.

        Returns:
            ConfigParser: Complete configuration for D-FAST Bank Erosion analysis.
        """
        self._build_general_section()
        self._build_detect_section()
        self._build_erosion_section()
        return self.config

    def _build_general_section(self) -> None:
        """Build the [General] section of the configuration."""
        self.config.add_section("General")
        section = self.config["General"]

        section["Version"] = "1.0"
        section["RiverKM"] = self.state["chainFileEdit"].text()
        section["Boundaries"] = (
            self.state["startRange"].text() + ":" + self.state["endRange"].text()
        )
        section["BankDir"] = self.state["bankDirEdit"].text()
        section["BankFile"] = self.state["bankFileName"].text()
        section["Plotting"] = str(self.state["makePlotsEdit"].isChecked())
        section["SavePlots"] = str(self.state["savePlotsEdit"].isChecked())
        section["SaveZoomPlots"] = str(self.state["saveZoomPlotsEdit"].isChecked())
        section["ZoomStepKM"] = self.state["zoomPlotsRangeEdit"].text()
        section["FigureDir"] = self.state["figureDirEdit"].text()
        section["ClosePlots"] = str(self.state["closePlotsEdit"].isChecked())
        section["DebugOutput"] = str(self.state["debugOutputEdit"].isChecked())

    def _build_detect_section(self) -> None:
        """Build the [Detect] section of the configuration."""
        self.config.add_section("Detect")
        section = self.config["Detect"]

        section["SimFile"] = self.state["simFileEdit"].text()
        section["WaterDepth"] = self.state["waterDepth"].text()

        nbank = self.state["searchLines"].topLevelItemCount()
        section["NBank"] = str(nbank)

        dlines = "[ "
        for i in range(nbank):
            istr = str(i + 1)
            section["Line" + istr] = self.state["searchLines"].topLevelItem(i).text(1)
            dlines += self.state["searchLines"].topLevelItem(i).text(2) + ", "
        dlines = dlines[:-2] + " ]"
        section["DLines"] = dlines

    def _build_erosion_section(self) -> None:
        """Build the [Erosion] section of the configuration."""
        self.config.add_section("Erosion")
        section = self.config["Erosion"]

        # Basic erosion parameters
        section["TErosion"] = self.state["tErosion"].text()
        section["RiverAxis"] = self.state["riverAxisEdit"].text()
        section["Fairway"] = self.state["fairwayEdit"].text()
        section["OutputInterval"] = self.state["chainageOutStep"].text()
        section["OutputDir"] = self.state["outDirEdit"].text()
        section["BankNew"] = self.state["newBankFile"].text()
        section["BankEq"] = self.state["newEqBankFile"].text()
        section["EroVol"] = self.state["eroVol"].text()
        section["EroVolEqui"] = self.state["eroVolEqui"].text()

        # Ship parameters
        self._build_ship_parameters(section)

        # Bank strength parameters
        self._build_bank_strength_parameters(section)

        # Filter parameters
        self._build_filters(section)

        # Discharge levels
        self._build_erosion_levels(section)

    def _build_ship_parameters(self, section) -> None:
        """Build ship-related parameters in the Erosion section.

        Args:
            section: The ConfigParser section to populate.
        """
        if self.state["shipTypeType"].currentText() == "Constant":
            section["ShipType"] = str(
                self.state["shipTypeSelect"].currentIndex() + 1
            )  # index 0 -> shipType 1
        else:
            section["ShipType"] = self.state["shipTypeEdit"].text()

        section["VShip"] = self.state["shipVelocEdit"].text()
        section["NShip"] = self.state["nShipsEdit"].text()
        section["NWaves"] = self.state["shipNWavesEdit"].text()
        section["Draught"] = self.state["shipDraughtEdit"].text()
        section["Wave0"] = self.state["wavePar0Edit"].text()
        section["Wave1"] = self.state["wavePar1Edit"].text()

    def _build_bank_strength_parameters(self, section) -> None:
        """Build bank strength parameters in the Erosion section.

        Args:
            section: The ConfigParser section to populate.
        """
        if self.state["strengthPar"].currentText() == "Bank Type":
            section["Classes"] = "true"
            if self.state["bankTypeType"].currentText() == "Constant":
                section["BankType"] = str(self.state["bankTypeSelect"].currentIndex())
            else:
                section["BankType"] = self.state["bankTypeEdit"].text()
        else:
            section["Classes"] = "false"
            section["BankType"] = self.state["bankShearEdit"].text()

        section["ProtectionLevel"] = self.state["bankProtectEdit"].text()
        section["Slope"] = self.state["bankSlopeEdit"].text()
        section["Reed"] = self.state["bankReedEdit"].text()

    def _build_filters(self, section) -> None:
        """Build filter parameters in the Erosion section.

        Args:
            section: The ConfigParser section to populate.
        """
        if self.state["velFilterActive"].isChecked():
            section["VelFilterDist"] = self.state["velFilterWidth"].text()
        if self.state["bedFilterActive"].isChecked():
            section["BedFilterDist"] = self.state["bedFilterWidth"].text()

    def _build_erosion_levels(self, section) -> None:
        """Build discharge level parameters in the Erosion section.

        Args:
            section: The ConfigParser section to populate.
        """
        nlevel = self.state["discharges"].topLevelItemCount()
        section["NLevel"] = str(nlevel)
        section["RefLevel"] = self.state["refLevel"].text()

        for i in range(nlevel):
            istr = str(i + 1)
            section["SimFile" + istr] = (
                self.state["discharges"].topLevelItem(i).text(1)
            )
            section["PDischarge" + istr] = (
                self.state["discharges"].topLevelItem(i).text(2)
            )

            # Optional ship type parameter
            if self.state[istr + "_shipTypeType"].currentText() != "Use Default":
                if self.state[istr + "_shipTypeType"].currentText() == "Constant":
                    section["ShipType" + istr] = str(
                        self.state[istr + "_shipTypeSelect"].currentIndex() + 1
                    )  # index 0 -> shipType 1
                else:
                    section["ShipType" + istr] = self.state[istr + "_shipTypeEdit"].text()

            # Optional velocity parameter
            if self.state[istr + "_shipVelocType"].currentText() != "Use Default":
                section["VShip" + istr] = self.state[istr + "_shipVelocEdit"].text()

            # Optional number of ships parameter
            if self.state[istr + "_nShipsType"].currentText() != "Use Default":
                section["NShip" + istr] = self.state[istr + "_nShipsEdit"].text()

            # Optional number of waves parameter
            if self.state[istr + "_shipNWavesType"].currentText() != "Use Default":
                section["NWaves" + istr] = self.state[istr + "_shipNWavesEdit"].text()

            # Optional draught parameter
            if self.state[istr + "_shipDraughtType"].currentText() != "Use Default":
                section["Draught" + istr] = self.state[istr + "_shipDraughtEdit"].text()

            # Optional slope parameter
            if self.state[istr + "_bankSlopeType"].currentText() != "Use Default":
                section["Slope" + istr] = self.state[istr + "_bankSlopeEdit"].text()

            # Optional reed parameter
            if self.state[istr + "_bankReedType"].currentText() != "Use Default":
                section["Reed" + istr] = self.state[istr + "_bankReedEdit"].text()

            # Optional erosion volume file
            if self.state[istr + "_eroVolEdit"].text() != "":
                section["EroVol" + istr] = self.state[istr + "_eroVolEdit"].text()


def get_configuration() -> ConfigParser:
    """Extract a configuration from the GUI.

    Returns
    -------
    config : ConfigParser
        Configuration for the D-FAST Bank Erosion analysis.
    """
    exporter = ConfigurationExporter(StateStore.instance())
    return exporter.build()


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