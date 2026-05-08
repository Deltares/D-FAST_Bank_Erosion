from __future__ import annotations
from typing import ClassVar, cast
from pathlib import Path
from configparser import ConfigParser, SectionProxy
from PySide6.QtWidgets import (
    QTreeWidgetItem,
    QComboBox,
    QLineEdit,
)
from dataclasses import dataclass, field

from dfastbe.io.config import (
    ConfigFile,
    ConfigFileError
)
from dfastbe.gui.utils import (
    typeUpdatePar,
)
from dfastbe.io.file_utils import absolute_path
from dfastbe.gui.state_management import StateStore
from dfastbe.gui.tabs.discharge_levels import DischargeLevelsTabs

__all__ = [
    "get_configuration",
    "bankStrengthSwitch",
    "ConfigurationLoader",
    "ConfigurationExporter",
]

USE_DEFAULT = "Use Default"

BANK_TYPE = "Bank Type"
CRITICAL_SHEAR_STRESS = "Critical Shear Stress"
CONSTANT = "Constant"


@dataclass
class ConfigurationLoader:
    """Load a configuration file and apply its values to the GUI state.

    The loader reads a D-FAST Bank Erosion configuration from ``config_path``
    and immediately synchronizes the relevant widgets stored in
    :class:`~dfastbe.gui.state_management.StateStore`.

    The loading process is split into the main configuration sections used by
    the application:

    * ``General`` for project-wide paths and output settings
    * ``Detect`` for bank detection input and search-line settings
    * ``Erosion`` for erosion, discharge, ship, and bank-strength parameters

    Attributes:
        config_path: Path to the configuration file that will be loaded.
        state_management: Shared GUI state store used to access and update
            widgets by name.
    """
    config_path: Path
    state_management: StateStore = field(init=False, default_factory=StateStore.instance)

    _REQUIRED_SECTIONS: ClassVar[dict[str, list[str]]] = {
        "General": ["Version", "RiverKM", "Boundaries", "BankDir"],
        "Detect": ["SimFile"],
        "Erosion": [
            "TErosion",
            "RiverAxis",
            "Fairway",
            "OutputInterval",
            "OutputDir",
            "RefLevel",
        ],
    }

    def _find_missing(self) -> tuple[list[str], dict[str, list[str]]]:
        """Return missing sections and missing keys per existing section."""
        missing_sections: list[str] = []
        missing_elements: dict[str, list[str]] = {}
        for section, keys in self._REQUIRED_SECTIONS.items():
            if section not in self.config:
                missing_sections.append(section)
                continue
            for elem in keys:
                if elem not in self.config[section]:
                    missing_elements.setdefault(section, []).append(elem)
        return missing_sections, missing_elements

    def _validate_configuration(self) -> None:
        """Validate the loaded configuration for required sections and keys.

        Only keys that are read without a default value by their corresponding `load`
        method are added here. The other keys are considered optional.

        Raises:
            ConfigFileError: If an expected section or element are not in the config file
        """
        missing_sections, missing_elements = self._find_missing()
        if not (missing_sections or missing_elements):
            return

        messages: list[str] = []
        if missing_sections:
            messages.append(
                f"The following sections are missing: {', '.join(missing_sections)}"
            )
        messages.extend(
            f"Section {section} misses the following elements: {', '.join(elems)}"
            for section, elems in missing_elements.items()
        )
        raise ConfigFileError(
            f"Unsupported or invalid configuration file: {'; '.join(messages)}."
        )

    def __post_init__(self):
        """Read and validate the configuration file and load all supported sections.

        The configuration is parsed once into ``self.config_file`` and the raw
        :class:`configparser.ConfigParser` object is stored in ``self.config``
        for convenience. After that, the configuration is validated for all the
        non-default values and finally the General, Detect, and Erosion sections
        are loaded to the GUI in a fixed order.
        """
        self.config_file = ConfigFile.read(self.config_path)
        self.config = self.config_file.config
        self._validate_configuration()

        self._load_general_section()
        self._load_detect_section()
        self._load_erosion_section()

    def _load_general_section(self) -> None:
        """Load the General section from configuration."""
        section = self.config["General"]
        self.state_management["riverKMEdit"].setText(section["RiverKM"])

        study_range = self.config_file.get_range("General", "Boundaries")
        self.state_management["startRange"].setText(str(study_range[0]))
        self.state_management["endRange"].setText(str(study_range[1]))

        self.state_management["bankDirEdit"].setText(section["BankDir"])

        bank_file = self.config_file.get_str("General", "BankFile", default="bankfile")
        self.state_management["bankFileName"].setText(bank_file)

        flag = self.config_file.get_bool("General", "Plotting", default=True)
        self.state_management["makePlotsEdit"].setChecked(flag)

        flag = self.config_file.get_bool("General", "SavePlots", default=True)
        self.state_management["savePlotsEdit"].setChecked(flag)

        flag = self.config_file.get_bool("General", "SaveZoomPlots", default=False)
        self.state_management["saveZoomPlotsEdit"].setChecked(flag)

        zoom_step_km = self.config_file.get_float("General", "ZoomStepKM", default=1.0)
        self.state_management["zoomPlotsRangeEdit"].setText(str(zoom_step_km))

        fig_dir = self.config_file.get_str(
            "General",
            "FigureDir",
            default=absolute_path(self.config_file.root_dir, "figures"),
        )
        self.state_management["figureDirEdit"].setText(fig_dir)

        flag = self.config_file.get_bool("General", "ClosePlots", default=False)
        self.state_management["closePlotsEdit"].setChecked(flag)

        flag = self.config_file.get_bool("General", "DebugOutput", default=False)
        self.state_management["debugOutputEdit"].setChecked(flag)

    def _load_detect_section(self) -> None:
        """Load the Detect section from configuration."""
        section = self.config["Detect"]
        self.state_management["simFileEdit"].setText(section["SimFile"])

        water_depth = self.config_file.get_float("Detect", "WaterDepth", default=0.0)
        self.state_management["waterDepth"].setText(str(water_depth))

        n_bank = self.config_file.get_int("Detect", "NBank", default=0, positive=True)
        self._load_search_lines(n_bank)

    def _load_erosion_section(self) -> None:
        """Load the Erosion section from configuration."""
        section = self.config["Erosion"]

        # Load basic erosion parameters
        self.state_management["tErosion"].setText(section["TErosion"])
        self.state_management["riverAxisEdit"].setText(section["RiverAxis"])
        self.state_management["fairwayEdit"].setText(section["Fairway"])
        self.state_management["chainageOutStep"].setText(section["OutputInterval"])
        self.state_management["outDirEdit"].setText(section["OutputDir"])

        bank_new = self.config_file.get_str("Erosion", "BankNew", default="banknew")
        self.state_management["newBankFile"].setText(bank_new)

        bank_eq = self.config_file.get_str("Erosion", "BankEq", default="bankeq")
        self.state_management["newEqBankFile"].setText(bank_eq)

        txt = self.config_file.get_str("Erosion", "EroVol", default="erovol_standard.evo")
        self.state_management["eroVol"].setText(txt)

        txt = self.config_file.get_str("Erosion", "EroVolEqui", default="erovol_eq.evo")
        self.state_management["eroVolEqui"].setText(txt)

        # Load discharge levels
        n_level = self.config_file.get_int("Erosion", "NLevel", default=0, positive=True)
        self._load_discharges(n_level, section)

        # Load ship and erosion parameters
        self._load_ship_parameters()

        # Load bank strength configuration
        use_bank_type = self.config_file.get_bool("Erosion", "Classes", default=True)
        self._configure_bank_strength(use_bank_type)

        # Load additional erosion parameters
        self._load_param("bankProtect", "Erosion", "ProtectionLevel", "-1000")
        self._load_param("bankSlope", "Erosion", "Slope", "20.0")
        self._load_param("bankReed", "Erosion", "Reed", "0.0")

        # Load filters
        self._load_filter("velFilter", "Erosion", "VelFilterDist")
        self._load_filter("bedFilter", "Erosion", "BedFilterDist")

        # Configure tabs for discharge levels
        tabs_manager = DischargeLevelsTabs(self.config, self.config_file)
        tabs_manager.configure_tabs(n_level)

    def _load_search_lines(self, n_bank: int) -> None:
        """Load search lines from configuration.

        Args:
            n_bank (int): Number of bank lines to load.
        """
        d_lines = self.config_file.get_bank_search_distances(n_bank)
        self.state_management["searchLines"].invisibleRootItem().takeChildren()

        for i in range(n_bank):
            istr = str(i + 1)
            file_name = self.config_file.get_str("Detect", "Line" + istr)
            QTreeWidgetItem(
                self.state_management["searchLines"],
                [istr, file_name, str(d_lines[i])]
            )

        if n_bank > 0:
            self.state_management["searchLinesEdit"].setEnabled(True)
            self.state_management["searchLinesRemove"].setEnabled(True)

    def _load_discharges(self, n_level: int, section) -> None:
        """Load discharge levels from configuration.

        Args:
            n_level (int): Number of discharge levels.
            section: Configuration section containing RefLevel.
        """
        self.state_management["discharges"].invisibleRootItem().takeChildren()

        for i in range(n_level):
            istr = str(i + 1)
            file_name = self.config_file.get_str("Erosion", "SimFile" + istr)
            prob = self.config_file.get_str("Erosion", "PDischarge" + istr)
            QTreeWidgetItem(
                self.state_management["discharges"],
                [istr, file_name, prob]
            )

        if n_level > 0:
            self.state_management["dischargesEdit"].setEnabled(True)
            self.state_management["dischargesRemove"].setEnabled(True)

        self.state_management["refLevel"].validator().setTop(n_level)
        self.state_management["refLevel"].setText(section["RefLevel"])

    def _load_ship_parameters(self) -> None:
        """Load ship-related parameters from configuration."""
        self._load_param("shipType", "Erosion", "ShipType")
        self._load_param("shipVeloc", "Erosion", "VShip")
        self._load_param("nShips", "Erosion", "NShip")
        self._load_param("shipNWaves", "Erosion", "NWave", "5")
        self._load_param("shipDraught", "Erosion", "Draught")
        self._load_param("wavePar0", "Erosion", "Wave0", "200.0")

        wave0 = self.config_file.get_str("Erosion", "Wave0", "200.0")
        self._load_param("wavePar1", "Erosion", "Wave1", wave0)

    def _configure_bank_strength(self, use_bank_type: bool) -> None:
        """Configure bank strength settings based on configuration.

        Args:
            use_bank_type (bool): Whether to use bank type or critical shear stress.
        """
        # Enable/disable appropriate controls
        self.state_management["bankType"].setEnabled(use_bank_type)
        self.state_management["bankTypeType"].setEnabled(use_bank_type)
        self.state_management["bankTypeEdit"].setEnabled(use_bank_type)
        self.state_management["bankTypeEditFile"].setEnabled(use_bank_type)
        self.state_management["bankShear"].setEnabled(not use_bank_type)
        self.state_management["bankShearType"].setEnabled(not use_bank_type)
        self.state_management["bankShearEdit"].setEnabled(not use_bank_type)
        self.state_management["bankShearEditFile"].setEnabled(not use_bank_type)

        if use_bank_type:
            self.state_management["strengthPar"].setCurrentText(BANK_TYPE)
            bankStrengthSwitch()
            self._load_param("bankType", "Erosion", "BankType")
        else:
            self.state_management["strengthPar"].setCurrentText(CRITICAL_SHEAR_STRESS)
            bankStrengthSwitch()
            self._load_param("bankShear", "Erosion", "BankType")


    def _load_filter(self, field: str, group: str, key: str) -> None:
        """Load a filter configuration from the config file.

        Args:
            field (str): Short name of the parameter (e.g., "velFilter").
            group (str): Name of the group in the configuration (e.g., "Erosion").
            key (str): Name of the key in the configuration group (e.g., "VelFilterDist").
        """
        val = self.config_file.get_float(group, key, 0.0)
        if val > 0.0:
            self.state_management[field + "Active"].setChecked(True)
            self.state_management[field + "Width"].setText(str(val))
        else:
            self.state_management[field + "Active"].setChecked(False)

    def _load_param(
            self,
            field: str,
            group: str,
            key: str,
            default: str | None = None
    ) -> None:
        """Load a general parameter from configuration.

        Args:
            field (str): Short name of the parameter (e.g., "shipType").
            group (str): Name of the group in the configuration (e.g., "Erosion").
            key (str): Name of the key in the configuration group (e.g., "ShipType").
            default (str): Default value if the key doesn't exist.
        """
        config_value = self.config_file.get_str(group, key, default)

        try:
            val = float(config_value)
            cast(QComboBox, self.state_management[field + "Type"]).setCurrentText(CONSTANT)
            if field + "Select" in self.state_management:
                int_value = int(val)
                if field == "shipType":
                    int_value = int_value - 1
                cast(QComboBox, self.state_management[field + "Select"]).setCurrentIndex(int_value)
            else:
                cast(QLineEdit, self.state_management[field + "Edit"]).setText(config_value)
        except (ValueError, TypeError):
            cast(QComboBox, self.state_management[field + "Type"]).setCurrentText("Variable")
            cast(QLineEdit, self.state_management[field + "Edit"]).setText(config_value)


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
        section["RiverKM"] = self.state["riverKMEdit"].text()
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

        distances: list[str] = []
        for i in range(nbank):
            item = self.state["searchLines"].topLevelItem(i)
            section[f"Line{i + 1}"] = item.text(1)
            distances.append(item.text(2))

        section["DLines"] = f"[ {', '.join(distances)} ]" if distances else "[ ]"

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

    def _build_ship_parameters(self, section: SectionProxy) -> None:
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

    def _build_bank_strength_parameters(self, section: SectionProxy) -> None:
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

    def _build_filters(self, section: SectionProxy) -> None:
        """Build filter parameters in the Erosion section.

        Args:
            section: The ConfigParser section to populate.
        """
        if self.state["velFilterActive"].isChecked():
            section["VelFilterDist"] = self.state["velFilterWidth"].text()
        if self.state["bedFilterActive"].isChecked():
            section["BedFilterDist"] = self.state["bedFilterWidth"].text()

    _OPTIONAL_PER_LEVEL_PARAMS = (
        # (type_state_suffix, edit_state_suffix, section_key_prefix)
        ("_shipVelocType", "_shipVelocEdit", "VShip"),
        ("_nShipsType", "_nShipsEdit", "NShip"),
        ("_shipNWavesType", "_shipNWavesEdit", "NWaves"),
        ("_shipDraughtType", "_shipDraughtEdit", "Draught"),
        ("_bankSlopeType", "_bankSlopeEdit", "Slope"),
        ("_bankReedType", "_bankReedEdit", "Reed"),
    )

    def _build_erosion_levels(self, section: SectionProxy) -> None:
        """Build discharge level parameters in the Erosion section.

        Args:
            section: The ConfigParser section to populate.
        """
        nlevel = self.state["discharges"].topLevelItemCount()
        section["NLevel"] = str(nlevel)
        section["RefLevel"] = self.state["refLevel"].text()

        for i in range(nlevel):
            self._build_single_erosion_level(section, i + 1)

    def _build_single_erosion_level(self, section: SectionProxy, n: int) -> None:
        """Populate the per-level keys for one discharge level.

        Args:
            section: The ConfigParser section to populate.
            n: One-based discharge level index.
        """
        item = self.state["discharges"].topLevelItem(n - 1)
        section[f"SimFile{n}"] = item.text(1)
        section[f"PDischarge{n}"] = item.text(2)

        self._write_optional_ship_type(section, n)

        for type_suffix, edit_suffix, key in self._OPTIONAL_PER_LEVEL_PARAMS:
            if self.state[f"{n}{type_suffix}"].currentText() != USE_DEFAULT:
                section[f"{key}{n}"] = self.state[f"{n}{edit_suffix}"].text()

        ero_vol = self.state[f"{n}_eroVolEdit"].text()
        if ero_vol != "":
            section[f"EroVol{n}"] = ero_vol

    def _write_optional_ship_type(self, section: SectionProxy, n: int) -> None:
        """Write the optional per-level ShipType key when the user overrides the default.

        Args:
            section: The ConfigParser section to populate.
            n: One-based discharge level index.
        """
        ship_type_kind = self.state[f"{n}_shipTypeType"].currentText()
        if ship_type_kind == USE_DEFAULT:
            return
        if ship_type_kind == "Constant":
            section[f"ShipType{n}"] = str(
                self.state[f"{n}_shipTypeSelect"].currentIndex() + 1
            )  # index 0 -> shipType 1
        else:
            section[f"ShipType{n}"] = self.state[f"{n}_shipTypeEdit"].text()


def get_configuration() -> ConfigParser:
    """Extract a configuration from the GUI.

    Returns:
        Configuration for the D-FAST Bank Erosion analysis.
    """
    exporter = ConfigurationExporter(StateStore.instance())
    return exporter.build()


def setParam(field: str, config, group: str, key: str, default: str = "??") -> None:
    """Update the dialog for a general parameter based on configuration file.

    Args:
        field: Short name of the parameter.
        config: Configuration for the D-FAST Bank Erosion analysis with absolute
            or relative paths.
        group: Name of the group in the configuration.
        key: Name of the key in the configuration group.
        default: Default string if the group/key pair doesn't exist in the
            configuration.
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

    Args:
        field: Short name of the parameter.
        config: Configuration for the D-FAST Bank Erosion analysis with absolute
            or relative paths.
        group: Name of the group in the configuration.
        key: Name of the key in the configuration group.
    """
    state_management = StateStore.instance()
    config_file = ConfigFile(config)
    str = config_file.get_str(group, key, "")
    if str == "":
        state_management[field + "Type"].setCurrentText(USE_DEFAULT)
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
    if type == BANK_TYPE:
        state_management["bankType"].setEnabled(True)
        state_management["bankTypeType"].setEnabled(True)
        typeUpdatePar("bankType")
        state_management["bankShear"].setEnabled(False)
        state_management["bankShearType"].setEnabled(False)
        state_management["bankShearEdit"].setText("")
        state_management["bankShearEdit"].setEnabled(False)
        state_management["bankShearEditFile"].setEnabled(False)
    elif type == CRITICAL_SHEAR_STRESS:
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
    """Update the dialog for a filter based on configuration file.

    Args:
        field: Short name of the parameter.
        config: Configuration for the D-FAST Bank Erosion analysis with absolute
            or relative paths.
        group: Name of the group in the configuration.
        key: Name of the key in the configuration group.
    """
    state_management = StateStore.instance()
    config_file = ConfigFile(config)
    val = config_file.get_float(group, key, 0.0)
    if val > 0.0:
        state_management[field + "Active"].setChecked(True)
        state_management[field + "Width"].setText(str(val))
    else:
        state_management[field + "Active"].setChecked(False)