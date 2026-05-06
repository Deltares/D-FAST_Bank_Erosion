from dataclasses import dataclass, field
from configparser import ConfigParser
from dfastbe.io.config import ConfigFile
from dfastbe.gui.state_management import StateStore

from dfastbe.gui.utils import (
    addTabForLevel,
)

@dataclass
class DischargeLevelsTabs:
    """Manages discharge level tabs configuration and loading."""

    config: ConfigParser
    config_file: ConfigFile
    state_management: StateStore = field(init=False, default_factory=StateStore.instance)

    def configure_tabs(self, n_level: int) -> None:
        """Configure tabs for discharge levels.

        Args:
            n_level (int): Number of discharge levels.
        """
        tabs_manager = self.state_management["tabs"]

        # Remove existing level tabs
        for i in range(tabs_manager.count() - 1, 4, -1):
            tabs_manager.removeTab(i)

        # Add tabs for each level and load parameters
        for i in range(n_level):
            istr = str(i + 1)
            addTabForLevel(istr)

            # Load level-specific parameters
            self._load_optional_param(istr + "_shipType", "Erosion", "ShipType" + istr)
            self._load_optional_param(istr + "_shipVeloc", "Erosion", "VShip" + istr)
            self._load_optional_param(istr + "_nShips", "Erosion", "NShip" + istr)
            self._load_optional_param(istr + "_shipNWaves", "Erosion", "NWave" + istr)
            self._load_optional_param(istr + "_shipDraught", "Erosion", "Draught" + istr)
            self._load_optional_param(istr + "_bankSlope", "Erosion", "Slope" + istr)
            self._load_optional_param(istr + "_bankReed", "Erosion", "Reed" + istr)

            txt = self.config_file.get_str("Erosion", "EroVol" + istr, default="")
            self.state_management[istr + "_eroVolEdit"].setText(txt)

    def _load_optional_param(self, field: str, group: str, key: str) -> None:
        """Load an optional parameter from configuration.

        Args:
            field (str): Short name of the parameter (e.g., "1_shipType").
            group (str): Name of the group in the configuration (e.g., "Erosion").
            key (str): Name of the key in the configuration group (e.g., "ShipType1").
        """
        value_str = self.config_file.get_str(group, key, "")

        if value_str == "":
            self.state_management[field + "Type"].setCurrentText("Use Default")
            self.state_management[field + "Edit"].setText("")
        else:
            try:
                val = float(value_str)
                self.state_management[field + "Type"].setCurrentText("Constant")
                if field + "Select" in self.state_management.keys():
                    index = int(val) - 1  # shipType 1 -> index 0
                    self.state_management[field + "Select"].setCurrentIndex(index)
                else:
                    self.state_management[field + "Edit"].setText(value_str)
            except ValueError:
                self.state_management[field + "Type"].setCurrentText("Variable")
                self.state_management[field + "Edit"].setText(value_str)