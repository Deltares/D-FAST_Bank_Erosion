"""
Unit tests for the ConfigurationLoader class.
"""

import pytest
from pathlib import Path
from configparser import ConfigParser
from unittest.mock import Mock, patch, call

from dfastbe.gui.configs import ConfigurationLoader
from dfastbe.gui.state_management import StateStore
from dfastbe.io.config import ConfigFile


class TestConfigurationLoader:
    """Test cases for the ConfigurationLoader class."""

    @pytest.fixture
    def mock_state_store(self):
        """Fixture to create a mock StateStore instance.

        Returns:
            dict: A mock dictionary simulating the StateStore with mock widgets.
        """
        # Initialize StateStore singleton properly
        StateStore._instance = None  # Reset any previous instance
        mock_store = StateStore.initialize()

        # Mock text input widgets
        text_fields = [
            # General section
            "chainFileEdit", "startRange", "endRange", "bankDirEdit",
            "bankFileName", "zoomPlotsRangeEdit", "figureDirEdit",
            # Detect section
            "simFileEdit", "waterDepth",
            # Erosion section
            "tErosion", "riverAxisEdit", "fairwayEdit", "chainageOutStep",
            "outDirEdit", "newBankFile", "newEqBankFile", "eroVol", "eroVolEqui",
            # Bank strength
            "bankTypeEdit", "bankShearEdit",
            # Ship parameters
            "shipTypeEdit", "shipVelocEdit", "nShipsEdit", "shipNWavesEdit",
            "shipDraughtEdit", "wavePar0Edit", "wavePar1Edit",
            # Erosion parameters
            "bankProtectEdit", "bankSlopeEdit", "bankReedEdit",
        ]
        for field in text_fields:
            mock_widget = Mock()
            mock_widget.setText = Mock()
            mock_widget.text = Mock(return_value="")
            mock_store[field] = mock_widget

        # Mock checkbox widgets (QCheckBox-like)
        checkbox_fields = [
            "makePlotsEdit", "savePlotsEdit", "saveZoomPlotsEdit",
            "closePlotsEdit", "debugOutputEdit",
            "velFilterActive", "bedFilterActive"
        ]
        for field in checkbox_fields:
            mock_widget = Mock()
            mock_widget.setChecked = Mock()
            mock_widget.isChecked = Mock(return_value=False)
            mock_store[field] = mock_widget

        # Mock filter width widgets
        for filter_field in ["velFilterWidth", "bedFilterWidth"]:
            mock_widget = Mock()
            mock_widget.setText = Mock()
            mock_widget.text = Mock(return_value="")
            mock_store[filter_field] = mock_widget

        # Mock combo boxes
        combo_fields = [
            "strengthPar", "bankTypeType", "bankShearType",
            # Ship parameter types
            "shipTypeType", "shipVelocType", "nShipsType", "shipNWavesType",
            "shipDraughtType", "wavePar0Type", "wavePar1Type",
            # Erosion parameter types
            "bankProtectType", "bankSlopeType", "bankReedType",
        ]
        for field in combo_fields:
            mock_combo = Mock()
            mock_combo.setCurrentText = Mock()
            mock_combo.currentText = Mock(return_value="")
            mock_store[field] = mock_combo

        # Mock shipTypeSelect combo box specifically
        mock_ship_type_select = Mock()
        mock_ship_type_select.setCurrentIndex = Mock()
        mock_ship_type_select.currentIndex = Mock(return_value=0)
        mock_store["shipTypeSelect"] = mock_ship_type_select

        # Mock tree widgets for searchLines and discharges
        for tree_field in ["searchLines", "discharges"]:
            mock_tree = Mock()
            mock_root = Mock()
            mock_root.takeChildren = Mock()
            mock_tree.invisibleRootItem = Mock(return_value=mock_root)
            mock_tree.topLevelItemCount = Mock(return_value=0)
            mock_store[tree_field] = mock_tree

        # Mock buttons/widgets that can be enabled/disabled
        enable_disable_fields = [
            "searchLinesEdit", "searchLinesRemove",
            "dischargesEdit", "dischargesRemove",
            "bankType", "bankTypeEdit", "bankTypeEditFile",
            "bankShear", "bankShearEdit", "bankShearEditFile"
        ]
        for field in enable_disable_fields:
            mock_widget = Mock()
            mock_widget.setEnabled = Mock()
            mock_store[field] = mock_widget

        # Mock refLevel with validator
        mock_ref_level = Mock()
        mock_ref_level.setText = Mock()
        mock_validator = Mock()
        mock_validator.setTop = Mock()
        mock_ref_level.validator = Mock(return_value=mock_validator)
        mock_store["refLevel"] = mock_ref_level

        # Mock tabs widget
        mock_tabs = Mock()
        mock_tabs.count = Mock(return_value=5)
        mock_tabs.removeTab = Mock()
        mock_store["tabs"] = mock_tabs

        for i in range(1, 5):
            istr = str(i)
            for suffix in ["_eroVolEdit", "_shipType", "_shipVeloc", "_nShips",
                          "_shipNWaves", "_shipDraught", "_bankSlope", "_bankReed"]:
                mock_widget = Mock()
                mock_widget.setText = Mock()
                mock_widget.text = Mock(return_value="")
                mock_store[istr + suffix] = mock_widget

        yield mock_store

        StateStore._instance = None


    @pytest.fixture
    def mock_config_file(self):
        """Fixture to create a mock ConfigFile instance.

        Returns:
            ConfigFile: A mock ConfigFile with sample data.
        """
        config = ConfigParser()
        config.read_dict({
            "General": {
                "Version": "1.0",
                "RiverKM": "inputs/rivkm_20m.xyc",
                "Boundaries": "123.0:128.0",
                "BankDir": "output/banklines",
                "BankFile": "bankfile",
                "Plotting": "True",
                "SavePlots": "True",
                "SaveZoomPlots": "False",
                "ZoomStepKM": "1.0",
                "FigureDir": "output/figures",
                "ClosePlots": "False",
                "DebugOutput": "True",
            },
            "Detect": {
                "SimFile": "test_sim.nc",
                "WaterDepth": "0.5",
                "NBank": "2",
                "Line1": "bank_line_1.xyc",
                "Line2": "bank_line_2.xyc",
                "DLines": "[ 50.0, 75.0 ]",
            },
            "Erosion": {
                "TErosion": "10.0",
                "RiverAxis": "river_axis.xyc",
                "Fairway": "fairway.xyc",
                "OutputInterval": "100.0",
                "OutputDir": "output/erosion",
                "BankNew": "banknew",
                "BankEq": "bankeq",
                "EroVol": "erovol_standard.evo",
                "EroVolEqui": "erovol_eq.evo",
                "NLevel": "2",
                "RefLevel": "1",
                "SimFile1": "discharge_file_1.nc",
                "PDischarge1": "0.6",
                "SimFile2": "discharge_file_2.nc",
                "PDischarge2": "0.4",
                "ShipType": "1",
                "VShip": "5.0",
                "NShip": "100",
                "NWave": "5",
                "Draught": "2.5",
                "Wave0": "200.0",
                "Wave1": "200.0",
                "Classes": "true",
                "BankType": "2",
                "ProtectionLevel": "-1000",
                "Slope": "20.0",
                "Reed": "0.0",
            }
        })

        mock_file = Mock(spec=ConfigFile)
        mock_file.config = config
        mock_file.root_dir = Path("/test")  # Add root_dir
        mock_file.get_range = Mock(return_value=(123.0, 128.0))
        mock_file.get_str = Mock(side_effect=lambda section, key, default=None: {
            ("General", "BankFile"): "bankfile",
            ("General", "FigureDir"): "output/figures",
            ("Detect", "Line1"): "bank_line_1.xyc",
            ("Detect", "Line2"): "bank_line_2.xyc",
            ("Erosion", "BankNew"): "banknew",
            ("Erosion", "BankEq"): "bankeq",
            ("Erosion", "EroVol"): "erovol_standard.evo",
            ("Erosion", "EroVolEqui"): "erovol_eq.evo",
            ("Erosion", "SimFile1"): "discharge_file_1.nc",
            ("Erosion", "PDischarge1"): "0.6",
            ("Erosion", "SimFile2"): "discharge_file_2.nc",
            ("Erosion", "PDischarge2"): "0.4",
            ("Erosion", "Wave0"): "200.0",
        }.get((section, key), default))
        mock_file.get_bool = Mock(side_effect=lambda section, key, default=None: {
            ("General", "Plotting"): True,
            ("General", "SavePlots"): True,
            ("General", "SaveZoomPlots"): False,
            ("General", "ClosePlots"): False,
            ("General", "DebugOutput"): True,
            ("Erosion", "Classes"): True,
        }.get((section, key), default))
        mock_file.get_float = Mock(side_effect=lambda section, key, default=None: {
            ("General", "ZoomStepKM"): 1.0,
            ("Detect", "WaterDepth"): 0.5,
        }.get((section, key), default))
        mock_file.get_int = Mock(side_effect=lambda section, key, default=None, positive=False: {
            ("Detect", "NBank"): 2,
            ("Erosion", "NLevel"): 2,
        }.get((section, key), default))
        mock_file.get_bank_search_distances = Mock(return_value=[50.0, 75.0])

        return mock_file

    @pytest.fixture
    def config_loader(self, mock_state_store, mock_config_file, tmp_path):
        """Fixture to create a ConfigurationLoader instance.

        Args:
            mock_state_store: Mocked StateStore fixture.
            mock_config_file: Mocked ConfigFile fixture.
            tmp_path: Pytest temporary directory.

        Returns:
            ConfigurationLoader: Instance with mocked dependencies.
        """
        config_path = tmp_path / "test_config.cfg"
        config_path.write_text("[General]\nVersion=1.0\n")

        with (patch('dfastbe.gui.configs.ConfigFile.read', return_value=mock_config_file),
             patch('dfastbe.gui.configs.StateStore.instance', return_value=mock_state_store),
             patch('dfastbe.gui.configs.QTreeWidgetItem'),
             patch('dfastbe.gui.configs.addTabForLevel'),
             patch('dfastbe.gui.configs.DischargeLevelsTabs'),
             patch('dfastbe.gui.configs.bankStrengthSwitch')):
                 loader = ConfigurationLoader(config_path)
                 loader.rootdir = str(tmp_path)
                 return loader


    def test_load_general_section_sets_parameters(self, config_loader, mock_state_store):
        """Test that _load_general_section sets the parameters fields correctly."""
        mock_state_store["chainFileEdit"].setText.assert_called_once_with(
            "inputs/rivkm_20m.xyc")
        mock_state_store["startRange"].setText.assert_called_once_with("123.0")
        mock_state_store["endRange"].setText.assert_called_once_with("128.0")
        mock_state_store["bankDirEdit"].setText.assert_called_once_with(
            "output/banklines")
        mock_state_store["bankFileName"].setText.assert_called_once_with("bankfile")
        mock_state_store["makePlotsEdit"].setChecked.assert_called_once_with(True)
        mock_state_store["savePlotsEdit"].setChecked.assert_called_once_with(True)
        mock_state_store["saveZoomPlotsEdit"].setChecked.assert_called_once_with(False)
        mock_state_store["closePlotsEdit"].setChecked.assert_called_once_with(False)
        mock_state_store["debugOutputEdit"].setChecked.assert_called_once_with(True)
        mock_state_store["zoomPlotsRangeEdit"].setText.assert_called_once_with("1.0")
        mock_state_store["figureDirEdit"].setText.assert_called_once_with(
            "output/figures")

    def test_load_detect_section_sets_parameters(self, config_loader, mock_state_store):
        """Test that _load_detect_section sets the parameters fields correctly."""
        mock_state_store["simFileEdit"].setText.assert_called_once_with("test_sim.nc")

        config_loader.config_file.get_float.assert_any_call("Detect", "WaterDepth", default=0.0)
        mock_state_store["waterDepth"].setText.assert_called_once_with("0.5")

        config_loader.config_file.get_int.assert_any_call("Detect", "NBank", default=0, positive=True)


    @pytest.mark.parametrize(
        "n_bank",
        [0, 1, 2, 3],
        ids=["no_banks", "one_bank", "two_banks", "three_banks"],
    )
    def test_load_search_lines_populates_tree_widget(
        self,
        config_loader,
        mock_state_store,
        n_bank
    ):
        """Test that _load_search_lines populates the searchLines tree widget correctly.

        Args:
            n_bank: Number of bank search lines.
        """
        # Set up mock return values based on n_bank
        mock_distances = [50.0 + i * 25.0 for i in range(n_bank)]
        config_loader.config_file.get_bank_search_distances = Mock(return_value=mock_distances)

        # Mock get_str to return line file names
        def mock_get_str(section, key, default=None):
            if section == "Detect" and key.startswith("Line"):
                line_num = key[4:]  # Extract number from "Line1", "Line2", etc.
                return f"bank_line_{line_num}.xyc"
            return default

        config_loader.config_file.get_str = Mock(side_effect=mock_get_str)

        mock_state_store["searchLines"].invisibleRootItem.reset_mock()
        mock_state_store["searchLinesEdit"].setEnabled.reset_mock()
        mock_state_store["searchLinesRemove"].setEnabled.reset_mock()

        with patch('dfastbe.gui.configs.QTreeWidgetItem') as mock_tree_item:
            config_loader._load_search_lines(n_bank)

            config_loader.config_file.get_bank_search_distances.assert_called_once_with(n_bank)

            mock_state_store["searchLines"].invisibleRootItem.assert_called_once()
            mock_state_store["searchLines"].invisibleRootItem().takeChildren.assert_called_once()

            assert mock_tree_item.call_count == n_bank

            for i in range(n_bank):
                expected_line_name = f"bank_line_{i + 1}.xyc"
                expected_distance = str(mock_distances[i])
                mock_tree_item.assert_any_call(
                    mock_state_store["searchLines"],
                    [str(i + 1), expected_line_name, expected_distance]
                )

            if n_bank > 0:
                mock_state_store["searchLinesEdit"].setEnabled.assert_called_once_with(True)
                mock_state_store["searchLinesRemove"].setEnabled.assert_called_once_with(True)
            else:
                mock_state_store["searchLinesEdit"].setEnabled.assert_not_called()
                mock_state_store["searchLinesRemove"].setEnabled.assert_not_called()

    def test_load_erosion_section_sets_basic_parameters(self, config_loader, mock_state_store):
        """Test that _load_erosion_section sets basic erosion parameters correctly."""
        for field in ["tErosion", "riverAxisEdit", "fairwayEdit", "chainageOutStep",
                      "outDirEdit", "newBankFile", "newEqBankFile", "eroVol", "eroVolEqui"]:
            mock_state_store[field].setText.reset_mock()

        with (patch('dfastbe.gui.configs.QTreeWidgetItem'),
             patch('dfastbe.gui.configs.bankStrengthSwitch'),
             patch('dfastbe.gui.configs.addTabForLevel'),
             patch('dfastbe.gui.configs.DischargeLevelsTabs') as mock_tabs_class,
             patch.object(config_loader, '_load_ship_parameters') as mock_load_ship_params,
             patch.object(config_loader, '_configure_bank_strength') as mock_configure_bank_strength,
             patch.object(config_loader, '_load_filter') as mock_load_filter,
             patch.object(config_loader, '_load_param') as mock_load_param,
             patch.object(config_loader, '_load_discharges') as mock_load_discharges):

            # Setup mock for DischargeLevelsTabs instance
            mock_tabs_instance = Mock()
            mock_tabs_class.return_value = mock_tabs_instance

            config_loader._load_erosion_section()

            mock_state_store["tErosion"].setText.assert_called_once_with("10.0")
            mock_state_store["riverAxisEdit"].setText.assert_called_once_with("river_axis.xyc")
            mock_state_store["fairwayEdit"].setText.assert_called_once_with("fairway.xyc")
            mock_state_store["chainageOutStep"].setText.assert_called_once_with("100.0")
            mock_state_store["outDirEdit"].setText.assert_called_once_with("output/erosion")

            mock_state_store["newBankFile"].setText.assert_called_once_with("banknew")
            mock_state_store["newEqBankFile"].setText.assert_called_once_with("bankeq")
            mock_state_store["eroVol"].setText.assert_called_once_with("erovol_standard.evo")
            mock_state_store["eroVolEqui"].setText.assert_called_once_with("erovol_eq.evo")

            mock_load_ship_params.assert_called_once()
            mock_load_discharges.assert_called_once_with(2, config_loader.config["Erosion"])
            mock_configure_bank_strength.assert_called_once_with(True)

            # Verify DischargeLevelsTabs was instantiated and configure_tabs was called
            mock_tabs_class.assert_called_once_with(config_loader.config, config_loader.config_file)
            mock_tabs_instance.configure_tabs.assert_called_once_with(2)

            # Check that _load_param was called with the expected arguments
            expected_load_param_calls = [
                call("bankProtect", "Erosion", "ProtectionLevel", "-1000"),
                call("bankSlope", "Erosion", "Slope", "20.0"),
                call("bankReed", "Erosion", "Reed", "0.0"),
            ]
            assert mock_load_param.call_count == len(expected_load_param_calls)
            mock_load_param.assert_has_calls(expected_load_param_calls, any_order=True)

            # Verify _load_filter calls
            expected_load_filter_calls = [
                call("velFilter", "Erosion", "VelFilterDist"),
                call("bedFilter", "Erosion", "BedFilterDist"),
            ]

            assert mock_load_filter.call_count == len(expected_load_filter_calls)
            mock_load_filter.assert_has_calls(expected_load_filter_calls, any_order=True)

    def test_load_ship_parameters_sets_all_parameters(self, config_loader, mock_state_store):
        """Test that _load_ship_parameters sets all ship-related parameters correctly."""
        with patch.object(config_loader, '_load_param') as mock_load_param:
            config_loader._load_ship_parameters()

            # Verify all _load_param calls for ship parameters
            expected_load_param_calls = [
                call("shipType", "Erosion", "ShipType"),
                call("shipVeloc", "Erosion", "VShip"),
                call("nShips", "Erosion", "NShip"),
                call("shipNWaves", "Erosion", "NWave", "5"),
                call("shipDraught", "Erosion", "Draught"),
                call("wavePar0", "Erosion", "Wave0", "200.0"),
                call("wavePar1", "Erosion", "Wave1", "200.0"),
            ]

            assert mock_load_param.call_count == len(expected_load_param_calls)
            mock_load_param.assert_has_calls(expected_load_param_calls, any_order=True)

            config_loader.config_file.get_str.assert_any_call("Erosion", "Wave0", "200.0")

    @pytest.mark.parametrize(
        "use_bank_type",
        [True, False],
        ids=["use_bank_type_true", "use_bank_type_false"],
    )
    def test_configure_bank_strength_sets_parameters(
            self,
            config_loader,
            mock_state_store,
            use_bank_type
    ):
        """Test that _configure_bank_strength sets parameters correctly for different use_bank_type values.

        Args:
            use_bank_type: Whether to use bank type (True) or critical shear stress (False).
        """
        for field in ["bankType", "bankTypeType", "bankTypeEdit", "bankTypeEditFile",
                      "bankShear", "bankShearType", "bankShearEdit", "bankShearEditFile"]:
            mock_state_store[field].setEnabled.reset_mock()
        mock_state_store["strengthPar"].setCurrentText.reset_mock()

        with patch.object(config_loader, '_load_param') as mock_load_param, \
             patch('dfastbe.gui.configs.bankStrengthSwitch') as mock_bank_strength_switch:

            config_loader._configure_bank_strength(use_bank_type)

            # Verify that bankType widgets are enabled/disabled based on use_bank_type
            mock_state_store["bankType"].setEnabled.assert_called_once_with(use_bank_type)
            mock_state_store["bankTypeType"].setEnabled.assert_called_once_with(use_bank_type)
            mock_state_store["bankTypeEdit"].setEnabled.assert_called_once_with(use_bank_type)
            mock_state_store["bankTypeEditFile"].setEnabled.assert_called_once_with(use_bank_type)

            # Verify that bankShear widgets are enabled/disabled oppositely
            mock_state_store["bankShear"].setEnabled.assert_called_once_with(not use_bank_type)
            mock_state_store["bankShearType"].setEnabled.assert_called_once_with(not use_bank_type)
            mock_state_store["bankShearEdit"].setEnabled.assert_called_once_with(not use_bank_type)
            mock_state_store["bankShearEditFile"].setEnabled.assert_called_once_with(not use_bank_type)

            if use_bank_type:
                mock_state_store["strengthPar"].setCurrentText.assert_called_once_with("Bank Type")
                mock_load_param.assert_called_once_with(
                    "bankType",
                    "Erosion",
                    "BankType"
                )
            else:
                mock_state_store["strengthPar"].setCurrentText.assert_called_once_with("Critical Shear Stress")
                mock_load_param.assert_called_once_with(
                    "bankShear",
                    "Erosion",
                    "BankType"
                )

            mock_bank_strength_switch.assert_called_once()

    @pytest.mark.parametrize(
        "n_level",
        [0, 1, 2],
        ids=["no_levels", "one_level", "two_levels"],
    )
    def test_load_discharges_populates_tree_widget(
        self,
        config_loader,
        mock_state_store,
        n_level
    ):
        """Test that _load_discharges populates the discharges tree widget correctly.

        Args:
            n_level: Number of discharge levels.
        """
        mock_section = {"RefLevel": "1"}

        # Mock get_str to return discharge file names and probabilities
        def mock_get_str(section, key, default=None):
            if section == "Erosion":
                if key.startswith("SimFile"):
                    level_num = key[-1]
                    return f"discharge_file_{level_num}.nc"
                elif key.startswith("PDischarge"):
                    level_num = key[-1]
                    probabilities = ["0.5", "0.3", "0.15", "0.05"]
                    return probabilities[int(level_num) - 1] if int(level_num) <= len(probabilities) else "0.1"
            return default

        config_loader.config_file.get_str = Mock(side_effect=mock_get_str)

        # Reset mocks that were called during __post_init__
        mock_state_store["discharges"].invisibleRootItem.reset_mock()
        mock_state_store["dischargesEdit"].setEnabled.reset_mock()
        mock_state_store["dischargesRemove"].setEnabled.reset_mock()
        mock_state_store["refLevel"].setText.reset_mock()
        mock_state_store["refLevel"].validator().setTop.reset_mock()

        with patch('dfastbe.gui.configs.QTreeWidgetItem') as mock_tree_item:
            config_loader._load_discharges(n_level, mock_section)

            mock_state_store["discharges"].invisibleRootItem.assert_called_once()
            mock_state_store["discharges"].invisibleRootItem().takeChildren.assert_called_once()

            assert mock_tree_item.call_count == n_level

            for i in range(n_level):
                level_num = str(i + 1)
                expected_file_name = f"discharge_file_{level_num}.nc"
                probabilities = ["0.5", "0.3", "0.15", "0.05"]
                expected_prob = probabilities[i] if i < len(probabilities) else "0.1"
                mock_tree_item.assert_any_call(
                    mock_state_store["discharges"],
                    [level_num, expected_file_name, expected_prob]
                )

            if n_level > 0:
                mock_state_store["dischargesEdit"].setEnabled.assert_called_once_with(True)
                mock_state_store["dischargesRemove"].setEnabled.assert_called_once_with(True)
            else:
                mock_state_store["dischargesEdit"].setEnabled.assert_not_called()
                mock_state_store["dischargesRemove"].setEnabled.assert_not_called()

            mock_state_store["refLevel"].validator().setTop.assert_called_once_with(n_level)

            mock_state_store["refLevel"].setText.assert_called_once_with("1")
