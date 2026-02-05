"""
Unit tests for the ConfigurationLoader class.
"""

import pytest
from configparser import ConfigParser
from pathlib import Path
from unittest.mock import Mock, patch, call, PropertyMock

from dfastbe.gui.configs import ConfigurationLoader
from dfastbe.io.config import ConfigFile


class TestConfigurationLoader:
    """Test cases for the ConfigurationLoader class."""

    @pytest.fixture
    def mock_state_store(self):
        """Fixture to create a mock StateStore instance.

        Returns:
            dict: A mock dictionary simulating the StateStore with mock widgets.
        """
        mock_store = {}

        # Mock text input widgets (QLineEdit-like)
        text_fields = [
            "chainFileEdit", "startRange", "endRange", "bankDirEdit",
            "bankFileName", "zoomPlotsRangeEdit", "figureDirEdit",
            # Detect section
            "simFileEdit", "waterDepth",
            # Erosion section
            "tErosion", "riverAxisEdit", "fairwayEdit", "chainageOutStep",
            "outDirEdit", "newBankFile", "newEqBankFile", "eroVol", "eroVolEqui",
            # Bank strength
            "bankTypeEdit", "bankShearEdit",
        ]
        for field in text_fields:
            mock_widget = Mock()
            mock_widget.setText = Mock()
            mock_widget.text = Mock(return_value="")
            mock_store[field] = mock_widget

        # Mock checkbox widgets (QCheckBox-like)
        checkbox_fields = [
            "makePlotsEdit", "savePlotsEdit", "saveZoomPlotsEdit",
            "closePlotsEdit", "debugOutputEdit"
        ]
        for field in checkbox_fields:
            mock_widget = Mock()
            mock_widget.setChecked = Mock()
            mock_widget.isChecked = Mock(return_value=False)
            mock_store[field] = mock_widget

        # Mock combo boxes
        combo_fields = [
            "strengthPar", "bankTypeType", "bankShearType"
        ]
        for field in combo_fields:
            mock_combo = Mock()
            mock_combo.setCurrentText = Mock()
            mock_combo.currentText = Mock(return_value="")
            mock_store[field] = mock_combo

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

        # Mock level-specific widgets (for _configure_tabs_for_levels)
        for i in range(1, 5):  # Support up to 4 levels
            istr = str(i)
            for suffix in ["_eroVolEdit", "_shipType", "_shipVeloc", "_nShips",
                          "_shipNWaves", "_shipDraught", "_bankSlope", "_bankReed"]:
                mock_widget = Mock()
                mock_widget.setText = Mock()
                mock_widget.text = Mock(return_value="")
                mock_store[istr + suffix] = mock_widget

        return mock_store

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
        mock_file.version = "1.0"
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

        with patch('dfastbe.gui.configs.StateStore.instance',
                   return_value=mock_state_store):
            loader = ConfigurationLoader(config_path)
            loader.config_file = mock_config_file
            loader.config = mock_config_file.config
            loader.rootdir = str(tmp_path)
            return loader

    @pytest.mark.parametrize(
        "has_version,expected_result",
        [
            (True, True),
            (False, False),
        ],
        ids=["with_version", "without_version"],
    )
    def test_read_config_file(
        self,
        mock_state_store,
        tmp_path,
        has_version,
        expected_result
    ):
        """Test that _read_config_file reads and validates configuration correctly.

        Args:
            has_version: Whether the config file has version information.
            expected_result: Expected return value (True for success, False for failure).
        """
        # Create a test config file
        config_path = tmp_path / "test_config.cfg"
        if has_version:
            config_path.write_text("[General]\nVersion=1.0\n")
        else:
            config_path.write_text("[General]\nRiverKM=test.xyc\n")

        with patch('dfastbe.gui.configs.StateStore.instance', return_value=mock_state_store), \
             patch('dfastbe.gui.configs.ConfigFile.read') as mock_config_read, \
             patch('dfastbe.gui.configs.absolute_path') as mock_absolute_path, \
             patch('dfastbe.gui.configs.show_error') as mock_show_error:

            # Setup mocks
            mock_absolute_path.return_value = str(config_path)
            mock_config_file = Mock()
            mock_config_file.config = ConfigParser()
            mock_config_file.config.read(str(config_path))

            if has_version:
                mock_config_file.version = "1.0"
            else:
                # Simulate KeyError when accessing version property
                type(mock_config_file).version = PropertyMock(side_effect=KeyError("version"))

            mock_config_read.return_value = mock_config_file

            # Create loader and test _read_config_file
            loader = ConfigurationLoader(config_path)
            result = loader._read_config_file()

            # Verify result
            assert result == expected_result

            # Verify absolute_path was called
            mock_absolute_path.assert_called_once()

            # Verify ConfigFile.read was called with absolute path
            mock_config_read.assert_called_once_with(str(config_path))

            if has_version:
                # Verify success case: no error shown, config is set
                mock_show_error.assert_not_called()
                assert loader.config_file == mock_config_file
                assert loader.config == mock_config_file.config
                assert loader.rootdir == str(tmp_path)
                assert loader.config_file.path == str(config_path)
            else:
                # Verify failure case: error was shown, config might not be fully set
                mock_show_error.assert_called_once()
                error_message = mock_show_error.call_args[0][0]
                assert "No version information" in error_message
                assert str(config_path) in error_message

    @pytest.mark.parametrize(
        "file_exists,use_tmp_path,file_name,expected_result,should_show_error",
        [
            (True, True, "test_config.cfg", True, False),
            (False, True, "missing_config.cfg", False, True),
            (False, False, "dfastbe.cfg", False, False),
        ],
        ids=["file_exists", "file_missing_show_error", "default_file_missing_no_error"],
    )
    def test_validate_path(
        self,
        mock_state_store,
        tmp_path,
        file_exists,
        use_tmp_path,
        file_name,
        expected_result,
        should_show_error
    ):
        """Test that _validate_path validates configuration file path correctly.

        Args:
            file_exists: Whether the config file exists.
            use_tmp_path: Whether to use tmp_path for the config file location.
            file_name: Name of the config file.
            expected_result: Expected return value (True if valid, False if invalid).
            should_show_error: Whether an error message should be shown.
        """
        # Create config path
        if use_tmp_path:
            config_path = tmp_path / file_name
        else:
            # For the special "dfastbe.cfg" case, use Path directly without directory
            config_path = Path(file_name)

        # Create the file only if it should exist
        if file_exists:
            config_path.write_text("[General]\nVersion=1.0\n")

        with patch('dfastbe.gui.configs.StateStore.instance', return_value=mock_state_store), \
             patch('dfastbe.gui.configs.show_error') as mock_show_error:

            # Create loader and test _validate_path
            loader = ConfigurationLoader(config_path)
            result = loader._validate_path()

            # Verify result
            assert result == expected_result

            # Verify error was shown or not shown as expected
            if should_show_error:
                mock_show_error.assert_called_once()
                error_message = mock_show_error.call_args[0][0]
                assert "does not exist" in error_message
                assert str(config_path) in error_message
            else:
                mock_show_error.assert_not_called()

    def test_load_general_section_sets_parameters(self, config_loader, mock_state_store):
        """Test that _load_general_section sets the parameters fields correctly."""
        config_loader._load_general_section()

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
        with patch.object(config_loader, '_load_search_lines') as mock_load_search_lines:
            config_loader._load_detect_section()

            # Verify simFileEdit was set
            mock_state_store["simFileEdit"].setText.assert_called_once_with("test_sim.nc")

            # Verify waterDepth was set
            config_loader.config_file.get_float.assert_any_call("Detect", "WaterDepth", default=0.0)
            mock_state_store["waterDepth"].setText.assert_called_once_with("0.5")

            # Verify n_bank was retrieved
            config_loader.config_file.get_int.assert_any_call("Detect", "NBank", default=0, positive=True)

            # Verify _load_search_lines was called with the correct n_bank value
            mock_load_search_lines.assert_called_once_with(2)

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

        with patch('dfastbe.gui.configs.QTreeWidgetItem') as mock_tree_item:
            config_loader._load_search_lines(n_bank)

            # Verify get_bank_search_distances was called with correct n_bank
            config_loader.config_file.get_bank_search_distances.assert_called_once_with(n_bank)

            # Verify the tree was cleared
            mock_state_store["searchLines"].invisibleRootItem.assert_called_once()
            mock_state_store["searchLines"].invisibleRootItem().takeChildren.assert_called_once()

            # Verify correct number of tree items were created
            assert mock_tree_item.call_count == n_bank

            # Verify each tree item was created with correct parameters
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
        with (patch('dfastbe.gui.configs.QTreeWidgetItem'), \
             patch('dfastbe.gui.configs.setParam') as mock_set_param, \
             patch('dfastbe.gui.configs.setOptParam'), \
             patch('dfastbe.gui.configs.setFilter') as mock_set_filter, \
             patch('dfastbe.gui.configs.bankStrengthSwitch'), \
             patch('dfastbe.gui.configs.addTabForLevel'), \
             patch.object(config_loader, '_configure_tabs_for_levels') as mock_configure_tabs,  \
             patch.object(config_loader, '_load_ship_parameters') as mock_load_ship_params, \
             patch.object(config_loader, '_configure_bank_strength') as mock_configure_bank_strength, \
             patch.object(config_loader, '_load_discharges') as mock_load_discharges):

            config_loader._load_erosion_section()

            # Verify basic erosion parameters were set via setText
            mock_state_store["tErosion"].setText.assert_called_once_with("10.0")
            mock_state_store["riverAxisEdit"].setText.assert_called_once_with("river_axis.xyc")
            mock_state_store["fairwayEdit"].setText.assert_called_once_with("fairway.xyc")
            mock_state_store["chainageOutStep"].setText.assert_called_once_with("100.0")
            mock_state_store["outDirEdit"].setText.assert_called_once_with("output/erosion")

            # Verify bank file parameters were set
            mock_state_store["newBankFile"].setText.assert_called_once_with("banknew")
            mock_state_store["newEqBankFile"].setText.assert_called_once_with("bankeq")
            mock_state_store["eroVol"].setText.assert_called_once_with("erovol_standard.evo")
            mock_state_store["eroVolEqui"].setText.assert_called_once_with("erovol_eq.evo")

            # Assert that helper methods were called correctly
            mock_load_ship_params.assert_called_once()
            mock_load_discharges.assert_called_once_with(2, config_loader.config["Erosion"])
            mock_configure_bank_strength.assert_called_once_with(True)
            mock_configure_tabs.assert_called_once_with(2)

            # Verify setParam calls made directly in _load_erosion_section (not in mocked methods)
            expected_set_param_calls = [
                (("bankProtect", config_loader.config, "Erosion", "ProtectionLevel", "-1000"), {}),
                (("bankSlope", config_loader.config, "Erosion", "Slope", "20.0"), {}),
                (("bankReed", config_loader.config, "Erosion", "Reed", "0.0"), {}),
            ]

            # Check that setParam was called with the expected arguments
            assert mock_set_param.call_count == len(expected_set_param_calls)
            mock_set_param.assert_has_calls([
                call(*call_args[0], **call_args[1]) for call_args in expected_set_param_calls
            ], any_order=True)

            # Verify setFilter calls
            expected_set_filter_calls = [
                (("velFilter", config_loader.config, "Erosion", "VelFilterDist"), {}),
                (("bedFilter", config_loader.config, "Erosion", "BedFilterDist"), {}),
            ]

            assert mock_set_filter.call_count == len(expected_set_filter_calls)
            mock_set_filter.assert_has_calls([
                call(*call_args[0], **call_args[1]) for call_args in expected_set_filter_calls
            ], any_order=True)

    def test_load_ship_parameters_sets_all_parameters(self, config_loader, mock_state_store):
        """Test that _load_ship_parameters sets all ship-related parameters correctly."""
        with patch('dfastbe.gui.configs.setParam') as mock_set_param:
            config_loader._load_ship_parameters()

            # Verify all setParam calls for ship parameters
            expected_set_param_calls = [
                (("shipType", config_loader.config, "Erosion", "ShipType"), {}),
                (("shipVeloc", config_loader.config, "Erosion", "VShip"), {}),
                (("nShips", config_loader.config, "Erosion", "NShip"), {}),
                (("shipNWaves", config_loader.config, "Erosion", "NWave", "5"), {}),
                (("shipDraught", config_loader.config, "Erosion", "Draught"), {}),
                (("wavePar0", config_loader.config, "Erosion", "Wave0", "200.0"), {}),
                (("wavePar1", config_loader.config_file.config, "Erosion", "Wave1", "200.0"), {}),
            ]

            # Check that setParam was called with the expected arguments
            assert mock_set_param.call_count == len(expected_set_param_calls)
            mock_set_param.assert_has_calls([
                call(*call_args[0], **call_args[1]) for call_args in expected_set_param_calls
            ], any_order=True)

            # Verify that config_file.get_str was called to retrieve Wave0 value
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
        with patch('dfastbe.gui.configs.setParam') as mock_set_param, \
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

            # Verify strengthPar was set correctly
            if use_bank_type:
                mock_state_store["strengthPar"].setCurrentText.assert_called_once_with("Bank Type")
                # Verify setParam was called with bankType
                mock_set_param.assert_called_once_with(
                    "bankType",
                    config_loader.config_file.config,
                    "Erosion",
                    "BankType"
                )
            else:
                mock_state_store["strengthPar"].setCurrentText.assert_called_once_with("Critical Shear Stress")
                # Verify setParam was called with bankShear
                mock_set_param.assert_called_once_with(
                    "bankShear",
                    config_loader.config,
                    "Erosion",
                    "BankType"
                )

            # Verify bankStrengthSwitch was called
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
        # Mock section containing RefLevel
        mock_section = {"RefLevel": "1"}

        # Mock get_str to return discharge file names and probabilities
        def mock_get_str(section, key, default=None):
            if section == "Erosion":
                if key.startswith("SimFile"):
                    level_num = key[-1]  # Extract number from "SimFile1", "SimFile2", etc.
                    return f"discharge_file_{level_num}.nc"
                elif key.startswith("PDischarge"):
                    level_num = key[-1]  # Extract number from "PDischarge1", "PDischarge2", etc.
                    # Return different probabilities for variety
                    probabilities = ["0.5", "0.3", "0.15", "0.05"]
                    return probabilities[int(level_num) - 1] if int(level_num) <= len(probabilities) else "0.1"
            return default

        config_loader.config_file.get_str = Mock(side_effect=mock_get_str)

        with patch('dfastbe.gui.configs.QTreeWidgetItem') as mock_tree_item:
            config_loader._load_discharges(n_level, mock_section)

            # Verify the tree was cleared
            mock_state_store["discharges"].invisibleRootItem.assert_called_once()
            mock_state_store["discharges"].invisibleRootItem().takeChildren.assert_called_once()

            # Verify correct number of tree items were created
            assert mock_tree_item.call_count == n_level

            # Verify each tree item was created with correct parameters
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

            # Verify refLevel validator was configured with correct n_level
            mock_state_store["refLevel"].validator().setTop.assert_called_once_with(n_level)

            # Verify refLevel text was set from section
            mock_state_store["refLevel"].setText.assert_called_once_with("1")

    @pytest.mark.parametrize(
        "version,should_load",
        [
            ("1.0", True),
            ("2.0", False),
            ("0.9", False),
            ("invalid", False),
        ],
        ids=["valid_version_1.0", "unsupported_version_2.0", "unsupported_version_0.9", "invalid_version"],
    )
    def test_load_with_different_versions(
        self,
        config_loader,
        mock_state_store,
        mock_config_file,
        version,
        should_load
    ):
        """Test that load() handles different version numbers correctly.

        Args:
            version: The version string to test.
            should_load: Whether the configuration should be loaded for this version.
        """
        # Set the version in the mock config file
        mock_config_file.version = version
        config_loader.config_file = mock_config_file

        with patch('dfastbe.gui.configs.show_error') as mock_show_error, \
             patch.object(config_loader, '_validate_path', return_value=True), \
             patch.object(config_loader, '_read_config_file', return_value=True), \
             patch.object(config_loader, '_load_general_section') as mock_load_general, \
             patch.object(config_loader, '_load_detect_section') as mock_load_detect, \
             patch.object(config_loader, '_load_erosion_section') as mock_load_erosion:

            config_loader.load()

            if should_load:
                # Verify all section loaders were called for version 1.0
                mock_load_general.assert_called_once()
                mock_load_detect.assert_called_once()
                mock_load_erosion.assert_called_once()
                mock_show_error.assert_not_called()
            else:
                # Verify section loaders were not called for unsupported versions
                mock_load_general.assert_not_called()
                mock_load_detect.assert_not_called()
                mock_load_erosion.assert_not_called()
                # Verify error was shown
                mock_show_error.assert_called_once()
                error_message = mock_show_error.call_args[0][0]
                assert f"Unsupported version number {version}" in error_message


