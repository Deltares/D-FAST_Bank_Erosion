"""
Unit tests for the ConfigurationLoader class.
"""

import pytest
from configparser import ConfigParser
from unittest.mock import Mock, patch, call

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



