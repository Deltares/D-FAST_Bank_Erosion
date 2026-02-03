"""
Unit tests for the ConfigurationLoader class.
"""

import pytest
from configparser import ConfigParser
from unittest.mock import Mock, patch

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

    @pytest.mark.parametrize(
        "n_bank,search_distances,expected_lines,buttons_enabled",
        [
            (
                2,
                [50.0, 75.0],
                [
                    ("1", "bank_line_1.xyc", "50.0"),
                    ("2", "bank_line_2.xyc", "75.0"),
                ],
                True,
            ),
            (
                0,
                [],
                [],
                False,
            ),
        ],
        ids=["n_bank_greater_than_0", "n_bank_equals_0"],
    )
    def test_load_detect_section_sets_parameters(
            self,
            config_loader,
            mock_state_store,
            n_bank,
            search_distances,
            expected_lines,
            buttons_enabled
    ):
        """Test that _load_detect_section sets parameters correctly for different n_bank values.

        Args:
            n_bank: Number of bank lines.
            search_distances: List of search distances for each bank line.
            expected_lines: Expected tree widget items (line number, file name, distance).
            buttons_enabled: Whether edit/remove buttons should be enabled.
        """
        # Configure mock to return the specified n_bank value
        config_loader.config_file.get_int = Mock(side_effect=lambda section, key, default=None, positive=False: {
            ("Detect", "NBank"): n_bank,
            ("Erosion", "NLevel"): 2,
        }.get((section, key), default))
        config_loader.config_file.get_bank_search_distances = Mock(return_value=search_distances)

        with patch('dfastbe.gui.configs.QTreeWidgetItem') as mock_tree_item:
            config_loader._load_detect_section()

            # Verify SimFile is set
            mock_state_store["simFileEdit"].setText.assert_called_once_with("test_sim.nc")

            # Verify WaterDepth is set
            mock_state_store["waterDepth"].setText.assert_called_once_with("0.5")

            # Verify get_int was called with correct parameters
            config_loader.config_file.get_int.assert_called_with(
                "Detect", "NBank", default=0, positive=True
            )

            # Verify get_bank_search_distances was called with correct n_bank
            config_loader.config_file.get_bank_search_distances.assert_called_once_with(n_bank)

            # Verify search lines tree was cleared
            mock_state_store["searchLines"].invisibleRootItem().takeChildren.assert_called_once()

            # Verify tree items were created for search lines
            if expected_lines:
                assert mock_tree_item.call_count == len(expected_lines)
                for line_data in expected_lines:
                    mock_tree_item.assert_any_call(
                        mock_state_store["searchLines"],
                        list(line_data)
                    )
            else:
                mock_tree_item.assert_not_called()

            # Verify buttons enabled/disabled state
            if buttons_enabled:
                mock_state_store["searchLinesEdit"].setEnabled.assert_called_once_with(True)
                mock_state_store["searchLinesRemove"].setEnabled.assert_called_once_with(True)
            else:
                mock_state_store["searchLinesEdit"].setEnabled.assert_not_called()
                mock_state_store["searchLinesRemove"].setEnabled.assert_not_called()

    @pytest.mark.parametrize(
        "n_level,discharge_levels",
        [
            (
                2,
                [
                    ("1", "discharge_file_1.nc", "0.6"),
                    ("2", "discharge_file_2.nc", "0.4"),
                ],
            ),
            (
                0,
                [],
            ),
        ],
        ids=["n_level_equals_2", "n_level_equals_0"],
    )
    def test_load_erosion_section_sets_parameters(
            self,
            config_loader,
            mock_state_store,
            n_level,
            discharge_levels,
    ):
        """Test that _load_erosion_section sets parameters correctly for different n_level values.

        Args:
            n_level: Number of discharge levels.
            discharge_levels: Expected tree widget items (level number, file name, probability).
        """
        # Configure mock to return the specified n_level value
        config_loader.config_file.get_int = Mock(side_effect=lambda section, key, default=None, positive=False: {
            ("Erosion", "NLevel"): n_level,
        }.get((section, key), default))

        # Configure mock for discharge file retrieval
        discharge_files = {f"SimFile{i+1}": level[1] for i, level in enumerate(discharge_levels)}
        discharge_probs = {f"PDischarge{i+1}": level[2] for i, level in enumerate(discharge_levels)}

        config_loader.config_file.get_str = Mock(side_effect=lambda section, key, default=None: {
            ("General", "BankFile"): "bankfile",
            ("General", "FigureDir"): "output/figures",
            ("Erosion", "BankNew"): "banknew",
            ("Erosion", "BankEq"): "bankeq",
            ("Erosion", "EroVol"): "erovol_standard.evo",
            ("Erosion", "EroVolEqui"): "erovol_eq.evo",
            ("Erosion", "Wave0"): "200.0",
            **{("Erosion", k): v for k, v in discharge_files.items()},
            **{("Erosion", k): v for k, v in discharge_probs.items()},
        }.get((section, key), default))

        # Add level-specific widgets to mock_state_store for _configure_tabs_for_levels
        for i in range(1, n_level + 1):
            istr = str(i)
            mock_widget = Mock()
            mock_widget.setText = Mock()
            mock_state_store[istr + "_eroVolEdit"] = mock_widget

        with patch('dfastbe.gui.configs.QTreeWidgetItem') as mock_tree_item, \
             patch('dfastbe.gui.configs.setParam'), \
             patch('dfastbe.gui.configs.setOptParam'), \
             patch('dfastbe.gui.configs.setFilter'), \
             patch('dfastbe.gui.configs.bankStrengthSwitch'), \
             patch('dfastbe.gui.configs.addTabForLevel'), \
             patch('dfastbe.gui.configs.StateStore.instance', return_value=mock_state_store):

            config_loader._load_erosion_section()

            # Verify basic erosion parameters are set
            mock_state_store["tErosion"].setText.assert_called_once_with("10.0")
            mock_state_store["riverAxisEdit"].setText.assert_called_once_with("river_axis.xyc")
            mock_state_store["fairwayEdit"].setText.assert_called_once_with("fairway.xyc")
            mock_state_store["chainageOutStep"].setText.assert_called_once_with("100.0")
            mock_state_store["outDirEdit"].setText.assert_called_once_with("output/erosion")

            # Verify bank file parameters
            mock_state_store["newBankFile"].setText.assert_called_once_with("banknew")
            mock_state_store["newEqBankFile"].setText.assert_called_once_with("bankeq")
            mock_state_store["eroVol"].setText.assert_called_once_with("erovol_standard.evo")
            mock_state_store["eroVolEqui"].setText.assert_called_once_with("erovol_eq.evo")

            # Verify get_int was called for NLevel
            config_loader.config_file.get_int.assert_any_call(
                "Erosion", "NLevel", default=0, positive=True
            )

            # Verify discharges tree was cleared
            mock_state_store["discharges"].invisibleRootItem().takeChildren.assert_called_once()

            # Verify tree items were created for discharge levels
            if discharge_levels:
                assert mock_tree_item.call_count >= len(discharge_levels)
                for level_data in discharge_levels:
                    mock_tree_item.assert_any_call(
                        mock_state_store["discharges"],
                        list(level_data)
                    )
            else:
                # When n_level is 0, no discharge tree items should be created
                # but QTreeWidgetItem might still be called by other parts
                pass

            # Verify buttons enabled/disabled state
            if n_level > 0:
                mock_state_store["dischargesEdit"].setEnabled.assert_called_once_with(True)
                mock_state_store["dischargesRemove"].setEnabled.assert_called_once_with(True)
            else:
                mock_state_store["dischargesEdit"].setEnabled.assert_not_called()
                mock_state_store["dischargesRemove"].setEnabled.assert_not_called()

            # Verify refLevel was set
            mock_state_store["refLevel"].validator().setTop.assert_called_once_with(n_level)
            mock_state_store["refLevel"].setText.assert_called_once_with("1")


