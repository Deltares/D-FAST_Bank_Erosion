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
            "bankFileName", "zoomPlotsRangeEdit", "figureDirEdit"
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
            }
        })

        mock_file = Mock(spec=ConfigFile)
        mock_file.config = config
        mock_file.version = "1.0"
        mock_file.get_range = Mock(return_value=(123.0, 128.0))
        mock_file.get_str = Mock(side_effect=lambda section, key, default=None: {
            ("General", "BankFile"): "bankfile",
            ("General", "FigureDir"): "output/figures",
        }.get((section, key), default))
        mock_file.get_bool = Mock(side_effect=lambda section, key, default=None: {
            ("General", "Plotting"): True,
            ("General", "SavePlots"): True,
            ("General", "SaveZoomPlots"): False,
            ("General", "ClosePlots"): False,
            ("General", "DebugOutput"): True,
        }.get((section, key), default))
        mock_file.get_float = Mock(side_effect=lambda section, key, default=None: {
            ("General", "ZoomStepKM"): 1.0,
        }.get((section, key), default))

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

        with patch('dfastbe.gui.configs.StateStore.instance', return_value=mock_state_store):
            loader = ConfigurationLoader(config_path)
            loader.config_file = mock_config_file
            loader.config = mock_config_file.config
            loader.rootdir = str(tmp_path)
            return loader

    def test_load_general_section_sets_river_km(self, config_loader, mock_state_store):
        """Test that _load_general_section sets the RiverKM field correctly."""
        config_loader._load_general_section()

        mock_state_store["chainFileEdit"].setText.assert_called_once_with("inputs/rivkm_20m.xyc")

    def test_load_general_section_sets_boundaries(self, config_loader, mock_state_store):
        """Test that _load_general_section sets the start and end range correctly."""
        config_loader._load_general_section()

        mock_state_store["startRange"].setText.assert_called_once_with("123.0")
        mock_state_store["endRange"].setText.assert_called_once_with("128.0")

    def test_load_general_section_sets_bank_dir(self, config_loader, mock_state_store):
        """Test that _load_general_section sets the BankDir field correctly."""
        config_loader._load_general_section()

        mock_state_store["bankDirEdit"].setText.assert_called_once_with("output/banklines")

    def test_load_general_section_sets_bank_file(self, config_loader, mock_state_store):
        """Test that _load_general_section sets the BankFile field correctly."""
        config_loader._load_general_section()

        mock_state_store["bankFileName"].setText.assert_called_once_with("bankfile")

    def test_load_general_section_sets_plotting_flags(self, config_loader, mock_state_store):
        """Test that _load_general_section sets all plotting checkbox flags correctly."""
        config_loader._load_general_section()

        mock_state_store["makePlotsEdit"].setChecked.assert_called_once_with(True)
        mock_state_store["savePlotsEdit"].setChecked.assert_called_once_with(True)
        mock_state_store["saveZoomPlotsEdit"].setChecked.assert_called_once_with(False)
        mock_state_store["closePlotsEdit"].setChecked.assert_called_once_with(False)

    def test_load_general_section_sets_debug_output(self, config_loader, mock_state_store):
        """Test that _load_general_section sets the DebugOutput flag correctly."""
        config_loader._load_general_section()

        mock_state_store["debugOutputEdit"].setChecked.assert_called_once_with(True)

    def test_load_general_section_sets_zoom_step_km(self, config_loader, mock_state_store):
        """Test that _load_general_section sets the ZoomStepKM value correctly."""
        config_loader._load_general_section()

        mock_state_store["zoomPlotsRangeEdit"].setText.assert_called_once_with("1.0")

    def test_load_general_section_sets_figure_dir(self, config_loader, mock_state_store):
        """Test that _load_general_section sets the FigureDir field correctly."""
        config_loader._load_general_section()

        mock_state_store["figureDirEdit"].setText.assert_called_once_with("output/figures")

    def test_load_general_section_with_defaults(self, mock_state_store, tmp_path):
        """Test that _load_general_section uses default values when keys are missing."""
        # Create minimal config without optional keys
        config = ConfigParser()
        config.read_dict({
            "General": {
                "Version": "1.0",
                "RiverKM": "inputs/rivkm_20m.xyc",
                "Boundaries": "100.0:200.0",
                "BankDir": "output/banklines",
            }
        })

        mock_file = Mock(spec=ConfigFile)
        mock_file.config = config
        mock_file.version = "1.0"
        mock_file.get_range = Mock(return_value=(100.0, 200.0))
        mock_file.get_str = Mock(side_effect=lambda section, key, default=None: default)
        mock_file.get_bool = Mock(side_effect=lambda section, key, default=None: default)
        mock_file.get_float = Mock(side_effect=lambda section, key, default=None: default)

        config_path = tmp_path / "test_minimal_config.cfg"
        config_path.write_text("[General]\nVersion=1.0\n")

        with patch('dfastbe.gui.configs.StateStore.instance', return_value=mock_state_store):
            loader = ConfigurationLoader(config_path)
            loader.config_file = mock_file
            loader.config = mock_file.config
            loader.rootdir = str(tmp_path)

            loader._load_general_section()

        # Verify defaults are used
        mock_state_store["bankFileName"].setText.assert_called_once_with("bankfile")
        mock_state_store["makePlotsEdit"].setChecked.assert_called_once_with(True)
        mock_state_store["savePlotsEdit"].setChecked.assert_called_once_with(True)
        mock_state_store["saveZoomPlotsEdit"].setChecked.assert_called_once_with(False)
        mock_state_store["zoomPlotsRangeEdit"].setText.assert_called_once_with("1.0")
        mock_state_store["closePlotsEdit"].setChecked.assert_called_once_with(False)
        mock_state_store["debugOutputEdit"].setChecked.assert_called_once_with(False)

    def test_load_general_section_calls_config_methods(self, config_loader, mock_config_file):
        """Test that _load_general_section calls appropriate ConfigFile methods."""
        config_loader._load_general_section()

        # Verify ConfigFile methods are called
        mock_config_file.get_range.assert_called_once_with("General", "Boundaries")
        mock_config_file.get_str.assert_any_call("General", "BankFile", default="bankfile")
        mock_config_file.get_bool.assert_any_call("General", "Plotting", default=True)
        mock_config_file.get_bool.assert_any_call("General", "SavePlots", default=True)
        mock_config_file.get_bool.assert_any_call("General", "SaveZoomPlots", default=False)
        mock_config_file.get_float.assert_called_once_with("General", "ZoomStepKM", default=1.0)
        mock_config_file.get_bool.assert_any_call("General", "ClosePlots", default=False)
        mock_config_file.get_bool.assert_any_call("General", "DebugOutput", default=False)
