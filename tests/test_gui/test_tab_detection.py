import pytest
import configparser

from unittest.mock import patch
from pathlib import Path

from PySide6.QtWidgets import QLineEdit
from PySide6.QtGui import QDoubleValidator, Qt

from dfastbe.gui.tabs.detection import DetectionTab
from dfastbe.gui.state_management import StateStore
from dfastbe.gui.tabs.main_components import menu_save_configuration
from dfastbe.gui.utils import validator

@pytest.fixture
def initialize_detection_tab(setup_tab_state, qapp):
    return DetectionTab(
        setup_tab_state['tabs'],
        setup_tab_state['window'],
        qapp
    )

class TestDetectionTab:
    def test_widgets_registered(self, initialize_detection_tab):
        """Check that all expected widgets are registered in the state."""
        initialize_detection_tab.create()
        state = StateStore.instance()
        # These are the widgets DetectionTab actually registers
        for key in [
            "simFileEdit",
            "waterDepth",
            "searchLines"
        ]:
            assert key in state

    def test_water_depth_validator(self, initialize_detection_tab):
        """Check that waterDepth uses QLineEdit with QDoubleValidator (positive real)."""
        detection_tab = initialize_detection_tab
        detection_tab.create()
        state = StateStore.instance()
        water_depth = state["waterDepth"]
        assert isinstance(water_depth, QLineEdit)
        water_depth_validator = water_depth.validator()
        assert isinstance(water_depth_validator, QDoubleValidator)
        ref_validator = validator("positive_real")
        assert type(water_depth_validator) is type(ref_validator)

    def test_search_lines_widget_column_headers(self, initialize_detection_tab):
        """Check that searchLines widget has correct column headers."""
        initialize_detection_tab.create()
        state = StateStore.instance()
        search_lines = state["searchLines"]
        assert search_lines.headerItem().text(0) == "Index"
        assert search_lines.headerItem().text(1) == "FileName"
        assert search_lines.headerItem().text(2) == "Search Distance [m]"

    def test_search_lines_widget_column_widths(self, initialize_detection_tab):
        """Check that searchLines widget has correct column widths."""
        initialize_detection_tab.create()
        state = StateStore.instance()
        search_lines = state["searchLines"]
        assert search_lines.columnWidth(0) == 50
        assert search_lines.columnWidth(1) == 200


def mock_edit_search_lines_dialog(key, istr, file_name="", prob=""):
    return "test_file.xyc", "50.0"

class TestGuiBehaviorDetectionTab:
    def test_add_search_line(self, qtbot, initialize_detection_tab):
        detection_tab = initialize_detection_tab
        detection_tab.create()
        state = StateStore.instance()

        search_lines = state["searchLines"]

        assert search_lines.topLevelItemCount() == 0

        # Patch the dialog function to return test values
        with patch("dfastbe.gui.base.edit_search_line", mock_edit_search_lines_dialog):
            qtbot.mouseClick(state["searchLinesAdd"], Qt.LeftButton)

        # Check that a new item was added with the expected values
        assert search_lines.topLevelItemCount() == 1
        item = search_lines.topLevelItem(0)
        assert item.text(0) == "1"  # Index should be "1"
        assert item.text(1) == "test_file.xyc"  # File name from mock dialog
        assert item.text(2) == "50.0"  # Search distance from mock dialog

    def test_removing_search_line(self, qtbot, setup_tab_state, initialize_detection_tab):
        detection_tab = initialize_detection_tab
        detection_tab.create()
        state = StateStore.instance()

        state["tabs"] = setup_tab_state['tabs']

        # Add a search line row
        with patch("dfastbe.gui.base.edit_search_line", mock_edit_search_lines_dialog):
            qtbot.mouseClick(state["searchLinesAdd"], Qt.LeftButton)

        # Check if there is indeed one search line item before removal
        assert state["searchLines"].topLevelItemCount() == 1

        # Select the item to be removed using selection model
        item = state["searchLines"].topLevelItem(0)
        state["searchLines"].setCurrentItem(item)

        # Ensure the remove button is enabled
        remove_btn = state["searchLinesRemove"]
        edit_btn = state["searchLinesEdit"]
        if not remove_btn.isEnabled():
            remove_btn.setEnabled(True)
            edit_btn.setEnabled(True)

        qtbot.mouseClick(remove_btn, Qt.LeftButton)

        assert state["searchLines"].topLevelItemCount() == 0
        assert remove_btn.isEnabled() == False
        assert edit_btn.isEnabled() == False
        assert state["tabs"].count() == 1

    def test_menu_save_configuration_saves_detection_tab_state(
            self,
            qtbot,
            setup_tab_state,
            initialize_detection_tab,
            tmp_path,
            create_widget_configuration
        ):
        """
        Alters widgets in the Detection tab, calls menu_save_configuration, and checks
        that the saved config file contains the correct state.
        """
        window = setup_tab_state['window']
        tabs = setup_tab_state['tabs']
        qtbot.addWidget(window)
        qtbot.addWidget(tabs)
        initialize_detection_tab.create()
        state = create_widget_configuration
        # Set values for detection widgets
        state["waterDepth"].setText("5.0")

        # Ensure StateStore uses this widget state
        StateStore._instance = state
        # Patch QFileDialog.getSaveFileName to return a temp file path
        save_path = tmp_path / "saved_detection_config.cfg"
        with patch("PySide6.QtWidgets.QFileDialog.getSaveFileName", return_value=(str(save_path), "")):
            menu_save_configuration()

        # Read the saved config file and check for expected values
        config = configparser.ConfigParser()
        config.optionxform = str  # preserve case
        config.read(str(save_path))

        assert config["Detect"]["WaterDepth"] == "5.0"
        assert config["Detect"]["NBank"] == "2"
        assert Path(config["Detect"]["Line1"]).name == "line1.xy"
        assert Path(config["Detect"]["Line2"]).name == "line2.xy"
        assert config["Detect"]["DLines"] == "[ 50.0, 100.0 ]"
