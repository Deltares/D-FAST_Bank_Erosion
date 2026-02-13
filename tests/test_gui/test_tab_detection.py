import pytest

from unittest.mock import patch

from PySide6.QtWidgets import QLineEdit
from PySide6.QtGui import QDoubleValidator, Qt

from dfastbe.gui.tabs.detection import DetectionTab
from dfastbe.gui.state_management import StateStore
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
        initialize_detection_tab.create()
        state = StateStore.instance()
        search_lines = state["searchLines"]
        assert search_lines.headerItem().text(0) == "Index"
        assert search_lines.headerItem().text(1) == "FileName"
        assert search_lines.headerItem().text(2) == "Search Distance [m]"

    def test_discharge_widget_column_widths(self, initialize_detection_tab):
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

