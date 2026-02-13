"""
Shared fixtures for GUI testing.
"""

import os
import pytest
import sys

from PySide6.QtWidgets import (
    QTabWidget,
    QMainWindow,
    QCheckBox,
    QLineEdit,
    QComboBox,
    QTreeWidget,
    QTreeWidgetItem
)
from pathlib import Path

from dfastbe import __path__
from dfastbe.gui.state_management import StateStore
from dfastbe.io.logger import LogData


@pytest.fixture(scope="session")
def qapp_args():
    """Arguments to pass to QApplication."""
    args = []
    if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        # Use offscreen platform to avoid Qt aborts in headless CI.
        args = ["-platform", "offscreen"]
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return args


@pytest.fixture(autouse=True)
def initialize_log_data() -> LogData:
    """
    Initialize LogData singleton with messages file before each test.

    This fixture automatically runs before each test in this directory,
    ensuring that LogData is properly initialized and reset between tests.
    """
    # Reset LogData to ensure clean state
    LogData.reset()

    return LogData(Path(__path__[0]) / "io/log_data/messages.UK.ini")


@pytest.fixture(autouse=True)
def setup_tab_state(qapp, qtbot):
    """
    Ensures QApplication exists and sets up StateStore, QMainWindow, and QTabWidget
    for TestGeneralTab.
    """
    state = StateStore.initialize()
    window = QMainWindow()
    tabs = QTabWidget(window)
    yield {'state': state, 'window': window, 'tabs': tabs}


def widget_specifications():
    """
    Returns a list of widget specifications for automated creation.
    Each entry is a tuple: (key, widget_type, optional_args)
    """
    specs = [
        ("makePlotsEdit", QCheckBox, {}),
        ("savePlotsEdit", QCheckBox, {}),
        ("saveZoomPlotsEdit", QCheckBox, {}),
        ("zoomPlotsRangeEdit", QLineEdit, {}),
        ("figureDirEdit", QLineEdit, {}),
        ("closePlotsEdit", QCheckBox, {}),
        ("debugOutputEdit", QCheckBox, {}),
        ("bankFileName", QLineEdit, {}),
        ("startRange", QLineEdit, {}),
        ("endRange", QLineEdit, {}),
        ("chainFileEdit", QLineEdit, {}),
        ("bankDirEdit", QLineEdit, {}),
        ("fairwayEdit", QLineEdit, {}),
        ("simFileEdit", QLineEdit, {}),
        ("waterDepth", QLineEdit, {}),
        ("searchLines", QTreeWidget, {"items": [["1", "line1.xy", "50"]]}),
        ("tErosion", QLineEdit, {}),
        ("riverAxisEdit", QLineEdit, {}),
        ("bankSlopeEdit", QLineEdit, {}),
        ("bankReedEdit", QLineEdit, {}),
        ("velFilterActive", QCheckBox, {}),
        ("velFilterWidth", QLineEdit, {}),
        ("bedFilterActive", QCheckBox, {}),
        ("bedFilterWidth", QLineEdit, {}),
        ("discharges", QTreeWidget, {"items": [["1", "simfile.nc", "1.0"]]}),
        ("refLevel", QLineEdit, {}),
        ("chainageOutStep", QLineEdit, {}),
        ("outDirEdit", QLineEdit, {}),
        ("bankType", QLineEdit, {}),
        ("bankTypeEditFile", QLineEdit, {}),
        ("bankShear", QLineEdit, {}),
        ("bankShearType", QLineEdit, {}),
        ("bankShearEditFile", QLineEdit, {}),
        ("newBankFile", QLineEdit, {}),
        ("newEqBankFile", QLineEdit, {}),
        ("eroVol", QLineEdit, {}),
        ("eroVolEqui", QLineEdit, {}),
        ("shipTypeType", QComboBox, {"items": ["Constant", "Other"], "current": "Constant"}),
        ("shipTypeSelect", QComboBox, {"items": ["Type1", "Type2"], "index": 0}),
        ("shipTypeEdit", QLineEdit, {}),
        ("shipVelocEdit", QLineEdit, {}),
        ("nShipsEdit", QLineEdit, {}),
        ("shipNWavesEdit", QLineEdit, {}),
        ("shipDraughtEdit", QLineEdit, {}),
        ("wavePar0Edit", QLineEdit, {}),
        ("wavePar1Edit", QLineEdit, {}),
        ("strengthPar", QComboBox, {"items": ["Bank Type", "Critical Shear Stress"], "current": "Bank Type"}),
        ("bankTypeType", QComboBox, {"items": ["Constant", "Other"], "current": "Constant"}),
        ("bankTypeSelect", QComboBox, {"items": ["Type1", "Type2"], "index": 0}),
        ("bankTypeEdit", QLineEdit, {}),
        ("bankShearEdit", QLineEdit, {}),
        ("bankProtectEdit", QLineEdit, {}),
        # Per-discharge widgets for nlevel=1
        ("1_shipTypeType", QComboBox, {"items": ["Use Default", "Constant", "Other"], "current": "Use Default"}),
        ("1_shipTypeSelect", QComboBox, {"items": ["Type1", "Type2"], "index": 0}),
        ("1_shipTypeEdit", QLineEdit, {}),
        ("1_shipVelocType", QComboBox, {"items": ["Use Default", "Other"], "current": "Use Default"}),
        ("1_shipVelocEdit", QLineEdit, {}),
        ("1_nShipsType", QComboBox, {"items": ["Use Default", "Other"], "current": "Use Default"}),
        ("1_nShipsEdit", QLineEdit, {}),
        ("1_shipNWavesType", QComboBox, {"items": ["Use Default", "Other"], "current": "Use Default"}),
        ("1_shipNWavesEdit", QLineEdit, {}),
        ("1_shipDraughtType", QComboBox, {"items": ["Use Default", "Other"], "current": "Use Default"}),
        ("1_shipDraughtEdit", QLineEdit, {}),
        ("1_bankSlopeType", QComboBox, {"items": ["Use Default", "Other"], "current": "Use Default"}),
        ("1_bankSlopeEdit", QLineEdit, {}),
        ("1_bankReedType", QComboBox, {"items": ["Use Default", "Other"], "current": "Use Default"}),
        ("1_bankReedEdit", QLineEdit, {}),
        ("1_eroVolEdit", QLineEdit, {}),
    ]
    return specs


def widget_factory():
    """
    Creates widgets based on widget_specifications and returns a state dict.
    """
    state = {}
    for key, widget_type, args in widget_specifications():
        if widget_type is QComboBox:
            widget = QComboBox()
            items = args.get("items", [])
            for item in items:
                widget.addItem(item)
            if "current" in args:
                widget.setCurrentText(args["current"])
            if "index" in args:
                widget.setCurrentIndex(args["index"])
        elif widget_type is QTreeWidget:
            widget = QTreeWidget()
            for item_data in args.get("items", []):
                item = QTreeWidgetItem(item_data)
                widget.addTopLevelItem(item)
        else:
            widget = widget_type()
        state[key] = widget
    return state


@pytest.fixture
def create_widget_configuration():
    """
    Initializes all widgets required for get_configuration and returns the state dict.
    Tests can use this fixture and modify widget values as needed.
    """
    return widget_factory()
