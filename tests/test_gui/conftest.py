"""
Shared fixtures for GUI testing.
"""

import pytest
from PySide6.QtWidgets import QTabWidget, QMainWindow
from pathlib import Path

from dfastbe import __path__
from dfastbe.gui.state_management import StateStore
from dfastbe.io.logger import LogData


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
def setup_general_tab_state(qapp, qtbot):
    """
    Ensures QApplication exists and sets up StateStore, QMainWindow, and QTabWidget
    for TestGeneralTab. Cleans up StateStore after each test.
    """
    StateStore._instance = None
    state = StateStore.initialize()
    window = QMainWindow()
    tabs = QTabWidget(window)
    yield {'state': state, 'window': window, 'tabs': tabs}
    StateStore._instance = None

