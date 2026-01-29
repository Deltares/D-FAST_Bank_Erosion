"""
Shared fixtures for GUI testing.
"""

import os
import pytest
import sys

from pathlib import Path
from dfastbe import __path__
from dfastbe.io.logger import LogData
from dfastbe.gui import gui
from dfastbe.gui.gui import dialog
from PySide6 import QtWidgets


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


@pytest.fixture
def setup_dialog(qtbot):
    """
    Fixture that creates the dialog and provides qtbot.

    The create_dialog function will use the existing QApplication.
    """
    # TODO: this fixture would need an update when switching to OOP in gui.py
    # setUp phase
    dialog.clear()

    gui.create_dialog()

    yield dialog


@pytest.fixture
def mock_menubar(qapp):
    """Create a mock menubar for testing."""
    menubar = QtWidgets.QMenuBar()
    return menubar