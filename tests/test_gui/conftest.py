"""
Shared fixtures for GUI testing.
"""

import pytest

from pathlib import Path
from dfastbe import __path__
from dfastbe.io.logger import LogData
from dfastbe.gui import gui
from dfastbe.gui.gui import dialog

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


@pytest.fixture
def dialog_window(qtbot):
    """
    Fixture that creates the dialog and provides qtbot.

    The create_dialog function will use the existing QApplication.
    """
    # TODO: this fixture would need an update when switching to OOP in gui.py
    # setUp phase
    dialog.clear()

    gui.create_dialog()

    yield dialog