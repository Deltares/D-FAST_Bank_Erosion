"""
Shared fixtures for GUI testing.
"""

import os
import pytest
import sys

from pathlib import Path

from dfastbe import __path__
from dfastbe.io.logger import LogData
from dfastbe.gui.application import GUI, StateStore


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
def setup_gui(qapp, monkeypatch):
    """
    Create and initialize a complete GUI instance for testing.

    This fixture:
    - Resets the StateStore singleton to ensure clean state
    - Patches QApplication to use the existing qapp instance from pytest-qt
    - Creates a GUI instance with all tabs and components
    - Returns the StateManagement dictionary containing all GUI elements
    - Ensures proper cleanup after each test

    The returned dictionary contains:
        - 'application': QApplication instance
        - 'window': Main QMainWindow
        - 'tabs': QTabWidget containing all tabs
        - And all other GUI components registered in StateManagement

    Args:
        qapp: QApplication fixture from pytest-qt
        monkeypatch: pytest fixture for patching

    Yields:
        StateStore: Dictionary-like object containing all GUI components

    Example:
        def test_gui_feature(setup_gui):
            window = setup_gui["window"]
            tabs = setup_gui["tabs"]
            assert tabs.count() == 5
    """
    # Patch QApplication to return the existing qapp instance
    # This prevents creating multiple QApplication instances which causes errors
    monkeypatch.setattr(
        "dfastbe.gui.application.QApplication",
        lambda: qapp
    )

    # Create GUI instance
    gui = GUI()

    # Create all tabs and components
    gui.create()

    # Yield the StateManagement (which is accessible via StateStore.instance())
    yield gui.state

    # Cleanup: close the GUI properly
    gui.close()



@pytest.fixture
def mock_menubar(qapp):
    """Create a mock menubar for testing."""
    from PySide6.QtWidgets import QMainWindow

    # Create a minimal window for the menubar
    window = QMainWindow()
    menubar_instance = window.menuBar()

    # Return both the menubar and window for cleanup
    yield menubar_instance

    # Cleanup
    window.close()
