"""
Shared fixtures for GUI testing.
"""

import os
import pytest
import sys

from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QBoxLayout
)
from dfastbe.gui.tabs.main_components import (
    MenuBar,
    ButtonBar
)
from dfastbe import __path__
from dfastbe.io.logger import LogData
from dfastbe.gui.application import GUI

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
def setup_menubar(qapp):
    """
    Create and initialize a MenuBar instance with a window for testing.

    This fixture sets up all the components needed to test MenuBar functionality:
    - Creates a QMainWindow with a menubar
    - Instantiates and initializes a MenuBar instance
    - Returns a dictionary with the window, menubar widget, and MenuBar instance

    Returns:
        dict: Dictionary containing:
            - 'window': QMainWindow instance
            - 'menubar': QMenuBar widget
            - 'menu_bar_instance': MenuBar instance (from tabs.main_components)

    Example:
        def test_menu_feature(setup_menubar):
            menubar = setup_menubar["menubar"]
            actions = menubar.actions()
            assert len(actions) > 0
    """

    # Create window and menubar
    window = QMainWindow()
    menubar_widget = window.menuBar()

    # Create and initialize MenuBar instance
    menu_bar_instance = MenuBar(window=window, app=qapp)
    menu_bar_instance.create()

    # Return all components in a dictionary
    result = {
        'window': window,
        'menubar': menubar_widget,
        'menu_bar_instance': menu_bar_instance
    }

    yield result

    # Cleanup
    window.close()


@pytest.fixture
def setup_button_bar(qapp):
    """
    Create and initialize a ButtonBar instance with a window for testing.

    This fixture sets up all the components needed to test ButtonBar functionality:
    - Creates a QMainWindow with a central widget and layout
    - Instantiates and initializes a ButtonBar instance
    - Returns a dictionary with the window, layout, and ButtonBar instance

    Returns:
        dict: Dictionary containing:
            - 'window': QMainWindow instance
            - 'layout': QBoxLayout instance
            - 'button_bar_instance': ButtonBar instance (from tabs.main_components)

    Example:
        def test_button_feature(setup_button_bar):
            window = setup_button_bar["window"]
            buttons = window.findChildren(QtWidgets.QPushButton)
            assert len(buttons) == 3
    """
    # Create window with central widget and layout
    window = QMainWindow()
    central_widget = QWidget()
    layout = QBoxLayout(QBoxLayout.Direction.TopToBottom, central_widget)
    window.setCentralWidget(central_widget)

    # Create and initialize ButtonBar instance
    button_bar_instance = ButtonBar(window=window, layout=layout, app=qapp)
    button_bar_instance.create()

    # Return all components in a dictionary
    result = {
        'window': window,
        'layout': layout,
        'button_bar_instance': button_bar_instance
    }

    yield result

    # Cleanup
    window.close()

