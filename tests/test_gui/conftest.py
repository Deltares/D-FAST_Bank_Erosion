"""
Shared fixtures for GUI testing.
"""

import os
import pytest
import sys

from PySide6.QtWidgets import (
    QTabWidget,
    QMainWindow,
    QWidget,
    QBoxLayout,
    QCheckBox,
    QLineEdit,
    QComboBox,
    QTreeWidget,
    QTreeWidgetItem
)
from pathlib import Path

from dfastbe import __path__
from dfastbe.gui.application import GUI
from dfastbe.gui.state_management import StateStore
from dfastbe.gui.tabs.main_components import (
    MenuBar,
    ButtonBar
)
from dfastbe.io.logger import LogData

USE_DEFAULT = "Use Default"

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
        ("riverKMEdit", QLineEdit, {}),
        ("bankDirEdit", QLineEdit, {}),
        ("fairwayEdit", QLineEdit, {}),
        ("simFileEdit", QLineEdit, {}),
        ("waterDepth", QLineEdit, {}),
        ("searchLines", QTreeWidget, {"items": [["1", "line1.xy", "50.0"],
                                                ["2", "line2.xy", "100.0"]]}),
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
        ("1_shipTypeType", QComboBox, {"items": [USE_DEFAULT, "Constant", "Other"], "current": USE_DEFAULT}),
        ("1_shipTypeSelect", QComboBox, {"items": ["Type1", "Type2"], "index": 0}),
        ("1_shipTypeEdit", QLineEdit, {}),
        ("1_shipVelocType", QComboBox, {"items": [USE_DEFAULT, "Other"], "current": USE_DEFAULT}),
        ("1_shipVelocEdit", QLineEdit, {}),
        ("1_nShipsType", QComboBox, {"items": [USE_DEFAULT, "Other"], "current": USE_DEFAULT}),
        ("1_nShipsEdit", QLineEdit, {}),
        ("1_shipNWavesType", QComboBox, {"items": [USE_DEFAULT, "Other"], "current": USE_DEFAULT}),
        ("1_shipNWavesEdit", QLineEdit, {}),
        ("1_shipDraughtType", QComboBox, {"items": [USE_DEFAULT, "Other"], "current": USE_DEFAULT}),
        ("1_shipDraughtEdit", QLineEdit, {}),
        ("1_bankSlopeType", QComboBox, {"items": [USE_DEFAULT, "Other"], "current": USE_DEFAULT}),
        ("1_bankSlopeEdit", QLineEdit, {}),
        ("1_bankReedType", QComboBox, {"items": [USE_DEFAULT, "Other"], "current": USE_DEFAULT}),
        ("1_bankReedEdit", QLineEdit, {}),
        ("1_eroVolEdit", QLineEdit, {}),
    ]
    return specs


def _build_combo_box(args):
    widget = QComboBox()
    for item in args.get("items", []):
        widget.addItem(item)
    if "current" in args:
        widget.setCurrentText(args["current"])
    if "index" in args:
        widget.setCurrentIndex(args["index"])
    return widget


def _build_tree_widget(args):
    widget = QTreeWidget()
    for item_data in args.get("items", []):
        widget.addTopLevelItem(QTreeWidgetItem(item_data))
    return widget


_WIDGET_BUILDERS = {
    QComboBox: _build_combo_box,
    QTreeWidget: _build_tree_widget,
}


def widget_factory():
    """Creates widgets based on widget_specifications and returns a state dict."""
    state = {}
    for key, widget_type, args in widget_specifications():
        builder = _WIDGET_BUILDERS.get(widget_type)
        state[key] = builder(args) if builder else widget_type()
    return state


@pytest.fixture
def create_widget_configuration():
    """
    Initializes all widgets required for get_configuration and returns the state dict.
    Tests can use this fixture and modify widget values as needed.
    """
    return widget_factory()
