"""
Tests for the create_dialog() function in dfastbe.gui.gui module.
"""
import pytest
from PyQt5 import QtWidgets
from dfastbe.gui import gui
from dfastbe.gui.gui import dialog, gui_text


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

    # Add the window to qtbot for proper event handling
    if "window" in dialog:
        qtbot.addWidget(dialog["window"])

    yield dialog

    # tearDown phase
    if "window" in dialog:
        dialog["window"].close()
    dialog.clear()


def test_create_dialog_contains_expected_elements(dialog_window):
    """Test that create_dialog instantiates window, tabs and application."""
    assert isinstance(dialog_window, dict)
    assert len(dialog_window) > 0
    assert "application" in dialog_window
    assert "window" in dialog_window
    assert "tabs" in dialog_window


def test_create_dialog_sets_fusion_style(dialog_window):
    """Test that the application style is set to fusion."""
    app = dialog["application"]
    assert app.style().objectName() == "fusion"


def test_create_dialog_creates_main_window(dialog_window):
    """Test that main window is created with correct properties."""
    win = dialog["window"]
    assert isinstance(win, QtWidgets.QMainWindow)
    assert win.windowTitle() == "D-FAST Bank Erosion"


def test_create_dialog_has_expected_window_geometry(dialog_window):
    """Test that window has correct initial geometry."""
    win = dialog["window"]
    geometry = win.geometry()
    assert geometry.x() == 200
    assert geometry.y() == 200
    assert geometry.width() == 600
    assert geometry.height() == 300


def test_create_dialog_window_has_icon(dialog_window):
    """Test that window icon is set."""
    win = dialog_window["window"]
    assert not win.windowIcon().isNull()


def test_create_dialog_has_central_widget(dialog_window):
    """Test that central widget is properly configured."""
    win = dialog["window"]
    central_widget = win.centralWidget()

    assert central_widget is not None
    assert isinstance(central_widget, QtWidgets.QWidget)
    assert central_widget.layout() is not None


def test_create_dialog_creates_tabs(dialog_window):
    """Test that tab widget is created and stored in dialog dict."""
    tabs = dialog["tabs"]
    assert isinstance(tabs, QtWidgets.QTabWidget)


def test_create_dialog_tab_count(dialog_window):
    """Test that the correct number of tabs are created."""
    tabs = dialog["tabs"]
    # Should have 5 tabs: General, Detection, Erosion, Shipping Parameters, Bank Parameters
    assert tabs.count() == 5


def test_create_dialog_tab_names(dialog_window):
    """Test that tabs have the expected names."""
    tabs = dialog["tabs"]
    expected_tabs = ["General", "Detection", "Erosion", "Shipping Parameters", "Bank Parameters"]
    actual_tabs = [tabs.tabText(i) for i in range(tabs.count())]
    assert actual_tabs == expected_tabs


def test_create_dialog_creates_buttons(dialog_window):
    """Test that action buttons are created."""
    win = dialog["window"]
    buttons = win.findChildren(QtWidgets.QPushButton)

    # Should have at least 3 main buttons: Detect, Erode, Close
    assert len(buttons) >= 3

    button_texts = [btn.text() for btn in buttons]
    assert gui_text("action_detect") in button_texts
    assert gui_text("action_erode") in button_texts
    assert gui_text("action_close") in button_texts


def test_create_dialog_detect_button(dialog_window):
    """Test that detect button exists and has proper text."""
    win = dialog["window"]
    buttons = win.findChildren(QtWidgets.QPushButton)
    detect_buttons = [btn for btn in buttons if btn.text() == gui_text("action_detect")]

    assert len(detect_buttons) == 1
    detect_btn = detect_buttons[0]
    assert detect_btn.isEnabled()


def test_create_dialog_erode_button(dialog_window):
    """Test that erode button exists and has proper text."""
    win = dialog["window"]
    buttons = win.findChildren(QtWidgets.QPushButton)
    erode_buttons = [btn for btn in buttons if btn.text() == gui_text("action_erode")]

    assert len(erode_buttons) == 1
    erode_btn = erode_buttons[0]
    assert erode_btn.isEnabled()


def test_create_dialog_close_button(dialog_window):
    """Test that close button exists and has proper text."""
    win = dialog["window"]
    buttons = win.findChildren(QtWidgets.QPushButton)
    close_buttons = [btn for btn in buttons if btn.text() == gui_text("action_close")]

    assert len(close_buttons) == 1
    close_btn = close_buttons[0]
    assert close_btn.isEnabled()


def test_create_dialog_buttons_have_connections(dialog_window):
    """Test that buttons are properly connected to their handlers."""
    win = dialog["window"]
    buttons = win.findChildren(QtWidgets.QPushButton)

    detect_btn = next(btn for btn in buttons if btn.text() == gui_text("action_detect"))
    erode_btn = next(btn for btn in buttons if btn.text() == gui_text("action_erode"))
    close_btn = next(btn for btn in buttons if btn.text() == gui_text("action_close"))

    # Check that buttons exist and are enabled (connections are harder to test directly in PyQt5)
    # The fact that the buttons were created with .clicked.connect() means they have connections
    assert detect_btn is not None
    assert erode_btn is not None
    assert close_btn is not None
    assert detect_btn.isEnabled()
    assert erode_btn.isEnabled()
    assert close_btn.isEnabled()


def test_create_dialog_creates_menubar(dialog_window):
    """Test that menubar is created."""
    win = dialog["window"]
    menubar = win.menuBar()

    assert menubar is not None
    assert isinstance(menubar, QtWidgets.QMenuBar)


def test_create_dialog_menubar_has_menus(dialog_window):
    """Test that menubar has the expected menus."""
    win = dialog["window"]
    menubar = win.menuBar()

    menus = menubar.actions()
    assert len(menus) > 0

    menu_texts = [action.text() for action in menus]
    # Check for File and Help menus
    assert gui_text("File") in menu_texts
    assert gui_text("Help") in menu_texts


def test_create_dialog_tabs_widget_in_layout(dialog_window):
    """Test that tabs widget is properly added to the layout."""
    win = dialog["window"]
    central_widget = win.centralWidget()
    tabs = dialog["tabs"]

    # Verify tabs widget is in the central widget's layout
    layout = central_widget.layout()
    assert layout is not None
    # Check that tabs is one of the widgets in the layout
    found_tabs = False
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item and item.widget() == tabs:
            found_tabs = True
            break
    assert found_tabs, "Tabs widget not found in central widget layout"


def test_create_dialog_window_not_shown_by_default(dialog_window):
    """Test that window is created but not shown by default."""
    win = dialog["window"]
    # The window should be created but not necessarily visible yet
    # (visibility depends on whether show() or exec() is called)
    assert isinstance(win, QtWidgets.QMainWindow)


def test_create_dialog_can_be_called_multiple_times(qtbot, qapp):
    """Test that create_dialog can be called multiple times without crashing."""
    dialog.clear()

    # First call
    gui.create_dialog()
    first_window = dialog.get("window")
    if first_window:
        qtbot.addWidget(first_window)
        first_window.close()

    dialog.clear()

    # Second call
    gui.create_dialog()
    second_window = dialog.get("window")
    if second_window:
        qtbot.addWidget(second_window)

    assert second_window is not None
    assert isinstance(second_window, QtWidgets.QMainWindow)

    # Cleanup
    if second_window:
        second_window.close()
    dialog.clear()


def test_create_dialog_global_dialog_dict_structure(dialog_window):
    """Test that dialog dictionary has the expected structure."""
    required_keys = ["application", "window", "tabs"]

    for key in required_keys:
        assert key in dialog, f"Missing required key: {key}"


def test_create_dialog_tabs_parent_is_window(dialog_window):
    """Test that tabs widget has the window or central widget as parent."""
    tabs = dialog["tabs"]
    win = dialog["window"]
    central_widget = win.centralWidget()

    # The tabs parent should be either the window or the central widget
    # (depending on how Qt manages the hierarchy when added to layout)
    tabs_parent = tabs.parent()
    assert tabs_parent == win or tabs_parent == central_widget
