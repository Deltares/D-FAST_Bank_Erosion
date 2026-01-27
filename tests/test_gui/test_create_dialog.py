"""
Tests for the create_dialog() function in dfastbe.gui.gui module.
"""
from PyQt5 import QtWidgets
from dfastbe.gui.gui import dialog, gui_text


def test_create_dialog_contains_expected_elements(dialog_window):
    """Test that create_dialog instantiates window, tabs and application."""
    assert isinstance(dialog_window, dict)
    assert len(dialog_window) > 0
    assert "application" in dialog_window
    assert "window" in dialog_window
    assert "tabs" in dialog_window


def test_create_dialog_sets_fusion_style(dialog_window):
    """Test that the application style is set to fusion."""
    app = dialog_window["application"]
    assert app.style().objectName() == "fusion"


def test_create_dialog_creates_main_window(dialog_window):
    """Test that main window is created with correct properties."""
    win = dialog_window["window"]
    assert isinstance(win, QtWidgets.QMainWindow)
    assert win.windowTitle() == "D-FAST Bank Erosion"


def test_create_dialog_has_expected_window_geometry(dialog_window):
    """Test that window has correct initial geometry."""
    win = dialog_window["window"]
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
    assert central_widget.layout().direction() == 2


def test_create_dialog_creates_tabs(dialog_window):
    """Test that tab widget is created and stored in dialog dict."""
    tabs = dialog_window["tabs"]
    assert isinstance(tabs, QtWidgets.QTabWidget)


def test_create_dialog_tab_count(dialog_window):
    """Test that the correct number of tabs are created."""
    tabs = dialog_window["tabs"]
    # Should have 5 tabs: General, Detection, Erosion, Shipping Parameters, Bank Parameters
    assert tabs.count() == 5


def test_create_dialog_tab_names(dialog_window):
    """Test that tabs have the expected names."""
    tabs = dialog_window["tabs"]
    expected_tab_names = ["General", "Detection", "Erosion", "Shipping Parameters", "Bank Parameters"]
    actual_tabs = [tabs.tabText(i) for i in range(tabs.count())]
    assert actual_tabs == expected_tab_names


def test_create_dialog_creates_buttons(dialog_window):
    """Test that action buttons are created."""
    win = dialog_window["window"]
    buttons = win.findChildren(QtWidgets.QPushButton)

    assert len(buttons) >= 3

    button_texts = [btn.text() for btn in buttons]
    assert gui_text("action_detect") in button_texts
    assert gui_text("action_erode") in button_texts
    assert gui_text("action_close") in button_texts


def test_create_dialog_check_buttons(dialog_window):
    """Test that buttons are enabled."""
    win = dialog_window["window"]
    buttons = win.findChildren(QtWidgets.QPushButton)

    detect_btn = next(btn for btn in buttons if btn.text() == gui_text("action_detect"))
    compute_btn = next(btn for btn in buttons if btn.text() == gui_text("action_erode"))
    close_btn = next(btn for btn in buttons if btn.text() == gui_text("action_close"))

    assert detect_btn.isEnabled()
    assert compute_btn.isEnabled()
    assert close_btn.isEnabled()


def test_create_dialog_creates_menubar(dialog_window):
    """Test that menubar is created."""
    win = dialog_window["window"]
    menubar = win.menuBar()

    assert isinstance(menubar, QtWidgets.QMenuBar)


def test_create_dialog_menubar_has_menus(dialog_window):
    """Test that menubar has the expected menus."""
    win = dialog_window["window"]
    menubar = win.menuBar()

    menus = menubar.actions()
    assert len(menus) > 0

    menu_texts = [action.text() for action in menus]
    # Check for File and Help menus
    assert gui_text("File") in menu_texts
    assert gui_text("Help") in menu_texts


def test_create_dialog_tabs_widget_in_layout(dialog_window):
    """Test that tabs widget is properly added to the layout."""
    win = dialog_window["window"]
    central_widget = win.centralWidget()
    expected_tab_names = ["General", "Detection", "Erosion", "Shipping Parameters", "Bank Parameters"]

    layout = central_widget.layout()
    assert layout is not None

    # Check that the tabs widgets are inside the layout
    item = layout.itemAt(0)
    actual_tab_names = [item.widget().tabText(idx)
                        for idx in range(item.widget().count())]

    assert expected_tab_names == actual_tab_names
