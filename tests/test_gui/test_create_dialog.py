"""
Tests for the create_dialog() function in dfastbe.gui.gui module.
"""
from PyQt5 import QtWidgets
from dfastbe.gui.gui import (
    dialog,
    gui_text,
    createMenus,
)


def test_create_dialog_contains_expected_elements(setup_dialog):
    """Test that create_dialog instantiates window, tabs and application."""
    assert isinstance(setup_dialog, dict)
    assert len(setup_dialog) > 0
    assert "application" in setup_dialog
    assert "window" in setup_dialog
    assert "tabs" in setup_dialog


def test_create_dialog_sets_fusion_style(setup_dialog):
    """Test that the application style is set to fusion."""
    app = setup_dialog["application"]
    assert app.style().objectName() == "fusion"


def test_create_dialog_creates_main_window(setup_dialog):
    """Test that main window is created with correct properties."""
    win = setup_dialog["window"]
    assert isinstance(win, QtWidgets.QMainWindow)
    assert win.windowTitle() == "D-FAST Bank Erosion"


def test_create_dialog_has_expected_window_geometry(setup_dialog):
    """Test that window has correct initial geometry."""
    win = setup_dialog["window"]
    geometry = win.geometry()
    assert geometry.x() == 200
    assert geometry.y() == 200
    assert geometry.width() == 600
    assert geometry.height() == 300


def test_create_setup_dialog_has_icon(setup_dialog):
    """Test that window icon is set."""
    win = setup_dialog["window"]
    assert not win.windowIcon().isNull()


def test_create_dialog_has_central_widget(setup_dialog):
    """Test that central widget is properly configured."""
    win = dialog["window"]
    central_widget = win.centralWidget()

    assert central_widget is not None
    assert isinstance(central_widget, QtWidgets.QWidget)
    assert central_widget.layout() is not None
    assert central_widget.layout().direction() == 2


def test_create_dialog_creates_tabs(setup_dialog):
    """Test that tab widget is created and stored in dialog dict."""
    tabs = setup_dialog["tabs"]
    assert isinstance(tabs, QtWidgets.QTabWidget)


def test_create_dialog_tab_count(setup_dialog):
    """Test that the correct number of tabs are created."""
    tabs = setup_dialog["tabs"]
    # Should have 5 tabs: General, Detection, Erosion, Shipping Parameters, Bank Parameters
    assert tabs.count() == 5


def test_create_dialog_tab_names(setup_dialog):
    """Test that tabs have the expected names."""
    tabs = setup_dialog["tabs"]
    expected_tab_names = ["General", "Detection", "Erosion", "Shipping Parameters", "Bank Parameters"]
    actual_tabs = [tabs.tabText(i) for i in range(tabs.count())]
    assert actual_tabs == expected_tab_names


def test_create_dialog_creates_buttons(setup_dialog):
    """Test that action buttons are created."""
    win = setup_dialog["window"]
    buttons = win.findChildren(QtWidgets.QPushButton)

    assert len(buttons) >= 3

    button_texts = [btn.text() for btn in buttons]
    assert gui_text("action_detect") in button_texts
    assert gui_text("action_erode") in button_texts
    assert gui_text("action_close") in button_texts


def test_create_dialog_check_buttons(setup_dialog):
    """Test that buttons are enabled."""
    win = setup_dialog["window"]
    buttons = win.findChildren(QtWidgets.QPushButton)

    detect_btn = next(btn for btn in buttons if btn.text() == gui_text("action_detect"))
    compute_btn = next(btn for btn in buttons if btn.text() == gui_text("action_erode"))
    close_btn = next(btn for btn in buttons if btn.text() == gui_text("action_close"))

    assert detect_btn.isEnabled()
    assert compute_btn.isEnabled()
    assert close_btn.isEnabled()


def test_create_dialog_creates_menubar(setup_dialog):
    """Test that menubar is created."""
    win = setup_dialog["window"]
    menubar = win.menuBar()

    assert isinstance(menubar, QtWidgets.QMenuBar)


def test_create_dialog_menubar_has_menus(setup_dialog):
    """Test that menubar has the expected menus."""
    win = setup_dialog["window"]
    menubar = win.menuBar()

    menus = menubar.actions()
    assert len(menus) > 0

    menu_texts = [action.text() for action in menus]
    # Check for File and Help menus
    assert gui_text("File") in menu_texts
    assert gui_text("Help") in menu_texts


def test_create_dialog_tabs_widget_in_layout(setup_dialog):
    """Test that tabs widget is properly added to the layout."""
    win = setup_dialog["window"]
    central_widget = win.centralWidget()
    expected_tab_names = ["General", "Detection", "Erosion", "Shipping Parameters", "Bank Parameters"]

    layout = central_widget.layout()
    assert layout is not None

    # Check that the tabs widgets are inside the layout
    item = layout.itemAt(0)
    actual_tab_names = [item.widget().tabText(idx)
                        for idx in range(item.widget().count())]

    assert expected_tab_names == actual_tab_names


def test_createMenus_creates_file_menu(mock_menubar, setup_dialog):
    """Test that createMenus creates a File menu."""
    createMenus(mock_menubar)

    actions = mock_menubar.actions()
    assert len(actions) >= 2
    assert gui_text("File") in actions[0].text()