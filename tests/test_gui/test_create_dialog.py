"""
Tests for the GUI creation using the new GUI class with StateManagement.
"""
from unittest.mock import patch

import pytest
from PySide6 import QtWidgets

from PySide6.QtWidgets import QMainWindow

from dfastbe.gui.utils import gui_text
from dfastbe.gui.tabs.main_components import (
    MenuBar,
    ButtonBar
)

# Expected tab names in the GUI
EXPECTED_TAB_NAMES = ["General", "Detection", "Erosion", "Shipping Parameters", "Bank Parameters"]


class TestCreateDialog:

    def test_dialog_contains_expected_elements(self, setup_gui):
        """Test that create_dialog instantiates window, tabs and application."""
        assert "application" in setup_gui
        assert "window" in setup_gui
        assert "tabs" in setup_gui


    def test_dialog_sets_fusion_style(self, setup_gui):
        """Test that the application style is set to fusion."""
        app = setup_gui["application"]
        assert app.style().objectName() == "fusion"


    def test_dialog_has_main_window(self, setup_gui):
        """Test that main window is created with correct properties."""
        win = setup_gui["window"]
        assert isinstance(win, QtWidgets.QMainWindow)
        assert win.windowTitle() == "D-FAST Bank Erosion"


    def test_dialog_has_expected_window_geometry(self, setup_gui):
        """Test that window has correct initial geometry."""
        win = setup_gui["window"]
        geometry = win.geometry()
        assert geometry.x() == 200
        assert geometry.y() == 200
        assert geometry.width() == 600
        assert geometry.height() == 300


    def test_dialog_has_icon(self, setup_gui):
        """Test that window icon is set."""
        win = setup_gui["window"]
        assert not win.windowIcon().isNull()


    def test_dialog_has_central_widget(self, setup_gui):
        """Test that central widget is properly configured."""
        win = setup_gui["window"]
        central_widget = win.centralWidget()

        assert central_widget is not None
        assert isinstance(central_widget, QtWidgets.QWidget)
        layout = central_widget.layout()
        assert layout is not None
        assert isinstance(layout, QtWidgets.QBoxLayout)
        assert layout.direction() == QtWidgets.QBoxLayout.Direction.TopToBottom


    def test_dialog_creates_tabs(self, setup_gui):
        """Test that tab widget is created and stored in dialog dict."""
        tabs = setup_gui["tabs"]
        assert isinstance(tabs, QtWidgets.QTabWidget)


    def test_dialog_tab_count(self, setup_gui):
        """Test that the correct number of tabs are created."""
        tabs = setup_gui["tabs"]
        # Should have 5 tabs: General, Detection, Erosion, Shipping Parameters, Bank Parameters
        assert tabs.count() == 5


    def test_dialog_tab_names(self, setup_gui):
        """Test that tabs have the expected names."""
        tabs = setup_gui["tabs"]
        actual_tabs = [tabs.tabText(i) for i in range(tabs.count())]
        assert actual_tabs == EXPECTED_TAB_NAMES


    def test_dialog_creates_buttons(self, setup_gui):
        """Test that action buttons are created."""
        win = setup_gui["window"]
        buttons = win.findChildren(QtWidgets.QPushButton)

        assert len(buttons) >= 3

        button_texts = [btn.text() for btn in buttons]
        assert gui_text("action_detect") in button_texts
        assert gui_text("action_erode") in button_texts
        assert gui_text("action_close") in button_texts


    def test_dialog_has_expected_buttons(self, setup_gui):
        """Test that buttons are enabled."""
        win = setup_gui["window"]
        buttons = win.findChildren(QtWidgets.QPushButton)

        detect_btn = next(btn for btn in buttons if btn.text() == gui_text("action_detect"))
        compute_btn = next(btn for btn in buttons if btn.text() == gui_text("action_erode"))
        close_btn = next(btn for btn in buttons if btn.text() == gui_text("action_close"))

        assert detect_btn.isEnabled()
        assert compute_btn.isEnabled()
        assert close_btn.isEnabled()


    def test_dialog_creates_menubar(self, setup_gui):
        """Test that menubar is created."""
        win = setup_gui["window"]
        menubar = win.menuBar()

        assert isinstance(menubar, QtWidgets.QMenuBar)


    def test_dialog_menu_texts_in_menubar(self, setup_gui):
        """Test that menubar has the expected menus."""
        win = setup_gui["window"]
        menubar = win.menuBar()

        menus = menubar.actions()
        assert len(menus) > 0

        menu_texts = [action.text() for action in menus]
        # Check for File and Help menus
        assert gui_text("File") in menu_texts
        assert gui_text("Help") in menu_texts


    def test_dialog_tabs_widget_in_layout(self, setup_gui):
        """Test that tabs widget is properly added to the layout."""
        win = setup_gui["window"]
        central_widget = win.centralWidget()

        layout = central_widget.layout()
        assert layout is not None

        # Check that the tabs widgets are inside the layout
        item = layout.itemAt(0)
        actual_tab_names = [item.widget().tabText(idx)
                            for idx in range(item.widget().count())]

        assert EXPECTED_TAB_NAMES == actual_tab_names


class TestCreateMenus:

    def test_menu_contains_file_option(self, setup_menubar):
        """Test that MenuBar.create() creates a File menu."""
        menubar = setup_menubar["menubar"]

        actions = menubar.actions()
        assert len(actions) >= 2
        assert gui_text("File") in actions[0].text()


    def test_menu_contains_help_option(self, setup_menubar):
        """Test that MenuBar.create() creates a Help menu."""
        menubar = setup_menubar["menubar"]

        actions = menubar.actions()
        assert len(actions) >= 2
        assert gui_text("Help") in actions[1].text()


    def test_menu_structure_file_dropdown(self, setup_menubar):
        """Test that File menu dropdown contains `Save`, `Load` and `Close`."""
        menubar = setup_menubar["menubar"]

        file_menu_action = menubar.actions()[0]
        file_menu = file_menu_action.menu()
        file_actions = file_menu.actions()

        assert len(file_actions) == 4
        assert file_actions[0].text() == gui_text("Load")
        assert file_actions[1].text() == gui_text("Save")
        assert file_actions[2].isSeparator()
        assert file_actions[3].text() == gui_text("Close")


    def test_menu_structure_help_dropdown(self, setup_menubar):
        """Test that Help menu dropdown contains `Manual`, `Version` and
        `About Qt`."""
        menubar = setup_menubar["menubar"]

        help_menu_action = menubar.actions()[1]
        help_menu = help_menu_action.menu()
        help_actions = help_menu.actions()

        assert len(help_actions) == 4
        assert help_actions[0].text() == gui_text("Manual")
        assert help_actions[1].isSeparator()
        assert help_actions[2].text() == gui_text("Version")
        assert help_actions[3].text() == gui_text("AboutQt")


class TestCreateButtons:
    """Test class for ButtonBar creation and structure."""

    def test_button_bar_creates_widget(self, setup_button_bar):
        """Test that ButtonBar.create() creates a button bar widget."""
        window = setup_button_bar["window"]

        # Find the button bar widget in the layout
        buttons = window.findChildren(QtWidgets.QPushButton)
        assert len(buttons) == 3


    def test_button_bar_has_detect_button(self, setup_button_bar):
        """Test that ButtonBar contains a Detect button."""
        window = setup_button_bar["window"]
        buttons = window.findChildren(QtWidgets.QPushButton)

        button_texts = [btn.text() for btn in buttons]
        assert gui_text("action_detect") in button_texts


    def test_button_bar_has_erode_button(self, setup_button_bar):
        """Test that ButtonBar contains an Erode button."""
        window = setup_button_bar["window"]
        buttons = window.findChildren(QtWidgets.QPushButton)

        button_texts = [btn.text() for btn in buttons]
        assert gui_text("action_erode") in button_texts


    def test_button_bar_has_close_button(self, setup_button_bar):
        """Test that ButtonBar contains a Close button."""
        window = setup_button_bar["window"]
        buttons = window.findChildren(QtWidgets.QPushButton)

        button_texts = [btn.text() for btn in buttons]
        assert gui_text("action_close") in button_texts


    def test_button_bar_buttons_are_enabled(self, setup_button_bar):
        """Test that all buttons in ButtonBar are enabled."""
        window = setup_button_bar["window"]
        buttons = window.findChildren(QtWidgets.QPushButton)

        detect_btn = next(btn for btn in buttons if btn.text() == gui_text("action_detect"))
        erode_btn = next(btn for btn in buttons if btn.text() == gui_text("action_erode"))
        close_btn = next(btn for btn in buttons if btn.text() == gui_text("action_close"))

        assert detect_btn.isEnabled()
        assert erode_btn.isEnabled()
        assert close_btn.isEnabled()


    def test_button_bar_layout_is_horizontal(self, setup_button_bar):
        """Test that ButtonBar uses a horizontal layout."""
        window = setup_button_bar["window"]

        # Find all buttons first
        buttons = window.findChildren(QtWidgets.QPushButton)
        assert len(buttons) == 3

        # Get the parent widget of the first button (should be the button bar)
        button_bar_widget = buttons[0].parentWidget()

        # Verify all buttons have the same parent
        assert all(btn.parentWidget() == button_bar_widget for btn in buttons)

        # Check the layout of the button bar
        layout = button_bar_widget.layout()
        assert layout is not None
        assert layout.direction() == QtWidgets.QBoxLayout.Direction.LeftToRight


class TestButtonActions:
    """Test class for mocking button clicks."""

    def _create_button_bar_with_patch(self, qapp, patch_target):
        """Helper to create ButtonBar with a specific function patched.

        Args:
            qapp: The QApplication instance
            patch_target: The target function to patch

        Returns:
            tuple: (window, buttons, mock_func, patcher) - The window, buttons list, mocked function, and patcher
        """
        patcher = patch(patch_target)
        mock_func = patcher.start()

        window = QMainWindow()
        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.Direction.TopToBottom, central_widget)
        window.setCentralWidget(central_widget)

        button_bar_instance = ButtonBar(window=window, layout=layout, app=qapp)
        button_bar_instance.create()

        buttons = window.findChildren(QtWidgets.QPushButton)

        return window, buttons, mock_func, patcher

    def _click_button(self, buttons: list, button_text: str):
        """Helper method to find and click a specific button.

        Args:
            buttons: List of QPushButton instances
            button_text: The text of the button to click
        """
        button = next(btn for btn in buttons if btn.text() == button_text)
        button.click()

    @pytest.mark.parametrize(
        "patch_target, button_text_key",
        [
            ('dfastbe.gui.tabs.main_components.run_detection', "action_detect"),
            ('dfastbe.gui.tabs.main_components.run_erosion', "action_erode"),
            ('dfastbe.gui.tabs.main_components.BaseBar.close', "action_close"),
        ],
        ids=[
            "detect_button_action",
            "erode_button_action",
            "close_button_action",
        ]
    )
    def test_button_action_triggered(self, qapp, patch_target, button_text_key):
        """Test that clicking buttons calls the expected functions.

        This parametrized test covers all button actions to reduce code duplication.
        """
        window, buttons, mock_func, patcher = self._create_button_bar_with_patch(
            qapp, patch_target
        )

        try:
            button_text = gui_text(button_text_key)
            self._click_button(buttons, button_text)
            # Process Qt events to ensure the button click signal is handled
            qapp.processEvents()
            mock_func.assert_called_once()
        finally:
            patcher.stop()
            window.close()


class TestMenuActions:
    """Test class for mocking menu button presses."""

    def _create_menu_window_with_patch(self, qapp, patch_target):
        """Helper to create MenuBar with a specific function patched.

        Args:
            qapp: The QApplication instance
            patch_target: The target function to patch

        Returns:
            tuple: (window, menubar, mock_func) - The window, menubar, and mocked function
        """
        patcher = patch(patch_target)
        mock_func = patcher.start()

        window = QMainWindow()
        menubar = window.menuBar()
        menu_bar_instance = MenuBar(window=window, app=qapp)
        menu_bar_instance.create()

        return window, menubar, mock_func, patcher

    def _trigger_menu_action(self, menubar, menu_index: int, action_index: int):
        """Helper method to navigate and trigger a specific menu action.

        Args:
            menubar: The QMenuBar instance
            menu_index: Index of the menu in the menubar (0 for File, 1 for Help)
            action_index: Index of the action within the menu
        """
        menu_action = menubar.actions()[menu_index]
        menu = menu_action.menu()
        action = menu.actions()[action_index]
        action.trigger()

    @pytest.mark.parametrize(
        "patch_target, menu_index, action_index",
        [
            ('dfastbe.gui.tabs.main_components.menu_load_configuration', 0, 0),
            ('dfastbe.gui.tabs.main_components.menu_save_configuration', 0, 1),
            ('dfastbe.gui.tabs.main_components.menu_open_manual', 1, 0),
            ('dfastbe.gui.tabs.main_components.menu_about_self', 1, 2),
            ('dfastbe.gui.tabs.main_components.menu_about_qt', 1, 3),
            ('dfastbe.gui.tabs.main_components.BaseBar.close', 0, 3),
        ],
        ids=[
            "file_load_action",
            "file_save_action",
            "help_manual_action",
            "help_version_action",
            "help_about_qt_action",
            "file_close_action",
        ]
    )
    def test_menu_action_triggered(self, qapp, patch_target, menu_index, action_index):
        """Test that triggering menu actions calls the expected functions.

        This parametrized test covers multiple menu actions to reduce code duplication.
        """
        window, menubar, mock_func, patcher = self._create_menu_window_with_patch(
            qapp, patch_target
        )

        try:
            self._trigger_menu_action(menubar, menu_index, action_index)
            mock_func.assert_called_once()
        finally:
            patcher.stop()
            window.close()
