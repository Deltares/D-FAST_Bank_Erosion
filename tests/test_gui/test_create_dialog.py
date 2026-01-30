"""
Tests for the GUI creation using the new GUI class with StateManagement.
"""
from unittest.mock import patch
from PySide6 import QtWidgets
from dfastbe.gui.utils import gui_text
from dfastbe.gui.tabs.main_components import MenuBar


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
        expected_tab_names = ["General", "Detection", "Erosion", "Shipping Parameters", "Bank Parameters"]
        actual_tabs = [tabs.tabText(i) for i in range(tabs.count())]
        assert actual_tabs == expected_tab_names


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
        expected_tab_names = ["General", "Detection", "Erosion", "Shipping Parameters", "Bank Parameters"]

        layout = central_widget.layout()
        assert layout is not None

        # Check that the tabs widgets are inside the layout
        item = layout.itemAt(0)
        actual_tab_names = [item.widget().tabText(idx)
                            for idx in range(item.widget().count())]

        assert expected_tab_names == actual_tab_names


class TestCreateMenus:

    def test_menu_contains_file_option(self, mock_menubar, qapp):
        """Test that MenuBar.create() creates a File menu."""
        from PySide6.QtWidgets import QMainWindow
        window = QMainWindow()
        window.setMenuBar(mock_menubar)
        menu_bar = MenuBar(window=window, app=qapp)
        menu_bar.create()

        actions = mock_menubar.actions()
        assert len(actions) >= 2
        assert gui_text("File") in actions[0].text()


    def test_menu_contains_help_option(self, mock_menubar, qapp):
        """Test that MenuBar.create() creates a Help menu."""
        from PySide6.QtWidgets import QMainWindow
        window = QMainWindow()
        window.setMenuBar(mock_menubar)
        menu_bar = MenuBar(window=window, app=qapp)
        menu_bar.create()

        actions = mock_menubar.actions()
        assert len(actions) >= 2
        assert gui_text("Help") in actions[1].text()


    def test_menu_structure_file_dropdown(self, mock_menubar, qapp):
        """Test that File menu dropdown contains `Save`, `Load` and `Close`."""
        from PySide6.QtWidgets import QMainWindow
        window = QMainWindow()
        window.setMenuBar(mock_menubar)
        menu_bar = MenuBar(window=window, app=qapp)
        menu_bar.create()

        file_menu_action = mock_menubar.actions()[0]
        file_menu = file_menu_action.menu()
        file_actions = file_menu.actions()

        assert len(file_actions) == 4
        assert file_actions[0].text() == gui_text("Load")
        assert file_actions[1].text() == gui_text("Save")
        assert file_actions[2].isSeparator()
        assert file_actions[3].text() == gui_text("Close")


    def test_menu_structure_help_dropdown(self, mock_menubar, qapp):
        """Test that Help menu dropdown contains `Manual`, `Version` and
        `About Qt`."""
        from PySide6.QtWidgets import QMainWindow
        window = QMainWindow()
        window.setMenuBar(mock_menubar)
        menu_bar = MenuBar(window=window, app=qapp)
        menu_bar.create()

        help_menu_action = mock_menubar.actions()[1]
        help_menu = help_menu_action.menu()
        help_actions = help_menu.actions()

        assert len(help_actions) == 4
        assert help_actions[0].text() == gui_text("Manual")
        assert help_actions[1].isSeparator()
        assert help_actions[2].text() == gui_text("Version")
        assert help_actions[3].text() == gui_text("AboutQt")


class TestMenuActions:
    """Test class for mocking menu button presses."""

    @patch('dfastbe.gui.tabs.main_components.menu_load_configuration')
    def test_file_menu_load_action_triggered(self, mock_load, mock_menubar, qapp):
        """Test that triggering Load menu action calls menu_load_configuration."""
        from PySide6.QtWidgets import QMainWindow
        window = QMainWindow()
        window.setMenuBar(mock_menubar)
        menu_bar = MenuBar(window=window, app=qapp)
        menu_bar.create()

        file_menu_action = mock_menubar.actions()[0]
        file_menu = file_menu_action.menu()
        load_action = file_menu.actions()[0]

        # Trigger the action
        load_action.trigger()

        # Verify the function was called
        mock_load.assert_called_once()

    @patch('dfastbe.gui.tabs.main_components.menu_save_configuration')
    def test_file_menu_save_action_triggered(self, mock_save, mock_menubar, qapp):
        """Test that triggering Save menu action calls menu_save_configuration."""
        from PySide6.QtWidgets import QMainWindow
        window = QMainWindow()
        window.setMenuBar(mock_menubar)
        menu_bar = MenuBar(window=window, app=qapp)
        menu_bar.create()

        file_menu_action = mock_menubar.actions()[0]
        file_menu = file_menu_action.menu()
        save_action = file_menu.actions()[1]

        # Trigger the action
        save_action.trigger()

        # Verify the function was called
        mock_save.assert_called_once()

    @patch.object(MenuBar, 'close')
    def test_file_menu_close_action_triggered(self, mock_close, mock_menubar, qapp):
        """Test that triggering Close menu action calls MenuBar.close()."""
        from PySide6.QtWidgets import QMainWindow
        window = QMainWindow()
        window.setMenuBar(mock_menubar)
        menu_bar = MenuBar(window=window, app=qapp)
        menu_bar.create()

        file_menu_action = mock_menubar.actions()[0]
        file_menu = file_menu_action.menu()
        close_action = file_menu.actions()[3]  # After Load, Save, and separator

        # Trigger the action
        close_action.trigger()

        # Verify the function was called
        mock_close.assert_called_once()

    @patch('dfastbe.gui.tabs.main_components.menu_open_manual')
    def test_help_menu_open_manual_action_triggered(self, mock_open_manual, mock_menubar, qapp):
        """Test that triggering Manual menu action calls menu_open_manual."""
        from PySide6.QtWidgets import QMainWindow
        window = QMainWindow()
        window.setMenuBar(mock_menubar)
        menu_bar = MenuBar(window=window, app=qapp)
        menu_bar.create()

        help_menu_action = mock_menubar.actions()[1]
        help_menu = help_menu_action.menu()
        manual_action = help_menu.actions()[0]

        # Trigger the action
        manual_action.trigger()

        # Verify the function was called
        mock_open_manual.assert_called_once()

    @patch('dfastbe.gui.tabs.main_components.menu_about_self')
    def test_help_menu_version_action_triggered(self, mock_version, mock_menubar, qapp):
        """Test that triggering Version menu action calls menu_about_self."""
        from PySide6.QtWidgets import QMainWindow
        window = QMainWindow()
        window.setMenuBar(mock_menubar)
        menu_bar = MenuBar(window=window, app=qapp)
        menu_bar.create()

        help_menu_action = mock_menubar.actions()[1]
        help_menu = help_menu_action.menu()
        version_action = help_menu.actions()[2]  # After manual and separator

        # Trigger the action
        version_action.trigger()

        # Verify the function was called
        mock_version.assert_called_once()

    @patch('dfastbe.gui.tabs.main_components.menu_about_qt')
    def test_help_menu_about_qt_action_triggered(self, mock_about_qt, mock_menubar, qapp):
        """Test that triggering About Qt menu action calls menu_about_qt."""
        from PySide6.QtWidgets import QMainWindow
        window = QMainWindow()
        window.setMenuBar(mock_menubar)
        menu_bar = MenuBar(window=window, app=qapp)
        menu_bar.create()

        help_menu_action = mock_menubar.actions()[1]
        help_menu = help_menu_action.menu()
        about_qt_action = help_menu.actions()[3]

        # Trigger the action
        about_qt_action.trigger()

        # Verify the function was called
        mock_about_qt.assert_called_once()
