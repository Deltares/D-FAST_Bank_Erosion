from PySide6.QtWidgets import (
    QLineEdit,
    QFormLayout,
    QTreeWidget,
    QWidget,
    QTabWidget,
    QApplication,
    QMainWindow
)

from dfastbe.gui.utils import addOpenFileRow, validator
from dfastbe.gui.base import BaseTab
from dfastbe.gui.state_management import StateStore


class DetectionTab(BaseTab):

    def __init__(self, tabs: QTabWidget, window: QMainWindow, app: QApplication):
        """Initialize the tab for the bank line detection settings.

        Args:
            tabs : QTabWidget
                Tabs object to which the tab should be added.
            window : QMainWindow
                The window object in which the tab item is located.
            app : QApplication
                The application object to which the window belongs, needed for font information.
        """
        super().__init__(tabs, window, app)

    def create(self) -> None:
        """Create the tab for the bank line detection settings."""
        detect_widget = QWidget()
        detect_layout = QFormLayout(detect_widget)
        self.tabs.addTab(detect_widget, "Detection")

        addOpenFileRow(detect_layout, "simFile", "Simulation File")

        water_depth = QLineEdit(self.window)
        water_depth.setValidator(validator("positive_real"))
        state_management = StateStore.instance()
        state_management["waterDepth"] = water_depth
        detect_layout.addRow("Water Depth [m]", water_depth)

        search_lines = QTreeWidget(self.window)
        search_lines.setHeaderLabels(["Index", "FileName", "Search Distance [m]"])
        search_lines.setFont(self.app.font())
        search_lines.setColumnWidth(0, 50)
        search_lines.setColumnWidth(1, 200)

        search_lines_layout = self.add_remove_edit_layout(search_lines, "searchLines")
        detect_layout.addRow("Search Lines", search_lines_layout)