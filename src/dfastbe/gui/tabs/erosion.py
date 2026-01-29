from PySide6.QtWidgets import (
    QLineEdit,
    QFormLayout,
    QTreeWidget,
    QWidget,
    QTabWidget,
    QApplication,
    QMainWindow
)
from PySide6.QtGui import QIntValidator
from dfastbe.gui.utils import addOpenFileRow, validator
from dfastbe.gui.base import BaseTab
from dfastbe.gui.state_management import StateStore


class ErosionTab(BaseTab):
    def __init__(self, tabs: QTabWidget, window: QMainWindow, app: QApplication):
        """Initialize the tab for the bank erosion settings.

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
        """Create the tab for the main bank erosion settings."""
        state_management = StateStore.instance()
        erosion_widget = QWidget()
        erosion_layout = QFormLayout(erosion_widget)
        self.tabs.addTab(erosion_widget, "Erosion")

        erosion_time = QLineEdit(self.window)
        erosion_time.setValidator(validator("positive_real"))
        state_management["tErosion"] = erosion_time
        erosion_layout.addRow("Simulation Time [yr]", erosion_time)

        addOpenFileRow(erosion_layout, "riverAxis", "River Axis File")

        addOpenFileRow(erosion_layout, "fairway", "Fairway File")

        discharges = QTreeWidget(self.window)
        discharges.setHeaderLabels(["Level", "FileName", "Probability [-]"])
        discharges.setFont(self.app.font())
        discharges.setColumnWidth(0, 50)
        discharges.setColumnWidth(1, 250)
        # c1 = QTreeWidgetItem(discharges, ["0", "test\\filename", "0.5"])

        discharge_layout = self.add_remove_edit_layout(discharges, "discharges")
        erosion_layout.addRow("Discharges", discharge_layout)

        ref_level = QLineEdit(self.window)
        ref_level.setValidator(QIntValidator(1, 1))
        state_management["refLevel"] = ref_level
        erosion_layout.addRow("Reference Case", ref_level)

        chainage_out_step = QLineEdit(self.window)
        chainage_out_step.setValidator(validator("positive_real"))
        state_management["chainageOutStep"] = chainage_out_step
        erosion_layout.addRow("Chainage Output Step [km]", chainage_out_step)

        addOpenFileRow(erosion_layout, "outDir", "Output Directory")

        new_bank_file = QLineEdit(self.window)
        state_management["newBankFile"] = new_bank_file
        erosion_layout.addRow("New Bank File Name", new_bank_file)

        new_eq_bank_file = QLineEdit(self.window)
        state_management["newEqBankFile"] = new_eq_bank_file
        erosion_layout.addRow("New Eq Bank File Name", new_eq_bank_file)

        erosion_volume = QLineEdit(self.window)
        state_management["eroVol"] = erosion_volume
        erosion_layout.addRow("EroVol File Name", erosion_volume)

        ero_vol_eq = QLineEdit(self.window)
        state_management["eroVolEqui"] = ero_vol_eq
        erosion_layout.addRow("EroVolEqui File Name", ero_vol_eq)