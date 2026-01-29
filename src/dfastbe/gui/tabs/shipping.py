from PySide6.QtWidgets import (
    QTabWidget,
    QWidget,
    QGridLayout,
    QSizePolicy,
    QSpacerItem,
)
from dfastbe.gui.base import BaseTab
from dfastbe.gui.utils import SHIP_TYPES, generalParLayout


class ShippingTab(BaseTab):
    def __init__(self, tabs: QTabWidget):
        """Initialize the tab for the bank erosion settings.

        Args:
            tabs : QTabWidget
                Tabs object to which the tab should be added.
        """
        super().__init__(tabs)

    def create(self) -> None:
        """
        Create the tab for the general shipping settings.
        """
        eParamsWidget = QWidget()
        eParamsLayout = QGridLayout(eParamsWidget)
        self.tabs.addTab(eParamsWidget, "Shipping Parameters")

        generalParLayout(eParamsLayout, 0, "shipType", "Ship Type", selectList=SHIP_TYPES)
        generalParLayout(eParamsLayout, 2, "shipVeloc", "Velocity [m/s]")
        generalParLayout(eParamsLayout, 3, "nShips", "# Ships [1/yr]")
        generalParLayout(eParamsLayout, 4, "shipNWaves", "# Waves [1/ship]")
        generalParLayout(eParamsLayout, 5, "shipDraught", "Draught [m]")
        generalParLayout(eParamsLayout, 6, "wavePar0", "Wave0 [m]")
        generalParLayout(eParamsLayout, 7, "wavePar1", "Wave1 [m]")

        stretch = QSpacerItem(
            10, 10, QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )
        eParamsLayout.addItem(stretch, 8, 0)