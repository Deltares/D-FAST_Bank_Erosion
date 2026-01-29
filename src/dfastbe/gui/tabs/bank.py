from PySide6.QtWidgets import (
    QTabWidget,
    QWidget,
    QGridLayout,
    QLabel,
    QComboBox,
    QSizePolicy,
    QSpacerItem,
    QCheckBox,
    QLineEdit
)
from dfastbe.gui.base import BaseTab
from dfastbe.gui.utils import generalParLayout, validator
from dfastbe.gui.configs import bankStrengthSwitch
from dfastbe.gui.state_management import StateStore


class BankTab(BaseTab):
    def __init__(self, tabs: QTabWidget):
        """Initialize the tab for the bank erosion settings.

        Args:
            tabs : QTabWidget
                Tabs object to which the tab should be added.
        """
        super().__init__(tabs)

    def create(self) -> None:
        """Create the tab for the general bank properties."""
        state_management = StateStore.instance()
        eParamsWidget = QWidget()
        eParamsLayout = QGridLayout(eParamsWidget)
        self.tabs.addTab(eParamsWidget, "Bank Parameters")

        strength = QLabel("Strength Parameter")
        eParamsLayout.addWidget(strength, 0, 0)
        strengthPar = QComboBox()
        strengthPar.addItems(("Bank Type", "Critical Shear Stress"))
        strengthPar.currentIndexChanged.connect(bankStrengthSwitch)
        state_management["strengthPar"] = strengthPar
        eParamsLayout.addWidget(strengthPar, 0, 1, 1, 2)

        generalParLayout(
            eParamsLayout,
            1,
            "bankType",
            "Bank Type",
            selectList=[
                "0 (Beschermde oeverlijn)",
                "1 (Begroeide oeverlijn)",
                "2 (Goede klei)",
                "3 (Matig / slechte klei)",
                "4 (Zand)",
            ],
        )
        generalParLayout(eParamsLayout, 3, "bankShear", "Critical Shear Stress [N/m2]")
        bankStrengthSwitch()
        generalParLayout(eParamsLayout, 4, "bankProtect", "Protection [m]")
        generalParLayout(eParamsLayout, 5, "bankSlope", "Slope [-]")
        generalParLayout(eParamsLayout, 6, "bankReed", "Reed [-]")

        addFilter(eParamsLayout, 7, "velFilter", "Velocity Filter [km]")
        addFilter(eParamsLayout, 8, "bedFilter", "Bank Elevation Filter [km]")

        stretch = QSpacerItem(
            10, 10, QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )
        eParamsLayout.addItem(stretch, 9, 0)


def addFilter(
    gridLayout: QGridLayout, row: int, key: str, labelString: str
) -> None:
    """
    Add a line of controls for a filter

    Arguments
    ---------
    gridLayout : QGridLayout
        Grid layout object in which to position the edit controls.
    row : int
        Grid row number to be used for this parameter.
    key : str
        Short name of the parameter.
    labelString : str
        String describing the parameter to be displayed as label.
    """
    state_management = StateStore.instance()
    widthEdit = QLineEdit("0.3")
    widthEdit.setValidator(validator("positive_real"))
    gridLayout.addWidget(widthEdit, row, 2)
    state_management[key + "Width"] = widthEdit

    useFilter = QCheckBox("")
    useFilter.setChecked(False)
    useFilter.stateChanged.connect(lambda: updateFilter(key))
    gridLayout.addWidget(useFilter, row, 1)
    state_management[key + "Active"] = useFilter

    filterTxt = QLabel(labelString)
    gridLayout.addWidget(filterTxt, row, 0)
    state_management[key + "Txt"] = filterTxt

    updateFilter(key)


def updateFilter(key: str) -> None:
    """
    Implements the dialog setting switching for both general and optional parameters.

    Arguments
    ---------
    key : str
        Short name of the parameter.
    """
    state_management = StateStore.instance()
    if state_management[key + "Active"].isChecked():
        state_management[key + "Width"].setEnabled(True)
    else:
        state_management[key + "Width"].setEnabled(False)