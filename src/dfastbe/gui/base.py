from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QTabWidget,
    QApplication,
    QMainWindow,
    QWidget,
    QBoxLayout,
    QPushButton,
    QGridLayout,
    QSpacerItem,
    QSizePolicy,
    QTreeWidgetItem,
    QLabel,
    QDialog,
    QLineEdit,
    QFormLayout
)

from dfastbe.gui.utils import (
    get_icon,
    ICONS_DIR,
    edit_search_line,
    addTabForLevel,
    selectFile,
    typeUpdatePar,
    set_dialog_size,
    addOpenFileRow,
    validator,
    close_edit
)
from dfastbe.gui.state_management import StateStore


class BaseTab:

    def __init__(self, tabs: QTabWidget, window: QMainWindow | None = None, app: QApplication | None = None):
        self.tabs = tabs
        self.window = window
        self.app = app

    def add_remove_edit_layout(
        self, main_widget: QWidget, key: str
    ) -> QWidget:
        """
        Create a standard layout with list control and add, edit and remove buttons.

        Arguments
        ---------
        main_widget : QWidget
            Main object on which the add, edit and remove buttons should operate.
        key : str
            Short name of the parameter.

        Returns
        -------
        parent : QWidget
            Parent QtWidget that contains the add, edit and remove buttons.
        """
        state_management = StateStore.instance()
        parent = QWidget()
        gridly = QGridLayout(parent)
        gridly.setContentsMargins(0, 0, 0, 0)

        state_management[key] = main_widget
        gridly.addWidget(main_widget, 0, 0)

        button_bar = QWidget()
        button_bar_layout = QBoxLayout(QBoxLayout.Direction.TopToBottom, button_bar)
        button_bar_layout.setContentsMargins(0, 0, 0, 0)
        gridly.addWidget(button_bar, 0, 1)

        add_button = QPushButton(get_icon(f"{ICONS_DIR}/add.png"), "")
        add_button.clicked.connect(lambda: add_an_item(key))
        state_management[key + "Add"] = add_button
        button_bar_layout.addWidget(add_button)

        edit_button = QPushButton(get_icon(f"{ICONS_DIR}/edit.png"), "")
        edit_button.clicked.connect(lambda: self.edit_an_item(key))
        edit_button.setEnabled(False)
        state_management[key + "Edit"] = edit_button
        button_bar_layout.addWidget(edit_button)

        delete_button = QPushButton(get_icon(f"{ICONS_DIR}/remove.png"), "")
        delete_button.clicked.connect(lambda: remove_an_item(key))
        delete_button.setEnabled(False)
        state_management[key + "Remove"] = delete_button
        button_bar_layout.addWidget(delete_button)

        stretch = QSpacerItem(
            10, 10, QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )
        button_bar_layout.addItem(stretch)

        return parent

    def edit_an_item(self, key: str) -> None:
        """
        Implements the actions for the edit item button.

        Dialog implemented in separate routines.

        Arguments
        ---------
        key : str
            Short name of the parameter.
        """
        state_management = StateStore.instance()
        selected = state_management[key].selectedItems()
        # root = dialog[key].invisibleRootItem()
        if len(selected) > 0:
            istr = selected[0].text(0)
            if key == "searchLines":
                file_name = selected[0].text(1)
                dist = selected[0].text(2)
                file_name, dist = edit_search_line(key, istr, file_name=file_name, dist=dist)
                selected[0].setText(1, file_name)
                selected[0].setText(2, dist)
            elif key == "discharges":
                file_name = selected[0].text(1)
                prob = selected[0].text(2)
                file_name, prob = editADischarge(key, istr, file_name=file_name, prob=prob)
                selected[0].setText(1, file_name)
                selected[0].setText(2, prob)


def add_an_item(key: str) -> None:
    """Implements the actions for the add item button.

    Args:
        key : str
            Short name of the parameter.
    """
    state_management = StateStore.instance()
    n_items = state_management[key].invisibleRootItem().childCount()
    i = n_items + 1
    istr = str(i)
    if key == "searchLines":
        file_name, dist = edit_search_line(key, istr)
        c1 = QTreeWidgetItem(state_management["searchLines"], [istr, file_name, dist])
    elif key == "discharges":
        prob = str(1 / (n_items + 1))
        file_name, prob = editADischarge(key, istr, prob=prob)
        c1 = QTreeWidgetItem(state_management["discharges"], [istr, file_name, prob])
        addTabForLevel(istr)
        state_management["refLevel"].validator().setTop(i)

    state_management[key + "Edit"].setEnabled(True)
    state_management[key + "Remove"].setEnabled(True)


def remove_an_item(key: str) -> None:
    """Implements the actions for the remove item button.

    Args:
        key : str
            Short name of the parameter.
    """
    state_management = StateStore.instance()
    selected = state_management[key].selectedItems()
    root = state_management[key].invisibleRootItem()
    if len(selected) > 0:
        istr = selected[0].text(0)
        root.removeChild(selected[0])
        i = int(istr) - 1
        for j in range(i, root.childCount()):
            root.child(j).setText(0, str(j + 1))
    else:
        istr = ""
    if root.childCount() == 0:
        state_management[key + "Edit"].setEnabled(False)
        state_management[key + "Remove"].setEnabled(False)
    if istr == "":
        pass
    elif key == "searchLines":
        pass
    elif key == "discharges":
        tabs = state_management["tabs"]
        state_management["refLevel"].validator().setTop(root.childCount())
        dj = 0
        for j in range(tabs.count()):
            if dj > 0:
                tabs.setTabText(j - 1, "Level " + str(j + dj))
                update_tab_keys(j + dj + 1)
            elif tabs.tabText(j) == "Level " + istr:
                tabs.removeTab(j)
                dj = i - j


def update_tab_keys(i: int) -> None:
    """Renumber tab i to tab i-1.

    Args:
        i : str
            Number of the tab to be updated.
    """
    state_management = StateStore.instance()
    iStart = str(i) + "_"
    newStart = str(i - 1) + "_"
    N = len(iStart)
    keys = [key for key in state_management.keys() if key[:N] == iStart]
    for key in keys:
        obj = state_management.pop(key)
        if key[-4:] == "Type":
            obj.currentIndexChanged.disconnect()
            obj.currentIndexChanged.connect(
                lambda: typeUpdatePar(newStart + key[N:-4])
            )
        elif key[-4:] == "File":
            obj.clicked.disconnect()
            obj.clicked.connect(lambda: selectFile(newStart + key[N:-4]))
        state_management[newStart + key[N:]] = obj


def editADischarge(key: str, istr: str, file_name: str = "", prob: str = ""):
    """Create an edit dialog for simulation file and weighing.

    Args:
        key : str
            Short name of the parameter.
        istr : str
            String representation of the simulation in the list.
        file_name : str
            Name of the simulation file.
        prob : str
            String representation of the weight for this simulation.
    """
    state_management = StateStore.instance()
    edit_dialog = QDialog()
    set_dialog_size(edit_dialog, 600, 100)
    edit_dialog.setWindowFlags(
        Qt.WindowTitleHint | Qt.WindowSystemMenuHint
    )
    edit_dialog.setWindowTitle("Edit Discharge")
    edit_layout = QFormLayout(edit_dialog)

    label = QLabel(istr)
    edit_layout.addRow("Level Nr", label)

    addOpenFileRow(edit_layout, "editDischarge", "Simulation File")
    state_management["editDischargeEdit"].setText(file_name)

    probability = QLineEdit()
    probability.setText(prob)
    probability.setValidator(validator("positive_real"))
    edit_layout.addRow("Probability [-]", probability)

    done = QPushButton("Done")
    done.clicked.connect(lambda: close_edit(edit_dialog))

    edit_layout.addRow(" ", done)

    edit_dialog.exec()

    file_name = state_management["editDischargeEdit"].text()
    prob = probability.text()
    return file_name, prob