from dfastbe.gui.state_management import StateStore

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
)

from dfastbe.gui.utils import (
    get_icon,
    ICONS_DIR,
    add_an_item,
    remove_an_item,
    edit_search_line,
    editADischarge
)


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