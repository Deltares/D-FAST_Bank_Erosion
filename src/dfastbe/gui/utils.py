import os
from typing import Any, Dict, Tuple, List
from pathlib import Path
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QMessageBox,
    QDialog,
    QPushButton,
    QLabel,
    QLineEdit,
    QFormLayout,
    QGridLayout,
    QWidget,
    QFileDialog,
    QTreeWidgetItem,
    QSizePolicy,
    QSpacerItem,
    QComboBox
)
from PySide6.QtGui import QValidator, QDoubleValidator
from PySide6.QtCore import Qt

from dfastbe.io.logger import LogData

from dfastbe import __path__
from dfastbe.gui.state_management import StateStore


SHIP_TYPES = ["1 (multiple barge convoy set)", "2 (RHK ship / motorship)", "3 (towboat)"]
r_dir = Path(__path__[0])
ICONS_DIR = r_dir / "gui/icons"


__all__ = [
    "get_icon",
    "gui_text",
    "SHIP_TYPES",
    "show_error",
    "validator",
    "ICONS_DIR",
    "edit_search_line",
    "addOpenFileRow",
    "typeUpdatePar",
    "addTabForLevel",
    "add_an_item",
    "remove_an_item",
    "editADischarge",
    "generalParLayout"
]


def get_icon(file_name: str) -> QIcon:
    """Opens the icon file relative to the location of the program.

    Args:
        file_name : str
            Name of the icon file.
    """
    root_dir = Path(__file__).parent.absolute()
    return QIcon(f"{root_dir / file_name}")


def gui_text(
    key: str,
    prefix: str = "gui_",
    placeholder_dict: Dict[str, Any] | None = None
) -> str:
    """
    Query the global dictionary of texts for a single string in the GUI.

    This routine concatenates the prefix and the key to query the global
    dictionary of texts. It selects the first line of the text obtained and
    expands and placeholders in the string using the optional dictionary
    provided.

    Args:
        key : str
            The key string used to query the dictionary (extended with prefix).
        prefix : str
            The prefix used in combination with the key (default "gui_").
        placeholder_dict : Optional[Dict[str, Any]]
            A dictionary used for placeholder expansions (it defaults to None).

    Returns:
        The first line of the text in the dictionary expanded with the keys.
    """
    if placeholder_dict is None:
        placeholder_dict = {}

    text = LogData().get_text(prefix + key)
    text_str = text[0].format(**placeholder_dict)
    return text_str


def show_error(message: str, detailed_message: str | None = None) -> None:
    """Display an error message box with specified string.

    Args:
        message : str
            Text to be displayed in the message box.
        detailed_message : Option[str]
            Text to be displayed when the user clicks the Details button.
    """
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText(message)
    if detailed_message:
        msg.setDetailedText(detailed_message)

    msg.setWindowTitle("Error")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec()


def validator(valid_str: str) -> QValidator:
    """Wrapper to easily create a validator.

    Args:
        valid_str : str
            Identifier for the requested validation method.

    Returns:
        validator : QValidator
            Validator for the requested validation method.
    """
    if valid_str == "positive_real":
        validator = QDoubleValidator()
        validator.setBottom(0)
    else:
        raise ValueError(f"Unknown validator type: {valid_str}")
    return validator


def close_edit(q_dialog: QDialog) -> None:
    """Generic close function for edit dialogs.

    Args:
        q_dialog : QDialog
            Dialog object to be closed.
    """
    q_dialog.close()



def set_dialog_size(edit_dialog: QDialog, width: int, height: int) -> None:
    """Set the width and height of a dialog and position it centered relative to the main window.

    Args:
        edit_dialog : QDialog
            Dialog object to be positioned correctly.
        width : int
            Desired width of the dialog.
        height : int
            Desired height of the dialog.
    """
    state_management = StateStore.instance()
    parent = state_management["window"]
    x = parent.x()
    y = parent.y()
    pw = parent.width()
    ph = parent.height()
    edit_dialog.setGeometry(
        x + pw / 2 - width / 2, y + ph / 2 - height / 2, width, height
    )


def edit_search_line(
    key: str, istr: str, file_name: str = "", dist: str = "50"
) -> Tuple[str, str]:
    """
    Create an edit dialog for the search lines list.

    Arguments
    ---------
    key : str
        Short name of the parameter.
    istr : str
        String representation of the search line in the list.
    file_name : str
        Name of the search line file.
    dist : str
        String representation of the search distance.

    Returns
    -------
    fileName1 : str
        Updated name of the search line file.
    dist1 : str
        Updated string representation of the search distance.
    """
    state_management = StateStore.instance()
    edit_dialog = QDialog()
    set_dialog_size(edit_dialog, 600, 100)
    edit_dialog.setWindowFlags(
        Qt.WindowTitleHint | Qt.WindowSystemMenuHint
    )
    edit_dialog.setWindowTitle("Edit Search Line")
    edit_layout = QFormLayout(edit_dialog)

    label = QLabel(istr)
    edit_layout.addRow("Search Line Nr", label)

    addOpenFileRow(edit_layout, "editSearchLine", "Search Line File")
    state_management["editSearchLineEdit"].setText(file_name)

    search_distance = QLineEdit()
    search_distance.setText(dist)
    search_distance.setValidator(validator("positive_real"))
    edit_layout.addRow("Search Distance [m]", search_distance)

    done = QPushButton("Done")
    done.clicked.connect(lambda: close_edit(edit_dialog))
    # edit_SearchDistance.setValidator(validator("positive_real"))
    edit_layout.addRow(" ", done)

    edit_dialog.exec()

    file_name = state_management["editSearchLineEdit"].text()
    dist = search_distance.text()
    return file_name, dist


def addOpenFileRow(
    formLayout: QFormLayout, key: str, labelString: str
) -> None:
    """
    Add a line of controls for selecting a file or folder in a form layout.

    Arguments
    ---------
    formLayout : QFormLayout
        Form layout object in which to position the edit controls.
    key : str
        Short name of the parameter.
    labelString : str
        String describing the parameter to be displayed as label.
    """
    state_management = StateStore.instance()
    Label = QLabel(labelString)
    state_management[key] = Label
    fLayout = openFileLayout(key + "Edit")
    formLayout.addRow(Label, fLayout)


def openFileLayout(key, enabled=True) -> QWidget:
    """
    Create a standard layout with a file or folder edit field and selection button.

    Arguments
    ---------
    key : str
        Short name of the parameter.
    enabled : bool
        Flag indicating whether the file selection button should be enabed by default.

    Returns
    ------
    parent : QWidget
        Parent QtWidget that contains the edit field and selection button.
    """
    state_management = StateStore.instance()
    parent = QWidget()
    gridly = QGridLayout(parent)
    gridly.setContentsMargins(0, 0, 0, 0)

    myWidget = QLineEdit()
    state_management[key] = myWidget
    gridly.addWidget(myWidget, 0, 0)

    openFile = QPushButton(get_icon(f"{ICONS_DIR}/open.png"), "")
    openFile.clicked.connect(lambda: selectFile(key))
    openFile.setEnabled(enabled)
    state_management[key + "File"] = openFile
    gridly.addWidget(openFile, 0, 2)

    return parent

def selectFile(key: str) -> None:
    """Select a file or directory via a selection dialog.

    Args:
        key : str
            Short name of the parameter.
    """
    state_management = StateStore.instance()
    dnm: str
    if not state_management[key + "File"].hasFocus():
        # in the add/edit dialogs, the selectFile is triggered when the user presses enter in one of the lineEdit boxes ...
        # don't trigger the actual selectFile
        fil = ""
    elif key == "simFileEdit":
        fil, fltr = QFileDialog.getOpenFileName(
            caption="Select D-Flow FM Map File", filter="D-Flow FM Map Files (*map.nc)"
        )
        # getOpenFileName returns a tuple van file name and active file filter.
    elif key == "chainFileEdit":
        fil, fltr = QFileDialog.getOpenFileName(
            caption="Select Chainage File", filter="Chainage Files (*.xyc)"
        )
    elif key == "riverAxisEdit":
        fil, fltr = QFileDialog.getOpenFileName(
            caption="Select River Axis File", filter="River Axis Files (*.xyc)"
        )
    elif key == "fairwayEdit":
        fil, fltr = QFileDialog.getOpenFileName(
            caption="Select Fairway File", filter="Fairway Files (*.xyc)"
        )
    elif key == "editSearchLineEdit":
        fil, fltr = QFileDialog.getOpenFileName(
            caption="Select Search Line File", filter="Search Line Files (*.xyc)"
        )
    elif key == "editDischargeEdit":
        fil, fltr = QFileDialog.getOpenFileName(
            caption="Select Simulation File", filter="Simulation File (*map.nc)"
        )
    elif key == "bankDirEdit":
        fil = QFileDialog.getExistingDirectory(
            caption="Select Bank Directory"
        )
    elif key == "figureDirEdit":
        fil = QFileDialog.getExistingDirectory(
            caption="Select Figure Output Directory"
        )
    elif key == "outDirEdit":
        fil = QFileDialog.getExistingDirectory(
            caption="Select Output Directory"
        )
    else:
        if key[-4:] == "Edit":
            rkey = key[:-4]
            nr = ""
            while rkey[0] in "1234567890":
                nr = nr + rkey[0]
                rkey = rkey[1:]
            if rkey[0] == "_":
                rkey = rkey[1:]
            if not nr == "":
                nr = " for Level " + nr
            if rkey == "bankType":
                ftype = "Bank Type"
                ext = ".btp"
                oneFile = False
            elif rkey == "bankShear":
                ftype = "Critical Shear"
                ext = ".btp"
                oneFile = False
            elif rkey == "bankProtect":
                ftype = "Protection Level"
                ext = ".bpl"
                oneFile = False
            elif rkey == "bankSlope":
                ftype = "Bank Slope"
                ext = ".slp"
                oneFile = False
            elif rkey == "bankReed":
                ftype = "Reed Fraction"
                ext = ".rdd"
                oneFile = False
            elif rkey == "shipType":
                ftype = "Ship Type"
                ext = ""
                oneFile = True
            elif rkey == "shipVeloc":
                ftype = "Ship Velocity"
                ext = ""
                oneFile = True
            elif rkey == "nShips":
                ftype = "Number of Ships"
                ext = ""
                oneFile = True
            elif rkey == "shipNWaves":
                ftype = "Number of Ship Waves"
                ext = ""
                oneFile = True
            elif rkey == "shipDraught":
                ftype = "Ship Draught"
                ext = ""
                oneFile = True
            elif rkey == "wavePar0":
                ftype = "Wave0"
                ext = ""
                oneFile = True
            elif rkey == "wavePar1":
                ftype = "Wave1"
                ext = ""
                oneFile = True
            else:
                ftype = "Parameter"
                ext = "*"
            ftype = ftype + " File"
            fil, fltr = QFileDialog.getOpenFileName(
                caption="Select " + ftype + nr, filter=ftype + " (*" + ext + ")"
            )
            if fil != "":
                fil, fext = os.path.splitext(fil)
                if fext == ext:
                    if not oneFile:
                        # file should end on _<nr>
                        nr = ""
                        while fil[-1] in "1234567890":
                            nr = rkey[-1] + nr
                            fil = fil[:-1]
                        if nr == "" or fil[-1] != "_":
                            print("Missing bank number(s) at end of file name. Reference not updated.")
                            fil = ""
                        else:
                            fil = fil[:-1]
                else:
                    if ext == "":
                        print("Unsupported file extension {} while expecting no extension. Reference not updated.".format(fext))
                    else:
                        print("Unsupported file extension {} while expecting {}. Reference not updated.".format(fext,ext))
                    fil = ""
        else:
            print(key)
            fil = ""
    if fil != "":
        state_management[key].setText(fil)


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

def addTabForLevel(istr: str) -> None:
    """Create the tab for the settings associated with simulation i.

    Args:
        istr : str
            String representation of the simulation number.
    """
    state_management = StateStore.instance()
    newWidget = QWidget()
    newLayout = QGridLayout(newWidget)
    state_management["tabs"].addTab(newWidget, "Level " + istr)

    optionalParLayout(
        newLayout, 0, istr + "_shipType", "Ship Type", selectList=SHIP_TYPES
    )
    optionalParLayout(newLayout, 2, istr + "_shipVeloc", "Velocity [m/s]")
    optionalParLayout(newLayout, 3, istr + "_nShips", "# Ships [1/yr]")
    optionalParLayout(newLayout, 4, istr + "_shipNWaves", "# Waves [1/ship]")
    optionalParLayout(newLayout, 5, istr + "_shipDraught", "Draught [m]")
    optionalParLayout(newLayout, 6, istr + "_bankSlope", "Slope [-]")
    optionalParLayout(newLayout, 7, istr + "_bankReed", "Reed [-]")

    Label = QLabel("EroVol File Name")
    state_management[istr + "_eroVol"] = Label
    newLayout.addWidget(Label, 8, 0)
    Edit = QLineEdit()
    state_management[istr + "_eroVolEdit"] = Edit
    newLayout.addWidget(Edit, 8, 2)

    stretch = QSpacerItem(
        10, 10, QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
    )
    newLayout.addItem(stretch, 9, 0)


def optionalParLayout(
    gridLayout: QGridLayout, row: int, key, labelString, selectList=None
) -> None:
    """Add a line of controls for editing an optional parameter.

    Args:
        gridLayout : QGridLayout
            Grid layout object in which to position the edit controls.
        row : int
            Grid row number to be used for this parameter.
        key : str
            Short name of the parameter.
        labelString : str
            String describing the parameter to be displayed as label.
        selectList : Optional[List[str]]
            In case the parameter can only have a limited number of values: a list
            of strings describing the options.
    """
    state_management = StateStore.instance()
    Label = QLabel(labelString)
    state_management[key + "Label"] = Label
    gridLayout.addWidget(Label, row, 0)

    paramTypes = ("Use Default", "Constant", "Variable")
    Type = QComboBox()
    Type.addItems(paramTypes)
    Type.currentIndexChanged.connect(lambda: typeUpdatePar(key))
    state_management[key + "Type"] = Type
    gridLayout.addWidget(Type, row, 1)

    if selectList is None:
        fLayout = openFileLayout(key + "Edit", enabled=False)
        state_management[key + "Edit"].setEnabled(False)
        gridLayout.addWidget(fLayout, row, 2)
    else:
        Select = QComboBox()
        Select.addItems(selectList)
        Select.setEnabled(False)
        state_management[key + "Select"] = Select
        gridLayout.addWidget(Select, row, 2)

        fLayout = openFileLayout(key + "Edit", enabled=False)
        state_management[key + "Edit"].setEnabled(False)
        gridLayout.addWidget(fLayout, row + 1, 2)


def typeUpdatePar(key: str) -> None:
    """
    Implements the dialog setting switching for both general and optional parameters.

    Arguments
    ---------
    key : str
        Short name of the parameter.
    """
    state_management = StateStore.instance()
    type = state_management[key + "Type"].currentText()
    state_management[key + "Edit"].setText("")
    if type == "Use Default":
        state_management[key + "Edit"].setValidator(None)
        state_management[key + "Edit"].setEnabled(False)
        state_management[key + "EditFile"].setEnabled(False)
        if key + "Select" in state_management.keys():
            state_management[key + "Select"].setEnabled(False)
    elif type == "Constant":
        if key + "Select" in state_management.keys():
            state_management[key + "Select"].setEnabled(True)
            state_management[key + "Edit"].setEnabled(False)
        else:
            if key != "bankProtect":
                state_management[key + "Edit"].setValidator(validator("positive_real"))
            state_management[key + "Edit"].setEnabled(True)
        state_management[key + "EditFile"].setEnabled(False)
    elif type == "Variable":
        if key + "Select" in state_management.keys():
            state_management[key + "Select"].setEnabled(False)
        state_management[key + "Edit"].setEnabled(True)
        state_management[key + "Edit"].setValidator(None)
        state_management[key + "EditFile"].setEnabled(True)


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


def generalParLayout(
    gridLayout: QGridLayout,
    row: int,
    key: str,
    labelString: str,
    selectList: List[str] | None = None,
) -> None:
    """
    Add a line of controls for editing a general parameter.

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
    selectList : Optional[List[str]]
        In case the parameter can only have a limited number of values: a list
        of strings describing the options.
    """
    state_management = StateStore.instance()
    Label = QLabel(labelString)
    state_management[key] = Label
    gridLayout.addWidget(Label, row, 0)

    paramTypes = ("Constant", "Variable")
    Type = QComboBox()
    Type.addItems(paramTypes)
    Type.currentIndexChanged.connect(lambda: typeUpdatePar(key))
    state_management[key + "Type"] = Type
    gridLayout.addWidget(Type, row, 1)

    if selectList is None:
        fLayout = openFileLayout(key + "Edit", enabled=False)
        gridLayout.addWidget(fLayout, row, 2)
    else:
        Select = QComboBox()
        Select.addItems(selectList)
        state_management[key + "Select"] = Select
        gridLayout.addWidget(Select, row, 2)

        fLayout = openFileLayout(key + "Edit", enabled=False)
        state_management[key + "Edit"].setEnabled(False)
        gridLayout.addWidget(fLayout, row + 1, 2)

    typeUpdatePar(key)