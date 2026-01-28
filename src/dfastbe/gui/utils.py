import os
from typing import Any, Dict
from pathlib import Path
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMessageBox, QApplication, QDialog
from PySide6.QtGui import QValidator, QDoubleValidator

from dfastbe.io.logger import LogData

from dfastbe import __version__, __path__

SHIP_TYPES = ["1 (multiple barge convoy set)", "2 (RHK ship / motorship)", "3 (towboat)"]
r_dir = Path(__path__[0])
ICONS_DIR = r_dir / "gui/icons"
USER_MANUAL_FILE_NAME = "dfastbe_usermanual.pdf"


__all__ = [
    "get_icon",
    "gui_text",
    "SHIP_TYPES",
    "menu_open_manual",
    "show_error",
    "menu_about_qt",
    "menu_about_self",
    "validator",
    "close_edit",
    "ICONS_DIR"
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


def menu_open_manual():
    """Open the user manual."""
    manual_path = r_dir / USER_MANUAL_FILE_NAME
    if manual_path.exists():
        try:
            # bandit complains about os.startfile, but it is the only way to open a file in the default application on Windows.
            # On Linux and MacOS, opening the file might give a security warning.
            os.startfile(str(manual_path)) # nosec
        except Exception as e:
            show_error(f"Failed to open the user manual: {e}")
    else:
        show_error(f"User manual not found: {manual_path}")


def menu_about_self():
    """Show the about dialog for D-FAST Bank Erosion."""

    msg = QMessageBox()
    msg.setText(f"D-FAST Bank Erosion {__version__}")
    msg.setInformativeText("Copyright (c) 2025 Deltares.")
    msg.setDetailedText(gui_text("license"))
    msg.setWindowTitle(gui_text("about"))
    msg.setStandardButtons(QMessageBox.Ok)

    dfast_icon = get_icon(f"{ICONS_DIR}/D-FASTBE.png")
    available_sizes = dfast_icon.availableSizes()
    if available_sizes:
        icon_size = available_sizes[0]
        pixmap = dfast_icon.pixmap(icon_size).scaled(64,64)
        msg.setIconPixmap(pixmap)
    msg.setWindowIcon(dfast_icon)
    msg.exec()


def menu_about_qt():
    """Show the about dialog for Qt."""
    QApplication.aboutQt()


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