from typing import Any, Dict
from pathlib import Path
from PySide6.QtGui import QIcon

from dfastbe.io.logger import LogData


__all__ = ["get_icon", "gui_text"]


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