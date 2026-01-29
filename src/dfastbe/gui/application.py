"""
Copyright (C) 2025 Stichting Deltares.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation version 2.1.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, see <http://www.gnu.org/licenses/>.

contact: delft3d.support@deltares.nl
Stichting Deltares
P.O. Box 177
2600 MH Delft, The Netherlands

All indications and logos of, and references to, "Delft3D" and "Deltares"
are registered trademarks of Stichting Deltares, and remain the property of
Stichting Deltares. All rights reserved.

INFORMATION
This file is part of D-FAST Bank Erosion: https://github.com/Deltares/D-FAST_Bank_Erosion
"""
from __future__ import annotations
import sys
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QTabWidget,
    QApplication,
    QBoxLayout,
    QMainWindow,
    QWidget
)

from dfastbe.gui.utils import (
    get_icon,
    ICONS_DIR,
)
from dfastbe.gui.configs import (
    load_configuration,
)

from dfastbe.gui.tabs.detection import DetectionTab
from dfastbe.gui.tabs.erosion import ErosionTab
from dfastbe.gui.tabs.shipping import ShippingTab
from dfastbe.gui.tabs.bank import BankTab
from dfastbe.gui.tabs.main_components import ButtonBar, MenuBar
from dfastbe.gui.tabs.general import GeneralTab
from dfastbe.gui.state_management import StateStore

__all__ = ["GUI", "main"]

class _StateProxy(MutableMapping[str, Any]):
    """Lazy proxy that forwards mapping operations to the StateStore singleton.

    This keeps existing `StateManagement[...]` call sites working without
    reassigning a module-level global in `GUI.__init__`. Every access resolves
    the current singleton via `StateStore.instance()`, so all reads and writes
    target the same shared dictionary created by the GUI constructor.
    """

    def _state(self) -> StateStore:
        return StateStore.instance()

    def __getitem__(self, key: str) -> Any:
        return self._state()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._state()[key] = value

    def __delitem__(self, key: str) -> None:
        del self._state()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._state())

    def __len__(self) -> int:
        return len(self._state())

    def __repr__(self) -> str:
        return repr(self._state())


StateManagement: MutableMapping[str, Any] = _StateProxy()


class GUI:

    def __init__(self):
        self.state = StateStore.initialize()

        self.app = QApplication()
        self.app.setStyle("fusion")
        StateManagement["application"] = self.app
        self.window, self.layout = self.create_window()
        StateManagement["window"] = self.window

        self.tabs = QTabWidget(self.window)
        StateManagement["tabs"] = self.tabs
        self.layout.addWidget(self.tabs)

        self.menu_Bar = self.create_menu_bar()
        self.button_bar = self.create_action_buttons()

    @staticmethod
    def create_window():
        win = QMainWindow()
        win.setWindowTitle("D-FAST Bank Erosion")
        win.setGeometry(200, 200, 600, 300)
        win.setWindowIcon(get_icon(f"{ICONS_DIR}/D-FASTBE.png"))

        # win.resize(1000, 800)

        central_widget = QWidget()
        layout = QBoxLayout(QBoxLayout.Direction.TopToBottom, central_widget)
        win.setCentralWidget(central_widget)
        return win, layout

    def create(self) -> None:
        """Construct the D-FAST Bank Erosion user interface."""
        self.general_tab = GeneralTab(self.tabs, self.window)
        self.general_tab.create()

        self.detection_tab = DetectionTab(self.tabs, self.window, self.app)
        self.detection_tab.create()

        self.erosion_tab = ErosionTab(self.tabs, self.window, self.app)
        self.erosion_tab.create()

        self.shipping_tab = ShippingTab(self.tabs)
        self.shipping_tab.create()

        self.bank_tab = BankTab(self.tabs)
        self.bank_tab.create()

    def create_menu_bar(self) -> MenuBar:
        """Add the menus to the menubar."""
        menu = MenuBar(window=self.window, app=self.app)
        menu.create()
        return menu

    def create_action_buttons(self) -> ButtonBar:
        button_bar = ButtonBar(window=self.window, app=self.app, layout=self.layout)
        button_bar.create()
        return button_bar

    def activate(self) -> None:
        """Activate the user interface and run the program."""
        self.window.show()
        sys.exit(self.app.exec())

    def close(self) -> None:
        """Close the dialog and program."""
        plt.close("all")
        self.window.close()
        self.app.closeAllWindows()
        self.app.quit()


def main(config: Optional[Path] = None) -> None:
    """
    Start the user interface using default settings or optional configuration.

    Args:
        config : Optional[str]
            Optional name of configuration file.
    """
    gui = GUI()
    gui.create()
    if config is not None:
        load_configuration(config)

    gui.activate()
