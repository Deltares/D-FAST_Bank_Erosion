from __future__ import annotations
import os
from pathlib import Path
from PySide6.QtWidgets import (
    QPushButton,
    QWidget,
    QBoxLayout,
    QApplication,
    QMainWindow,
    QFileDialog
)
import matplotlib.pyplot as plt
from dfastbe.gui.utils import (
    gui_text,
    menu_about_self,
    menu_about_qt,
    show_error
)
from dfastbe.gui.configs import (
    get_configuration,
    load_configuration,
)
from dfastbe.gui.tabs.analysis_runner import run_detection, run_erosion
from dfastbe.io.config import ConfigFile
from dfastbe import __path__


r_dir = Path(__path__[0])
USER_MANUAL_FILE_NAME = "dfastbe_usermanual.pdf"


class BaseBar:
    def __init__(self, *, window: QMainWindow, app: QApplication, layout: QBoxLayout | None = None):
        self.window = window
        self.layout = layout
        self.app = app

    def create(self):
        ...

    def close(self) -> None:
        """Close the dialog and program."""
        plt.close("all")
        self.window.close()
        self.app.closeAllWindows()
        self.app.quit()


class MenuBar(BaseBar):
    def __init__(self, window: QMainWindow, app: QApplication):
        super().__init__(window=window, app=app)
        self.menubar = self.window.menuBar()

    def create(self):
        menu = self.menubar.addMenu(gui_text("File"))
        item = menu.addAction(gui_text("Load"))
        item.triggered.connect(menu_load_configuration)
        item = menu.addAction(gui_text("Save"))
        item.triggered.connect(menu_save_configuration)
        menu.addSeparator()
        item = menu.addAction(gui_text("Close"))
        item.triggered.connect(self.close)

        menu = self.menubar.addMenu(gui_text("Help"))
        item = menu.addAction(gui_text("Manual"))
        item.triggered.connect(menu_open_manual)
        menu.addSeparator()
        item = menu.addAction(gui_text("Version"))
        item.triggered.connect(menu_about_self)
        item = menu.addAction(gui_text("AboutQt"))
        item.triggered.connect(menu_about_qt)


class ButtonBar(BaseBar):
    def __init__(self, window: QMainWindow, layout: QBoxLayout, app: QApplication | None = None):
        super().__init__(window=window, app=app, layout=layout)

    def create(self):
        button_bar = QWidget(self.window)
        button_bar_layout = QBoxLayout(QBoxLayout.Direction.LeftToRight, button_bar)
        button_bar_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(button_bar)

        detect = QPushButton(gui_text("action_detect"), self.window)
        detect.clicked.connect(lambda: run_detection(self.app))
        button_bar_layout.addWidget(detect)

        erode = QPushButton(gui_text("action_erode"), self.window)
        erode.clicked.connect(lambda: run_erosion(self.app))
        button_bar_layout.addWidget(erode)

        done = QPushButton(gui_text("action_close"), self.window)
        done.clicked.connect(self.close)
        button_bar_layout.addWidget(done)


def menu_load_configuration() -> None:
    """Select and load a configuration file."""

    file = QFileDialog.getOpenFileName(
        caption="Select Configuration File", filter="Config Files (*.cfg)"
    )
    filename = file[0]
    if filename != "":
        load_configuration(Path(filename))


def menu_save_configuration() -> None:
    """Ask for a configuration file name and save GUI selection to that file."""

    fil = QFileDialog.getSaveFileName(
        caption="Save Configuration As", filter="Config Files (*.cfg)"
    )
    filename = fil[0]
    if filename != "":
        config = get_configuration()
        rootdir = os.path.dirname(filename)
        config_file = ConfigFile(config)
        config_file.relative_to(rootdir)
        config.write(filename)


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