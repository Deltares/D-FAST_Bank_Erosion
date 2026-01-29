"""Analysis runner for the ui."""

import os
import traceback
from typing import Callable

import matplotlib.pyplot as plt
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from dfastbe.bank_erosion.bank_erosion import Erosion
from dfastbe.bank_lines.bank_lines import BankLines
from dfastbe.gui.configs import get_configuration
from dfastbe.gui.utils import show_error
from dfastbe.io.config import ConfigFile

__all__ = ["run_erosion", "run_detection"]


class Invoker:

    def __init__(self, func: Callable[[ConfigFile], None], app: QApplication):
        """

        Args:
            func (Callable[[configparser.ConfigParser], None]):
                function containing the plain analysis steps
            app (QApplication):
                pyside6 application
        """
        self.callable = func
        self.app = app

    def __call__(self) -> None:
        """Run an analysis based on settings in the GUI.

        Use a dummy configuration name in the current work directory to create
        relative paths.
        """
        config = get_configuration()
        rootdir = os.getcwd()
        config_file = ConfigFile(config)
        config_file.root_dir = rootdir
        config_file.relative_to(rootdir)
        self.app.setOverrideCursor(Qt.WaitCursor)

        plt.close("all")
        # should maybe use a separate thread for this ...
        try:
            self.callable(config_file)
        except (SystemExit, KeyboardInterrupt) as exception:
            raise exception
        except:
            self.app.restoreOverrideCursor()
            stack_trace = traceback.format_exc()
            show_error(
                "A run-time exception occurred. Press 'Show Details...' for the full stack trace.",
                stack_trace,
            )
        else:
            self.app.restoreOverrideCursor()


def run_detection(app) -> None:
    """Trigger the bank line detection analysis with error handling."""
    Invoker(run_detection_steps, app)()


def run_detection_steps(config_file: ConfigFile) -> None:
    """
    Create bank line detection object and run the analysis.

    Arguments
    ---------
    config_file : configparser.ConfigParser
        Analysis configuration settings.
    """
    bank_line = BankLines(config_file, gui=True)
    bank_line.detect()
    bank_line.plot()
    bank_line.save()


def run_erosion(app) -> None:
    """Trigger the bank line erosion analysis with error handling."""
    Invoker(run_erosion_steps, app)()


def run_erosion_steps(config_file: ConfigFile) -> None:
    """Create bank line erosion object and run the analysis.

    Args:
        config_file: ConfigFile
            Analysis configuration settings.
    """
    erosion = Erosion(config_file, gui=True)
    erosion.run()
    erosion.plot()
    erosion.save()
