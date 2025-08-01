"""
Copyright (C) 2020 Stichting Deltares.

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
from pathlib import Path
from dfastbe.io.logger import load_program_texts
from dfastbe.io.config import ConfigFile
from dfastbe.bank_erosion.bank_erosion import Erosion
from dfastbe.bank_lines.bank_lines import BankLines
from dfastbe.io.logger import log_text, timed_logger
from dfastbe.gui.gui import main
from dfastbe import __file__
R_DIR = Path(__file__).resolve().parent
LOG_DATA_DIR = R_DIR / "io/log_data"


def run(
    language: str = "UK",
    run_mode: str = "GUI",
    configfile: str = "dfastbe.cfg",
) -> None:
    """
    Initializes the language file and starts the chosen run mode.

    This function loads the appropriate language file and executes one of the
    available modes: 'BANKLINES', 'BANKEROSION', or 'GUI'. The default configuration
    file is `dfastbe.cfg`.

    Args:
        language (str, optional):
            Display language code. Acceptable values are 'NL' (Dutch) or 'UK' (English).
            Defaults to 'UK'.
        run_mode (str, optional):
            Mode in which the program should run. Available options:

            - 'BANKLINES': Runs the bank lines processing.
            - 'BANKEROSION': Runs the bank erosion processing.
            - 'GUI': Launches the graphical user interface.

            Defaults to 'GUI'.
        configfile (str, optional):
            Path to the configuration file. Defaults to 'dfastbe.cfg'.

    Raises:
        Exception: If an invalid `run_mode` is provided. The valid options are
            'BANKLINES', 'BANKEROSION', or 'GUI'.

    Example:
        Running the program with Dutch language and bank erosion mode:

        ```python
        run(language="NL", run_mode="BANKEROSION", configfile="custom_config.cfg")
        ```

        Running the program in default mode (GUI) with the English language:

        ```python
        run()
        ```
    """
    language = language.upper()
    load_program_texts( LOG_DATA_DIR / f"messages.{language}.ini")
    run_mode = run_mode.upper()

    if run_mode == "GUI":
        main(configfile)
    else:
        config_file = ConfigFile.read(configfile)

        if run_mode == "BANKLINES":
            bank_lines = BankLines(config_file)
            bank_lines.detect()
            bank_lines.plot()
            bank_lines.save()
        elif run_mode == "BANKEROSION":
            erosion = Erosion(config_file)
            erosion.run()
            erosion.plot()
            erosion.save()
            log_text("end_bankerosion")
            timed_logger("-- end analysis --")
        else:
            raise ValueError(f"Invalid run mode {run_mode} specified. Should read 'BANKLINES', 'BANKEROSION' or 'GUI'.")