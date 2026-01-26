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
import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

from dfastbe.cmd import run
from dfastbe import __version__

matplotlib.use("Qt5Agg")


# ------------------------------------------------------------------------------
# Needed for Nuitka compilation
# ------------------------------------------------------------------------------

is_nuitka = "__compiled__" in globals()
if is_nuitka:
    root = str(Path(__file__).parent)
    os.environ["GDAL_DATA"] = root + os.sep + "gdal"
    os.environ["PROJ_LIB"] = root + os.sep + "proj"
    os.environ["MATPLOTLIBDATA"] = root + os.sep + "matplotlib" + os.sep + "mpl-data"
    os.environ["TCL_LIBRARY"] = root + os.sep + "lib" + os.sep + "tcl8.6"
    proj_lib_dirs = os.environ.get("PROJ_LIB", "")
    import pyproj.datadir

    pyproj.datadir.set_data_dir(root + os.sep + "proj")
    import pyproj

# ------------------------------------------------------------------------------


def _existing_path(value: str) -> Path:
    """Return a Path for an existing file, or raise argparse error."""
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"config file not found: {path}")
    return path


def parse_arguments() -> Tuple[str, str, Optional[Path]]:
    """Parse the command line arguments.

    Raises:
        LanguageError: If invalid language is specified.
    Returns:
        language (str):
            Language identifier ("NL" or "UK").
        run_mode (str):
            Specification of the run mode ("BANKLINES", "BANKEROSION" or "GUI")
        config_name (Path | None):
            Path to the configuration file, required for non-GUI modes.
    """
    parser = argparse.ArgumentParser(
        description="D-FAST Bank Erosion. Example: python -m dfastbe --mode BANKEROSION --config settings.cfg"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--language",
        default="UK",
        choices=["NL", "UK"],
        type=str.upper,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mode",
        default="GUI",
        choices=["BANKLINES", "BANKEROSION", "GUI"],
        type=str.upper,
        help="run mode 'BANKLINES', 'BANKEROSION' or 'GUI' (%(default)s is default)",
    )
    parser.add_argument(
        "--config",
        default=None,
        type=_existing_path,
        help="path to the configuration file (required unless --mode GUI)",
    )
    args = parser.parse_args()
    if args.mode != "GUI" and args.config is None:
        parser.error("--config is required when --mode is BANKLINES or BANKEROSION")

    language = args.language
    run_mode = args.mode
    configfile = args.config

    return language, run_mode, configfile


def main():
    """Main function to run the D-FAST Bank Erosion application."""
    language, run_mode, configfile = parse_arguments()
    run(language, run_mode, configfile)


if __name__ == "__main__":
    main()
