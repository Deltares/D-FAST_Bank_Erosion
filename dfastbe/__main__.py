# -*- coding: utf-8 -*-
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

from typing import Optional, Tuple

# ------------------------------------------------------------------------------
# Needed for Nuitka compilation
# ------------------------------------------------------------------------------
import os
import pathlib

is_nuitka = "__compiled__" in globals()
if is_nuitka:
    root = str(pathlib.Path(__file__).parent)
    os.environ["GDAL_DATA"] = root + os.sep + "gdal"
    os.environ["PROJ_LIB"] = root + os.sep + "proj"
    os.environ["MATPLOTLIBDATA"] = root + os.sep + "matplotlib" + os.sep + "mpl-data"
    os.environ["TCL_LIBRARY"] = root + os.sep + "lib" + os.sep + "tcl8.6"
    proj_lib_dirs = os.environ.get("PROJ_LIB", "")
    import pyproj.datadir

    pyproj.datadir.set_data_dir(root + os.sep + "proj")
    import pyproj

import fiona.enums
import fiona.ogrext
import fiona.schema
import _ctypes
import pandas._libs.tslibs.base

import six
import netCDF4.utils
import cftime

# ------------------------------------------------------------------------------
import matplotlib

matplotlib.use("Qt5Agg")

import argparse
import dfastbe.cmd


def parse_arguments() -> Tuple[str, str, str]:
    """
    Parse the command line arguments.

    Arguments
    ---------
    None

    Raises
    ------
    Exception
        If invalid language is specified.

    Returns
    -------
    language : str
        Language identifier ("NL" or "UK").
    runmode : str
        Specification of the run mode ("BANKLINES", "BANKEROSION" or "GUI")
    config_name : str
        Name of the configuration file.
    """
    parser = argparse.ArgumentParser(description="D-FAST Morphological Impact.")
    parser.add_argument(
        "--language", help="display language 'NL' or 'UK' ('UK' is default)"
    )
    parser.set_defaults(language="UK")
    parser.add_argument(
        "--mode", help="run mode 'BANKLINES', 'BANKEROSION' or 'GUI' (GUI is default)"
    )
    parser.set_defaults(mode="GUI")
    parser.add_argument(
        "--config", help="name of configuration file ('dfastbe.cfg' is default)"
    )
    parser.set_defaults(config="dfastbe.cfg")
    args = parser.parse_args()

    language = args.__dict__["language"].upper()
    runmode = args.__dict__["mode"].upper()
    configfile = args.__dict__["config"]
    if language not in ["NL", "UK"]:
        raise Exception(
            "Incorrect language '{}' specified. Should read 'NL' or 'UK'.".format(
                language
            )
        )
    return language, runmode, configfile


if __name__ == "__main__":
    language, runmode, configfile = parse_arguments()
    dfastbe.cmd.run(language, runmode, configfile)
