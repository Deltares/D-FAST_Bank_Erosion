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
    parser = argparse.ArgumentParser(description="D-FAST Bank Erosion.")
    
    parser.add_argument(
        "--language",
        choices=["UK", "NL"],
        default="UK",
        help=argparse.SUPPRESS
    )

    parser.add_argument(
        "--mode",
        default="GUI",
        help="run mode 'BANKLINES', 'BANKEROSION' or 'GUI' (GUI is default)"
    )

    parser.add_argument(
        "--config",
        default="dfastbe.cfg",
        help="name of configuration file ('dfastbe.cfg' is default)"
    )

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

def main():
    language, runmode, configfile = parse_arguments()
    dfastbe.cmd.run(language, runmode, configfile)

if __name__ == "__main__":
    main()
