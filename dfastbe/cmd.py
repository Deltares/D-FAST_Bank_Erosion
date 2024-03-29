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

import matplotlib.pyplot

import sys
import os
import numpy
import dfastbe.batch
import dfastbe.gui
import dfastbe.io


def run(
    language: str = "UK", runmode: str = "GUI", configfile: str = "dfastbe.cfg",
) -> None:
    """
    Main routine initializing the language file and starting the chosen run mode.
    
    Arguments
    ---------
    language: str
        Display language 'NL' or 'UK' ('UK' is default)
    runmode: str
        Run mode 'BANKLINES', 'BANKEROSION' or 'GUI' ('GUI' is default)
    configfile: str
        Configuration file ('dfastbe.cfg' is default)
    """
    progloc = dfastbe.io.get_progloc()
    LANGUAGE = language.upper()
    try:
        dfastbe.io.load_program_texts(
            progloc + os.path.sep + "messages." + LANGUAGE + ".ini"
        )
    except:
        if LANGUAGE == "NL":
            print(
                "Het taalbestand 'messages."
                + LANGUAGE
                + ".ini' kan niet worden geladen."
            )
        else:
            print("Unable to load language file 'messages." + LANGUAGE + ".ini'")
    else:
        RUNMODE = runmode.upper()
        if RUNMODE == "BANKLINES":
            dfastbe.batch.banklines(configfile)
        elif RUNMODE == "BANKEROSION":
            dfastbe.batch.bankerosion(configfile)
        elif RUNMODE == "GUI":
            dfastbe.gui.main(configfile)
        else:
            raise Exception(
                "Invalid run mode '{}' specified. Should read 'BANKLINES', 'BANKEROSION' or 'GUI'.".format(
                    runmode
                )
            )
