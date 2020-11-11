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

from typing import Union, Dict, List, Optional, Tuple

import numpy
SimulationObject = Dict[str, numpy.ndarray]

import netCDF4
import configparser
import os
import pandas
import geopandas
import shapely

PROGTEXTS: Dict[str, List[str]]


def load_program_texts(filename: str) -> None:
    """
    Load texts from configuration file, and store globally for access.

    This routine reads the text file "filename", and detects the keywords
    indicated by lines starting with [ and ending with ]. The content is
    placed in a global dictionary PROGTEXTS which may be queried using the
    routine "program_texts". These routines are used to implement multi-
    language support.

    Parameters
    ----------
    filename : str
        The name of the file to be read and parsed.
    """
    text: List[str]
    dict: Dict[str, List[str]]

    global PROGTEXTS

    all_lines = open(filename, "r").read().splitlines()
    dict = {}
    text = []
    key = None
    for line in all_lines:
        rline = line.strip()
        if rline.startswith("[") and rline.endswith("]"):
            if not key is None:
                dict[key] = text
            key = rline[1:-1]
            text = []
        else:
            text.append(line)
    if key in dict.keys():
        raise Exception('Duplicate entry for "{}" in "{}".'.format(key, filename))
    if not key is None:
        dict[key] = text
    PROGTEXTS = dict


def program_texts(key: str) -> List[str]:
    """
    Query the global dictionary of texts via a string key.

    Query the global dictionary PROGTEXTS by means of a string key and return
    the list of strings contained in the dictionary. If the dictionary doesn't
    include the key, a default string is returned.

    Parameters
    ----------
    key : str
        The key string used to query the dictionary.

    Returns
    -------
    text : List[str]
        The list of strings returned contain the text stored in the dictionary
        for the key. If the key isn't available in the dictionary, the routine
        returns the default string "No message found for <key>"
    """

    global PROGTEXTS

    try:
        str = PROGTEXTS[key]
    except:
        str = ["No message found for " + key]
    return str


def read_xyc(filename: str, ncol: int = 2):
    """

    Arguments
    ---------
    filename : str
        Name of the file to be read.
    ncol : int
        Number of columns to be read (2 or 3)

    Returns
    -------
    L :
        Line string.
    """
    fileroot, ext = os.path.splitext(filename)
    if ext.lower() == ".xyc":
        if ncol == 3:
            colnames = ["Val", "X", "Y"]
        else:
            colnames = ["X", "Y"]
        P = pandas.read_csv(
            filename, names=colnames, skipinitialspace=True, delim_whitespace=True
        )
        nPnts = len(P.X)
        x = P.X.to_numpy().reshape((nPnts, 1))
        y = P.Y.to_numpy().reshape((nPnts, 1))
        if ncol == 3:
            z = P.Val.to_numpy().reshape((nPnts, 1))
            LC = numpy.concatenate((x, y, z), axis=1)
        else:
            LC = numpy.concatenate((x, y), axis=1)
        L = shapely.geometry.asLineString(LC)
    else:
        GEO = geopandas.read_file(filename)["geometry"]
        L = [object for object in GEO]
    return L


def write_km_eroded_volumes(km, vol, filename):
    """
    Write a text file with eroded volume data binned per kilometre.

    Arguments
    ---------
    km :
        Array containing chainage values.
    vol :
        Array containing erosion volume values.
    filename : str
        Name of the file to be written.

    Returns
    -------
    None
    """
    with open(filename, "w") as erofile:
        for i in range(len(km)):
            valstr = "\t".join(["{:.2f}".format(x) for x in vol[i, :]])
            erofile.write("{:.2f}\t".format(km[i]) + valstr + "\n")


def read_config(filename: str) -> configparser.ConfigParser:
    """Read a configParser object (configuration file).

    This function ...
        reads the config file using the standard configParser.
        falls back to a dedicated reader compatible with old waqbank files.

    Arguments
    ---------
    filename : str
        Name of configuration file to be read.

    Returns
    -------
    config : configparser.ConfigParser
        Settings for the D-FAST Bank Erosion analysis.
    """
    try:
        config = configparser.ConfigParser(comment_prefixes=("%"))
        with open(filename, "r") as configfile:
            config.read_file(configfile)
    except:
        config = configparser.ConfigParser()
        config["General"] = {}
        all_lines = open(filename, "r").read().splitlines()
        for line in all_lines:
            perc = line.find("%")
            if perc >= 0:
                line = line[:perc]
            data = line.split()
            if len(data) >= 3:
                config["General"][data[0]] = data[2]
    return upgrade_config(config)


def upgrade_config(config: configparser.ConfigParser):
    """
    Upgrade the configuration data structure to version 1.0 format.

    Arguments
    ---------
    config : configparser.ConfigParser
        Settings for the D-FAST Bank Erosion analysis.

    Results
    -------
    config1 : configparser.ConfigParser
        Settings for the D-FAST Bank Erosion analysis in 1.0 format.

    """
    try:
        version = config["General"]["Version"]
    except:
        version = "0.1"

    if version == "0.1":
        config["General"]["Version"] = "1.0"

        config["Detect"] = {}
        config = movepar(config, "General", "Delft3Dfile", "Detect", "SimFile")
        config = movepar(config, "General", "SDSfile", "Detect", "SimFile")
        config = movepar(config, "General", "SimFile", "Detect")
        config = movepar(config, "General", "NBank", "Detect")
        NBank = config_get_int(config, "Detect", "NBank", default=0, positive=True)
        for i in range(NBank):
            istr = str(i + 1)
            config = movepar(config, "General", "Line" + istr, "Detect")
        config = movepar(config, "General", "WaterDepth", "Detect")
        config = movepar(config, "General", "DLines", "Detect")

        config["Erosion"] = {}
        config = movepar(config, "General", "TErosion", "Erosion")
        config = movepar(config, "General", "RiverAxis", "Erosion")
        config = movepar(config, "General", "Fairway", "Erosion")
        config = movepar(config, "General", "RefLevel", "Erosion")
        config = movepar(config, "General", "OutputInterval", "Erosion")
        config = movepar(config, "General", "OutputDir", "Erosion")
        config = movepar(config, "General", "BankNew", "Erosion")
        config = movepar(config, "General", "BankEq", "Erosion")
        config = movepar(config, "General", "EroVol", "Erosion")
        config = movepar(config, "General", "EroVolEqui", "Erosion")
        config = movepar(config, "General", "NLevel", "Erosion")
        NLevel = config_get_int(config, "Erosion", "NLevel", default=0, positive=True)
        for i in range(NLevel):
            istr = str(i + 1)
            config = movepar(
                config, "General", "Delft3Dfile" + istr, "Erosion", "SimFile" + istr
            )
            config = movepar(
                config, "General", "SDSfile" + istr, "Erosion", "SimFile" + istr
            )
            config = movepar(config, "General", "SimFile" + istr, "Erosion")
            config = movepar(config, "General", "PDischarge" + istr, "Erosion")

        config = movepar(config, "General", "ShipType", "Erosion")
        config = movepar(config, "General", "VShip", "Erosion")
        config = movepar(config, "General", "NShip", "Erosion")
        config = movepar(config, "General", "NWave", "Erosion")
        config = movepar(config, "General", "Draught", "Erosion")
        config = movepar(config, "General", "Wave0", "Erosion")
        config = movepar(config, "General", "Wave1", "Erosion")

        config = movepar(config, "General", "Classes", "Erosion")
        config = movepar(config, "General", "BankType", "Erosion")
        config = movepar(config, "General", "ProtectLevel", "Erosion")
        config = movepar(config, "General", "Slope", "Erosion")
        config = movepar(config, "General", "Reed", "Erosion")
        config = movepar(config, "General", "VelFilter", "Erosion")

        for i in range(NLevel):
            istr = str(i + 1)
            config = movepar(config, "General", "ShipType" + istr, "Erosion")
            config = movepar(config, "General", "VShip" + istr, "Erosion")
            config = movepar(config, "General", "NShip" + istr, "Erosion")
            config = movepar(config, "General", "NWave" + istr, "Erosion")
            config = movepar(config, "General", "Draught" + istr, "Erosion")
            config = movepar(config, "General", "Slope" + istr, "Erosion")
            config = movepar(config, "General", "Reed" + istr, "Erosion")
            config = movepar(config, "General", "EroVol" + istr, "Erosion")

    return config


def movepar(
    config: configparser.ConfigParser,
    group1: str,
    key1: str,
    group2: str,
    key2: Optional[str] = None,
) -> configparser.ConfigParser:
    """
    Move a parameter from one group/keyword to another.

    Arguments
    ---------
    config : configparser.ConfigParser
        Original settings for the D-FAST Bank Erosion analysis.
    group1 : str
        Name of the group in the original configuration.
    key1 : str
        Name of the keyword in the original configuration.
    group2 : str
        Name of the group in the target configuration.
    key2 : Optional[str]
        Name of the keyword in the target configuration (can be None if equal to the keyword in the original file).

    Results
    -------
    config1 : configparser.ConfigParser
        Updated settings for the D-FAST Bank Erosion analysis.
    """
    if group1 in config.sections() and key1 in config[group1]:
        if key2 is None:
            key2 = key1
        config[group2][key2] = config[group1][key1]
        config[group1].pop(key1)
    return config


def write_config(filename: str, config: configparser.ConfigParser) -> None:
    """Pretty print a configParser object (configuration file) to file.

    This function ...
        aligns the equal signs for all keyword/value pairs.
        adds a two space indentation to all keyword lines.
        adds an empty line before the start of a new block.

    Arguments
    ---------
    filename : str
        Name of the configuration file to bewritten.
    config : configparser.ConfigParser
        Settings for the D-FAST Bank Erosion analysis.
    """
    sections = config.sections()
    ml = 0
    for s in sections:
        options = config.options(s)
        if len(options) > 0:
            ml = max(ml, max([len(x) for x in options]))

    OPTIONLINE = "  {{:{}s}} = {{}}\n".format(ml)
    with open(filename, "w") as configfile:
        first = True
        for s in sections:
            if first:
                first = False
            else:
                configfile.write("\n")
            configfile.write("[{}]\n".format(s))
            options = config.options(s)
            for o in options:
                configfile.write(OPTIONLINE.format(o, config[s][o]))


def config_get_xykm(config: configparser.ConfigParser):
    """

    Arguments
    ---------
    config : configparser.ConfigParser
        Settings for the D-FAST Bank Erosion analysis.

    Returns
    -------
    xykm :

    """
    # get km bounds
    kmbounds = config_get_range(config, "General", "Boundaries")
    if kmbounds[0] > kmbounds[1]:
        kmbounds = kmbounds[::-1]

    # get the chainage file
    kmfile = config_get_str(config, "General", "RiverKM")
    xykm = read_xyc(kmfile, ncol=3)

    # make sure that chainage is increasing with node index
    if xykm.coords[0][2] > xykm.coords[1][2]:
        xykm = shapely.geometry.asLineString(xykm.coords[::-1])

    # clip the chainage path to the range of chainages of interest
    xykm = clip_chainage_path(xykm, kmfile, kmbounds)

    return xykm


def clip_chainage_path(xykm, kmfile: str, kmbounds: Tuple[float, float]):
    """
    Clip a chainage line to the relevant reach.

    Arguments
    ---------
    xykm :
        Original river chainage line.
    kmfile : str
        Name of chainage file (from which xykm was obtained).
    kmbounds : Tuple(float, float)
        Lower and upper limit for the chainage.

    Returns
    -------
    xykm1 :
        Clipped river chainage line.
    """
    start_i = None
    end_i = None
    for i, c in enumerate(xykm.coords):
        if start_i is None:
            if c[2] >= kmbounds[0]:
                start_i = i
        if c[2] >= kmbounds[1]:
            end_i = i
            break

    if start_i is None:
        raise Exception(
            'Start chainage {} is larger than the maximum chainage {} listed in "{}"'.format(
                kmbounds[0], xykm.coords[-1][2], kmfile
            )
        )
    elif start_i == 0:
        # lower bound (potentially) clipped to available reach
        if xykm.coords[0][2] - kmbounds[0] > 0.1:
            raise Exception(
                'Start chainage {} is smaller than the minimum chainage {} listed in "{}"'.format(
                    kmbounds[0], xykm.coords[0][2], kmfile
                )
            )
        x0 = None
    else:
        alpha = (kmbounds[0] - xykm.coords[start_i - 1][2]) / (
            xykm.coords[start_i][2] - xykm.coords[start_i - 1][2]
        )
        x0 = tuple(
            (c1 + alpha * (c2 - c1))
            for c1, c2 in zip(xykm.coords[start_i - 1], xykm.coords[start_i])
        )
        if alpha > 0.9:
            # value close to first node (start_i), so let's skip that one
            start_i = start_i + 1

    if end_i is None:
        if kmbounds[1] - xykm.coords[-1][2] > 0.1:
            raise Exception(
                'End chainage {} is larger than the maximum chainage {} listed in "{}"'.format(
                    kmbounds[1], xykm.coords[-1][2], kmfile
                )
            )
        # else kmbounds[1] matches chainage of last point
        if x0 is None:
            # whole range available selected
            pass
        else:
            xykm = shapely.geometry.LineString([x0] + xykm.coords[start_i:])
    elif end_i == 0:
        raise Exception(
            'End chainage {} is smaller than the minimum chainage {} listed in "{}"'.format(
                kmbounds[1], xykm.coords[0][2], kmfile
            )
        )
    else:
        alpha = (kmbounds[1] - xykm.coords[end_i - 1][2]) / (
            xykm.coords[end_i][2] - xykm.coords[end_i - 1][2]
        )
        x1 = tuple(
            (c1 + alpha * (c2 - c1))
            for c1, c2 in zip(xykm.coords[end_i - 1], xykm.coords[end_i])
        )
        if alpha < 0.1:
            # value close to previous point (end_i - 1), so let's skip that one
            end_i = end_i - 1
        if x0 is None:
            xykm = shapely.geometry.LineString(xykm.coords[:end_i] + [x1])
        else:
            xykm = shapely.geometry.LineString([x0] + xykm.coords[start_i:end_i] + [x1])
    return xykm


def config_get_bank_guidelines(config: configparser.ConfigParser) -> List[numpy.ndarray]:
    """
    Get the guide lines for the bank lines from the analysis settings.

    Arguments
    ---------
    config : configparser.ConfigParser
        Settings for the D-FAST Bank Erosion analysis.

    Returns
    -------
    line : List[numpy.ndarray]
        List of arrays containing the x,y-coordinates of a bank guide lines.
    """
    # read guiding bank line
    nbank = config_get_int(config, "Detect", "NBank")
    line = [None] * nbank
    for b in range(nbank):
        bankfile = config["Detect"]["Line{}".format(b + 1)]
        line[b] = read_xyc(bankfile)
    return line


def config_get_bank_search_distances(
    config: configparser.ConfigParser, nbank: int
) -> List[float]:
    """
    Get the search distance per bank line from the analysis settings.

    Arguments
    ---------
    config : configparser.ConfigParser
        Settings for the D-FAST Bank Erosion analysis.
    nbank : int
        Number of bank (guide) lines.

    Returns
    -------
    dlines : List[float]
        Array of length nbank containing the search distance value per bank line (default value: 50).
    """
    dlines_key = config["Detect"].get("DLines", None)
    if dlines_key is None:
        dlines = [50] * nbank
    elif dlines_key[0] == "[" and dlines_key[-1] == "]":
        dlines_split = dlines_key[1:-1].split(",")
        dlines = [float(d) for d in dlines_split]
        if not all([d > 0 for d in dlines]):
            raise Exception(
                "keyword DLINES should contain positive values in configuration file."
            )
        if len(dlines) != nbank:
            raise Exception(
                "keyword DLINES should contain NBANK values in configuration file."
            )
    return dlines


def config_get_simfile(config: configparser.ConfigParser, group: str, istr: str) -> str:
    """
    Get the name of the simulation file from the analysis settings.

    Arguments
    ---------
    config : configparser.ConfigParser
        Settings for the D-FAST Bank Erosion analysis.
    group : str
        Name of the group in which to search for the simulation file name.
    istr : str
        Postfix for the simulation file name keyword; typically a string representation of the index.

    Returns
    -------
    simfile : str
        Name of the simulation file (empty string if keywords are not found).
    """
    simfile = config[group].get("Delft3Dfile" + istr, "")
    simfile = config[group].get("SDSfile" + istr, simfile)
    simfile = config[group].get("simfile" + istr, simfile)
    return simfile


def config_get_range(
    config: configparser.ConfigParser, group: str, key: str
) -> Tuple[float, float]:
    """
    Get a start and end value from a selected group and keyword in the analysis settings.

    Arguments
    ---------
    config : configparser.ConfigParser
        Settings for the D-FAST Bank Erosion analysis.
    group : str
        Name of the group from which to read.
    key : str
        Name of the keyword from which to read.

    Returns
    -------
    val : Tuple[float,float]
        Lower and upper limit of the range.
    """
    str = config_get_str(config, group, key)
    try:
        vallist = [float(fstr) for fstr in str.split(":")]
        val = (vallist[0], vallist[1])
    except:
        raise Exception(
            'No range specified for required keyword "{}" in block "{}".'.format(
                key, group
            )
        )
    return val


def config_get_bool(
    config: configparser.ConfigParser,
    group: str,
    key: str,
    default: Optional[bool] = None,
) -> bool:
    """
    Get a boolean from a selected group and keyword in the analysis settings.

    Arguments
    ---------
    config : configparser.ConfigParser
        Settings for the D-FAST Bank Erosion analysis.
    group : str
        Name of the group from which to read.
    key : str
        Name of the keyword from which to read.
    default : Optional[bool]
        Optional default value.

    Raises
    ------
    Exception
        If the keyword isn't specified and no default value is given.

    Returns
    -------
    val : bool
        Boolean value.
    """
    try:
        str = config[group][key].lower()
        val = (
            (str == "yes")
            or (str == "y")
            or (str == "true")
            or (str == "t")
            or (str == "1")
        )
    except:
        if not default is None:
            val = default
        else:
            raise Exception(
                'No boolean value specified for required keyword "{}" in block "{}".'.format(
                    key, group
                )
            )
    return val


def config_get_int(
    config: configparser.ConfigParser,
    group: str,
    key: str,
    default: Optional[int] = None,
    positive: bool = False,
) -> int:
    """
    Get an integer from a selected group and keyword in the analysis settings.

    Arguments
    ---------
    config : configparser.ConfigParser
        Settings for the D-FAST Bank Erosion analysis.
    group : str
        Name of the group from which to read.
    key : str
        Name of the keyword from which to read.
    default : Optional[int]
        Optional default value.
    positive : bool
        Flag specifying whether all integers are accepted (if False), or only strictly positive integers (if True).

    Raises
    ------
    Exception
        If the keyword isn't specified and no default value is given.
        If a negative or zero value is specified when a positive value is required.

    Returns
    -------
    val : int
        Integer value.
    """
    try:
        val = int(config[group][key])
        if positive:
            if val <= 0:
                raise Exception(
                    'Value for "{}" in block "{}" must be positive, not {}.'.format(
                        key, group, val
                    )
                )
    except:
        if not default is None:
            val = default
        else:
            raise Exception(
                'No integer value specified for required keyword "{}" in block "{}".'.format(
                    key, group
                )
            )
    return val


def config_get_float(
    config: configparser.ConfigParser,
    group: str,
    key: str,
    default: Optional[float] = None,
) -> float:
    """
    Get a floating point value from a selected group and keyword in the analysis settings.

    Arguments
    ---------
    config : configparser.ConfigParser
        Settings for the D-FAST Bank Erosion analysis.
    group : str
        Name of the group from which to read.
    key : str
        Name of the keyword from which to read.
    default : Optional[float]
        Optional default value.

    Raises
    ------
    Exception
        If the keyword isn't specified and no default value is given.

    Returns
    -------
    val : float
        Floating point value.
    """
    try:
        val = float(config[group][key])
    except:
        if not default is None:
            val = default
        else:
            raise Exception(
                'No floating point value specified for required keyword "{}" in block "{}".'.format(
                    key, group
                )
            )
    return val


def config_get_str(
    config: configparser.ConfigParser,
    group: str,
    key: str,
    default: Optional[str] = None,
) -> str:
    """
    Get a string from a selected group and keyword in the analysis settings.

    Arguments
    ---------
    config : configparser.ConfigParser
        Settings for the D-FAST Bank Erosion analysis.
    group : str
        Name of the group from which to read.
    key : str
        Name of the keyword from which to read.
    default : Optional[str]
        Optional default value.

    Raises
    ------
    Exception
        If the keyword isn't specified and no default value is given.

    Returns
    -------
    val : str
        String.
    """
    try:
        val = config[group][key]
    except:
        if not default is None:
            val = default
        else:
            raise Exception(
                'No value specified for required keyword "{}" in block "{}".'.format(
                    key, group
                )
            )
    return val


def config_get_parameter(
    config: configparser.ConfigParser,
    group: str,
    key: str,
    bank_km: List[numpy.ndarray],
    default=None,
    ext: str = "",
    positive: bool = False,
    valid: Optional[List[float]] = None,
    onefile: bool = False,
):
    """
    Get a parameter field from a selected group and keyword in the analysis settings.

    Arguments
    ---------
    config : configparser.ConfigParser
        Settings for the D-FAST Bank Erosion analysis.
    group : str
        Name of the group from which to read.
    key : str
        Name of the keyword from which to read.
    bank_km : List[numpy.ndarray]
        For each bank a listing of the bank points (bank chainage locations).
    default : Optional[Union[float, List[numpy.ndarray]]]
        Optional default value or default parameter field; default None.
    ext : str
        File name extension; default empty string.
    positive : bool
        Flag specifying whether all values are accepted (if False), or only strictly positive values (if True); default False.
    valid : Optional[List[float]]
        Optional list of valid values; default None.
    onefile : bool
        Flag indicating whether for one file should be used for all bank lines (True) or one file per bank line (False; default).

    Raises
    ------
    Exception
        If a parameter isn't provided in the configuration, but no default value provided either.
        If the value is negative while a positive value is required (positive = True).
        If the value doesn't match one of the value values (valid is not None).

    Returns
    -------
    parfield : List[numpy.ndarray]
        Parameter field: for each bank a parameter value per bank point (bank chainage location).
    """
    try:
        filename = config[group][key]
        use_default = False
    except:
        if default is None:
            raise Exception(
                'No value specified for required keyword "{}" in block "{}".'.format(
                    key, group
                )
            )
        use_default = True

    # if val is value then use that value globally
    parfield = [None] * len(bank_km)
    try:
        if use_default:
            if isinstance(default, list):
                return default
            rval = default
        else:
            rval = float(filename)
            if positive:
                if rval < 0:
                    raise Exception(
                        'Value of "{}" should be positive, not {}.'.format(key, rval)
                    )
            if not valid is None:
                if valid.count(rval) == 0:
                    raise Exception(
                        'Value of "{}" should be in {}, not {}.'.format(
                            key, valid, rval
                        )
                    )
        for ib, bkm in enumerate(bank_km):
            parfield[ib] = numpy.zeros(len(bkm)) + rval
    except:
        if onefile:
            km_thr, val = get_kmval(filename, key, positive, valid)
        for ib, bkm in enumerate(bank_km):
            if not onefile:
                filename_i = filename + "_{}".format(ib + 1) + ext
                km_thr, val = get_kmval(filename_i, key, positive, valid)
            if km_thr is None:
                parfield[ib] = numpy.zeros(len(bkm)) + val[0]
            else:
                idx = numpy.zeros(len(bkm), dtype=numpy.int64)
                for thr in km_thr:
                    idx[bkm >= thr] += 1
                parfield[ib] = val[idx]
            # print("Min/max of data: ", parfield[ib].min(), parfield[ib].max())
    return parfield


def get_kmval(filename: str, key: str, positive: bool, valid: Optional[List[float]]):
    """
    Read a parameter file, check its contents and return arrays of chainages and values.

    Arguments
    ---------
    filename : str
        Name of the parameter file to be read.
    key : str
        Name of the quantity that we're reading.
    positive : bool
        Flag specifying whether all values are accepted (if False), or only positive values (if True).
    valid : Optional[List[float]]
        Optional list of valid values.

    Raises
    ------
    Exception
        If negative values are read while values are required to be positive (positive = True).
        If some values are not valid when cross validated against the valid list (valid is not None).
        If the chainage values in the file are not strictly increasing.

    Returns
    -------
    km_thr : Optional[numpy.ndarray]
        Array containing the chainage of the midpoints between the values.
    val : numpy.ndarray
        Array containing the values.
    """
    # print("Trying to read: ",filename)
    P = pandas.read_csv(
        filename,
        names=["Chainage", "Val"],
        skipinitialspace=True,
        delim_whitespace=True,
    )
    nPnts = len(P.Chainage)
    km = P.Chainage.to_numpy()
    val = P.Val.to_numpy()
    if len(km.shape) == 0:
        km = km[None]
        val = val[None]
    if positive:
        if (val < 0).any():
            raise Exception(
                'Values of "{}" in "{}" should be positive. Negative value read for chainage(s): {}'.format(
                    key, filename, km[val < 0]
                )
            )
    # if not valid is None:
    #    isvalid = False
    #    for valid_val in valid:
    #        isvalid = isvalid | (val == valid_val)
    #    if not isvalid.all():
    #        raise Exception('Value of "{}" in "{}" should be in {}. Invalid value read for chainage(s): {}.'.format(key, filename, km[~isvalid]))
    if len(km) == 1:
        km_thr = None
    else:
        if not (km[1:] > km[:-1]).all():
            raise Exception(
                'Chainage values are not increasing in the file "{}" read for "{}".'.format(
                    filename, key
                )
            )
        km_thr = (km[:-1] + km[1:]) / 2
    return km_thr, val


def read_simdata(filename: str) -> Tuple[SimulationObject, float]:
    """
    Read a deault set of quantities from a UGRID netCDF file coming from D-Flow FM (or similar).

    Arguments
    ---------
    filename : str
        Name of the simulation output file to be read.

    Raises
    ------
    Exception
        If the file is not recognized as a D-Flow FM map-file.

    Returns
    -------
    sim : SimulationObject
        Dictionary containing the data read from the simulation output file.
    dh0 : float
        Threshold depth for detecting drying and flooding.
    """
    sim = {}
    # determine file type
    path, name = os.path.split(filename)
    if name[-6:] == "map.nc":
        sim["x_node"] = read_fm_map(filename, "x", location="node")
        sim["y_node"] = read_fm_map(filename, "y", location="node")
        FNC = read_fm_map(filename, "face_node_connectivity")
        if FNC.mask.shape == ():
            # all faces have the same number of nodes
            sim["nnodes"] = (
                numpy.ones(FNC.data.shape[0], dtype=numpy.int) * FNC.data.shape[1]
            )
        else:
            # varying number of nodes
            sim["nnodes"] = FNC.mask.shape[1] - FNC.mask.sum(axis=1)
        FNC.data[FNC.mask] = 0
        sim["facenode"] = FNC.data
        sim["zb_location"] = "node"
        sim["zb_val"] = read_fm_map(filename, "altitude", location="node")
        sim["zw_face"] = read_fm_map(filename, "Water level")
        sim["h_face"] = read_fm_map(filename, "sea_floor_depth_below_sea_surface")
        sim["ucx_face"] = read_fm_map(filename, "sea_water_x_velocity")
        sim["ucy_face"] = read_fm_map(filename, "sea_water_y_velocity")
        sim["chz_face"] = read_fm_map(filename, "Chezy roughness")

        rootgrp = netCDF4.Dataset(filename)
        try:
            filesource = rootgrp.converted_from
            if filesource == "SIMONA":
                dh0 = 0.1
            else:
                dh0 = 0.01
        except:
            dh0 = 0.01

    elif name[:3] == "SDS":
        dh0 = 0.1
        raise Exception(
            'WAQUA output files not yet supported. Unable to process "{}"'.format(name)
        )
    elif name[:4] == "trim":
        dh0 = 0.01
        raise Exception(
            'Delft3D map files not yet supported. Unable to process "{}"'.format(name)
        )
    else:
        raise Exception('Unable to determine file type for "{}"'.format(name))
    return sim, dh0


def read_fm_map(filename: str, varname: str, location: str = "face") -> numpy.ndarray:
    """
    Read the last time step of any quantity defined at faces from a D-Flow FM map-file.

    Arguments
    ---------
    filename : str
        Name of the D-Flow FM map.nc file to read the data.
    varname : str
        Name of the netCDF variable to be read.
    location : str
        Name of the stagger location at which the data should be located
        (default is "face")

    Raises
    ------
    Exception
        If the data file doesn't include a 2D mesh.
        If it cannot uniquely identify the variable to be read.

    Returns
    -------
    data
        Data of the requested variable (for the last time step only if the variable is
        time dependent).
    """
    # open file
    rootgrp = netCDF4.Dataset(filename)

    # locate 2d mesh variable
    mesh2d = rootgrp.get_variables_by_attributes(
        cf_role="mesh_topology", topology_dimension=2
    )
    if len(mesh2d) != 1:
        raise Exception(
            "Currently only one 2D mesh supported ... this file contains {} 2D meshes.".format(
                len(mesh2d)
            )
        )
    meshname = mesh2d[0].name

    # define a default start_index
    start_index = 0

    # locate the requested variable ... start with some special cases
    if varname == "x":
        # the x-coordinate or longitude
        crdnames = mesh2d[0].getncattr(location + "_coordinates").split()
        for n in crdnames:
            stdname = rootgrp.variables[n].standard_name
            if stdname == "projection_x_coordinate" or stdname == "longitude":
                var = rootgrp.variables[n]
                break

    elif varname == "y":
        # the y-coordinate or latitude
        crdnames = mesh2d[0].getncattr(location + "_coordinates").split()
        for n in crdnames:
            stdname = rootgrp.variables[n].standard_name
            if stdname == "projection_y_coordinate" or stdname == "latitude":
                var = rootgrp.variables[n]
                break

    elif varname[-12:] == "connectivity":
        # a mesh connectivity variable with corrected index
        varname = mesh2d[0].getncattr(varname)
        var = rootgrp.variables[varname]
        if "start_index" in var.ncattrs():
            start_index = var.getncattr("start_index")

    else:
        # find any other variable by standard_name or long_name
        var = rootgrp.get_variables_by_attributes(
            standard_name=varname, mesh=meshname, location=location
        )
        if len(var) == 0:
            var = rootgrp.get_variables_by_attributes(
                long_name=varname, mesh=meshname, location=location
            )
        if len(var) != 1:
            raise Exception(
                'Expected one variable for "{}", but obtained {}.'.format(
                    varname, len(var)
                )
            )
        var = var[0]

    # read data checking for time dimension
    dims = var.dimensions
    if var.get_dims()[0].isunlimited():
        # assume that time dimension is unlimited and is the first dimension
        # slice to obtain last time step
        data = var[-1, :]
    else:
        data = var[...] - start_index

    # close file
    rootgrp.close()

    # return data
    return data


def get_mesh_and_facedim_names(filename: str) -> Tuple[str, str]:
    """
    Obtain the names of 2D mesh and face dimension from netCDF UGRID file.

    Arguments
    ---------
    filename : str
        Name of the netCDF file.

    Raises
    ------
    Exception
        If there is not one mesh in the netCDF file.

    Returns
    -------
    tuple : Tuple[str, str]
        Name of the 2D mesh variable
        Name of the face dimension of that 2D mesh
    """
    # open file
    rootgrp = netCDF4.Dataset(filename)

    # locate 2d mesh variable
    mesh2d = rootgrp.get_variables_by_attributes(
        cf_role="mesh_topology", topology_dimension=2
    )
    if len(mesh2d) != 1:
        raise Exception(
            "Currently only one 2D mesh supported ... this file contains {} 2D meshes.".format(
                len(mesh2d)
            )
        )

    #
    facenodeconnect_varname = mesh2d[0].face_node_connectivity
    fnc = rootgrp.get_variables_by_attributes(name=facenodeconnect_varname)[0]

    # default
    facedim = fnc.dimensions[0]
    return mesh2d[0].name, facedim


def copy_ugrid(src: netCDF4.Dataset, meshname: str, dst: netCDF4.Dataset):
    """
    Copy UGRID mesh data from one netCDF file to another.

    Copy UGRID mesh data (mesh variable, all attributes, all variables that the
    UGRID attributes depend on) from source file to destination file.

    Arguments
    ---------
    src : UNION[str, netCDF4.Dataset]
        Name of source file, or dataset object representing the source file.
    meshname : str
        Name of the UGRID mesh to be copied from source to destination.
    dst : UNION[str, netCDF4.Dataset]
        Name of destination file, or dataset object representing the destination
        file.
    """
    # if src is string, then open the file
    if isinstance(src, str):
        src = netCDF4.Dataset(src)
        srcclose = True
    else:
        srcclose = False

    # locate source mesh
    mesh = src.variables[meshname]

    # if dst is string, then open the file
    if isinstance(dst, str):
        dst = netCDF4.Dataset(dst, "w", format="NETCDF4")
        dstclose = True
    else:
        dstclose = False

    # copy mesh variable
    copy_var(src, meshname, dst)
    atts = [
        "face_node_connectivity",
        "edge_node_connectivity",
        "edge_face_connectivity",
        "face_coordinates",
        "edge_coordinates",
        "node_coordinates",
    ]
    for att in atts:
        try:
            varlist = mesh.getncattr(att).split()
        except:
            varlist = []
        for varname in varlist:
            copy_var(src, varname, dst)

            # check if variable has bounds attribute, if so copy those as well
            var = src.variables[varname]
            atts2 = ["bounds"]
            for att2 in atts2:
                try:
                    varlist2 = var.getncattr(att2).split()
                except:
                    varlist2 = []
                for varname2 in varlist2:
                    copy_var(src, varname2, dst)

    # close files if strings where provided
    if srcclose:
        src.close()
    if dstclose:
        dst.close()


def copy_var(src: netCDF4.Dataset, varname: str, dst: netCDF4.Dataset) -> None:
    """
    Copy a single variable from one netCDF file to another.

    Copy a single netCDF variable including all attributes from source file to
    destination file. Create dimensions as necessary.

    Arguments
    ---------
    src : netCDF4.Dataset
        Dataset object representing the source file.
    varname : str
        Name of the netCDF variable to be copied from source to destination.
    dst : netCDF4.Dataset
        Dataset object representing the destination file.
    """
    # locate the
    srcvar = src.variables[varname]

    # copy dimensions
    for name in srcvar.dimensions:
        dimension = src.dimensions[name]
        if name not in dst.dimensions.keys():
            dst.createDimension(
                name, (len(dimension) if not dimension.isunlimited() else None)
            )

    # copy variable
    dstvar = dst.createVariable(varname, srcvar.datatype, srcvar.dimensions)

    # copy variable attributes all at once via dictionary
    dstvar.setncatts(srcvar.__dict__)
    dstvar[:] = srcvar[:]


def ugrid_add(
    dstfile: str,
    varname: str,
    ldata: numpy.array,
    meshname: str,
    facedim: str,
    long_name: str = "None",
    units: str = "None",
) -> None:
    """
    Add a new variable defined at faces to an existing UGRID netCDF file

    Arguments
    ---------
    dstfile : str
        Name of netCDF file to write data to.
    varname : str
        Name of netCDF variable to be written.
    ldata : numpy.array
        Linear array containing the data to be written.
    meshname : str
        Name of mesh variable in the netCDF file.
    facedim : str
        Name of the face dimension of the selected mesh.
    long_name : str
        Long descriptive name for the variable ("None" if no long name attribute
        should be written).
    units : str
        String indicating the unit ("None" if no unit attribute should be written).
    """
    # open destination file
    dst = netCDF4.Dataset(dstfile, "a")

    # check if face dimension exists
    dim = dst.dimensions[facedim]

    # add variable and write data
    var = dst.createVariable(varname, "f8", (facedim,))
    var.mesh = meshname
    var.location = "face"
    if long_name != "None":
        var.long_name = long_name
    if units != "None":
        var.units = units
    var[:] = ldata[:]

    # close destination file
    dst.close()


def read_waqua_xyz(filename: str, cols: Tuple[int, ...] = (2,)) -> numpy.ndarray:
    """
    Read data columns from a SIMONA XYZ file.

    Arguments
    ---------
    filename : str
        Name of file to be read.
    cols : Tuple[int]
        List of column numbers for which to return the data.

    Returns
    -------
    data : numpy.ndarray
        Data read from the file.
    """
    data = numpy.genfromtxt(filename, delimiter=",", skip_header=1, usecols=cols)
    return data


def write_simona_box(
    filename: str, rdata: numpy.ndarray, firstm: int, firstn: int
) -> None:
    """
    Write a SIMONA BOX file.

    Arguments
    ---------
    filename : str
        Name of the file to be written.
    rdata : numpy.ndarray
        Two-dimensional NumPy array containing the data to be written.
    firstm : int
        Firt M index to be written.
    firstn : int
        First N index to be written.
    """
    # open the data file
    boxfile = open(filename, "w")

    # get shape and prepare block header; data will be written in blocks of 10
    # N-lines
    shp = numpy.shape(rdata)
    mmax = shp[0]
    nmax = shp[1]
    boxheader = "      BOX MNMN=({m1:4d},{n1:5d},{m2:5d},{n2:5d}), VARIABLE_VAL=\n"
    nstep = 10

    # Loop over all N-blocks and write data to file
    for j in range(firstn, nmax, nstep):
        k = min(nmax, j + nstep)
        boxfile.write(boxheader.format(m1=firstm + 1, n1=j + 1, m2=mmax, n2=k))
        nvalues = (mmax - firstm) * (k - j)
        boxdata = ("   " + "{:12.3f}" * (k - j) + "\n") * (mmax - firstm)
        values = tuple(rdata[firstm:mmax, j:k].reshape(nvalues))
        boxfile.write(boxdata.format(*values))

    # close the file
    boxfile.close()


def absolute_path(rootdir: str, file: str) -> str:
    """
    Convert a relative path to an absolute path.

    Arguments
    ---------
    rootdir : str
        Any relative paths should be given relative to this location.
    file : str
        A relative or absolute location.

    Returns
    -------
    afile : str
        An absolute location.
    """
    if file == "":
        return file
    else:
        try:
            return os.path.normpath(os.path.join(rootdir, file))
        except:
            return file


def relative_path(rootdir: str, file: str) -> str:
    """
    Convert an absolute path to a relative path.

    Arguments
    ---------
    rootdir : str
        Any relative paths will be given relative to this location.
    file : str
        An absolute location.

    Returns
    -------
    rfile : str
        An absolute or relative location (relative only if it's on the same drive as rootdir).
    """
    if file == "":
        return file
    else:
        try:
            rfile = os.path.relpath(file, rootdir)
            return rfile
        except:
            return file
