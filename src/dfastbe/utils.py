from typing import Tuple
import time
import os
import configparser
from dfastbe import io


def timed_logger(label: str) -> None:
    """
    Write a message with time information.

    Arguments
    ---------
    label : str
        Message string.
    """
    time, diff = _timer()
    print(time + diff + label)


def _timer() -> Tuple[str, str]:
    """
    Return text string representation of time since previous call.

    The routine uses the global variable LAST_TIME to store the time of the
    previous call.

    Arguments
    ---------
    None

    Returns
    -------
    time_str : str
        String representing duration since first call.
    diff_str : str
        String representing duration since previous call.
    """
    global FIRST_TIME
    global LAST_TIME
    new_time = time.time()
    if "LAST_TIME" in globals():
        time_str = "{:6.2f} ".format(new_time - FIRST_TIME)
        diff_str = "{:6.2f} ".format(new_time - LAST_TIME)
    else:
        time_str = "   0.00"
        diff_str = "       "
        FIRST_TIME = new_time
    LAST_TIME = new_time
    return time_str, diff_str


def config_to_absolute_paths(
        rootdir: str, config: configparser.ConfigParser
) -> configparser.ConfigParser:
    """
    Convert a configuration object to contain absolute paths (for editing).

    Arguments
    ---------
    rootdir : str
        The path to be used as base for the absolute paths.
    config : configparser.ConfigParser
        Configuration for the D-FAST Bank Erosion analysis with absolute or relative paths.

    Returns
    -------
    config1 : configparser.ConfigParser
        Configuration for the D-FAST Bank Erosion analysis with only absolute paths.
    """
    if "General" in config:
        config = parameter_absolute_path(config, "General", "RiverKM", rootdir)
        config = parameter_absolute_path(config, "General", "BankDir", rootdir)
        config = parameter_absolute_path(config, "General", "FigureDir", rootdir)

    if "Detect" in config:
        config = parameter_absolute_path(config, "Detect", "SimFile", rootdir)
        i = 0
        while True:
            i = i + 1
            Line = "Line" + str(i)
            if Line in config["Detect"]:
                config = parameter_absolute_path(config, "Detect", Line, rootdir)
            else:
                break

    if "Erosion" in config:
        config = parameter_absolute_path(config, "Erosion", "RiverAxis", rootdir)
        config = parameter_absolute_path(config, "Erosion", "Fairway", rootdir)
        config = parameter_absolute_path(config, "Erosion", "OutputDir", rootdir)

        config = parameter_absolute_path(config, "Erosion", "ShipType", rootdir)
        config = parameter_absolute_path(config, "Erosion", "VShip", rootdir)
        config = parameter_absolute_path(config, "Erosion", "NShip", rootdir)
        config = parameter_absolute_path(config, "Erosion", "NWave", rootdir)
        config = parameter_absolute_path(config, "Erosion", "Draught", rootdir)
        config = parameter_absolute_path(config, "Erosion", "Wave0", rootdir)
        config = parameter_absolute_path(config, "Erosion", "Wave1", rootdir)

        config = parameter_absolute_path(config, "Erosion", "BankType", rootdir)
        config = parameter_absolute_path(config, "Erosion", "ProtectionLevel", rootdir)
        config = parameter_absolute_path(config, "Erosion", "Slope", rootdir)
        config = parameter_absolute_path(config, "Erosion", "Reed", rootdir)

        NLevel = io.config_get_int(config, "Erosion", "NLevel", default=0)
        for i in range(NLevel):
            istr = str(i + 1)
            config = parameter_absolute_path(
                config, "Erosion", "SimFile" + istr, rootdir
            )
            config = parameter_absolute_path(
                config, "Erosion", "ShipType" + istr, rootdir
            )
            config = parameter_absolute_path(config, "Erosion", "VShip" + istr, rootdir)
            config = parameter_absolute_path(config, "Erosion", "NShip" + istr, rootdir)
            config = parameter_absolute_path(config, "Erosion", "NWave" + istr, rootdir)
            config = parameter_absolute_path(
                config, "Erosion", "Draught" + istr, rootdir
            )
            config = parameter_absolute_path(config, "Erosion", "Slope" + istr, rootdir)
            config = parameter_absolute_path(config, "Erosion", "Reed" + istr, rootdir)

    return config


def config_to_relative_paths(
        rootdir: str, config: configparser.ConfigParser
) -> configparser.ConfigParser:
    """
    Convert a configuration object to contain relative paths (for saving).

    Arguments
    ---------
    rootdir : str
        The path to be used as base for the relative paths.
    config : configparser.ConfigParser
        Configuration for the D-FAST Bank Erosion analysis with only absolute paths.

    Returns
    -------
    config1 : configparser.ConfigParser
        Updated configuration for D-FAST Bank Erosion analysis with as much as possible relative paths.
    """
    if "General" in config:
        config = parameter_relative_path(config, "General", "RiverKM", rootdir)
        config = parameter_relative_path(config, "General", "BankDir", rootdir)
        config = parameter_relative_path(config, "General", "FigureDir", rootdir)

    if "Detect" in config:
        config = parameter_relative_path(config, "Detect", "SimFile", rootdir)
        i = 0
        while True:
            i = i + 1
            Line = "Line" + str(i)
            if Line in config["Detect"]:
                config = parameter_relative_path(config, "Detect", Line, rootdir)
            else:
                break

    if "Erosion" in config:
        config = parameter_relative_path(config, "Erosion", "RiverAxis", rootdir)
        config = parameter_relative_path(config, "Erosion", "Fairway", rootdir)
        config = parameter_relative_path(config, "Erosion", "OutputDir", rootdir)

        config = parameter_relative_path(config, "Erosion", "ShipType", rootdir)
        config = parameter_relative_path(config, "Erosion", "VShip", rootdir)
        config = parameter_relative_path(config, "Erosion", "NShip", rootdir)
        config = parameter_relative_path(config, "Erosion", "NWave", rootdir)
        config = parameter_relative_path(config, "Erosion", "Draught", rootdir)
        config = parameter_relative_path(config, "Erosion", "Wave0", rootdir)
        config = parameter_relative_path(config, "Erosion", "Wave1", rootdir)

        config = parameter_relative_path(config, "Erosion", "BankType", rootdir)
        config = parameter_relative_path(config, "Erosion", "ProtectionLevel", rootdir)
        config = parameter_relative_path(config, "Erosion", "Slope", rootdir)
        config = parameter_relative_path(config, "Erosion", "Reed", rootdir)

        NLevel = io.config_get_int(config, "Erosion", "NLevel", default=0)
        for i in range(NLevel):
            istr = str(i + 1)
            config = parameter_relative_path(
                config, "Erosion", "SimFile" + istr, rootdir
            )
            config = parameter_relative_path(
                config, "Erosion", "ShipType" + istr, rootdir
            )
            config = parameter_relative_path(config, "Erosion", "VShip" + istr, rootdir)
            config = parameter_relative_path(config, "Erosion", "NShip" + istr, rootdir)
            config = parameter_relative_path(config, "Erosion", "NWave" + istr, rootdir)
            config = parameter_relative_path(
                config, "Erosion", "Draught" + istr, rootdir
            )
            config = parameter_relative_path(config, "Erosion", "Slope" + istr, rootdir)
            config = parameter_relative_path(config, "Erosion", "Reed" + istr, rootdir)

    return config

def parameter_absolute_path(
        config: configparser.ConfigParser, group: str, key: str, rootdir: str
) -> configparser.ConfigParser:
    """
    Convert a parameter value to contain an absolute path.

    Determine whether the string represents a number.
    If not, try to convert to an absolute path.

    Arguments
    ---------
    config : configparser.ConfigParser
        Configuration for the D-FAST Bank Erosion analysis.
    group : str
        Name of the group in the configuration.
    key : str
        Name of the key in the configuration.
    rootdir : str
        The path to be used as base for the absolute paths.

    Returns
    -------
    config1 : configparser.ConfigParser
        Updated configuration for the D-FAST Bank Erosion analysis.
    """
    if key in config[group]:
        valstr = config[group][key]
        try:
            val = float(valstr)
        except:
            config[group][key] = io.absolute_path(rootdir, valstr)
    return config


def parameter_relative_path(
        config: configparser.ConfigParser, group: str, key: str, rootdir: str
) -> configparser.ConfigParser:
    """
    Convert a parameter value to contain a relative path.

    Determine whether the string represents a number.
    If not, try to convert to a relative path.

    Arguments
    ---------
    config : configparser.ConfigParser
        Configuration for the D-FAST Bank Erosion analysis.
    group : str
        Name of the group in the configuration.
    key : str
        Name of the key in the configuration.
    rootdir : str
        The path to be used as base for the relative paths.

    Returns
    -------
    config1 : configparser.ConfigParser
        Updated configuration for the D-FAST Bank Erosion analysis.
    """
    if key in config[group]:
        valstr = config[group][key]
        try:
            val = float(valstr)
        except:
            config[group][key] = io.relative_path(rootdir, valstr)
    return config
