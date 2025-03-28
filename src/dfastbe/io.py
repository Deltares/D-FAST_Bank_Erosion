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
from typing import Tuple, Any, List, Union, Dict, Optional, TextIO, Callable, TypedDict
import numpy as np

import netCDF4
import configparser
import os
import os.path
import pandas
import geopandas
import shapely
from shapely.geometry import Point
import pathlib

MAX_RIVER_WIDTH = 1000


class SimulationObject(TypedDict):
    x_node: np.ndarray
    y_node: np.ndarray
    nnodes: np.ndarray
    facenode: np.ma.masked_array
    zb_location: np.ndarray
    zb_val: np.ndarray
    zw_face: np.ndarray
    h_face: np.ndarray
    ucx_face: np.ndarray
    ucy_face: np.ndarray
    chz_face: np.ndarray


PROGTEXTS: Dict[str, List[str]]


class ConfigFile:

    def __init__(self, config:configparser.ConfigParser, path: Union[Path, str] = None):
        """

        Args:
            config : configparser.ConfigParser
                Settings for the D-FAST Bank Erosion analysis.
            path : str
                Name of configuration file to be read.
        """
        self._config = config
        # the directory where the configuration file is located
        if path:
            self.path = path
            self.root_dir = Path(path).parent
            self.adjust_filenames()

    @property
    def config(self) -> configparser.ConfigParser:
        return self._config

    @config.setter
    def config(self, value:configparser.ConfigParser):
        self._config = value

    @property
    def version(self) -> str:
        return self.get_str("General", "Version")

    @property
    def root_dir(self):
        return self._root_dir

    @root_dir.setter
    def root_dir(self, value: str):
        self._root_dir = value

    @classmethod
    def read(cls, path: Union[str, pathlib.Path]):
        """
        Read a configParser object (configuration file).

        reads the config file using the standard configParser.
        falls back to a dedicated reader compatible with old waqbank files.

        Returns
        -------
        config : configparser.ConfigParser
            Settings for the D-FAST Bank Erosion analysis.
        """
        try:
            config = configparser.ConfigParser(comment_prefixes="%")
            with open(path, "r") as configfile:
                config.read_file(configfile)
        except Exception as e:
            print(f"Error during reading the config file: {e}")
            config = configparser.ConfigParser()
            config["General"] = {}
            all_lines = open(path, "r").read().splitlines()
            for line in all_lines:
                perc = line.find("%")
                if perc >= 0:
                    line = line[:perc]
                data = line.split()
                if len(data) >= 3:
                    config["General"][data[0]] = data[2]

        # if version != "1.0":
        config = cls._upgrade(config)
        return cls(config, path=path)

    @staticmethod
    def _upgrade(config: configparser.ConfigParser):
        """
        Upgrade the configuration data structure to version 1.0 format.

        Results
        -------
        config1 : configparser.ConfigParser
            Settings for the D-FAST Bank Erosion analysis in 1.0 format.

        """
        try:
            version = config["General"]["Version"]
        except KeyError:
            version = "0.1"

        if version == "0.1":
            config["General"]["Version"] = "1.0"

            config["Detect"] = {}
            config = move_parameter_location(
                config, "General", "Delft3Dfile", "Detect", "SimFile", convert=sim2nc
            )
            config = move_parameter_location(config, "General", "SDSfile", "Detect", "SimFile", convert=sim2nc)
            config = move_parameter_location(config, "General", "SimFile", "Detect")
            config = move_parameter_location(config, "General", "NBank", "Detect")
            config_file = ConfigFile(config)
            n_bank = config_file.get_int("Detect", "NBank", default=0, positive=True)
            for i in range(1, n_bank + 1):
                config = move_parameter_location(config, "General", f"Line{i}", "Detect")

            config = move_parameter_location(config, "General", "WaterDepth", "Detect")
            config = move_parameter_location(config, "General", "DLines", "Detect")

            config["Erosion"] = {}
            config = move_parameter_location(config, "General", "TErosion", "Erosion")
            config = move_parameter_location(config, "General", "RiverAxis", "Erosion")
            config = move_parameter_location(config, "General", "Fairway", "Erosion")
            config = move_parameter_location(config, "General", "RefLevel", "Erosion")
            config = move_parameter_location(config, "General", "OutputInterval", "Erosion")
            config = move_parameter_location(config, "General", "OutputDir", "Erosion")
            config = move_parameter_location(config, "General", "BankNew", "Erosion")
            config = move_parameter_location(config, "General", "BankEq", "Erosion")
            config = move_parameter_location(config, "General", "EroVol", "Erosion")
            config = move_parameter_location(config, "General", "EroVolEqui", "Erosion")
            config = move_parameter_location(config, "General", "NLevel", "Erosion")
            config_file = ConfigFile(config)
            n_level = config_file.get_int("Erosion", "NLevel", default=0, positive=True)

            for i in range(1, n_level + 1):
                config = move_parameter_location(
                    config,"General", f"Delft3Dfile{i}", "Erosion", f"SimFile{i}", convert=sim2nc,
                    )
                config = move_parameter_location(
                    config, "General", f"SDSfile{i}", "Erosion", f"SimFile{i}", convert=sim2nc,
                    )
                config = move_parameter_location(config, "General", f"SimFile{i}", "Erosion")
                config = move_parameter_location(config, "General", f"PDischarge{i}", "Erosion")

            config = move_parameter_location(config, "General", "ShipType", "Erosion")
            config = move_parameter_location(config, "General", "VShip", "Erosion")
            config = move_parameter_location(config, "General", "NShip", "Erosion")
            config = move_parameter_location(config, "General", "NWave", "Erosion")
            config = move_parameter_location(config, "General", "Draught", "Erosion")
            config = move_parameter_location(config, "General", "Wave0", "Erosion")
            config = move_parameter_location(config, "General", "Wave1", "Erosion")

            config = move_parameter_location(config, "General", "Classes", "Erosion")
            config = move_parameter_location(config, "General", "BankType", "Erosion")
            config = move_parameter_location(config, "General", "ProtectLevel", "Erosion", "ProtectionLevel")
            config = move_parameter_location(config, "General", "Slope", "Erosion")
            config = move_parameter_location(config, "General", "Reed", "Erosion")
            config = move_parameter_location(config, "General", "VelFilter", "Erosion")

            for i in range(1, n_level + 1):
                config = move_parameter_location(config, "General", f"ShipType{i}", "Erosion")
                config = move_parameter_location(config, "General", f"VShip{i}", "Erosion")
                config = move_parameter_location(config, "General", f"NShip{i}", "Erosion")
                config = move_parameter_location(config, "General", f"NWave{i}", "Erosion")
                config = move_parameter_location(config, "General", f"Draught{i}", "Erosion")
                config = move_parameter_location(config, "General", f"Slope{i}", "Erosion")
                config = move_parameter_location(config, "General", f"Reed{i}", "Erosion")
                config = move_parameter_location(config, "General", f"EroVol{i}", "Erosion")

        return config

    def write(self, filename: str) -> None:
        """Pretty print a configParser object (configuration file) to file.

        This function ...
            aligns the equal signs for all keyword/value pairs.
            adds a two space indentation to all keyword lines.
            adds an empty line before the start of a new block.

        Arguments
        ---------
        filename : str
            Name of the configuration file to be written.
        config : configparser.ConfigParser
            The variable containing the configuration.
        """
        config = self.config
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

    def adjust_filenames(self) -> Tuple[str, configparser.ConfigParser]:
        """
        Convert all paths to relative to current working directory.

        Returns
        -------
        rootdir : str
            Location of configuration file relative to current working directory.
        config : configparser.ConfigParser
            Analysis configuration settings using paths relative to current working directory.
        """
        # rootdir = os.path.dirname(self.path)
        cwd = os.getcwd()
        self.resolve(self.root_dir)
        self.relative_to(cwd)
        rootdir = relative_path(cwd, self.root_dir)

        return rootdir

    def get_str(
        self,
        group: str,
        key: str,
        default: Optional[str] = None,
    ) -> str:
        """
        Get a string from a selected group and keyword in the analysis settings.

        Arguments
        ---------
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
            val = self.config[group][key]
        except KeyError:
            if default is not None:
                val = default
            else:
                raise ConfigFileError(f"No value specified for required keyword {key} in block {group}.")
        return val

    def get_bool(
        self,
        group: str,
        key: str,
        default: Optional[bool] = None,
    ) -> bool:
        """
        Get a boolean from a selected group and keyword in the analysis settings.

        Arguments
        ---------
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
            str_val = self.config[group][key].lower()
            val = (
                    (str_val == "yes")
                    or (str_val == "y")
                    or (str_val == "true")
                    or (str_val == "t")
                    or (str_val == "1")
            )
        except:
            if default is not None:
                val = default
            else:
                raise ConfigFileError(f"No boolean value specified for required keyword {key} in block {group}.")

        return val

    def get_float(
        self,
        group: str,
        key: str,
        default: Optional[float] = None,
        positive: bool = False,
    ) -> float:
        """
        Get a floating point value from a selected group and keyword in the analysis settings.

        Arguments
        ---------
        group : str
            Name of the group from which to read.
        key : str
            Name of the keyword from which to read.
        default : Optional[float]
            Optional default value.
        positive : bool
            Flag specifying whether all floats are accepted (if False), or only positive floats (if True).

        Raises
        ------
        Exception
            If the keyword isn't specified and no default value is given.
            If a negative value is specified when a positive value is required.

        Returns
        -------
        val : float
            Floating point value.
        """
        try:
            val = float(self.config[group][key])
        except (KeyError, ValueError):
            if default is not None:
                val = default
            else:
                raise ConfigFileError(
                    f"No floating point value specified for required keyword {key} in block {group}."
                )
        if positive and val < 0.0:
            raise ConfigFileError(
                f"Value for {key} in block {group} must be positive, not {val}."
            )
        return val

    def get_int(
        self,
        group: str,
        key: str,
        default: Optional[int] = None,
        positive: bool = False,
    ) -> int:
        """
        Get an integer from a selected group and keyword in the analysis settings.

        Arguments
        ---------
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
            val = int(self.config[group][key])
        except (KeyError, ValueError):
            if default is not None:
                val = default
            else:
                raise ConfigFileError(
                    f"No integer value specified for required keyword {key} in block {group}."
                )
        if positive and val <= 0:
            raise ConfigFileError(
                f"Value for {key} in block {group} must be positive, not {val}."
            )
        return val

    def get_sim_file(self, group: str, istr: str) -> str:
        """
        Get the name of the simulation file from the analysis settings.

        Arguments
        ---------
        group : str
            Name of the group in which to search for the simulation file name.
        istr : str
            Postfix for the simulation file name keyword; typically a string representation of the index.

        Returns
        -------
        simfile : str
            Name of the simulation file (empty string if keywords are not found).
        """
        sim_file = self.config[group].get("SimFile" + istr, "")
        return sim_file

    def get_km_bounds(self) -> Tuple[float, float]:
        """

        Returns:
            km_bounds : Tuple[float, float]
                Lower and upper limit for the chainage.
        """
        km_bounds = self.get_range("General", "Boundaries")

        return km_bounds

    def get_search_lines(self) -> List[shapely.geometry.linestring.LineStringAdapter]:
        """
        Get the search lines for the bank lines from the analysis settings.

        Returns
        -------
        line : List[np.ndarray]
            List of arrays containing the x,y-coordinates of a bank search lines.
        """
        # read guiding bank line
        n_bank = self.get_int("Detect", "NBank")
        line = [None] * n_bank
        for b in range(n_bank):
            bankfile = self.config["Detect"][f"Line{b + 1}"]
            log_text("read_search_line", dict={"nr": b + 1, "file": bankfile})
            line[b] = read_xyc(bankfile)
        return line

    def get_bank_lines(self, bank_dir: str) -> List[np.ndarray]:
        """
        Get the bank lines from the detection step.

        Arguments
        ---------
        bank_dir : str
            Name of directory in which the bank lines files are located.

        Returns
        -------
        line : List[np.ndarray]
            List of arrays containing the x,y-coordinates of a bank lines.
        """
        bank_name = self.get_str("General", "BankFile", "bankfile")
        bankfile = f"{bank_dir}{os.sep}{bank_name}.shp"
        if os.path.exists(bankfile):
            log_text("read_banklines", dict={"file": bankfile})
            banklines = geopandas.read_file(bankfile)
        else:
            bankfile = bank_dir + os.sep + bank_name + "_#.xyc"
            log_text("read_banklines", dict={"file": bankfile})
            bankline_list = []
            b = 1
            while True:
                bankfile = bank_dir + os.sep + bank_name + "_" + str(b) + ".xyc"
                if os.path.exists(bankfile):
                    xy_bank = read_xyc(bankfile)
                    bankline_list.append(shapely.geometry.LineString(xy_bank))
                    b = b + 1
                else:
                    break
            bankline_series = geopandas.geoseries.GeoSeries(bankline_list)
            banklines = geopandas.geodataframe.GeoDataFrame.from_features(bankline_series)
        return banklines

    def get_parameter(
        self,
        group: str,
        key: str,
        bank_km: List[np.ndarray],
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
        group : str
            Name of the group from which to read.
        key : str
            Name of the keyword from which to read.
        bank_km : List[np.ndarray]
            For each bank a listing of the bank points (bank chainage locations).
        default : Optional[Union[float, List[np.ndarray]]]
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
        parfield : List[np.ndarray]
            Parameter field: for each bank a parameter value per bank point (bank chainage location).
        """
        try:
            filename = self.config[group][key]
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
                if positive and rval < 0:
                    raise Exception(
                        'Value of "{}" should be positive, not {}.'.format(key, rval)
                    )
                if valid is not None and valid.count(rval) == 0:
                    raise Exception(
                        'Value of "{}" should be in {}, not {}.'.format(
                            key, valid, rval
                        )
                    )
            for ib, bkm in enumerate(bank_km):
                parfield[ib] = np.zeros(len(bkm)) + rval
        except:
            if onefile:
                log_text("read_param", dict={"param": key, "file": filename})
                km_thr, val = get_kmval(filename, key, positive, valid)
            for ib, bkm in enumerate(bank_km):
                if not onefile:
                    filename_i = filename + "_{}".format(ib + 1) + ext
                    log_text(
                        "read_param_one_bank",
                        dict={"param": key, "i": ib + 1, "file": filename_i},
                    )
                    km_thr, val = get_kmval(filename_i, key, positive, valid)
                if km_thr is None:
                    parfield[ib] = np.zeros(len(bkm)) + val[0]
                else:
                    idx = np.zeros(len(bkm), dtype=np.int64)
                    for thr in km_thr:
                        idx[bkm >= thr] += 1
                    parfield[ib] = val[idx]
                # print("Min/max of data: ", parfield[ib].min(), parfield[ib].max())
        return parfield

    def get_bank_search_distances(self, nbank: int) -> List[float]:
        """
        Get the search distance per bank line from the analysis settings.

        Arguments
        ---------
        nbank : int
            Number of bank search lines.

        Returns
        -------
        dlines : List[float]
            Array of length nbank containing the search distance value per bank line (default value: 50).
        """
        dlines_key = self.config["Detect"].get("DLines", None)
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

    def get_range(self, group: str, key: str) -> Tuple[float, float]:
        """
        Get a start and end value from a selected group and keyword in the analysis settings.

        Arguments
        ---------
        group : str
            Name of the group from which to read.
        key : str
            Name of the keyword from which to read.

        Returns
        -------
        val : Tuple[float,float]
            Lower and upper limit of the range.
        """
        str_val = self.get_str(group, key)
        try:
            obrack = str_val.find("[")
            cbrack = str_val.find("]")
            if obrack >= 0 and cbrack >= 0:
                str_val = str_val[obrack + 1: cbrack - 1]
            val_list = [float(fstr) for fstr in str_val.split(":")]
            if val_list[0] > val_list[1]:
                val = (val_list[1], val_list[0])
            else:
                val = (val_list[0], val_list[1])
        except:
            raise Exception(
                'Invalid range specification "{}" for required keyword "{}" in block "{}".'.format(
                    str_val, key, group
                )
            )
        return val

    def get_xy_km(self) -> shapely.geometry.linestring.LineStringAdapter:
        """

        Returns
        -------
        xykm : shapely.geometry.linestring.LineStringAdapter

        """
        # get the chainage file
        km_file = self.get_str("General", "RiverKM")
        log_text("read_chainage", dict={"file": km_file})
        xy_km = read_xyc(km_file, num_columns=3)

        # make sure that chainage is increasing with node index
        if xy_km.coords[0][2] > xy_km.coords[1][2]:
            xy_km = shapely.geometry.asLineString(xy_km.coords[::-1])

        return xy_km

    def resolve(self, rootdir: str):
        """
        Convert a configuration object to contain absolute paths (for editing).

        Arguments
        ---------
        rootdir : str
            The path to be used as base for the absolute paths.
        """
        if "General" in self.config:
            self.resolve_parameter("General", "RiverKM", rootdir)
            self.resolve_parameter("General", "BankDir", rootdir)
            self.resolve_parameter("General", "FigureDir", rootdir)

        if "Detect" in self.config:
            self.resolve_parameter("Detect", "SimFile", rootdir)
            i = 0
            while True:
                i = i + 1
                line_i = "Line" + str(i)
                if line_i in self.config["Detect"]:
                    self.resolve_parameter("Detect", line_i, rootdir)
                else:
                    break

        if "Erosion" in self.config:
            self.resolve_parameter("Erosion", "RiverAxis", rootdir)
            self.resolve_parameter("Erosion", "Fairway", rootdir)
            self.resolve_parameter("Erosion", "OutputDir", rootdir)

            self.resolve_parameter("Erosion", "ShipType", rootdir)
            self.resolve_parameter("Erosion", "VShip", rootdir)
            self.resolve_parameter("Erosion", "NShip", rootdir)
            self.resolve_parameter("Erosion", "NWave", rootdir)
            self.resolve_parameter("Erosion", "Draught", rootdir)
            self.resolve_parameter("Erosion", "Wave0", rootdir)
            self.resolve_parameter("Erosion", "Wave1", rootdir)

            self.resolve_parameter("Erosion", "BankType", rootdir)
            self.resolve_parameter("Erosion", "ProtectionLevel", rootdir)
            self.resolve_parameter("Erosion", "Slope", rootdir)
            self.resolve_parameter("Erosion", "Reed", rootdir)

            n_level = self.get_int("Erosion", "NLevel", default=0)
            for i in range(1, n_level + 1):
                self.resolve_parameter("Erosion", f"SimFile{i}", rootdir)
                self.resolve_parameter("Erosion", f"ShipType{i}", rootdir)
                self.resolve_parameter("Erosion", f"VShip{i}", rootdir)
                self.resolve_parameter("Erosion", f"NShip{i}", rootdir)
                self.resolve_parameter("Erosion", f"NWave{i}", rootdir)
                self.resolve_parameter("Erosion", f"Draught{i}", rootdir)
                self.resolve_parameter("Erosion", f"Slope{i}", rootdir)
                self.resolve_parameter("Erosion", f"Reed{i}", rootdir)


    def relative_to(self, rootdir: str):
        """
        Convert a configuration object to contain relative paths (for saving).

        Args:
            rootdir : str
                The path to be used as base for the relative paths.
        """
        if "General" in self.config:
            self.parameter_relative_to("General", "RiverKM", rootdir)
            self.parameter_relative_to("General", "BankDir", rootdir)
            self.parameter_relative_to("General", "FigureDir", rootdir)

        if "Detect" in self.config:
            self.parameter_relative_to("Detect", "SimFile", rootdir)

            i = 0
            while True:
                i = i + 1
                line_i = f"Line{i}"
                if line_i in self.config["Detect"]:
                    self.parameter_relative_to("Detect", line_i, rootdir)
                else:
                    break

        if "Erosion" in self.config:
            self.parameter_relative_to("Erosion", "RiverAxis", rootdir)
            self.parameter_relative_to("Erosion", "Fairway", rootdir)
            self.parameter_relative_to("Erosion", "OutputDir", rootdir)

            self.parameter_relative_to("Erosion", "ShipType", rootdir)
            self.parameter_relative_to("Erosion", "VShip", rootdir)
            self.parameter_relative_to("Erosion", "NShip", rootdir)
            self.parameter_relative_to("Erosion", "NWave", rootdir)
            self.parameter_relative_to("Erosion", "Draught", rootdir)
            self.parameter_relative_to("Erosion", "Wave0", rootdir)
            self.parameter_relative_to("Erosion", "Wave1", rootdir)

            self.parameter_relative_to("Erosion", "BankType", rootdir)
            self.parameter_relative_to("Erosion", "ProtectionLevel", rootdir)
            self.parameter_relative_to("Erosion", "Slope", rootdir)
            self.parameter_relative_to("Erosion", "Reed", rootdir)

            n_level = self.get_int("Erosion", "NLevel", default=0)
            for i in range(1, n_level + 1):
                self.parameter_relative_to("Erosion", f"SimFile{i}", rootdir)
                self.parameter_relative_to("Erosion", f"ShipType{i}", rootdir)
                self.parameter_relative_to("Erosion", f"VShip{i}", rootdir)
                self.parameter_relative_to("Erosion", f"NShip{i}", rootdir)
                self.parameter_relative_to("Erosion", f"NWave{i}", rootdir)
                self.parameter_relative_to("Erosion", f"Draught{i}", rootdir)
                self.parameter_relative_to("Erosion", f"Slope{i}", rootdir)
                self.parameter_relative_to("Erosion", f"Reed{i}", rootdir)


    def resolve_parameter(self, group: str, key: str, rootdir: str):
        """
        Convert a parameter value to contain an absolute path.

        Determine whether the string represents a number.
        If not, try to convert to an absolute path.

        Arguments
        ---------
        group : str
            Name of the group in the configuration.
        key : str
            Name of the key in the configuration.
        rootdir : str
            The path to be used as base for the absolute paths.
        """
        if key in self.config[group]:
            val_str = self.config[group][key]
            try:
                float(val_str)
            except ValueError:
                self.config[group][key] = absolute_path(rootdir, val_str)


    def parameter_relative_to(self, group: str, key: str, rootdir: str):
        """
        Convert a parameter value to contain a relative path.

        Determine whether the string represents a number.
        If not, try to convert to a relative path.

        Arguments
        ---------
        group : str
            Name of the group in the configuration.
        key : str
            Name of the key in the configuration.
        rootdir : str
            The path to be used as base for the relative paths.
        """
        if key in self.config[group]:
            val_str = self.config[group][key]

            try:
                float(val_str)
            except ValueError:
                self.config[group][key] = relative_path(rootdir, val_str)


class RiverData:

    def __init__(self, config_file: ConfigFile):
        self.config_file = config_file
        self.profile: shapely.geometry.linestring.LineString = config_file.get_xy_km()
        self.station_bounds: Tuple = config_file.get_km_bounds() # start and end station
        self.start_station: float = self.station_bounds[0]
        self.end_station: float = self.station_bounds[1]
        log_text("clip_chainage", dict={"low": self.start_station, "high": self.end_station})
        self.masked_profile: shapely.geometry.linestring.LineString = self.mask_profile(self.station_bounds)
        self.masked_profile_arr = np.array(self.masked_profile)


    @property
    def bank_search_lines(self) -> List[shapely.geometry.linestring.LineStringAdapter]:
        """
        Get the bank search lines.

        Returns
        -------
        search_lines : List[shapely.geometry.linestring.LineStringAdapter]
            List of bank search lines.
        """
        return self.config_file.get_search_lines()

    @property
    def num_search_lines(self) -> int:
        return len(self.bank_search_lines)

    def mask_profile(self, bounds: Tuple[float, float]) -> shapely.geometry.linestring.LineStringAdapter:
        """
        Clip a chainage line to the relevant reach.

        Arguments
        ---------
        bounds : Tuple[float, float]
            Lower and upper limit for the chainage.

        Returns
        -------
        xykm1 : shapely.geometry.linestring.LineStringAdapter
            Clipped river chainage line.
        """
        xy_km = self.profile
        start_i = None
        end_i = None
        for i, c in enumerate(xy_km.coords):
            if start_i is None:
                if c[2] >= bounds[0]:
                    start_i = i
            if c[2] >= bounds[1]:
                end_i = i
                break

        if start_i is None:
            raise Exception(
                "Lower chainage bound {} is larger than the maximum chainage {} available".format(
                    bounds[0], xy_km.coords[-1][2]
                )
            )
        elif start_i == 0:
            # lower bound (potentially) clipped to available reach
            if xy_km.coords[0][2] - bounds[0] > 0.1:
                raise Exception(
                    "Lower chainage bound {} is smaller than the minimum chainage {} available".format(
                        bounds[0], xy_km.coords[0][2]
                    )
                )
            x0 = None
        else:
            alpha = (bounds[0] - xy_km.coords[start_i - 1][2]) / (
                    xy_km.coords[start_i][2] - xy_km.coords[start_i - 1][2]
            )
            x0 = tuple(
                (c1 + alpha * (c2 - c1))
                for c1, c2 in zip(xy_km.coords[start_i - 1], xy_km.coords[start_i])
            )
            if alpha > 0.9:
                # value close to the first node (start_i), so let's skip that one
                start_i = start_i + 1

        if end_i is None:
            if bounds[1] - xy_km.coords[-1][2] > 0.1:
                raise Exception(
                    "Upper chainage bound {} is larger than the maximum chainage {} available".format(
                        bounds[1], xy_km.coords[-1][2]
                    )
                )
            # else kmbounds[1] matches chainage of last point
            if x0 is None:
                # whole range available selected
                pass
            else:
                xy_km = shapely.geometry.LineString([x0] + xy_km.coords[start_i:])
        elif end_i == 0:
            raise Exception(
                "Upper chainage bound {} is smaller than the minimum chainage {} available".format(
                    bounds[1], xy_km.coords[0][2]
                )
            )
        else:
            alpha = (bounds[1] - xy_km.coords[end_i - 1][2]) / (
                    xy_km.coords[end_i][2] - xy_km.coords[end_i - 1][2]
            )
            x1 = tuple(
                (c1 + alpha * (c2 - c1))
                for c1, c2 in zip(xy_km.coords[end_i - 1], xy_km.coords[end_i])
            )
            if alpha < 0.1:
                # value close to the previous point (end_i - 1), so let's skip that one
                end_i = end_i - 1
            if x0 is None:
                xy_km = shapely.geometry.LineString(xy_km.coords[:end_i] + [x1])
            else:
                xy_km = shapely.geometry.LineString([x0] + xy_km.coords[start_i:end_i] + [x1])
        return xy_km

    def clip_search_lines(
        self,
        max_river_width: float = MAX_RIVER_WIDTH,
    ) -> Tuple[List[shapely.geometry.linestring.LineStringAdapter], float]:
        """
        Clip the list of lines to the envelope of certain size surrounding a reference line.

        Arg:
            search_lines : List[shapely.geometry.linestring.LineStringAdapter]
                List of search lines to be clipped.
            river_profile : shapely.geometry.linestring.LineStringAdapter
                Reference line.
            max_river_width: float
                Maximum distance away from river_profile.

        Returns:
            search_lines : List[shapely.geometry.linestring.LineStringAdapter]
                List of clipped search lines.
            max_distance: float
                Maximum distance from any point within line to reference line.
        """
        search_lines = self.bank_search_lines
        profile_buffer = self.masked_profile.buffer(max_river_width, cap_style=2)

        # The algorithm uses simplified geometries for determining the distance between lines for speed.
        # Stay accurate to within about 1 m
        profile_simplified = self.masked_profile.simplify(1)

        max_distance = 0
        for ind in range(self.num_search_lines):
            # Clip the bank search lines to the reach of interest (indicated by the reference line).
            search_lines[ind] = search_lines[ind].intersection(profile_buffer)

            # If the bank search line breaks into multiple parts, select the part closest to the reference line.
            if search_lines[ind].geom_type == "MultiLineString":
                distance_min = max_river_width
                i_min = 0
                for i in range(len(search_lines[ind])):
                    line_simplified = search_lines[ind][i].simplify(1)
                    distance_min_i = line_simplified.distance(profile_simplified)
                    if distance_min_i < distance_min:
                        distance_min = distance_min_i
                        i_min = i
                search_lines[ind] = search_lines[ind][i_min]

            # Determine the maximum distance from a point on this line to the reference line.
            line_simplified = search_lines[ind].simplify(1)
            max_distance = max(
                [
                    Point(c).distance(profile_simplified)
                    for c in line_simplified.coords
                ]
            )

            # Increase the value of max_distance by 2 to account for error introduced by using simplified lines.
            max_distance = max(max_distance, max_distance + 2)

        return search_lines, max_distance


def load_program_texts(filename: str) -> None:
    """
    Load texts from configuration file, and store globally for access.

    This routine reads the text file "filename", and detects the keywords
    indicated by lines starting with [ and ending with ]. The content is
    placed in a global dictionary PROGTEXTS which may be queried using the
    routine "get_text". These routines are used to implement multi-
    language support.

    Arguments
    ---------
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


def log_text(
    key: str,
    file: Optional[TextIO] = None,
    dict: Dict[str, Any] = {},
    repeat: int = 1,
    indent: str = "",
) -> None:
    """
    Write a text to standard out or file.

    Arguments
    ---------
    key : str
        The key for the text to show to the user.
    file : Optional[TextIO]
        The file to write to (None for writing to standard out).
    dict : Dict[str, Any]
        A dictionary used for placeholder expansions (default empty).
    repeat : int
        The number of times that the same text should be repeated (default 1).
    indent : str
        String to use for each line as indentation (default empty).

    Returns
    -------
    None
    """
    str = get_text(key)
    for r in range(repeat):
        for s in str:
            sexp = s.format(**dict)
            if file is None:
                print(indent + sexp)
            else:
                file.write(indent + sexp + "\n")


def get_filename(key: str) -> str:
    """
    Query the global dictionary of texts for a file name.

    The file name entries in the global dictionary have a prefix "filename_"
    which will be added to the key by this routine.

    Arguments
    ---------
    key : str
        The key string used to query the dictionary.

    Results
    -------
    filename : str
        File name.
    """
    filename = get_text("filename_" + key)[0]
    return filename


def get_text(key: str) -> List[str]:
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


def read_fm_map(filename: str, varname: str, location: str = "face") -> np.ndarray:
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
    if var.get_dims()[0].isunlimited():
        # assume that time dimension is unlimited and is the first dimension
        # slice to obtain last time step
        data = var[-1, :]
    else:
        data = var[...] - start_index

    rootgrp.close()

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


def copy_ugrid(srcname: str, meshname: str, dstname: str) -> None:
    """
    Copy UGRID mesh data from one netCDF file to another.

    Copy UGRID mesh data (mesh variable, all attributes, all variables that the
    UGRID attributes depend on) from source file to destination file.

    Arguments
    ---------
    srcname : str
        Name of source file.
    meshname : str
        Name of the UGRID mesh to be copied from source to destination.
    dstname : str
        Name of destination file, or dataset object representing the destination
        file.
    """
    # open source and destination files
    src = netCDF4.Dataset(srcname)
    dst = netCDF4.Dataset(dstname, "w", format="NETCDF4")

    # locate source mesh
    mesh = src.variables[meshname]

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

    # close files
    src.close()
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
    # locate the variable to be copied
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
    ldata: np.array,
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
    ldata : np.array
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


def read_waqua_xyz(filename: str, cols: Tuple[int, ...] = (2,)) -> np.ndarray:
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
    data : np.ndarray
        Data read from the file.
    """
    data = np.genfromtxt(filename, delimiter=",", skip_header=1, usecols=cols)
    return data


def write_simona_box(
    filename: str, rdata: np.ndarray, firstm: int, firstn: int
) -> None:
    """
    Write a SIMONA BOX file.

    Arguments
    ---------
    filename : str
        Name of the file to be written.
    rdata : np.ndarray
        Two-dimensional np array containing the data to be written.
    firstm : int
        Firt M index to be written.
    firstn : int
        First N index to be written.
    """
    # open the data file
    boxfile = open(filename, "w")

    # get shape and prepare block header; data will be written in blocks of 10
    # N-lines
    shp = np.shape(rdata)
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


def read_xyc(
    filename: str, num_columns: int = 2
) -> shapely.geometry.linestring.LineStringAdapter:
    """
    Read lines from a file.

    Arguments
    ---------
    filename : str
        Name of the file to be read.
    num_columns : int
        Number of columns to be read (2 or 3)

    Returns
    -------
    L : shapely.geometry.linestring.LineStringAdapter
        Line strings.
    """
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    if filename.suffix.lower() == ".xyc":
        if num_columns == 3:
            column_names = ["Val", "X", "Y"]
        else:
            column_names = ["X", "Y"]
        point_coordinates = pandas.read_csv(
            filename, names=column_names, skipinitialspace=True, delim_whitespace=True
        )
        num_points = len(point_coordinates.X)
        x = point_coordinates.X.to_numpy().reshape((num_points, 1))
        y = point_coordinates.Y.to_numpy().reshape((num_points, 1))
        if num_columns == 3:
            z = point_coordinates.Val.to_numpy().reshape((num_points, 1))
            coords = np.concatenate((x, y, z), axis=1)
        else:
            coords = np.concatenate((x, y), axis=1)
        line_string = shapely.geometry.LineString(coords)
    else:
        gdf = geopandas.read_file(filename)["geometry"]
        line_string = gdf[0]

    return line_string


def write_xyc(xy: np.ndarray, val: np.ndarray, filename: str) -> None:
    """
    Write a text file with x, y, and values.

    Arguments
    ---------
    xy : np.ndarray
        N x 2 array containing x and y coordinates.
    val : np.ndarray
        N x k array containing values.
    filename : str
        Name of the file to be written.

    Returns
    -------
    None
    """
    with open(filename, "w") as xyc:
        if val.ndim == 1:
            for i in range(len(val)):
                valstr = "{:.2f}".format(val[i])
                xyc.write("{:.2f}\t{:.2f}\t".format(xy[i, 0], xy[i, 1]) + valstr + "\n")
        else:
            for i in range(len(val)):
                valstr = "\t".join(["{:.2f}".format(x) for x in val[i, :]])
                xyc.write("{:.2f}\t{:.2f}\t".format(xy[i, 0], xy[i, 1]) + valstr + "\n")


def write_shp_pnt(
    xy: np.ndarray, dict: Dict[str, np.ndarray], filename: str
) -> None:
    """
    Write a shape point file with x, y, and values.

    Arguments
    ---------
    xy : np.ndarray
        N x 2 array containing x and y coordinates.
    dict : Dict[str, np.ndarray]
        Dictionary of quantities to be written, each np array should have length k.
    filename : str
        Name of the file to be written.

    Returns
    -------
    None
    """
    xy_Points = [shapely.geometry.Point(xy1) for xy1 in xy]
    geom = geopandas.geoseries.GeoSeries(xy_Points)
    write_shp(geom, dict, filename)


def write_shp(
    geom: geopandas.geoseries.GeoSeries, dict: Dict[str, np.ndarray], filename: str
) -> None:
    """
    Write a shape file for a given GeoSeries and dictionary of np arrays.
    The GeoSeries and all np should have equal length.

    Arguments
    ---------
    geom : geopandas.geoseries.GeoSeries
        geopandas GeoSeries containing k geometries.
    dict : Dict[str, np.ndarray]
        Dictionary of quantities to be written, each np array should have length k.
    filename : str
        Name of the file to be written.

    Returns
    -------
    None
    """
    val_DataFrame = pandas.DataFrame(dict)
    geopandas.GeoDataFrame(val_DataFrame, geometry=geom).to_file(filename)


def write_csv(dict: Dict[str, np.ndarray], filename: str) -> None:
    """
    Write a data to csv file.

    Arguments
    ---------
    dict : Dict[str, np.ndarray]
        Value(s) to be written.
    filename : str
        Name of the file to be written.

    Returns
    -------
    None
    """
    keys = [key for key in dict.keys()]
    header = ""
    for i in range(len(keys)):
        if i < len(keys) - 1:
            header = header + '"' + keys[i] + '", '
        else:
            header = header + '"' + keys[i] + '"'

    data = np.column_stack([array for array in dict.values()])
    np.savetxt(filename, data, delimiter=", ", header=header, comments="")


def write_km_eroded_volumes(
    km: np.ndarray, vol: np.ndarray, filename: str
) -> None:
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

def move_parameter_location(
    config: configparser.ConfigParser,
    group1: str,
    key1: str,
    group2: str,
    key2: Optional[str] = None,
    convert: Optional[Callable[[str], str]] = None,
) -> configparser.ConfigParser:
    """
    Move a parameter from one group/keyword to another.

    Args:
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
        convert: Optional[Callable[[str], str]]
            Function to convert the original value into new value.

    Returns:
        config : configparser.ConfigParser
            Updated settings for the D-FAST Bank Erosion analysis.
    """
    val2: str
    if group1 in config.sections() and key1 in config[group1]:
        if key2 is None:
            key2 = key1
        val1 = config[group1][key1]
        if convert is None:
            val2 = val1
        else:
            val2 = convert(val1)
        config[group2][key2] = val2
        config[group1].pop(key1)
    return config


def sim2nc(oldfile: str) -> str:
    """
    Convert an SDS file name to an NC file (mirrors sim2ugrid.m).

    Arguments
    ---------
    oldfile : str
        Name of the original SIMONA SDS or Delft3D-FLOW TRIM file.

    Results
    -------
    ncfile : str
        Name of the netCDF file as created by sim2ugrid.m.
    """
    path, name = os.path.split(oldfile)
    if name[:3] == "SDS":
        # SDS-case_map.nc
        ncfile = oldfile + "_map.nc"
    elif name[:4] == "trim":
        # trim-case_map.nc
        basename, ext = os.path.splitext(oldfile)
        ncfile = basename + "_map.nc"
    else:
        raise Exception('Unable to determine file type for "{}"'.format(oldfile))
    return ncfile


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
    km_thr : Optional[np.ndarray]
        Array containing the chainage of the midpoints between the values.
    val : np.ndarray
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
        # km_thr = (km[:-1] + km[1:]) / 2
        km_thr = km[1:]
    return km_thr, val


def read_simdata(filename: str, indent: str = "") -> Tuple[SimulationObject, float]:
    """
    Read a deault set of quantities from a UGRID netCDF file coming from D-Flow FM (or similar).

    Arguments
    ---------
    filename : str
        Name of the simulation output file to be read.
    indent : str
        String to use for each line as indentation (default empty).

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
    dum = np.array([])
    sim: SimulationObject = {
        "x_node": dum,
        "y_node": dum,
        "nnodes": dum,
        "facenode": dum,
        "zb_location": dum,
        "zb_val": dum,
        "zw_face": dum,
        "h_face": dum,
        "ucx_face": dum,
        "ucy_face": dum,
        "chz_face": dum,
    }
    # determine file type
    path, name = os.path.split(filename)
    if name[-6:] == "map.nc":
        log_text("read_grid", indent=indent)
        sim["x_node"] = read_fm_map(filename, "x", location="node")
        sim["y_node"] = read_fm_map(filename, "y", location="node")
        FNC = read_fm_map(filename, "face_node_connectivity")
        if FNC.mask.shape == ():
            # all faces have the same number of nodes
            sim["nnodes"] = (
                np.ones(FNC.data.shape[0], dtype=np.int) * FNC.data.shape[1]
            )
        else:
            # varying number of nodes
            sim["nnodes"] = FNC.mask.shape[1] - FNC.mask.sum(axis=1)
        FNC.data[FNC.mask] = 0
        # sim["facenode"] = FNC.data
        sim["facenode"] = FNC
        log_text("read_bathymetry", indent=indent)
        sim["zb_location"] = "node"
        sim["zb_val"] = read_fm_map(filename, "altitude", location="node")
        log_text("read_water_level", indent=indent)
        sim["zw_face"] = read_fm_map(filename, "Water level")
        log_text("read_water_depth", indent=indent)
        sim["h_face"] = np.maximum(
            read_fm_map(filename, "sea_floor_depth_below_sea_surface"), 0.0
        )
        log_text("read_velocity", indent=indent)
        sim["ucx_face"] = read_fm_map(filename, "sea_water_x_velocity")
        sim["ucy_face"] = read_fm_map(filename, "sea_water_y_velocity")
        log_text("read_chezy", indent=indent)
        sim["chz_face"] = read_fm_map(filename, "Chezy roughness")

        log_text("read_drywet", indent=indent)
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


def get_progloc() -> str:
    """
    Get the location of the program.

    Arguments
    ---------
    None
    """
    progloc = str(pathlib.Path(__file__).parent.absolute())
    return progloc


class ConfigFileError(Exception):
    """Custom exception for configuration file errors."""
    pass
