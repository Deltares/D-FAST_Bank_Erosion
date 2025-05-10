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

from configparser import ConfigParser
from configparser import Error as ConfigparserError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from dfastio.xyc.models import XYCModel
from geopandas.geodataframe import GeoDataFrame
from geopandas.geoseries import GeoSeries
from shapely.geometry import LineString
from dfastbe.io.file_utils import absolute_path, relative_path
from dfastbe.io.logger import log_text

__all__ = ["ConfigFile", "get_bbox", "ConfigFileError", "SimulationFilesError"]


class ConfigFile:
    """Class to read configuration files for D-FAST Bank Erosion.

    This class provides methods to read, write, and manage configuration files
    for the D-FAST Bank Erosion analysis. It also allows access to configuration
    settings and supports upgrading older configuration formats.
    """

    def __init__(self, config: ConfigParser, path: Union[Path, str] = None):
        """
        Initialize the ConfigFile object.

        Args:
            config (ConfigParser):
                Settings for the D-FAST Bank Erosion analysis.
            path (Union[Path, str]):
                Path to the configuration file.

        Examples:
            Reading a configuration file:
                ```python
                >>> import tempfile
                >>> from dfastbe.io.config import ConfigFile
                >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
                >>> print(config_file.config["General"]["Version"])
                1.0

                ```
            Writing a configuration file:
                ```python
                >>> from dfastbe.io.config import ConfigFile
                >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
                >>> with tempfile.TemporaryDirectory() as tmpdirname:
                ...     config_file.write(f"{tmpdirname}/meuse_manual_out.cfg")

                ```
        """
        self._config = config
        self.crs = "EPSG:28992"
        if path:
            self.path = Path(path)
            self.root_dir = self.path.parent
            self.make_paths_absolute()

    @property
    def config(self) -> ConfigParser:
        """ConfigParser: Get the configuration settings."""
        return self._config

    @config.setter
    def config(self, value: ConfigParser):
        self._config = value

    @property
    def version(self) -> str:
        """str: Get the version of the configuration file."""
        return self.get_str("General", "Version")

    @property
    def debug(self) -> bool:
        return self.get_bool("General", "DebugOutput", False)

    @property
    def root_dir(self) -> Path:
        """Path: Get the root directory of the configuration file."""
        return self._root_dir

    @root_dir.setter
    def root_dir(self, value: str):
        self._root_dir = value

    @classmethod
    def read(cls, path: Union[str, Path]) -> "ConfigFile":
        """Read a configParser object (configuration file).

        Reads the config file using the standard `configparser`. Falls back to a
        dedicated reader compatible with old waqbank files.

        Args:
            path (Union[str, Path]): Path to the configuration file.

        Returns:
            ConfigFile: Settings for the D-FAST Bank Erosion analysis.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            Exception: If there is an error reading the config file.

        Examples:
            Read a config file:
            ```python
            >>> from dfastbe.io.config import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")

            ```
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"The Config-File: {path} does not exist")

        try:
            config = ConfigParser(comment_prefixes="%")
            with open(path, "r") as configfile:
                config.read_file(configfile)
        except ConfigparserError as e:
            print(f"Error during reading the config file: {e}")
            config = cls.config_file_callback_parser(path)

        # if version != "1.0":
        config = cls._upgrade(config)
        return cls(config, path=path)

    @staticmethod
    def config_file_callback_parser(path: str) -> ConfigParser:
        """Parse a configuration file as fallback to the read method.

        Args:
            path (str): Path to the configuration file.

        Returns:
            ConfigParser: Parsed configuration file.
        """
        config = ConfigParser()
        config["General"] = {}
        all_lines = open(path, "r").read().splitlines()
        for line in all_lines:
            perc = line.find("%")
            if perc >= 0:
                line = line[:perc]
            data = line.split()
            if len(data) >= 3:
                config["General"][data[0]] = data[2]
        return config

    @staticmethod
    def _upgrade(config: ConfigParser) -> ConfigParser:
        """Upgrade the configuration data structure to version 1.0 format.

        Args:
            config (ConfigParser): D-FAST Bank Erosion settings in 0.1 format.

        Returns:
            ConfigParser: D-FAST Bank Erosion settings in 1.0 format.

        Examples:
            ```python
            >>> from dfastbe.io.config import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> result = config_file._upgrade(config_file.config)
            >>> isinstance(result, ConfigParser)
            True

            ```
        """
        try:
            version = config["General"]["Version"]
        except KeyError:
            version = "0.1"

        if version == "0.1":
            config["General"]["Version"] = "1.0"

            config["Detect"] = {}
            config = _move_parameter_location(
                config, "General", "Delft3Dfile", "Detect", "SimFile", convert=_sim2nc
            )
            config = _move_parameter_location(
                config, "General", "SDSfile", "Detect", "SimFile", convert=_sim2nc
            )
            config = _move_parameter_location(config, "General", "SimFile", "Detect")
            config = _move_parameter_location(config, "General", "NBank", "Detect")
            config_file = ConfigFile(config)
            n_bank = config_file.get_int("Detect", "NBank", default=0, positive=True)
            for i in range(1, n_bank + 1):
                config = _move_parameter_location(
                    config, "General", f"Line{i}", "Detect"
                )

            config = _move_parameter_location(config, "General", "WaterDepth", "Detect")
            config = _move_parameter_location(config, "General", "DLines", "Detect")

            config["Erosion"] = {}
            config = _move_parameter_location(config, "General", "TErosion", "Erosion")
            config = _move_parameter_location(config, "General", "RiverAxis", "Erosion")
            config = _move_parameter_location(config, "General", "Fairway", "Erosion")
            config = _move_parameter_location(config, "General", "RefLevel", "Erosion")
            config = _move_parameter_location(
                config, "General", "OutputInterval", "Erosion"
            )
            config = _move_parameter_location(config, "General", "OutputDir", "Erosion")
            config = _move_parameter_location(config, "General", "BankNew", "Erosion")
            config = _move_parameter_location(config, "General", "BankEq", "Erosion")
            config = _move_parameter_location(config, "General", "EroVol", "Erosion")
            config = _move_parameter_location(
                config, "General", "EroVolEqui", "Erosion"
            )
            config = _move_parameter_location(config, "General", "NLevel", "Erosion")
            config_file = ConfigFile(config)
            n_level = config_file.get_int("Erosion", "NLevel", default=0, positive=True)

            for i in range(1, n_level + 1):
                config = _move_parameter_location(
                    config,
                    "General",
                    f"Delft3Dfile{i}",
                    "Erosion",
                    f"SimFile{i}",
                    convert=_sim2nc,
                )
                config = _move_parameter_location(
                    config,
                    "General",
                    f"SDSfile{i}",
                    "Erosion",
                    f"SimFile{i}",
                    convert=_sim2nc,
                )
                config = _move_parameter_location(
                    config, "General", f"SimFile{i}", "Erosion"
                )
                config = _move_parameter_location(
                    config, "General", f"PDischarge{i}", "Erosion"
                )

            config = _move_parameter_location(config, "General", "ShipType", "Erosion")
            config = _move_parameter_location(config, "General", "VShip", "Erosion")
            config = _move_parameter_location(config, "General", "NShip", "Erosion")
            config = _move_parameter_location(config, "General", "NWave", "Erosion")
            config = _move_parameter_location(config, "General", "Draught", "Erosion")
            config = _move_parameter_location(config, "General", "Wave0", "Erosion")
            config = _move_parameter_location(config, "General", "Wave1", "Erosion")

            config = _move_parameter_location(config, "General", "Classes", "Erosion")
            config = _move_parameter_location(config, "General", "BankType", "Erosion")
            config = _move_parameter_location(
                config, "General", "ProtectLevel", "Erosion", "ProtectionLevel"
            )
            config = _move_parameter_location(config, "General", "Slope", "Erosion")
            config = _move_parameter_location(config, "General", "Reed", "Erosion")
            config = _move_parameter_location(config, "General", "VelFilter", "Erosion")

            for i in range(1, n_level + 1):
                config = _move_parameter_location(
                    config, "General", f"ShipType{i}", "Erosion"
                )
                config = _move_parameter_location(
                    config, "General", f"VShip{i}", "Erosion"
                )
                config = _move_parameter_location(
                    config, "General", f"NShip{i}", "Erosion"
                )
                config = _move_parameter_location(
                    config, "General", f"NWave{i}", "Erosion"
                )
                config = _move_parameter_location(
                    config, "General", f"Draught{i}", "Erosion"
                )
                config = _move_parameter_location(
                    config, "General", f"Slope{i}", "Erosion"
                )
                config = _move_parameter_location(
                    config, "General", f"Reed{i}", "Erosion"
                )
                config = _move_parameter_location(
                    config, "General", f"EroVol{i}", "Erosion"
                )

        return config

    def write(self, filename: str) -> None:
        """Pretty print a configParser object (configuration file) to file.

        Pretty prints a `configparser` object to a file. Aligns the equal signs for
        all keyword/value pairs, adds a two-space indentation to all keyword lines,
        and adds an empty line before the start of a new block.

        Args:
            filename (str): Name of the configuration file to be written.

        Examples:
            ```python
            >>> import tempfile
            >>> from dfastbe.io.config import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> with tempfile.TemporaryDirectory() as tmpdirname:
            ...     config_file.write(f"{tmpdirname}/meuse_manual_out.cfg")

            ```
        """
        sections = self.config.sections()
        max_length = 0
        for section in sections:
            options = self.config.options(section)
            max_length = max(max_length, *[len(option) for option in options])

        with open(filename, "w") as configfile:
            for index, section in enumerate(sections):
                if index > 0:
                    configfile.write("\n")
                configfile.write(f"[{section}]\n")

                for option in self.config.options(section):
                    configfile.write(
                        f"  {option:<{max_length}} = {self.config[section][option]}\n"
                    )

    def make_paths_absolute(self) -> str:
        """Convert all relative paths in the configuration to absolute paths.

        Returns:
            str: Absolute path of the configuration file's root directory.
        """
        self.resolve(self.root_dir)

        return self.root_dir

    def get_str(
        self,
        group: str,
        key: str,
        default: Optional[str] = None,
    ) -> str:
        """Get a string from a selected group and keyword in the analysis settings.

        Args:
            group (str): Name of the group from which to read.
            key (str): Name of the keyword from which to read.
            default (Optional[str]): Optional default value.

        Raises:
            ConfigFileError: If the keyword isn't specified and no default value is given.

        Returns:
            str: value of the keyword as string.

        Examples:
            ```python
            >>> from dfastbe.io.config import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> result = config_file.get_str("General", "BankDir")
            >>> expected = Path("tests/data/erosion/output/banklines").resolve()
            >>> str(expected) == result
            True

            ```
        """
        try:
            val = self.config[group][key]
        except KeyError as e:
            if default is not None:
                val = default
            else:
                raise ConfigFileError(
                    f"No value specified for required keyword {key} in block {group}."
                ) from e
        return val

    def get_bool(
        self,
        group: str,
        key: str,
        default: Optional[bool] = None,
    ) -> bool:
        """Get a boolean from a selected group and keyword in the analysis settings.

        Args:
            group (str): Name of the group from which to read.
            key (str): Name of the keyword from which to read.
            default (Optional[bool]): Optional default value.

        Raises:
            ConfigFileError: If the keyword isn't specified and no default value is given.

        Returns:
            bool: value of the keyword as boolean.

        Examples:
            ```python
            >>> from dfastbe.io.config import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> config_file.get_bool("General", "Plotting")
            True

            ```
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
        except KeyError as e:
            if default is not None:
                val = default
            else:
                raise ConfigFileError(
                    f"No boolean value specified for required keyword {key} in block {group}."
                ) from e

        return val

    def get_float(
        self,
        group: str,
        key: str,
        default: Optional[float] = None,
        positive: bool = False,
    ) -> float:
        """Get a floating point value from a selected group and keyword in the analysis settings.

        Args:
            group (str): Name of the group from which to read.
            key (str): Name of the keyword from which to read.
            default (Optional[float]): Optional default value.
            positive (bool): Flag specifying which floats are accepted.
                All floats are accepted (if False), or only positive floats (if True).

        Raises:
            ConfigFileError: If the keyword isn't specified and no default value is given.
            ConfigFileError: If a negative value is specified when a positive value is required.


        Returns:
            float: value of the keyword as float.

        Examples:
            ```python
            >>> from dfastbe.io.config import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> config_file.get_float("General", "ZoomStepKM")
            1.0

            ```
        """
        try:
            val = float(self.config[group][key])
        except (KeyError, ValueError) as e:
            if default is not None:
                val = default
            else:
                raise ConfigFileError(
                    f"No floating point value specified for required keyword {key} in block {group}."
                ) from e
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
        """Get an integer from a selected group and keyword in the analysis settings.

        Args:
            group (str): Name of the group from which to read.
            key (str): Name of the keyword from which to read.
            default (Optional[int]): Optional default value.
            positive (bool): Flag specifying which floats are accepted.
                All floats are accepted (if False), or only positive floats (if True).

        Raises:
            ConfigFileError: If the keyword isn't specified and no default value is given.
            ConfigFileError: If a negative or zero value is specified when a positive value is required.


        Returns:
            int: value of the keyword as int.

        Examples:
            ```python
            >>> from dfastbe.io.config import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> config_file.get_int("Detect", "NBank")
            2

            ```
        """
        try:
            val = int(self.config[group][key])
        except (KeyError, ValueError) as e:
            if default is not None:
                val = default
            else:
                raise ConfigFileError(
                    f"No integer value specified for required keyword {key} in block {group}."
                ) from e
        if positive and val <= 0:
            raise ConfigFileError(
                f"Value for {key} in block {group} must be positive, not {val}."
            )
        return val

    def get_sim_file(self, group: str, istr: str) -> str:
        """Get the name of the simulation file from the analysis settings.

        Args:
            group (str): Name of the group in which to search for the simulation file name.
            istr (str): Postfix for the simulation file name keyword;
                typically a string representation of the index.

        Returns:
            str: Name of the simulation file (empty string if keywords are not found).

        Examples:
            ```python
            >>> from dfastbe.io.config import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> result = config_file.get_sim_file("Erosion", "1")
            >>> expected = Path("tests/data/erosion/inputs/sim0075/SDS-j19_map.nc").resolve()
            >>> str(expected) == result
            True

            ```
        """
        sim_file = self.config[group].get(f"SimFile{istr}", "")
        return sim_file

    def get_start_end_stations(self) -> Tuple[float, float]:
        """Get the start and end station for the river.

        Returns:
            Tuple[float, float]: start and end station.

        Examples:
            ```python
            >>> from dfastbe.io.config import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> config_file.get_start_end_stations()
            (123.0, 128.0)

            ```
        """
        stations = self.get_range("General", "Boundaries")

        return stations

    def get_search_lines(self) -> List[LineString]:
        """Get the search lines for the bank lines from the analysis settings.

        Returns:
            List[np.ndarray]: List of arrays containing the x,y-coordinates of a bank search lines.
        """
        # read guiding bank line
        n_bank = self.get_int("Detect", "NBank")
        line = [None] * n_bank
        for b in range(n_bank):
            bankfile = self.config["Detect"][f"Line{b + 1}"]
            log_text("read_search_line", data={"nr": b + 1, "file": bankfile})
            line[b] = XYCModel.read(bankfile)
        return line

    def read_bank_lines(self, bank_dir: str) -> List[np.ndarray]:
        """Get the bank lines from the detection step.

        Args:
            bank_dir (str): Name of directory in which the bank lines files are located.

        Returns:
            List[np.ndarray]: List of arrays containing the x,y-coordinates of a bank lines.
        """
        bank_name = self.get_str("General", "BankFile", "bankfile")
        bankfile = Path(bank_dir) / f"{bank_name}.shp"
        if bankfile.exists():
            log_text("read_banklines", data={"file": str(bankfile)})
            return gpd.read_file(bankfile)

        bankfile = Path(bank_dir) / f"{bank_name}_#.xyc"
        log_text("read_banklines", data={"file": str(bankfile)})
        bankline_list = []
        b = 1
        while True:
            xyc_file = Path(bank_dir) / f"{bank_name}_{b}.xyc"
            if not xyc_file.exists():
                break

            xy_bank = XYCModel.read(xyc_file)
            bankline_list.append(LineString(xy_bank))
            b += 1
        bankline_series = GeoSeries(bankline_list, crs=self.crs)
        banklines = GeoDataFrame(geometry=bankline_series)
        return banklines

    def get_parameter(
        self,
        group: str,
        key: str,
        num_stations_per_bank: List[int],
        default: Any = None,
        ext: str = "",
        positive: bool = False,
        valid: Optional[List[float]] = None,
        onefile: bool = False,
    ) -> List[np.ndarray]:
        """Get a parameter field from a selected group and keyword in the analysis settings.

        Args:
            group (str):
                Name of the group from which to read.
            key (str):
                Name of the keyword from which to read.
            num_stations_per_bank (List[int]):
                Number of stations (points) For each bank (bank chainage locations).
            default (Optional[Union[float, List[np.ndarray]]]):
                Optional default value or default parameter field; default None.
            ext (str):
                File name extension; default empty string.
            positive (bool):
                Flag specifying which boolean values are accepted.
                All values are accepted (if False), or only strictly positive values (if True); default False.
            valid (Optional[List[float]]):
                Optional list of valid values; default None.
            onefile (bool):
                Flag indicating whether parameters are read from one file.
                One file should be used for all bank lines (True) or one file per bank line (False; default).

        Raises:
            Exception:
                If a parameter isn't provided in the configuration, but no default value provided either.
                If the value is negative while a positive value is required (positive = True).
                If the value doesn't match one of the value values (valid is not None).

        Returns:
            List[np.ndarray]: Parameter field
                For each bank a parameter value per bank point (bank chainage location).

        Examples:
            ```python
            >>> from dfastbe.io.config import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> bank_km = [np.array([0, 1, 2]), np.array([3, 4, 5, 6, 7])]
            >>> num_stations_per_bank = [len(bank) for bank in bank_km]
            >>> config_file.get_parameter("General", "ZoomStepKM", num_stations_per_bank)
            [array([1., 1., 1.]), array([1., 1., 1., 1., 1.])]

            ```
        """
        try:
            value = self.config[group][key]
            use_default = False
        except (KeyError, TypeError) as e:
            value = None
            if default is None:
                raise ConfigFileError(
                    f'No value specified for required keyword "{key}" in block "{group}".'
                ) from e
            use_default = True

        return self.process_parameter(
            value=value,
            key=key,
            num_stations_per_bank=num_stations_per_bank,
            use_default=use_default,
            default=default,
            ext=ext,
            positive=positive,
            valid=valid,
            onefile=onefile,
        )

    def validate_parameter_value(
        self,
        value: float,
        key: str,
        positive: bool = False,
        valid: Optional[List[float]] = None,
    ) -> float:
        """Validate a parameter value against constraints.

        Args:
            value (float): The parameter value to validate.
            key (str): Name of the parameter for error messages.
            positive (bool): Flag specifying whether all values are accepted (if False),
                or only positive values (if True); default False.
            valid (Optional[List[float]]): Optional list of valid values; default None.

        Raises:
            ValueError: If the value doesn't meet the constraints.

        Returns:
            float: The validated parameter value.
        """
        if positive and value < 0:
            raise ValueError(
                f'Value of "{key}" should be positive, not {value}.'
            )
        if valid is not None and valid.count(value) == 0:
            raise ValueError(
                f'Value of "{key}" should be in {valid}, not {value}.'
            )
        return value

    def process_parameter(
        self,
        value: Union[str, float],
        key: str,
        num_stations_per_bank: List[int],
        use_default: bool = False,
        default=None,
        ext: str = "",
        positive: bool = False,
        valid: Optional[List[float]] = None,
        onefile: bool = False,
    ) -> List[np.ndarray]:
        """Process a parameter value into arrays for each bank.

        Args:
            value (Union[str, float]): The parameter value or filename.
            key (str): Name of the parameter for error messages.
            num_stations_per_bank (List[int]): Number of stations for each bank.
            use_default (bool): Flag indicating whether to use the default value; default False.
            default: Default value to use if use_default is True; default None.
            ext (str): File name extension; default empty string.
            positive (bool): Flag specifying whether all values are accepted (if False),
                or only positive values (if True); default False.
            valid (Optional[List[float]]): Optional list of valid values; default None.
            onefile (bool): Flag indicating whether parameters are read from one file.
                One file should be used for all bank lines (True) or one file per bank line (False; default).

        Returns:
            List[np.ndarray]: Parameter values for each bank.
        """
        # if val is value then use that value globally
        num_banks = len(num_stations_per_bank)
        parameter_values = [None] * num_banks
        try:
            if use_default:
                if isinstance(default, list):
                    return default
                real_val = default
            else:
                real_val = float(value)
                self.validate_parameter_value(real_val, key, positive, valid)

            for ib, num_stations in enumerate(num_stations_per_bank):
                parameter_values[ib] = np.zeros(num_stations) + real_val

        except (ValueError, TypeError):
            if onefile:
                log_text("read_param", data={"param": key, "file": value})
                km_thr, val = _get_stations(value, key, positive)

            for ib, num_stations in enumerate(num_stations_per_bank):
                if not onefile:
                    filename_i = f"{value}_{ib + 1}{ext}"
                    log_text(
                        "read_param_one_bank",
                        data={"param": key, "i": ib + 1, "file": filename_i},
                    )
                    km_thr, val = _get_stations(filename_i, key, positive)

                if km_thr is None:
                    parameter_values[ib] = np.zeros(num_stations) + val[0]
                else:
                    idx = np.zeros(num_stations, dtype=int)

                    for thr in km_thr:
                        idx[num_stations >= thr] += 1
                    parameter_values[ib] = val[idx]
                # print("Min/max of data: ", parfield[ib].min(), parfield[ib].max())

        return parameter_values

    def get_bank_search_distances(self, num_search_lines: int) -> List[float]:
        """Get the search distance per bank line from the analysis settings.

        Args:
            num_search_lines (int): Number of bank search lines.

        Returns:
            List[float]: Array of length nbank containing the search distance value per bank line (default value: 50).

        Examples:
            ```python
            >>> from dfastbe.io.config import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> config_file.get_bank_search_distances(2)
            [50.0, 50.0]

            ```
        """
        d_lines_key = self.config["Detect"].get("DLines", None)
        if d_lines_key is None:
            d_lines = [50] * num_search_lines
        elif d_lines_key[0] == "[" and d_lines_key[-1] == "]":
            d_lines_split = d_lines_key[1:-1].split(",")
            d_lines = [float(d) for d in d_lines_split]
            if not all([d > 0 for d in d_lines]):
                raise ValueError(
                    "keyword DLINES should contain positive values in the configuration file."
                )
            if len(d_lines) != num_search_lines:
                raise ConfigFileError(
                    "keyword DLINES should contain NBANK values in the configuration file."
                )
        return d_lines

    def get_range(self, group: str, key: str) -> Tuple[float, float]:
        """Get a start and end value from a selected group and keyword in the analysis settings.

        Args:
            group (str): Name of the group from which to read.
            key (str): Name of the keyword from which to read.

        Returns:
            Tuple[float,float]: Lower and upper limit of the range.

        Examples:
            ```python
            >>> from dfastbe.io.config import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> config_file.get_range("General", "Boundaries")
            (123.0, 128.0)

            ```
        """
        str_val = self.get_str(group, key)
        try:
            obrack = str_val.find("[")
            cbrack = str_val.find("]")
            if obrack >= 0 and cbrack >= 0:
                str_val = str_val[obrack + 1 : cbrack - 1]
            val_list = [float(fstr) for fstr in str_val.split(":")]
            if val_list[0] > val_list[1]:
                val = (val_list[1], val_list[0])
            else:
                val = (val_list[0], val_list[1])
        except ValueError as e:
            raise ValueError(
                f'Invalid range specification "{str_val}" for required keyword "{key}" in block "{group}".'
            ) from e
        return val

    def get_river_center_line(self) -> LineString:
        """Get the river center line from the xyc file as a linestring.

        Returns:
            LineString: Chainage line.
        """
        # get the chainage file
        river_center_line_file = self.get_str("General", "RiverKM")
        log_text("read_chainage", data={"file": river_center_line_file})
        river_center_line = XYCModel.read(river_center_line_file, num_columns=3)

        # make sure that chainage is increasing with node index
        if river_center_line.coords[0][2] > river_center_line.coords[1][2]:
            river_center_line = LineString(river_center_line.coords[::-1])

        return river_center_line

    def resolve(self, rootdir: str):
        """Convert a configuration object to contain absolute paths (for editing).

        Args:
            rootdir (str): The path to be used as base for the absolute paths.

        Examples:
            ```python
            >>> from dfastbe.io.config import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> config_file.resolve("tests/data/erosion")

            ```
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

    def relative_to(self, rootdir: str) -> None:
        """Convert a configuration object to contain relative paths (for saving).

        Args:
            rootdir (str): The path to be used as base for the relative paths.

        Examples:
            ```python
            >>> from dfastbe.io.config import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> config_file.relative_to("testing/data/erosion")

            ```
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
        """Convert a parameter value to contain an absolute path.

        Determine whether the string represents a number.
        If not, try to convert to an absolute path.

        Args:
            group (str): Name of the group in the configuration.
            key (str): Name of the key in the configuration.
            rootdir (str): The path to be used as base for the absolute paths.

        Examples:
            ```python
            >>> from dfastbe.io.config import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> config_file.resolve_parameter("General", "RiverKM", "tests/data/erosion")

            ```
        """
        if key in self.config[group]:
            val_str = self.config[group][key]
            try:
                float(val_str)
            except ValueError:
                self.config[group][key] = absolute_path(rootdir, val_str)

    def parameter_relative_to(self, group: str, key: str, rootdir: str):
        """Convert a parameter value to contain a relative path.

        Determine whether the string represents a number.
        If not, try to convert to a relative path.

        Args:
            group (str): Name of the group in the configuration.
            key (str): Name of the key in the configuration.
            rootdir (str): The path to be used as base for the relative paths.

        Examples:
            ```python
            >>> from dfastbe.io.config import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> config_file.parameter_relative_to("General", "RiverKM", "tests/data/erosion")

            ```
        """
        if key in self.config[group]:
            val_str = self.config[group][key]

            try:
                float(val_str)
            except ValueError:
                self.config[group][key] = relative_path(rootdir, val_str)

    def get_plotting_flags(self, root_dir: Path | str) -> Dict[str, bool]:
        """Get the plotting flags from the configuration file.

        Returns:
            data (Dict[str, bool]):
                Dictionary containing the plotting flags.
                save_plot (bool): Flag indicating whether to save the plot.
                save_plot_zoomed (bool): Flag indicating whether to save the zoomed plot.
                zoom_km_step (float): Step size for zooming in on the plot.
                close_plot (bool): Flag indicating whether to close the plot.
        """
        plot_data = self.get_bool("General", "Plotting", True)

        if plot_data:
            save_plot = self.get_bool("General", "SavePlots", True)
            save_plot_zoomed = self.get_bool("General", "SaveZoomPlots", True)
            zoom_km_step = self.get_float("General", "ZoomStepKM", 1.0)
            if zoom_km_step < 0.01:
                save_plot_zoomed = False
            close_plot = self.get_bool("General", "ClosePlots", False)
        else:
            save_plot = False
            save_plot_zoomed = False
            close_plot = False

        data = {
            "plot_data": plot_data,
            "save_plot": save_plot,
            "save_plot_zoomed": save_plot_zoomed,
            "zoom_km_step": zoom_km_step,
            "close_plot": close_plot,
        }

        # as appropriate, check output dir for figures and file format
        if save_plot:
            fig_dir = self.get_str("General", "FigureDir", Path(root_dir) / "figure")
            log_text("figure_dir", data={"dir": fig_dir})
            path_fig_dir = Path(fig_dir)
            if path_fig_dir.exists():
                log_text("overwrite_dir", data={"dir": fig_dir})
            path_fig_dir.mkdir(parents=True, exist_ok=True)
            plot_ext = self.get_str("General", "FigureExt", ".png")
            data = data | {
                "fig_dir": fig_dir,
                "plot_ext": plot_ext,
            }

        return data

    def get_output_dir(self, option: str) -> Path:
        """Get the output directory for the analysis.

        Args:
            option (str):
                Option for which to get the output directory. "banklines" for bank lines, else the erosion output
                directory will be returned.
        Returns:
            output_dir (Path):
                Path to the output directory.
        """
        if option == "banklines":
            output_dir = self.get_str("General", "BankDir")
        else:
            output_dir = self.get_str("Erosion", "OutputDir")

        output_dir = Path(output_dir)
        log_text(f"{option}_out", data={"dir": output_dir})
        if output_dir.exists():
            log_text("overwrite_dir", data={"dir": output_dir})
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir


def get_bbox(
    coords: np.ndarray, buffer: float = 0.1
) -> Tuple[float, float, float, float]:
    """
    Derive the bounding box from a line.

    Args:
        coords (np.ndarray):
            An N x M array containing x- and y-coordinates as first two M entries
        buffer : float
            Buffer fraction surrounding the tight bounding box

    Returns:
        bbox (Tuple[float, float, float, float]):
            Tuple bounding box consisting of [min x, min y, max x, max y)
    """
    x = coords[:, 0]
    y = coords[:, 1]
    x_min = x.min()
    y_min = y.min()
    x_max = x.max()
    y_max = y.max()
    d = buffer * max(x_max - x_min, y_max - y_min)
    bbox = (x_min - d, y_min - d, x_max + d, y_max + d)

    return bbox


def _move_parameter_location(
    config: ConfigParser,
    group1: str,
    key1: str,
    group2: str,
    key2: Optional[str] = None,
    convert: Optional[Callable[[str], str]] = None,
) -> ConfigParser:
    """
    Move a parameter from one group/keyword to another.

    Args:
        config : ConfigParser
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
        config : ConfigParser
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


def _sim2nc(oldfile: str) -> str:
    """
    Convert an SDS file name to an NC file (mirrors sim2ugrid.m).

    Args:
        oldfile (str):Name of the original SIMONA SDS or Delft3D-FLOW TRIM file.

    Returns:
        str: Name of the netCDF file as created by sim2ugrid.m.
    """
    name = Path(oldfile).name
    if name.startswith("SDS"):
        # SDS-case_map.nc
        nc_file = f"{oldfile}_map.nc"
    elif name.startswith("trim"):
        # trim-case_map.nc
        nc_file = f"{Path(oldfile).stem}_map.nc"
    else:
        raise SimulationFilesError(f'Unable to determine file type for "{oldfile}"')
    return nc_file


def _get_stations(filename: str, key: str, positive: bool):
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
    points = pd.read_csv(
        filename,
        names=["Chainage", "Val"],
        skipinitialspace=True,
        delim_whitespace=True,
    )

    km = points.Chainage.to_numpy()
    val = points.Val.to_numpy()

    if len(km.shape) == 0:
        km = km[None]
        val = val[None]

    if positive and (val < 0).any():
        raise ValueError(
            f'Values of "{key}" in {filename} should be positive. Negative value read for chainage(s): {km[val < 0]}'
        )

    if len(km) == 1:
        km_thr = None
    else:
        if not (km[1:] > km[:-1]).all():
            raise ValueError(
                f"Chainage values are not increasing in the file {filename} read for {key}."
            )
        km_thr = km[1:]

    return km_thr, val


class ConfigFileError(Exception):
    """Custom exception for configuration file errors."""

    pass


class SimulationFilesError(Exception):
    """Custom exception for configuration file errors."""

    pass