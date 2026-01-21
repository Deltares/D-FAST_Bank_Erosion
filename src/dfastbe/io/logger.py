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
from typing import List, Dict, Optional, TextIO, Any, Tuple
from pathlib import Path
import time

__all__ = ["LogData", "timed_logger"]


class LogData:
    _instance: Optional["LogData"] = None

    def __new__(cls, file_name: Optional[Path] = None):
        """Singleton implementation using __new__."""
        if cls._instance is None:
            if file_name is None:
                raise ValueError("file_name must be provided on first instantiation")
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, file_name: Optional[Path] = None):
        """Initialize the LogData singleton instance.

        Arguments
        ---------
        file_name : Optional[Path]
            The name of the file to be read and parsed. Only used on first instantiation.
        """
        if not self._initialized:
            if file_name is None:
                raise ValueError("file_name must be provided on first instantiation")
            self.data = self.read_data(file_name)
            self._initialized = True

    def read_data(self, file_name: str | Path) -> dict[str, list[str]]:
        """Load texts from a configuration file, and store globally for access.

        This routine reads the text file "file_name", and detects the keywords
        indicated by lines starting with [ and ending with ]. The content is
        placed in a global dictionary PROGTEXTS which may be queried using the
        routine "get_text". These routines are used to implement multi-language support.

        Arguments
        ---------
        file_name : str
            The name of the file to be read and parsed.
        """
        all_lines = open(file_name, "r").read().splitlines()
        data: Dict[str, List[str]] = {}
        text: List[str] = []
        key = None
        for line in all_lines:
            r_line = line.strip()
            if r_line.startswith("[") and r_line.endswith("]"):
                if key is not None:
                    data[key] = text
                key = r_line[1:-1]
                text = []
            else:
                text.append(line)
        if key in data.keys():
            raise ValueError(f"Duplicate entry for {key} in {file_name}.")

        if key is not None:
            data[key] = text

        return data


    def get_text(self, key: str) -> List[str]:
        """
        Query the global dictionary of texts via a string key.

        Query the global dictionary PROGTEXTS by means of a string key and return
        the list of strings contained in the dictionary. If the dictionary doesn't
        include the key, a default string is returned.

        Parameters
        ----------
        data: dict[str, list[str]]
            logger data
        key : str
            The key string used to query the dictionary.

        Returns
        -------
        text : List[str]
            The list of strings returned contain the text stored in the dictionary
            for the key. If the key isn't available in the dictionary, the routine
            returns the default string "No message found for <key>"
        """

        if key in self.data.keys():
            str_value = self.data[key]
        else:
            str_value = ["No message found for " + key]
        return str_value

    def log_text(
        self,
        key: str,
        file: Optional[TextIO] = None,
        data: Optional[Dict[str, Any]] = None,
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
        data : Optional[Dict[str, Any]]
            A dictionary used for placeholder expansions (default None, treated as empty dict).
        repeat : int
            The number of times that the same text should be repeated (default 1).
        indent : str
            String to use for each line as indentation (default empty).

        Returns
        -------
        None
        """
        if data is None:
            data = {}
        str_value = self.get_text(key)
        for _ in range(repeat):
            for s in str_value:
                sexp = s.format(**data)
                if file is None:
                    print(indent + sexp)
                else:
                    file.write(indent + sexp + "\n")


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