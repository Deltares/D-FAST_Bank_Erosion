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
from typing import Union, List, Dict, Optional, TextIO, Any
from pathlib import Path

PROGTEXTS: Dict[str, List[str]]


def load_program_texts(file_name: Union[str, Path]) -> None:
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
    global PROGTEXTS

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

    PROGTEXTS = data


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
        str_value = PROGTEXTS[key]
    except:
        str_value = ["No message found for " + key]
    return str_value

def log_text(
        key: str,
        file: Optional[TextIO] = None,
        data: Dict[str, Any] = {},
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
    data : Dict[str, Any]
        A dictionary used for placeholder expansions (default empty).
    repeat : int
        The number of times that the same text should be repeated (default 1).
    indent : str
        String to use for each line as indentation (default empty).

    Returns
    -------
    None
    """
    str_value = get_text(key)
    for _ in range(repeat):
        for s in str_value:
            sexp = s.format(**data)
            if file is None:
                print(indent + sexp)
            else:
                file.write(indent + sexp + "\n")

