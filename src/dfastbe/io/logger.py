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
    """Singleton class for managing localized log messages from INI-style configuration files.

    LogData provides a centralized mechanism for loading and retrieving localized
    text messages used throughout the application. It implements the singleton pattern
    to ensure only one instance exists, maintaining consistent access to log messages
    across the entire application.

    The class reads INI-style configuration files where keys are enclosed in square
    brackets (e.g., ``[key_name]``) and the following lines until the next key contain
    the associated text content. This supports multi-language logging by loading
    different message files (e.g., ``messages.UK.ini``, ``messages.NL.ini``).

    Attributes:
        data (Dict[str, List[str]]): Dictionary mapping message keys to lists of
            text lines.

    Examples:
        - Create a LogData instance with a message file:
            ```python
            >>> from pathlib import Path
            >>> from dfastbe.io.logger import LogData
            >>> LogData.reset()
            >>> log_data = LogData(Path("tests/data/files/messages.UK.ini"))
            >>> type(log_data)
            <class 'dfastbe.io.logger.LogData'>

            ```
        - Verify the singleton pattern returns the same instance:
            ```python
            >>> log_data2 = LogData()
            >>> log_data is log_data2
            True

            ```
        - Access the loaded message data structure:
            ```python
            >>> sorted(log_data.data.keys())[:4]
            ['', 'confirm', 'filename_report.out', 'reach']

            ```
        - Retrieve a specific message by key:
            ```python
            >>> log_data.get_text("confirm")
            ['Confirm using "y" ...', '']

            ```

    See Also:
        log_text: Module-level convenience function for logging.
        timed_logger: Function for logging with timing information.
    """

    _instance: Optional["LogData"] = None

    def __new__(cls, file_name: Optional[str | Path] = None) -> "LogData":
        """Create or return the singleton LogData instance.

        Implements the singleton pattern using ``__new__``. On first call,
        creates a new instance and stores it. Subsequent calls return the
        existing instance.

        Args:
            file_name: Path to the INI-style message file. Required on first
                instantiation, optional on subsequent calls.

        Returns:
            The singleton LogData instance.

        Raises:
            ValueError: If ``file_name`` is None on first instantiation.

        Examples:
            - First instantiation requires a file path:
                ```python
                >>> from pathlib import Path
                >>> from dfastbe.io.logger import LogData
                >>> LogData.reset()
                >>> log_data = LogData(Path("tests/data/files/messages.UK.ini"))
                >>> log_data._initialized
                True

                ```
            - Subsequent calls without file_name return the same instance:
                ```python
                >>> log_data2 = LogData()
                >>> log_data is log_data2
                True

                ```
        """
        if cls._instance is None:
            if file_name is None:
                raise ValueError("file_name must be provided on first instantiation")
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, file_name: Optional[str | Path] = None) -> None:
        """Initialize the LogData singleton instance.

        Loads and parses the message file on first initialization. Subsequent
        calls to ``__init__`` are ignored since the instance is already initialized.

        Args:
            file_name: Path to the INI-style message file. Required on first
                instantiation, ignored on subsequent calls.

        Raises:
            ValueError: If ``file_name`` is None on first instantiation.

        Examples:
            - Initialize with a message file and access loaded data:
                ```python
                >>> from pathlib import Path
                >>> from dfastbe.io.logger import LogData
                >>> LogData.reset()
                >>> log_data = LogData(Path("tests/data/files/messages.UK.ini"))
                >>> len(log_data.data) > 0
                True
                >>> "confirm" in log_data.data
                True

                ```

        """
        if not self._initialized:
            if file_name is None:
                raise ValueError("file_name must be provided on first instantiation")
            self.data = self.read_data(file_name)
            self._initialized = True

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance to allow re-initialization.

        This class method clears the singleton instance, allowing a new instance
        to be created with a different message file. Useful for testing or
        switching between language files.

        Examples:
            - Reset and reinitialize with a different file:
                ```python
                >>> from pathlib import Path
                >>> from dfastbe.io.logger import LogData
                >>> LogData.reset()
                >>> log_data1 = LogData(Path("tests/data/files/messages.UK.ini"))
                >>> id1 = id(log_data1)
                >>> LogData.reset()
                >>> log_data2 = LogData(Path("tests/data/files/messages.UK.ini"))
                >>> id1 != id(log_data2)
                True

                ```
        """
        cls._instance = None

    def read_data(self, file_name: str | Path) -> dict[str, list[str]]:
        """Load and parse texts from an INI-style configuration file.

        Reads the specified file and parses it into a dictionary where keys are
        identified by lines enclosed in square brackets (e.g., ``[key_name]``).
        All lines following a key until the next key become the value (as a list
        of strings). This format supports multi-language localization.

        Args:
            file_name: Path to the INI-style message file to read and parse.

        Returns:
            Dictionary mapping string keys to lists of text lines. Each key
            corresponds to a bracketed section header, and the value contains
            all lines between that header and the next.

        Raises:
            ValueError: If a duplicate key is found in the file.
            FileNotFoundError: If the specified file does not exist.

        Examples:
            - Read and parse a message file:
                ```python
                >>> from pathlib import Path
                >>> from dfastbe.io.logger import LogData
                >>> LogData.reset()
                >>> log_data = LogData(Path("tests/data/files/messages.UK.ini"))
                >>> data = log_data.data
                >>> type(data)
                <class 'dict'>

                ```
            - Examine the structure of parsed data:
                ```python
                >>> data["confirm"]
                ['Confirm using "y" ...', '']
                >>> data["reach"]
                ['The measure is located on reach {reach}']

                ```
            - Empty key captures content before first bracketed section:
                ```python
                >>> data[""]
                ['']

                ```
        """
        with open(file_name, "r", encoding="utf-8") as f:
            all_lines = f.read().splitlines()

        data: Dict[str, List[str]] = {}
        text: List[str] = []
        key = None

        for line in all_lines:
            r_line = line.strip()
            if r_line.startswith("[") and r_line.endswith("]"):
                if key is not None:
                    if key in data:
                        raise ValueError(f"Duplicate entry for '{key}' in {file_name}.")
                    data[key] = text
                key = r_line[1:-1]
                text = []
            else:
                text.append(line)

        # Handle the last key
        if key is not None:
            if key in data:
                raise ValueError(f"Duplicate entry for '{key}' in {file_name}.")
            data[key] = text

        return data


    def get_text(self, key: str) -> List[str]:
        """Retrieve text lines associated with a message key.

        Queries the internal message dictionary and returns the list of strings
        stored for the given key. If the key is not found, returns a default
        "not found" message.

        Args:
            key: The message key to look up in the dictionary.

        Returns:
            List of strings containing the text for the key. If the key is not
            found, returns ``["No message found for <key>"]``.

        Examples:
            - Retrieve an existing message:
                ```python
                >>> from pathlib import Path
                >>> from dfastbe.io.logger import LogData
                >>> LogData.reset()
                >>> log_data = LogData(Path("tests/data/files/messages.UK.ini"))
                >>> log_data.get_text("confirm")
                ['Confirm using "y" ...', '']

                ```
            - Retrieve a message with placeholder:
                ```python
                >>> log_data.get_text("reach")
                ['The measure is located on reach {reach}']

                ```
            - Handle non-existent key:
                ```python
                >>> log_data.get_text("nonexistent_key")
                ['No message found for nonexistent_key']

                ```
            - Empty string key retrieves empty line content:
                ```python
                >>> log_data.get_text("")
                ['']

                ```
        """
        if key in self.data:
            val = self.data[key]
        else:
            val = [f"No message found for {key}"]
        return val

    def log_text(
        self,
        key: str,
        file: Optional[TextIO] = None,
        data: Optional[Dict[str, Any]] = None,
        repeat: int = 1,
        indent: str = "",
    ) -> None:
        """Write a localized message to standard output or a file.

        Retrieves the text associated with the given key and outputs it to either
        standard output or a specified file. Supports placeholder expansion using
        Python's string formatting, repetition, and indentation.

        Args:
            key: The message key to retrieve and output.
            file: File object to write to. If None, writes to standard output.
            data: Dictionary for placeholder expansion in the message text.
                Placeholders in the message (e.g., ``{reach}``) are replaced
                with corresponding values from this dictionary.
            repeat: Number of times to output the message. Defaults to 1.
            indent: String to prepend to each output line for indentation.
                Defaults to empty string.

        Examples:
            - Write a message to a StringIO buffer:

                ```python
                >>> from pathlib import Path
                >>> from io import StringIO
                >>> from dfastbe.io.logger import LogData
                >>> LogData.reset()
                >>> log_data = LogData(Path("tests/data/files/messages.UK.ini"))
                >>> output = StringIO()
                >>> log_data.log_text("confirm", file=output)
                >>> output.getvalue().splitlines()
                ['Confirm using "y" ...', '']

                ```
            - Write a message with placeholder expansion:
                ```python
                >>> output = StringIO()
                >>> log_data.log_text("reach", file=output, data={"reach": "River_ABC"})
                >>> output.getvalue().strip()
                'The measure is located on reach River_ABC'

                ```
            - Write to a file with indentation:
                ```python
                >>> output = StringIO()
                >>> log_data.log_text("reach", file=output, data={"reach": "Test"}, indent="  ")
                >>> output.getvalue()
                '  The measure is located on reach Test\\n'

                ```
            - Repeat a message multiple times:
                ```python
                >>> output = StringIO()
                >>> log_data.log_text("", file=output, repeat=3)
                >>> len(output.getvalue().splitlines())
                3

                ```

        See Also:
            get_text: Retrieve raw message text without output.

        """
        if data is None:
            data = {}
        str_value = self.get_text(key)
        for _ in range(repeat):
            for s in str_value:
                try:
                    sexp = s.format(**data)
                except KeyError as e:
                    raise KeyError(
                        f"Missing placeholder {e} in message '{key}'. "
                        f"Available data keys: {list(data.keys())}"
                    ) from e
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