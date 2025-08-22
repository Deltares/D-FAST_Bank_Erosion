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
from pathlib import Path

__all__ = ["absolute_path", "relative_path"]


def absolute_path(rootdir: str, path: str) -> str:
    """
    Convert a relative path to an absolute path.

    Args:
        rootdir (str): Any relative paths should be given relative to this location.
        path (str): A relative or absolute location.

    Returns:
        str: An absolute location.
    """
    if not path:
        return path
    root_path = Path(rootdir).resolve()
    target_path = Path(path)

    if target_path.is_absolute():
        return str(target_path)

    resolved_path = (root_path / target_path).resolve()
    return str(resolved_path)

def relative_path(rootdir: str, file: str) -> str:
    """
    Convert an absolute path to a relative path.

    Args:
        rootdir (str): Any relative paths will be given relative to this location.
        file (str): An absolute location.

    Returns:
        str: A relative location if possible, otherwise the absolute location.
    """
    if not file:
        return file

    root_path = Path(rootdir).resolve()
    file_path = Path(file).resolve()

    try:
        return str(file_path.relative_to(root_path))
    except ValueError:
        return str(file_path)