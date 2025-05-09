from pathlib import Path


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