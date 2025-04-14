import pytest
from pathlib import Path


@pytest.fixture
def exe_path() -> Path:
    """dfast binary path"""
    repo_root = Path(__file__).resolve().parent.parent.parent
    exe_path = repo_root / "dfastbe.dist/dfastbe.exe"
    return exe_path