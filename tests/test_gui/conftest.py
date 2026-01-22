"""
Shared fixtures for GUI testing.
"""

import pytest

from pathlib import Path
from dfastbe import __path__
from dfastbe.io.logger import LogData


@pytest.fixture(autouse=True)
def initialize_log_data():
    """
    Initialize LogData singleton with messages file before each test.

    This fixture automatically runs before each test in this directory,
    ensuring that LogData is properly initialized and reset between tests.
    """
    # Reset LogData to ensure clean state
    LogData.reset()

    return LogData(Path(__path__[0]) / "io/log_data/messages.UK.ini")





