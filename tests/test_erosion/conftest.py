from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from dfastbe import __path__
from dfastbe.io.logger import LogData


@pytest.fixture(autouse=True)
def initialize_log_data() -> LogData:
    """Reset and seed the LogData singleton before each test.

    Several modules under test (e.g. ``bank_erosion.data_models.inputs``) call
    ``LogData()`` without a ``file_name``, which requires the singleton to
    already exist. Without this fixture, tests pass or fail depending on
    collection order and marker filters (e.g. ``pytest -m unit``).
    """
    LogData.reset()
    return LogData(Path(__path__[0]) / "io/log_data/messages.UK.ini")


@pytest.fixture
def shipping_dict() -> Dict[str, List[np.ndarray]]:
    """Fixture to provide mock shipping data for testing."""
    return {
        "velocity": [np.array([5.0, 5.0, 5.0]), np.array([5.0, 5.0, 5.0])],
        "number": [
            np.array([20912, 20912, 20912]),
            np.array([20912, 20912, 20912]),
        ],
        "num_waves": [np.array([5.0, 5.0, 5.0]), np.array([5.0, 5.0, 5.0])],
        "draught": [np.array([1.2, 1.2, 1.2]), np.array([1.2, 1.2, 1.2])],
        "type": [np.array([2.0, 2.0, 2.0]), np.array([2.0, 2.0, 2.0])],
        "slope": [np.array([20.0, 20.0, 20.0]), np.array([20.0, 20.0, 20.0])],
        "reed": [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])],
    }
