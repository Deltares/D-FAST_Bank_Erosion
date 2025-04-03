import matplotlib
matplotlib.use("Agg")

import pytest
import sys
from unittest.mock import MagicMock

@pytest.fixture(autouse=True, scope="session")
def patch_matplotlib_use():
    sys.modules["matplotlib"].use = MagicMock()
