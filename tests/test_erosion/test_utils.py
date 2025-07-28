import numpy as np
import pytest

from dfastbe.bank_erosion.utils import ErodedBankLine


@pytest.fixture
def folder_path():
    return "tests/data/input/meuse_cropped_data/move_line_right"


@pytest.fixture
def xylines(folder_path):
    return np.loadtxt(
        f"{folder_path}/xylines.txt", delimiter=","
    )


@pytest.fixture
def erosion_distance(folder_path):
    return np.loadtxt(f"{folder_path}/erosion_distance.txt", delimiter=",")

@pytest.fixture
def expected_moved_lines(folder_path):
    return np.loadtxt(f"{folder_path}/moved_lines.txt", delimiter=",")

def test_move_line_right(xylines, erosion_distance, expected_moved_lines):
    eroded_bank_line = ErodedBankLine(xylines, erosion_distance)
    moved_lines = eroded_bank_line.move_line_right()

    assert moved_lines.shape == (469, 2)
    assert np.allclose(moved_lines, expected_moved_lines)
