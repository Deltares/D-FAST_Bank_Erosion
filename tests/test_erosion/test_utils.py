import numpy as np
import pytest

from dfastbe.bank_erosion.utils import ErodedBankLine


class TestErodedBankLine:
    """Test suite for the ErodedBankLine class."""

    @pytest.fixture
    def folder_path(self):
        """Fixture to provide the path to the test data folder."""
        return "tests/data/input/meuse_cropped_data/move_line_right"

    @pytest.fixture
    def xylines(self, folder_path):
        """Fixture to provide the xylines data."""
        return np.loadtxt(f"{folder_path}/xylines.txt", delimiter=",")

    @pytest.fixture
    def erosion_distance(self, folder_path):
        """Fixture to provide the erosion distance data."""
        return np.loadtxt(f"{folder_path}/erosion_distance.txt", delimiter=",")

    @pytest.fixture
    def expected_moved_lines(self, folder_path):
        """Fixture to provide the expected moved lines data."""
        return np.loadtxt(f"{folder_path}/moved_lines.txt", delimiter=",")

    @pytest.mark.integration
    def test_move_line_by_erosion(
        self, xylines, erosion_distance, expected_moved_lines
    ):
        """Test the move_line_by_erosion method of ErodedBankLine.

        Args:
            xylines (np.ndarray): The input xylines data.
            erosion_distance (np.ndarray): The erosion distance data.
            expected_moved_lines (np.ndarray): The expected moved lines data.

        Asserts:
            The shape of the moved lines matches the expected shape.
            The moved lines are close to the expected moved lines within a tolerance.
        """
        eroded_bank_line = ErodedBankLine(xylines, erosion_distance)
        moved_lines = eroded_bank_line.move_line_by_erosion()

        assert moved_lines.shape == expected_moved_lines.shape
        assert np.allclose(moved_lines, expected_moved_lines)
