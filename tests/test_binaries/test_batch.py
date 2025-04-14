import os
import subprocess
import pytest
from pathlib import Path


@pytest.mark.binaries
class TestBatchMode:
    config_file = "meuse_manual.cfg"

    def test_bank_lines_error(self, exe_path: Path):
        """
        Testing batch_mode: missing configuration file.
        """
        cwd = os.getcwd()
        test_dir = "tests/data/bank_lines"
        try:
            os.chdir(test_dir)

            result = subprocess.run(
                [
                    exe_path,
                    "--mode",
                    "BANKLINES",
                    "--config",
                    "config.cfg",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            out_str = result.stdout.decode("UTF-8").splitlines()
        finally:
            os.chdir(cwd)

        assert out_str[-1] == "FileNotFoundError: The Config-File: config.cfg does not exist"

    def test_bank_lines(self, exe_path: Path):
        """
        Testing the bank line detection mode.
        """
        test_dir = "tests/data/bank_lines"
        cwd = os.getcwd()
        try:
            os.chdir(test_dir)
            result = subprocess.run(
                [
                    exe_path,
                    "--mode",
                    "BANKLINES",
                    "--config",
                    self.config_file,
                ],
                capture_output=True,
            )
            out_str = result.stdout.decode("UTF-8").splitlines()
        finally:
            os.chdir(cwd)

        success_string = "===    Bank line detection ended successfully!    ==="
        assert success_string in out_str

    def test_bank_erosion(self, exe_path: Path):
        """
        Testing the bank erosion mode.
        """
        test_dir = "tests/data/erosion"
        cwd = os.getcwd()
        try:
            os.chdir(test_dir)
            result = subprocess.run(
                [
                    exe_path,
                    "--mode",
                    "BANKEROSION",
                    "--config",
                    self.config_file,
                ],
                capture_output=True,
            )
            out_str = result.stdout.decode("UTF-8").splitlines()
        finally:
            os.chdir(cwd)

        success_string = "===   Bank erosion analysis ended successfully!   ==="
        assert success_string in out_str
