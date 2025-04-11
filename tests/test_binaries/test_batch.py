import os
import subprocess
import sys
from contextlib import contextmanager
from io import StringIO
import pytest
from pathlib import Path


# dfast binary path
repo_root = Path(__file__).resolve().parent.parent.parent
exe_path = repo_root / "dfastbe.dist/dfastbe.exe"


@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


@pytest.mark.binaries
class TestBatchMode:
    config_file = "meuse_manual.cfg"

    def test_batch_mode_00(self):
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

    def test_bank_lines(self):
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

    def test_bank_erosion(self):
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
