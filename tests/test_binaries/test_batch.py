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


class TestBatchMode:
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

    @pytest.mark.parametrize(
        "test_dir, config_file",
        [
            ("tests/data/bank_lines", "Meuse_manual.cfg"),
        ],
    )
    def test_bank_lines(self, test_dir, config_file):
        """
        Testing the bank line detection mode.
        """
        cwd = os.getcwd()
        try:
            os.chdir(test_dir)
            result = subprocess.run(
                [
                    exe_path,
                    "--mode",
                    "BANKLINES",
                    "--config",
                    config_file,
                ],
                capture_output=True,
            )
            out_str = result.stdout.decode("UTF-8").splitlines()
        finally:
            os.chdir(cwd)

        success_string = "===    Bank line detection ended successfully!    ==="
        assert success_string in out_str

    @pytest.mark.parametrize(
        "test_dir, config_file",
        [
            ("tests/data/erosion", "meuse_manual.cfg"),
        ],
    )
    def test_bank_erosion(self, test_dir, config_file):
        """
        Testing the bank erosion mode.
        """
        cwd = os.getcwd()
        try:
            os.chdir(test_dir)
            result = subprocess.run(
                [
                    exe_path,
                    "--mode",
                    "BANKEROSION",
                    "--config",
                    config_file,
                ],
                capture_output=True,
            )
            out_str = result.stdout.decode("UTF-8").splitlines()
        finally:
            os.chdir(cwd)

        success_string = "===   Bank erosion analysis ended successfully!   ==="
        assert success_string in out_str
