import pytest
import os
import subprocess
import sys
from contextlib import contextmanager
from io import StringIO
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
class TestBasic:
    def test_basic_00(self):
        """
        test getting the help message.
        """
        result = subprocess.run([exe_path, "--help"])
        success = result.returncode == 0

        assert success == True

    def test_compare_help_message(self):
        """
        Testing program help.
        """
        result = subprocess.run([exe_path, "--help"], capture_output=True)
        help_message = result.stdout.decode("UTF-8").splitlines()

        assert "usage: dfastbe.exe" in help_message

    def test_basic_gui(self):
        """
        Testing startup of the GUI.
        GUI will be started and closed.
        If GUI does not start test will fail.
        """
        cwd = os.getcwd()
        test_dir = "tests/data/bank_lines"

        try:
            os.chdir(test_dir)
            try:
                process = subprocess.Popen(
                    exe_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()

            assert (
                process.returncode is None
            ), f"Process returned exit code: {process.returncode}, please run the dfastbe.exe to find the specific error."

        finally:
            os.chdir(cwd)