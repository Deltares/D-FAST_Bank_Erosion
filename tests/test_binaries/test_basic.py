import pytest
import os
import subprocess
from pathlib import Path


@pytest.mark.binaries
class TestBasic:
    def test_help_runs_correctly(self, exe_path: Path):
        """
        test getting the help message.
        """
        result = subprocess.run([exe_path, "--help"])

        assert result.returncode == 0, (
            f"Process returned exit code: {result.returncode}, please run the dfastbe.exe to find the specific error."
        )

    def test_compare_help_message(self, exe_path: Path):
        """
        Testing program help.
        """
        result = subprocess.run([exe_path, "--help"], capture_output=True)
        help_message = result.stdout.decode("UTF-8")
        assert "usage: dfastbe.exe" in help_message

    def test_basic_gui(self, exe_path: Path):
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