import os
import subprocess
import sys
from contextlib import contextmanager
from io import StringIO

# dfast binary path relative to tstdir
dfastexe = "../../../dfastbe.dist/dfastbe.exe"


@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


class TestBasic:
    def test_basic_00(self):
        """
        test getting the help message.
        """
        result = subprocess.run([dfastexe, "--help"])
        success = result.returncode == 0

        assert success == True

    def test_compare_help_message(self):
        """
        Testing program help.
        """
        result = subprocess.run([dfastexe, "--help"], capture_output=True)
        help_message = result.stdout.decode("UTF-8").splitlines()

        assert help_message == [
            "usage: dfastbe.exe [-h] [--language {NL,UK}] [--mode {BANKLINES,BANKEROSION,GUI}] [--config CONFIG]",
            "",
            "D-FAST Bank Erosion. Example: python -m dfastbe --mode BANKEROSION --config settings.cfg",
            "",
            "optional arguments:",
            "  -h, --help       show this help message and exit",
            "  --language {NL,UK}    display language 'NL' or 'UK' ('UK' is default)"
            "  --mode {BANKLINES,BANKEROSION,GUI}",
            "                        run mode 'BANKLINES', 'BANKEROSION' or 'GUI' (GUI is default",
            "  --config CONFIG       name of the configuration file ('dfastbe.cfg' is default)",
        ]

    def test_basic_gui(self):
        """
        Testing startup of the GUI.
        GUI will be started and closed.
        If GUI does not start test will fail.
        """
        cwd = os.getcwd()
        tstdir = "tests/data/bank_lines"
        try:
            os.chdir(tstdir)
            try:
                process = subprocess.Popen(
                    dfastexe, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                process.wait(timeout=1)

            except subprocess.TimeoutExpired:
                process.kill()

            assert (
                process.returncode is None
            ), f"Process returned exit code: {process.returncode}, please run the dfastbe.exe to find the specific error."
        finally:
            os.chdir(cwd)
