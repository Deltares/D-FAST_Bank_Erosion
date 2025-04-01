import os
import subprocess
import sys
from contextlib import contextmanager
from io import StringIO

import context

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


class Test_basic:
    def test_basic_00(self):
        """
        Test whether program runs at all.
        """
        cwd = os.getcwd()
        tstdir = "tests/data/bank_lines"
        success = False
        try:
            os.chdir(tstdir)
            result = subprocess.run([dfastexe, "--help"])
            success = result.returncode == 0
        finally:
            os.chdir(cwd)
        #
        self.maxDiff = None
        assert success == True

    def test_basic_01(self):
        """
        Testing program help.
        """
        cwd = os.getcwd()
        tstdir = "tests/data/bank_lines"
        try:
            os.chdir(tstdir)
            result = subprocess.run([dfastexe, "--help"], capture_output=True)
            outstr = result.stdout.decode("UTF-8").splitlines()
        finally:
            os.chdir(cwd)
        #
        self.maxDiff = None
        assert outstr == [
            "usage: dfastbe.exe [-h] [--language LANGUAGE] [--mode MODE] [--config CONFIG]",
            "",
            "D-FAST Bank Erosion.",
            "",
            "option:",
            "  -h, --help           show this help message and exit",
            "  --mode MODE          run mode 'BANKLINES', 'BANKEROSION' or 'GUI' (GUI is",
            "                       default)",
            "  --config CONFIG      name of configuration file ('dfastbe.cfg' is default)",
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
