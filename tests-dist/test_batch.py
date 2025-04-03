import os
import subprocess
import sys
from contextlib import contextmanager
from io import StringIO

import context
import netCDF4
import numpy
import pytest

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


class Test_batch_mode:
    def test_batch_mode_00(self):
        """
        Testing batch_mode: missing configuration file.
        """
        cwd = os.getcwd()
        tstdir = "tests/data/bank_lines"
        try:
            os.chdir(tstdir)
            result = subprocess.run(
                [
                    dfastexe,
                    "--mode",
                    "BANKLINES",
                    "--config",
                    "config.cfg",
                ],
                capture_output=True,
            )
            outstr = result.stdout.decode("UTF-8").splitlines()
        finally:
            os.chdir(cwd)
        #
        # for s in outstr:
        #    print(s)
        self.maxDiff = None
        assert outstr[-1] == "FileNotFoundError: The Config-File: config.cfg does not exist"

    @pytest.mark.parametrize(
        "tstdir, cfgfile",
        [
            ("tests/data/bank_lines", "Meuse_manual.cfg"),
        ],
    )
    def test_bank_lines(self, tstdir, cfgfile):
        """
        Testing the bank line detection mode.
        """
        cwd = os.getcwd()
        try:
            os.chdir(tstdir)
            result = subprocess.run(
                [
                    dfastexe,
                    "--mode",
                    "BANKLINES",
                    "--config",
                    cfgfile,
                ],
                capture_output=True,
            )
            outstr = result.stdout.decode("UTF-8").splitlines()
        finally:
            os.chdir(cwd)
        #
        # for s in outstr:
        #    print(s)
        success_string = "===    Bank line detection ended successfully!    ==="
        assert success_string in outstr

    @pytest.mark.parametrize(
        "tstdir, cfgfile",
        [
            ("tests/data/erosion", "meuse_manual.cfg"),
        ],
    )
    def test_bank_erosion(self, tstdir, cfgfile):
        """
        Testing the bank erosion mode.
        """
        cwd = os.getcwd()
        try:
            os.chdir(tstdir)
            result = subprocess.run(
                [
                    dfastexe,
                    "--mode",
                    "BANKEROSION",
                    "--config",
                    cfgfile,
                ],
                capture_output=True,
            )
            outstr = result.stdout.decode("UTF-8").splitlines()
        finally:
            os.chdir(cwd)
        #
        # for s in outstr:
        #    print(s)
        success_string = "===   Bank erosion analysis ended successfully!   ==="
        assert success_string in outstr
