import sys
from contextlib import contextmanager
from io import StringIO

import pytest
from pathlib import Path
from dfastbe.io.logger import LogData


@pytest.fixture
def log_data() -> LogData:
    return LogData(Path("tests/data/files/messages.UK.ini"))


@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err

class TestLogText:
    def test_log_text_01(self, log_data: LogData):
        """
        Testing standard output of a single text without expansion.
        """
        key = "confirm"
        with captured_output() as (out, err):
            log_data.log_text(key)
        outstr = out.getvalue().splitlines()
        strref = ['Confirm using "y" ...', '']
        assert outstr == strref

    def test_log_text_02(self, log_data: LogData):
        """
        Testing standard output of a repeated text without expansion.
        """
        key = ""
        nr = 3
        with captured_output() as (out, err):
            log_data.log_text(key, repeat=nr)
        outstr = out.getvalue().splitlines()
        strref = ['', '', '']
        assert outstr == strref

    def test_log_text_03(self, log_data: LogData):
        """
        Testing standard output of a text with expansion.
        """
        key = "reach"
        data = {"reach": "ABC"}
        with captured_output() as (out, err):
            log_data.log_text(key, data=data)
        outstr = out.getvalue().splitlines()
        strref = ['The measure is located on reach ABC']
        assert outstr == strref

    def test_log_text_04(self, log_data: LogData):
        """
        Testing file output of a text with expansion.
        """
        key = "reach"
        data = {"reach": "ABC"}
        filename = "test.log"
        with open(filename, "w") as f:
            log_data.log_text(key, data=data, file=f)
        all_lines = open(filename, "r").read().splitlines()
        strref = ['The measure is located on reach ABC']
        assert all_lines == strref


class TestGetText:
    def test_get_text_01(self, log_data: LogData):
        """
        Testing get_text: key not found.
        """
        assert log_data.get_text("@") == ["No message found for @"]

    def test_get_text_02(self, log_data: LogData):
        """
        Testing get_text: empty line key.
        """
        assert log_data.get_text("") == [""]

    def test_get_text_03(self, log_data: LogData):
        """
        Testing get_text: "confirm" key.
        """
        assert log_data.get_text("confirm") == ['Confirm using "y" ...', '']