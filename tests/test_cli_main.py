import sys

import matplotlib

matplotlib.use("Agg")
from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

# Patch before importing dfastbe.__main__ to prevent Qt5Agg in headless CI
with patch("matplotlib.use", lambda *args, **kwargs: None):
    from dfastbe.__main__ import main, parse_arguments


@pytest.mark.parametrize(
    "args, expected, success",
    [
        (["prog"], ("UK", "GUI", None), True),
        (["prog", "--language", "NL"], ("NL", "GUI", None), True),
        (["prog", "--mode", "BANKLINES"], ("UK", "BANKLINES", None), False),
        (["prog", "--config", "custom.cfg"], ("UK", "GUI", Path("custom.cfg")), True),
        (
            [
                "prog",
                "--language",
                "NL",
                "--mode",
                "BANKEROSION",
                "--config",
                "custom.cfg",
            ],
            ("NL", "BANKEROSION", Path("custom.cfg")),
            True,
        ),
    ],
)
def test_parse_arguments_valid(args, expected, success, monkeypatch):
    monkeypatch.setattr(sys, "argv", args)

    with patch("pathlib.Path.exists", return_value=True):
        if not success:
            with pytest.raises(SystemExit):
                assert parse_arguments() == expected
        else:
            assert parse_arguments() == expected


@pytest.mark.parametrize(
    "args",
    [
        (["prog", "--language", "DE"]),
        (["prog", "--mode", "INVALID"]),
    ],
)
def test_parse_arguments_invalid(args, monkeypatch):
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit):  # argparse throws SystemExit on invalid args
        parse_arguments()


def test_main_runs(monkeypatch):
    mock_runner = MagicMock()
    mock_runner.run = MagicMock()
    mock_run = MagicMock(return_value=mock_runner)
    monkeypatch.setattr("dfastbe.__main__.Runner", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--language", "UK", "--mode", "GUI", "--config", "cfg.ini"],
    )
    with patch("pathlib.Path.exists", return_value=True):
        main()
    mock_run.assert_called_once_with("UK", "GUI", Path("cfg.ini"))
    mock_runner.run.assert_called_once_with()
