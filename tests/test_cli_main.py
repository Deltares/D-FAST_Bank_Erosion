import sys

import matplotlib

matplotlib.use("Agg")
from unittest.mock import patch

import pytest

# Patch before importing dfastbe.__main__ to prevent Qt5Agg in headless CI
with patch("matplotlib.use", lambda *args, **kwargs: None):
    from dfastbe.__main__ import main, parse_arguments


@pytest.mark.parametrize(
    "args, expected",
    [
        (["prog"], ("UK", "GUI", "dfastbe.cfg", False)),
        (["prog", "--language", "NL"], ("NL", "GUI", "dfastbe.cfg", False)),
        (["prog", "--mode", "BANKLINES"], ("UK", "BANKLINES", "dfastbe.cfg", False)),
        (["prog", "--config", "custom.cfg"], ("UK", "GUI", "custom.cfg", False)),
        (["prog", "--debug"], ("UK", "GUI", "dfastbe.cfg", True)),
        (["prog", "--language", "UK", "--debug"], ("UK", "GUI", "dfastbe.cfg", True)),
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
            ("NL", "BANKEROSION", "custom.cfg", False),
        ),
    ],
    ids=[
        "default_args",
        "language_nl",
        "mode_banklines",
        "config_custom",
        "debug",
        "language_uk_debug",
        "language_nl_mode_bankerosion_config_custom",
    ],
)
def test_parse_arguments_valid(args, expected, monkeypatch):
    monkeypatch.setattr(sys, "argv", args)
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
    mock_run = lambda lang, mode, cfg: print(f"Mock run with {lang}, {mode}, {cfg}")
    monkeypatch.setattr("dfastbe.__main__.run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--language", "UK", "--mode", "GUI", "--config", "cfg.ini"],
    )
    main()
