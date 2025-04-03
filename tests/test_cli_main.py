import pytest
from unittest.mock import patch
import sys
from dfastbe.__main__ import parse_arguments, main


@pytest.mark.parametrize("args, expected", [
    (["prog"], ("UK", "GUI", "dfastbe.cfg")),
    (["prog", "--language", "NL"], ("NL", "GUI", "dfastbe.cfg")),
    (["prog", "--mode", "BANKLINES"], ("UK", "BANKLINES", "dfastbe.cfg")),
    (["prog", "--config", "custom.cfg"], ("UK", "GUI", "custom.cfg")),
    (["prog", "--language", "NL", "--mode", "BANKEROSION", "--config", "custom.cfg"],
     ("NL", "BANKEROSION", "custom.cfg")),
])
def test_parse_arguments_valid(args, expected, monkeypatch):
    monkeypatch.setattr(sys, "argv", args)
    assert parse_arguments() == expected


@pytest.mark.parametrize("args", [
    (["prog", "--language", "DE"]),
    (["prog", "--mode", "INVALID"]),
])
def test_parse_arguments_invalid(args, monkeypatch):
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit):  # argparse throws SystemExit on invalid args
        parse_arguments()


def test_main_runs(monkeypatch):
    mock_run = lambda lang, mode, cfg: print(f"Mock run with {lang}, {mode}, {cfg}")
    monkeypatch.setattr("dfastbe.__main__.run", mock_run)
    monkeypatch.setattr(sys, "argv", ["prog", "--language", "UK", "--mode", "GUI", "--config", "cfg.ini"])
    main()
