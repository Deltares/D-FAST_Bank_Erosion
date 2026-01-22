"""
Unit tests for GUI utility functions.

These tests cover pure functions that don't require Qt event loop or complex setup.
"""

import pytest
from dfastbe.gui.gui import gui_text, validator, shipTypes
from PyQt5.QtGui import QDoubleValidator


class TestGuiText:
    """Test the gui_text function for text retrieval."""
    @pytest.mark.parametrize("key,prefix,placeholder_dict,expected", [
        ("action_close", None, None, "Close"),
        ("some_key", "custom_", None, "No message found for custom_some_key"),
        ("read_param", "", {"param": "Parameter", "file": "filename"}, "reading Parameter from file: filename"),
        ("action_close", None, {"param": "Parameter", "file": "filename"}, "Close"),
    ],
    ids=[
        "existing_key_and_default_prefix",
        "non_existing_key_with_custom_prefix",
        "existing_key_with_placeholder_formatting",
        "existing_key_but_not_formattable",
    ])
    def test_gui_text(self, key, prefix, placeholder_dict, expected):
        """Test that gui_text returns the expected string."""
        kwargs = {"key": key}
        if prefix is not None:
            kwargs["prefix"] = prefix
        if placeholder_dict is not None:
            kwargs["placeholder_dict"] = placeholder_dict
        result = gui_text(**kwargs)
        assert result == expected


class TestValidator:
    """Test the validator function for input validation."""

    def test_validator_positive_real_returns_validator(self):
        """Test that validator returns a QDoubleValidator for positive_real."""
        val = validator("positive_real")
        assert val is not None
        assert isinstance(val, QDoubleValidator)

    def test_validator_raises_error(self):
        """Test that validator raises an error when it is unknown."""
        with pytest.raises(ValueError):
            validator("not_existing_validator")


class TestShipTypes:
    """Test the shipTypes function."""

    def test_amount_of_ship_types(self):
        """Test that shipTypes returns a list of 3 elements."""
        types = shipTypes()
        assert isinstance(types, list)
        assert len(types) == 3

    @pytest.mark.parametrize("index,expected", [
        (0, "1 (multiple barge convoy set)"),
        (1, "2 (RHK ship / motorship)"),
        (2, "3 (towboat)"),
    ],
    ids=[
        "first_ship_type",
        "second_ship_type",
        "third_ship_type",
    ])
    def test_ship_types_contains_expected_values(self, index, expected):
        """Test that shipTypes returns the correct ship type strings."""
        result = shipTypes()
        assert result[index] == expected
