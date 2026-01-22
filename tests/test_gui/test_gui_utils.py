"""
Unit tests for GUI utility functions.

These tests cover pure functions that don't require Qt event loop or complex setup.
"""

import pytest
from dfastbe.gui.gui import gui_text, validator, shipTypes
from PyQt5.QtGui import QValidator, QDoubleValidator


class TestGuiText:
    """Test the gui_text function for text retrieval."""

    def test_gui_text_returns_string(self):
        """Test that gui_text returns the correct string."""
        result = gui_text("action_close")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_gui_text_with_custom_prefix(self):
        """Test gui_text with custom prefix."""
        # This will depend on your actual text dictionary
        result = gui_text("some_key", prefix="custom_")
        assert isinstance(result, str)
        assert result == "No message found for custom_some_key"


    def test_gui_text_with_format_dict(self):
        """Test gui_text with formatting text."""
        # Example: if your text has placeholders like {name}
        result = gui_text(key="read_param",
                          prefix= "",
                          placeholder_dict={"param": "Parameter", "file": "filename"})
        assert isinstance(result, str)
        assert result == "reading Parameter from file: filename"

    def test_gui_text_with_format_dict_no_format_possible(self):
        """Test gui_text with formatting text, but the string doesn't allow it."""
        # Example: if your text has placeholders like {name}
        result = gui_text(key="action_close",
                          placeholder_dict={"param": "Parameter", "file": "filename"})
        assert isinstance(result, str)
        assert result == "Close"


class TestValidator:
    """Test the validator function for input validation."""

    def test_validator_positive_real_returns_validator(self):
        """Test that validator returns a QValidator for positive_real."""
        val = validator("positive_real")
        assert val is not None
        assert isinstance(val, QDoubleValidator)

    def test_validator_raises_error(self):
        """Test that validator raises an error when it is unknown."""
        with pytest.raises(ValueError):
            validator("not_existing_validator")
