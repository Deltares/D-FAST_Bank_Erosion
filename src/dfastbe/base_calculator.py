"""Base Calculator Class."""

from typing import Dict, Any
from dfastbe.io.config import ConfigFile


class BaseCalculator:
    """BaseCalculator Class.

    Base class for calculators.
    This class provides a common interface for all calculators.
    """

    def __init__(self, config_file: ConfigFile, gui: bool = False):
        """Initialize the BaseCalculator."""
        self.root_dir = config_file.root_dir
        self._config_file = config_file
        self.gui = gui
        self._results = None

    @property
    def config_file(self) -> ConfigFile:
        """Configuration file object."""
        return self._config_file

    @property
    def results(self) -> Dict[str, Any]:
        """dict: Results of the bank line detection analysis.

        Returns:
            dict: A dictionary containing the results of the bank erosion analysis.
            The keys and values depend on the specific implementation of the calculator.
            Erosion:
                Returns:
                bank (List[LineString]):
                    List of bank lines.
                banklines (GeoSeries):
                    Un-ordered set of bank line segments.
                masked_bank_lines (List[MultiLineString]):
                    Un-ordered set of bank line segments, clipped to bank area.
                bank_areas (List[Polygon]):
                    A search area corresponding to one of the bank search lines.
        """
        return self._results

    @results.setter
    def results(self, value: Dict[str, Any]):
        """Set the results of the bank erosion analysis."""
        self._results = value

    def __str__(self):
        """String representation of the calculator."""
        return f"Calculator: {self.name}"
