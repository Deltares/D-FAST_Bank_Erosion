import numpy as np

from dfastbe.bank_erosion.data_models import (
    BankData,
    ErosionInputs,
    ErosionResults,
    WaterLevelData,
)


class ErosionPlotter:
    """class to plot the results of the bank erosion analysis."""

    def __init__(
        self,
        erosion_results: ErosionResults,
        bank_data: BankData,
        water_level_data: WaterLevelData,
        erosion_inputs: ErosionInputs,
        midpoint_chainages: np.ndarray,
    ):
        """Initialize the ErosionPlotter with the required data.
        
        Args:
            erosion_results (ErosionResults):
                The results of the erosion analysis.
            bank_data (BankData):
                The bank data used in the analysis.
            water_level_data (WaterLevelData):
                The water level data used in the analysis.
            erosion_inputs (ErosionInputs):
                The inputs for the erosion analysis.
            midpoint_chainages (np.ndarray):
                The midpoint chainages for the analysis.
        """
        self._erosion_results = erosion_results
        self._bank_data = bank_data
        self._water_level_data = water_level_data
        self._erosion_inputs = erosion_inputs
        self._midpoint_chainages = midpoint_chainages
