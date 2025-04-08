from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class ErosionInputs:
    """Class to hold erosion inputs."""

    ship_data: Dict[str, np.ndarray]
    dfw0: List[np.ndarray]
    dfw1: List[np.ndarray]
    zss: List[np.ndarray]
    tauc: List[np.ndarray]
    banktype: List[np.ndarray]
    taucls_str: List[str]


@dataclass
class WaterLevelData:
    hfw_max: float
    water_level: List[List[np.ndarray]]
    ship_wave_max: List[List[np.ndarray]]
    ship_wave_min: List[List[np.ndarray]]
