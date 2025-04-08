from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class ErosionInputs:
    """Class to hold erosion inputs."""

    ship_data: Dict[str, np.ndarray]
    wave_0: List[np.ndarray]
    wave_1: List[np.ndarray]
    bank_protection_level: List[np.ndarray]
    tauc: List[np.ndarray]
    bank_type: List[np.ndarray]
    taucls_str: List[str]
