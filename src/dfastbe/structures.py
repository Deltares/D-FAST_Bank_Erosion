from dataclasses import dataclass
from typing import List, Dict, Tuple
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
    taucls: np.array = np.array([1e20, 95, 3.0, 0.95, 0.15])
    taucls_str: Tuple[str] = (
        "protected",
        "vegetation",
        "good clay",
        "moderate/bad clay",
        "sand",
    )
