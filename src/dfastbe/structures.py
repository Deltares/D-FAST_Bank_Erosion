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


@dataclass
class WaterLevelData:
    """Class to hold water level data."""

    hfw_max: float
    water_level: List[List[np.ndarray]]
    ship_wave_max: List[List[np.ndarray]]
    ship_wave_min: List[List[np.ndarray]]
    velocity: List[List[np.ndarray]]


@dataclass
class MeshData:
    """Class to hold mesh-related data."""
    x_face_coords: np.ndarray
    y_face_coords: np.ndarray
    x_edge_coords: np.ndarray
    y_edge_coords: np.ndarray
    face_node: np.ndarray
    n_nodes: np.ndarray
    edge_node: np.ndarray
    edge_face: np.ndarray
    face_edge_connectivity: np.ndarray
    river_axis_km: np.ndarray = np.array([])
    bbox: np.ndarray = np.array([])
