from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
from geopandas import GeoDataFrame

@dataclass
class ErosionInputs:
    """Class to hold erosion inputs."""

    ship_data: Dict[str, np.ndarray]
    wave_fairway_distance_0: List[np.ndarray]
    wave_fairway_distance_1: List[np.ndarray]
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
    bank_height: List[np.ndarray]
    chezy: List[List[np.ndarray]]


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
    edge_face_connectivity: np.ndarray
    face_edge_connectivity: np.ndarray
    boundary_edge_nrs: np.ndarray


@dataclass
class BankData:
    bank_idx: List[int]
    is_right_bank: List[bool]
    bank_km_mid: List[np.ndarray]
    bank_line_coords: List[np.ndarray]
    bank_lines: GeoDataFrame
    n_bank_lines: int
    xy_line_eq_list: List[np.ndarray] = np.array([])
    bank_type: List[np.ndarray] = np.array([])
