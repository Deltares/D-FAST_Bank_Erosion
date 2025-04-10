from dataclasses import dataclass, field
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
    """Class to hold water level data.
    args:
        zfw_ini (List[np.ndarray]): Initial water level.
        hfw_max (float): Maximum water level.
        water_level (List[List[np.ndarray]]): Water level data.
        ship_wave_max (List[List[np.ndarray]]): Maximum ship wave data.
        ship_wave_min (List[List[np.ndarray]]): Minimum ship wave data.
        velocity (List[List[np.ndarray]]): Velocity data.
        bank_height (List[np.ndarray]): Bank height data.
        chezy (List[List[np.ndarray]]): Chezy coefficient data.
    """
    hfw_max: float
    water_level: List[List[np.ndarray]]
    ship_wave_max: List[List[np.ndarray]]
    ship_wave_min: List[List[np.ndarray]]
    velocity: List[List[np.ndarray]]
    bank_height: List[np.ndarray]
    chezy: List[List[np.ndarray]]


@dataclass
class MeshData:
    """Class to hold mesh-related data.

    args:
        x_face_coords (np.ndarray): X-coordinates of the mesh faces.
        y_face_coords (np.ndarray): Y-coordinates of the mesh faces.
        x_edge_coords (np.ndarray): X-coordinates of the mesh edges.
        y_edge_coords (np.ndarray): Y-coordinates of the mesh edges.
        face_node (np.ndarray): Node connectivity for each face.
        n_nodes (np.ndarray): Number of nodes in the mesh.
        edge_node (np.ndarray): Node connectivity for each edge.
        edge_face_connectivity (np.ndarray): Edge-face connectivity matrix.
        face_edge_connectivity (np.ndarray): Face-edge connectivity matrix.
        boundary_edge_nrs (np.ndarray): Boundary edge numbers.
    """
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
    """Class to hold bank-related data.

    args:
        is_right_bank (List[bool]): List indicating if the bank is right or not.
        bank_km_mid (List[np.ndarray]): Midpoint coordinates of the banks.
        bank_line_coords (List[np.ndarray]): Coordinates of the bank lines.
        bank_face_indices (List[np.ndarray]): Indices of the faces associated with the banks.
        bank_lines (GeoDataFrame): GeoDataFrame containing the bank lines.
        n_bank_lines (int): Number of bank lines.
        bank_line_size (List[np.ndarray]): Size of each individual bank line.
    """
    is_right_bank: List[bool]
    bank_km_mid: List[np.ndarray]
    bank_line_coords: List[np.ndarray]
    bank_face_indices: List[np.ndarray]
    bank_lines: GeoDataFrame
    n_bank_lines: int
    bank_line_size: List[np.ndarray] = field(default_factory=list)


@dataclass
class FairwayData:
    """Class to hold fairway-related data.

    args:
        ifw_face_idx (np.ndarray): Index of the fairway faces.
        ifw_numpy (np.ndarray): Numpy array for fairway data.
        bp_fw_face_idx (np.ndarray): Index of the bank protection fairway faces.
        distance_fw (np.ndarray): Distance to the fairway.
        zfw_ini (List[np.ndarray]): Initial water level in the fairway.
    """
    ifw_face_idx: np.ndarray
    ifw_numpy: np.ndarray
    bp_fw_face_idx: np.ndarray
    distance_fw: np.ndarray
    zfw_ini: List[np.ndarray]
