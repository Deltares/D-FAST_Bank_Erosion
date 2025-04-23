"""Erosion-related data structures."""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, ClassVar
import numpy as np
from geopandas import GeoDataFrame


@dataclass
class ErosionInputs:
    """Class to hold erosion inputs.

    args:
        shipping_data (Dict[str, np.ndarray]):
            Data on all the vessels that travel through the river.
        wave_fairway_distance_0 (List[np.ndarray]):
            Threshold fairway distance 0 for wave attenuation.
        wave_fairway_distance_1 (List[np.ndarray]):
            Threshold fairway distance 1 for wave attenuation.
        bank_protection_level (List[np.ndarray]):
            Bank protection level.
        tauc (List[np.ndarray]):
            Critical bank shear stress values.
        bank_type (List[np.ndarray]):
            Integer representation of the bank type. Represents an index into the taucls_str array.
        taucls (np.ndarray):
            Critical bank shear stress values for different bank types.
        taucls_str (Tuple[str]):
            String representation for different bank types.
    """

    shipping_data: Dict[str, np.ndarray]
    wave_fairway_distance_0: List[np.ndarray]
    wave_fairway_distance_1: List[np.ndarray]
    bank_protection_level: List[np.ndarray]
    tauc: List[np.ndarray]
    bank_type: List[np.ndarray]
    taucls: ClassVar[np.ndarray] = np.array([1e20, 95, 3.0, 0.95, 0.15])
    taucls_str: ClassVar[Tuple[str]] = (
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
        hfw_max (float): Maximum water depth along the fairway.
        water_level (List[List[np.ndarray]]): Water level data.
        ship_wave_max (List[List[np.ndarray]]): Maximum bank height subject to ship waves [m]
        ship_wave_min (List[List[np.ndarray]]): Minimum bank height subject to ship waves [m]
        velocity (List[List[np.ndarray]]): Flow velocity magnitude along the bank [m/s]
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
        x_face_coords (np.ndarray):
            X-coordinates of the mesh faces.
        y_face_coords (np.ndarray):
            Y-coordinates of the mesh faces.
        x_edge_coords (np.ndarray):
            X-coordinates of the mesh edges.
        y_edge_coords (np.ndarray):
            Y-coordinates of the mesh edges.
        face_node (np.ndarray):
            Node connectivity for each face.
        n_nodes (np.ndarray):
            Number of nodes in the mesh.
        edge_node (np.ndarray):
            Node connectivity for each edge.
        edge_face_connectivity (np.ndarray):
            Per edge a list of the indices of the faces on the left and right side of that edge.
        face_edge_connectivity (np.ndarray):
            Per face a list of indices of the edges that together form the boundary of that face.
        boundary_edge_nrs (np.ndarray):
            List of edge indices that together form the boundary of the whole mesh.
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
        is_right_bank (List[bool]):
            List indicating if the bank is right or not.
        bank_chainage_midpoints (List[np.ndarray]):
            River chainage for the midpoints of each segment of the bank line
        bank_line_coords (List[np.ndarray]):
            Coordinates of the bank lines.
        bank_face_indices (List[np.ndarray]):
            Indices of the faces associated with the banks.
        bank_lines (GeoDataFrame):
            GeoDataFrame containing the bank lines.
        n_bank_lines (int):
            Number of bank lines.
        bank_line_size (List[np.ndarray]):
            Size of each individual bank line.
        fairway_distances (List[np.ndarray]):
            The distance of each bank line point to the closest fairway point.
        fairway_face_indices (List[np.ndarray]):
            The face index of the closest fairway point for each bank line point.
    """

    is_right_bank: List[bool]
    bank_chainage_midpoints: List[np.ndarray]
    bank_line_coords: List[np.ndarray]
    bank_face_indices: List[np.ndarray]
    bank_lines: GeoDataFrame
    n_bank_lines: int
    bank_line_size: List[np.ndarray] = field(default_factory=list)
    fairway_distances: List[np.ndarray] = field(default_factory=list)
    fairway_face_indices: List[np.ndarray] = field(default_factory=list)


@dataclass
class FairwayData:
    """Class to hold fairway-related data.

    args:
        fairway_face_indices (np.ndarray):
            Mesh face indices matching to the fairway points.
        intersection_coords (np.ndarray):
            The x, y coordinates of the intersection points of the fairway with the simulation mesh.
        fairway_initial_water_levels (List[np.ndarray]):
            Reference water level at the fairway
    """

    fairway_face_indices: np.ndarray
    intersection_coords: np.ndarray
    fairway_initial_water_levels: List[np.ndarray] = field(default_factory=list)


@dataclass
class ErosionResults:
    """Class to hold erosion results.

    args:
        eq_erosion_dist (List[np.ndarray]):
            Erosion distance at equilibrium for each bank line.
        total_erosion_dist (List[np.ndarray]):
            Total erosion distance for each bank line.
        flow_erosion_dist (List[np.ndarray]):
            Total erosion distance caused by flow for each bank line.
        ship_erosion_dist (List[np.ndarray]):
            Total erosion distance caused by ship waves for each bank line.
        vol_per_discharge (List[List[np.ndarray]]):
            Eroded volume per discharge level for each bank line.
        eq_eroded_vol (List[np.ndarray]):
            Eroded volume at equilibrium for each bank line.
        total_eroded_vol (List[np.ndarray]):
            Total eroded volume for each bank line.
        erosion_time (int):
            Time over which erosion is calculated.
        avg_erosion_rate (np.ndarray):
            Average erosion rate data.
        eq_eroded_vol_per_km (np.ndarray):
            Equilibrium eroded volume calculated per kilometer bin.
        total_eroded_vol_per_km (np.ndarray):
            Total eroded volume calculated per kilometer bin.

    Examples:
        - You can create an instance of the ErosionResults class as follows:
        ```python
        >>> from dfastbe.erosion.structures import ErosionResults
        >>> import numpy as np
        >>> erosion_results = ErosionResults(
        ...     eq_erosion_dist=[np.array([0.1, 0.2])],
        ...     total_erosion_dist=[np.array([0.3, 0.4])],
        ...     flow_erosion_dist=[np.array([0.5, 0.6])],
        ...     ship_erosion_dist=[np.array([0.7, 0.8])],
        ...     vol_per_discharge=[[np.array([0.9, 1.0])]],
        ...     eq_eroded_vol=[np.array([1.1, 1.2])],
        ...     total_eroded_vol=[np.array([1.3, 1.4])],
        ...     erosion_time=10,
        ...     avg_erosion_rate=np.array([0.1, 0.2]),
        ...     eq_eroded_vol_per_km=np.array([0.3, 0.4]),
        ...     total_eroded_vol_per_km=np.array([0.5, 0.6]),
        ... )
        >>> print(erosion_results)
        ErosionResults(eq_erosion_dist=[array([0.1, 0.2])], total_erosion_dist=[array([0.3, 0.4])], flow_erosion_dist=[array([0.5, 0.6])], ship_erosion_dist=[array([0.7, 0.8])], vol_per_discharge=[[array([0.9, 1. ])]], eq_eroded_vol=[array([1.1, 1.2])], total_eroded_vol=[array([1.3, 1.4])], erosion_time=10, avg_erosion_rate=array([0.1, 0.2]), eq_eroded_vol_per_km=array([0.3, 0.4]), total_eroded_vol_per_km=array([0.5, 0.6]))

        ```

        - The `avg_erosion_rate`, `eq_eroded_vol_per_km`, and `total_eroded_vol_per_km` attributes are optional and
        can be set to empty arrays if not needed, .

        ```python
        >>> from dfastbe.erosion.structures import ErosionResults
        >>> import numpy as np
        >>> erosion_results = ErosionResults(
        ...     eq_erosion_dist=[np.array([0.1, 0.2])],
        ...     total_erosion_dist=[np.array([0.3, 0.4])],
        ...     flow_erosion_dist=[np.array([0.5, 0.6])],
        ...     ship_erosion_dist=[np.array([0.7, 0.8])],
        ...     vol_per_discharge=[[np.array([0.9, 1.0])]],
        ...     eq_eroded_vol=[np.array([1.1, 1.2])],
        ...     total_eroded_vol=[np.array([1.3, 1.4])],
        ...     erosion_time=10,
        ... )
        >>> print(erosion_results)
        ErosionResults(eq_erosion_dist=[array([0.1, 0.2])], total_erosion_dist=[array([0.3, 0.4])], flow_erosion_dist=[array([0.5, 0.6])], ship_erosion_dist=[array([0.7, 0.8])], vol_per_discharge=[[array([0.9, 1. ])]], eq_eroded_vol=[array([1.1, 1.2])], total_eroded_vol=[array([1.3, 1.4])], erosion_time=10, avg_erosion_rate=array([], dtype=float64), eq_eroded_vol_per_km=array([], dtype=float64), total_eroded_vol_per_km=array([], dtype=float64))

        ```
    """

    eq_erosion_dist: List[np.ndarray]
    total_erosion_dist: List[np.ndarray]
    flow_erosion_dist: List[np.ndarray]
    ship_erosion_dist: List[np.ndarray]
    vol_per_discharge: List[List[np.ndarray]]
    eq_eroded_vol: List[np.ndarray]
    total_eroded_vol: List[np.ndarray]
    erosion_time: int
    avg_erosion_rate: np.ndarray = field(default_factory=lambda : np.empty(0))
    eq_eroded_vol_per_km: np.ndarray = field(default_factory=lambda : np.empty(0))
    total_eroded_vol_per_km: np.ndarray = field(default_factory=lambda : np.empty(0))
