"""Erosion-related data structures."""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, ClassVar
import numpy as np
from geopandas import GeoDataFrame
from shapely.geometry import LineString
from dfastio.xyc.models import XYCModel
from dfastbe.io import ConfigFile, BaseRiverData, BaseSimulationData, log_text


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
        >>> from dfastbe.erosion.data_models import ErosionResults
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
        can be set to empty arrays if not needed.

        ```python
        >>> from dfastbe.erosion.data_models import ErosionResults
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


class ErosionSimulationData(BaseSimulationData):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_mesh_topology(self) -> MeshData:
        """Derive secondary topology arrays from the face-node connectivity of the mesh.

        This function computes the edge-node, edge-face, and face-edge connectivity arrays,
        as well as the boundary edges of the mesh, based on the face-node connectivity provided
        in the simulation data.

        Returns:
            MeshData: a dataclass containing the following attributes:
                - `x_face_coords`: x-coordinates of face nodes
                - `y_face_coords`: y-coordinates of face nodes
                - `x_edge_coords`: x-coordinates of edge nodes
                - `y_edge_coords`: y-coordinates of edge nodes
                - `face_node`: the node indices for each of the mesh faces.
                - `n_nodes`: number of nodes per face
                - `edge_node`: the node indices for each of the mesh edges.
                - `edge_face_connectivity`: the face indices for each of the mesh edge
                - `face_edge_connectivity`: the edge indices for each of the mesh face
                - `boundary_edge_nrs`: indices of boundary edges

        Raises:
            KeyError:
                If required keys (e.g., `face_node`, `nnodes`, `x_node`, `y_node`) are missing from the `sim` object.

        Notes:
            - The function identifies unique edges by sorting and comparing node indices.
            - Boundary edges are identified as edges that belong to only one face.
            - The function assumes that the mesh is well-formed, with consistent face-node connectivity.
        """

        # get a sorted list of edge node connections (shared edges occur twice)
        # face_nr contains the face index to which the edge belongs
        n_faces = self.face_node.shape[0]
        n_edges = sum(self.n_nodes)
        edge_node = np.zeros((n_edges, 2), dtype=int)
        face_nr = np.zeros((n_edges,), dtype=int)
        i = 0
        for face_i in range(n_faces):
            num_edges = self.n_nodes[face_i]  # note: nEdges = nNodes
            for edge_i in range(num_edges):
                if edge_i == 0:
                    edge_node[i, 1] = self.face_node[face_i, num_edges - 1]
                else:
                    edge_node[i, 1] = self.face_node[face_i, edge_i - 1]
                edge_node[i, 0] = self.face_node[face_i, edge_i]
                face_nr[i] = face_i
                i = i + 1
        edge_node.sort(axis=1)
        i2 = np.argsort(edge_node[:, 1], kind="stable")
        i1 = np.argsort(edge_node[i2, 0], kind="stable")
        i12 = i2[i1]
        edge_node = edge_node[i12, :]
        face_nr = face_nr[i12]

        # detect which edges are equal to the previous edge, and get a list of all unique edges
        numpy_true = np.array([True])
        equal_to_previous = np.concatenate(
            (~numpy_true, (np.diff(edge_node, axis=0) == 0).all(axis=1))
        )
        unique_edge = ~equal_to_previous
        n_unique_edges = np.sum(unique_edge)
        # reduce the edge node connections to only the unique edges
        edge_node = edge_node[unique_edge, :]

        # number the edges
        edge_nr = np.zeros(n_edges, dtype=int)
        edge_nr[unique_edge] = np.arange(n_unique_edges, dtype=int)
        edge_nr[equal_to_previous] = edge_nr[
            np.concatenate((equal_to_previous[1:], equal_to_previous[:1]))
        ]

        # if two consecutive edges are unique, the first one occurs only once and represents a boundary edge
        is_boundary_edge = unique_edge & np.concatenate((unique_edge[1:], numpy_true))
        boundary_edge_nrs = edge_nr[is_boundary_edge]

        # go back to the original face order
        edge_nr_in_face_order = np.zeros(n_edges, dtype=int)
        edge_nr_in_face_order[i12] = edge_nr
        # create the face edge connectivity array
        face_edge_connectivity = np.zeros(self.face_node.shape, dtype=int)

        i = 0
        for face_i in range(n_faces):
            num_edges = self.n_nodes[face_i]  # note: num_edges = n_nodes
            for edge_i in range(num_edges):
                face_edge_connectivity[face_i, edge_i] = edge_nr_in_face_order[i]
                i = i + 1

        # determine the edge face connectivity
        edge_face = -np.ones((n_unique_edges, 2), dtype=int)
        edge_face[edge_nr[unique_edge], 0] = face_nr[unique_edge]
        edge_face[edge_nr[equal_to_previous], 1] = face_nr[equal_to_previous]

        x_face_coords = self.apply_masked_indexing(
            self.x_node, self.face_node
        )
        y_face_coords = self.apply_masked_indexing(
            self.y_node, self.face_node
        )
        x_edge_coords = self.x_node[edge_node]
        y_edge_coords = self.y_node[edge_node]

        return MeshData(
            x_face_coords=x_face_coords,
            y_face_coords=y_face_coords,
            x_edge_coords=x_edge_coords,
            y_edge_coords=y_edge_coords,
            face_node=self.face_node,
            n_nodes=self.n_nodes,
            edge_node=edge_node,
            edge_face_connectivity=edge_face,
            face_edge_connectivity=face_edge_connectivity,
            boundary_edge_nrs=boundary_edge_nrs,
        )

    @staticmethod
    def apply_masked_indexing(
            x0: np.array, idx: np.ma.masked_array
    ) -> np.ma.masked_array:
        """
        Index one array by another transferring the mask.

        Args:
            x0 : np.ndarray
                A linear array.
            idx : np.ma.masked_array
                An index array with possibly masked indices.

        returns:
            x1: np.ma.masked_array
                An array with same shape as idx, with mask.
        """
        idx_safe = idx.copy()
        idx_safe.data[np.ma.getmask(idx)] = 0
        x1 = np.ma.masked_where(np.ma.getmask(idx), x0[idx_safe])
        return x1


class ErosionRiverData(BaseRiverData):

    def __init__(self, config_file: ConfigFile):
        super().__init__(config_file)
        self.bank_dir = self._get_bank_line_dir()
        self.output_dir = config_file.get_output_dir("erosion")
        self.debug = config_file.get_bool("General", "DebugOutput", False)
        # set plotting flags
        self.plot_flags = config_file.get_plotting_flags(config_file.root_dir)
        # get filter settings for bank levels and flow velocities along banks
        self.zb_dx = config_file.get_float("Erosion", "BedFilterDist", 0.0, positive=True)
        self.vel_dx = config_file.get_float("Erosion", "VelFilterDist", 0.0, positive=True)
        log_text("get_levels")
        self.num_discharge_levels = config_file.get_int("Erosion", "NLevel")
        self.output_intervals = config_file.get_float("Erosion", "OutputInterval", 1.0)
        self.bank_lines = config_file.read_bank_lines(str(self.bank_dir))
        self.river_axis = self._read_river_axis()

    def simulation_data(self) -> ErosionSimulationData:

        ref_level = self.config_file.get_int("Erosion", "RefLevel") - 1
        # read simulation data (get_sim_data)
        sim_file = self.config_file.get_sim_file("Erosion", str(ref_level + 1))
        log_text("-")
        log_text("read_simdata", data={"file": sim_file})
        log_text("-")
        simulation_data = ErosionSimulationData.read(sim_file)

        return simulation_data

    def _get_bank_output_dir(self) -> Path:
        bank_output_dir = self.config_file.get_str("General", "BankDir")
        log_text("bankdir_out", data={"dir": bank_output_dir})
        if os.path.exists(bank_output_dir):
            log_text("overwrite_dir", data={"dir": bank_output_dir})
        else:
            os.makedirs(bank_output_dir)

        return Path(bank_output_dir)

    def _get_bank_line_dir(self) -> Path:
        bank_dir = self.config_file.get_str("General", "BankDir")
        log_text("bankdir_in", data={"dir": bank_dir})
        bank_dir = Path(bank_dir)
        if not bank_dir.exists():
            log_text("missing_dir", data={"dir": bank_dir})
            raise BankLinesResultsError(
                f"Required bank line directory:{bank_dir} does not exist. please use the banklines command to run the "
                "bankline detection tool first it."
            )
        else:
            return bank_dir

    def _read_river_axis(self) -> LineString:
        """Get the river axis from the analysis settings."""
        river_axis_file = self.config_file.get_str("Erosion", "RiverAxis")
        log_text("read_river_axis", data={"file": river_axis_file})
        river_axis = XYCModel.read(river_axis_file)
        return river_axis

class BankLinesResultsError(Exception):
    """Custom exception for BankLine results errors."""

    pass