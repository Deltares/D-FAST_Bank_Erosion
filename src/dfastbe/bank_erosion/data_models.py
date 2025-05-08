"""Erosion-related data structures."""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator, List, Dict, Tuple, ClassVar, TypeVar, Generic, Any, Type, Optional, Union
import numpy as np
from geopandas import GeoDataFrame
from shapely.geometry import LineString, Point
from geopandas.geoseries import GeoSeries
from dfastio.xyc.models import XYCModel
from dfastbe.io import ConfigFile, BaseRiverData, BaseSimulationData, log_text


GenericType = TypeVar("GenericType")

@dataclass
class BaseBank(Generic[GenericType]):
    left: GenericType
    right: GenericType
    id: Optional[int] = field(default=None)

    def get_bank(self, bank_index: int) -> GenericType:
        if bank_index == 0:
            return self.left
        elif bank_index == 1:
            return self.right
        else:
            raise ValueError("bank_index must be 0 (left) or 1 (right)")

    @classmethod
    def from_column_arrays(
        cls: Type["BaseBank[GenericType]"],
        data: Dict[str, Any],
        bank_cls: Type[GenericType],
        bank_order: Tuple[str, str] = ("left", "right")
    ) -> "BaseBank[GenericType]":
        if set(bank_order) != {"left", "right"}:
            raise ValueError("bank_order must be a permutation of ('left', 'right')")

        id_val = data.get("id")

        # Extract the first and second array for each parameter (excluding id)
        first_args = {}
        second_args = {}
        for key, value in data.items():
            if key == "id":
                continue
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"Expected 2-column array for key '{key}', got shape {value.shape}")

            split = dict(zip(bank_order, value))
            first_args[key] = split["left"]
            second_args[key] = split["right"]

        left = bank_cls(**first_args)
        right = bank_cls(**second_args)

        return cls(id=id_val, left=left, right=right)

    def __iter__(self) -> Iterator[GenericType]:
        """Iterate over the banks."""
        return iter([self.left, self.right])


@dataclass
class SingleErosion:
    wave_fairway_distance_0: np.ndarray
    wave_fairway_distance_1: np.ndarray
    bank_protection_level: np.ndarray
    tauc: np.ndarray


@dataclass
class ErosionInputs(BaseBank[SingleErosion]):
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
    shipping_data: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    bank_type: np.ndarray = field(default_factory=lambda: np.array([]))
    taucls: ClassVar[np.ndarray] = np.array([1e20, 95, 3.0, 0.95, 0.15])
    taucls_str: ClassVar[Tuple[str]] = (
        "protected",
        "vegetation",
        "good clay",
        "moderate/bad clay",
        "sand",
    )

    @classmethod
    def from_column_arrays(
        cls, data: dict, bank_cls: Type["SingleErosion"], shipping_data: Dict[str, List[np.ndarray]],
        bank_type: np.ndarray, bank_order: Tuple[str, str] = ("left", "right")
    ) -> "ErosionInputs":
        # Only include fields that belong to the bank-specific data
        base_fields = {k: v for k, v in data.items() if k != "id"}
        base = BaseBank.from_column_arrays(
            {"id": data.get("id"), **base_fields}, bank_cls, bank_order=bank_order
        )

        return cls(
            id=base.id,
            left=base.left,
            right=base.right,
            shipping_data=shipping_data,
            bank_type=bank_type,
        )

    @property
    def bank_protection_level(self) -> List[np.ndarray]:
        """Get the bank protection level."""
        return [self.left.bank_protection_level, self.right.bank_protection_level]

    @property
    def tauc(self) -> List[np.ndarray]:
        """Get the critical bank shear stress values."""
        return [self.left.tauc, self.right.tauc]

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
        vol_per_discharge (List[List[np.ndarray]]):
            Eroded volume per discharge level for each bank line.
    """

    hfw_max: float
    water_level: List[List[np.ndarray]]
    ship_wave_max: List[List[np.ndarray]]
    ship_wave_min: List[List[np.ndarray]]
    velocity: List[List[np.ndarray]]
    bank_height: List[np.ndarray]
    chezy: List[List[np.ndarray]]
    vol_per_discharge: List[List[np.ndarray]]


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
class SingleBank:
    is_right_bank: bool
    bank_line_coords: np.ndarray
    bank_face_indices: np.ndarray
    bank_line_size: np.ndarray = field(default_factory=lambda: np.array([]))
    fairway_distances: np.ndarray = field(default_factory=lambda: np.array([]))
    fairway_face_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    bank_chainage_midpoints: np.ndarray = field(default_factory=lambda: np.array([]))

    segment_length: np.ndarray = field(init=False)
    dx: np.ndarray = field(init=False)
    dy: np.ndarray = field(init=False)
    length: int = field(init=False)

    def __post_init__(self):
        """Post-initialization to ensure bank_line_coords is a list of numpy arrays."""
        self.segment_length = self._segment_length()
        self.dx = self._dx()
        self.dy = self._dy()
        self.length = len(self.bank_chainage_midpoints)

    def _segment_length(self) -> np.ndarray:
        """Calculate the length of each segment in the bank line.

        Returns:
            List[np.ndarray]: Length of each segment in the bank line.
        """
        return np.linalg.norm(np.diff(self.bank_line_coords, axis=0), axis=1)

    def _dx(self) -> np.ndarray:
        """Calculate the distance between each bank line point.

        Returns:
            List[np.ndarray]: Distance to the closest fairway point for each bank line point.
        """
        return np.diff(self.bank_line_coords[:, 0])

    def _dy(self) -> np.ndarray:
        """Calculate the distance between each bank line point.

        Returns:
            List[np.ndarray]: Distance to the closest fairway point for each bank line point.
        """
        return np.diff(self.bank_line_coords[:, 1])

    def get_mid_points(self, crs) -> GeoSeries:
        bank_coords = self.bank_line_coords
        bank_coords_mind = (bank_coords[:-1] + bank_coords[1:]) / 2

        bank_coords_points = [Point(xy) for xy in bank_coords_mind]
        geo_series = GeoSeries(bank_coords_points, crs=crs)
        return geo_series

@dataclass
class BankData(BaseBank[SingleBank]):
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
    bank_lines: GeoDataFrame = field(default_factory=GeoDataFrame)
    n_bank_lines: int = 0

    @classmethod
    def from_column_arrays(
        cls,
        data: dict,
        bank_cls: Type["SingleBank"],
        bank_lines: GeoDataFrame,
        n_bank_lines: int,
        bank_order: Tuple[str, str] = ("left", "right")
    ) -> "BankData":
        # Only include fields that belong to the bank-specific data
        base_fields = {k: v for k, v in data.items() if k != "id"}
        base = BaseBank.from_column_arrays(
            {"id": data.get("id"), **base_fields}, bank_cls, bank_order=bank_order
        )

        return cls(
            id=base.id,
            left=base.left,
            right=base.right,
            bank_lines=bank_lines,
            n_bank_lines=n_bank_lines,
        )

    @property
    def bank_line_coords(self) -> List[np.ndarray]:
        """Get the coordinates of the bank lines."""
        return [self.left.bank_line_coords, self.right.bank_line_coords]

    @property
    def is_right_bank(self) -> List[bool]:
        """Get the bank direction."""
        return [self.left.is_right_bank, self.right.is_right_bank]

    @property
    def bank_chainage_midpoints(self) -> List[np.ndarray]:
        """Get the chainage midpoints of the bank lines."""
        return [self.left.bank_chainage_midpoints, self.right.bank_chainage_midpoints]

    @property
    def num_stations_per_bank(self) -> List[int]:
        """Get the number of stations per bank."""
        return [self.left.length, self.right.length]


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
        >>> from dfastbe.bank_erosion.data_models import ErosionResults
        >>> import numpy as np
        >>> erosion_results = ErosionResults(
        ...     eq_erosion_dist=[np.array([0.1, 0.2])],
        ...     total_erosion_dist=[np.array([0.3, 0.4])],
        ...     flow_erosion_dist=[np.array([0.5, 0.6])],
        ...     ship_erosion_dist=[np.array([0.7, 0.8])],
        ...     eq_eroded_vol=[np.array([1.1, 1.2])],
        ...     total_eroded_vol=[np.array([1.3, 1.4])],
        ...     erosion_time=10,
        ...     avg_erosion_rate=np.array([0.1, 0.2]),
        ...     eq_eroded_vol_per_km=np.array([0.3, 0.4]),
        ...     total_eroded_vol_per_km=np.array([0.5, 0.6]),
        ... )
        >>> print(erosion_results)
        ErosionResults(eq_erosion_dist=[array([0.1, 0.2])], total_erosion_dist=[array([0.3, 0.4])], flow_erosion_dist=[array([0.5, 0.6])], ship_erosion_dist=[array([0.7, 0.8])], eq_eroded_vol=[array([1.1, 1.2])], total_eroded_vol=[array([1.3, 1.4])], erosion_time=10, avg_erosion_rate=array([0.1, 0.2]), eq_eroded_vol_per_km=array([0.3, 0.4]), total_eroded_vol_per_km=array([0.5, 0.6]))

        ```

        - The `avg_erosion_rate`, `eq_eroded_vol_per_km`, and `total_eroded_vol_per_km` attributes are optional and
        can be set to empty arrays if not needed.

        ```python
        >>> from dfastbe.bank_erosion.data_models import ErosionResults
        >>> import numpy as np
        >>> erosion_results = ErosionResults(
        ...     eq_erosion_dist=[np.array([0.1, 0.2])],
        ...     total_erosion_dist=[np.array([0.3, 0.4])],
        ...     flow_erosion_dist=[np.array([0.5, 0.6])],
        ...     ship_erosion_dist=[np.array([0.7, 0.8])],
        ...     eq_eroded_vol=[np.array([1.1, 1.2])],
        ...     total_eroded_vol=[np.array([1.3, 1.4])],
        ...     erosion_time=10,
        ... )
        >>> print(erosion_results)
        ErosionResults(eq_erosion_dist=[array([0.1, 0.2])], total_erosion_dist=[array([0.3, 0.4])], flow_erosion_dist=[array([0.5, 0.6])], ship_erosion_dist=[array([0.7, 0.8])], eq_eroded_vol=[array([1.1, 1.2])], total_eroded_vol=[array([1.3, 1.4])], erosion_time=10, avg_erosion_rate=array([], dtype=float64), eq_eroded_vol_per_km=array([], dtype=float64), total_eroded_vol_per_km=array([], dtype=float64))

        ```
    """

    eq_erosion_dist: List[np.ndarray]
    total_erosion_dist: List[np.ndarray]
    flow_erosion_dist: List[np.ndarray]
    ship_erosion_dist: List[np.ndarray]
    eq_eroded_vol: List[np.ndarray]
    total_eroded_vol: List[np.ndarray]
    erosion_time: int
    avg_erosion_rate: np.ndarray = field(default_factory=lambda : np.empty(0))
    eq_eroded_vol_per_km: np.ndarray = field(default_factory=lambda : np.empty(0))
    total_eroded_vol_per_km: np.ndarray = field(default_factory=lambda : np.empty(0))


class ErosionSimulationData(BaseSimulationData):

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

    def calculate_bank_velocity(self, single_bank: SingleBank, vel_dx) -> np.ndarray:
        from dfastbe.bank_erosion.utils import moving_avg
        bank_face_indices = single_bank.bank_face_indices
        vel_bank = (
                np.abs(
                    self.velocity_x_face[bank_face_indices] * single_bank.dx
                    + self.velocity_y_face[bank_face_indices] * single_bank.dy
                )
                / single_bank.segment_length
        )

        if vel_dx > 0.0:
            vel_bank = moving_avg(
                single_bank.bank_chainage_midpoints, vel_bank, vel_dx
            )

        return vel_bank

    def calculate_bank_height(self, single_bank: SingleBank, zb_dx):
        from dfastbe.bank_erosion.utils import moving_avg
        bank_index = single_bank.bank_face_indices
        if self.bed_elevation_location == "node":
            zb_nodes = self.bed_elevation_values
            zb_all = self.apply_masked_indexing(
                zb_nodes, self.face_node[bank_index, :]
            )
            zb_bank = zb_all.max(axis=1)
            if zb_dx > 0.0:
                zb_bank = moving_avg(
                    single_bank.bank_chainage_midpoints, zb_bank, zb_dx,
                )
        else:
            # don't know ... need to check neighbouring cells ...
            zb_bank = None

        return zb_bank

class ErosionRiverData(BaseRiverData):

    def __init__(self, config_file: ConfigFile):
        super().__init__(config_file)
        self.bank_dir = self._get_bank_line_dir()
        self.output_dir = config_file.get_output_dir("erosion")
        self.debug = config_file.debug
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
        self.erosion_time = self.config_file.get_int("Erosion", "TErosion", positive=True)

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
        log_text("bank_dir_out", data={"dir": bank_output_dir})
        if os.path.exists(bank_output_dir):
            log_text("overwrite_dir", data={"dir": bank_output_dir})
        else:
            os.makedirs(bank_output_dir)

        return Path(bank_output_dir)

    def _get_bank_line_dir(self) -> Path:
        bank_dir = self.config_file.get_str("General", "BankDir")
        log_text("bank_dir_in", data={"dir": bank_dir})
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


@dataclass
class ParametersPerBank:
    ship_velocity: float
    num_ship: float
    num_waves_per_ship: float
    ship_draught: float
    ship_type: float
    par_slope: float
    par_reed: float
    mu_slope: float
    mu_reed: float


@dataclass
class DischargeLevelParameters(BaseBank[ParametersPerBank]):
    pass


@dataclass
class DischargeCalculationParameters:
    bank_velocity: np.ndarray = field(default=lambda : np.array([]))
    water_level: np.ndarray = field(default=lambda : np.array([]))
    chezy: np.ndarray = field(default=lambda : np.array([]))
    ship_wave_max: np.ndarray = field(default=lambda : np.array([]))
    ship_wave_min: np.ndarray = field(default=lambda : np.array([]))
    volume_per_discharge: np.ndarray = field(default=lambda : np.array([]))
    erosion_distance_flow: np.ndarray = field(default=lambda : np.array([]))
    erosion_distance_shipping: np.ndarray = field(default=lambda : np.array([]))
    erosion_distance_tot: np.ndarray = field(default=lambda : np.array([]))
    erosion_volume_tot: np.ndarray = field(default=lambda : np.array([]))
    erosion_distance_eq: Optional[np.ndarray] = field(default=lambda : np.array([]))
    erosion_volume_eq: Optional[np.ndarray] = field(default=lambda : np.array([]))


@dataclass
class CalculationLevel(BaseBank[DischargeCalculationParameters]):
    hfw_max: float = field(default=0.0)

    @classmethod
    def from_column_arrays(
        cls, data: dict, bank_cls: Type["DischargeCalculationParameters"], hfw_max: float,
        bank_order: Tuple[str, str] = ("left", "right")
    ) -> "CalculationLevel":
        # Only include fields that belong to the bank-specific data
        # base_fields = {k: v for k, v in data.items() if k != "id"}
        base = BaseBank.from_column_arrays(data, bank_cls, bank_order=bank_order)

        return cls(
            id=base.id,
            left=base.left,
            right=base.right,
            hfw_max=hfw_max,
        )

class DischargeLevels:

    def __init__(self, levels: List[CalculationLevel]):
        self.levels = levels

    def __getitem__(self, index: int) -> CalculationLevel:
        return self.levels[index]

    def __len__(self) -> int:
        return len(self.levels)

    def append(self, level_calc: CalculationLevel):
        self.levels.append(level_calc)

    def get_max_hfw_level(self) -> float:
        return max(level.hfw_max for level in self.levels)

    def total_erosion_volume(self) -> float:
        return sum(
            np.sum(level.left.erosion_volume_tot) + np.sum(level.right.erosion_volume_tot)
            for level in self.levels
        )

    def __iter__(self):
        return iter(self.levels)

    def accumulate(self, attribute_name: str, bank_side: Union[str, List[str]] = None) -> List[np.ndarray]:
        if bank_side is None:
            bank_side = ["left", "right"]
        elif isinstance(bank_side, str):
                bank_side = [bank_side]

        if not all(side in ["left", "right"] for side in bank_side):
            raise ValueError("bank_side must be 'left', 'right', or a list of these.")

        total = [
            self._accumulate_attribute_side(attribute_name, side) for side in bank_side
        ]
        return total

    def _accumulate_attribute_side(self, attribute_name: str, bank_side: str) -> np.ndarray:
        for i, level in enumerate(self.levels):
            bank = getattr(level, bank_side)
            attr = getattr(bank, attribute_name, None)
            if attr is None:
                raise AttributeError(f"{attribute_name} not found in {bank_side} bank of level with id={level.id}")
            if i == 0:
                total = attr
            else:
                total += attr
        return total

    def _get_attr_both_sides_level(self, attribute_name: str, level) -> List[np.ndarray]:
        """Get the attributes of the levels for both left and right bank."""
        sides = [getattr(self.levels[level], side) for side in ["left", "right"]]
        attr = [getattr(side, attribute_name, None) for side in sides]
        return attr

    def get_attr_level(self, attribute_name: str) -> List[List[np.ndarray]]:
        """Get the attributes of the levels for both left and right bank."""
        return [self._get_attr_both_sides_level(attribute_name, level) for level in range(len(self.levels))]

    def get_water_level_data(self, bank_height) -> WaterLevelData:
        return WaterLevelData(
            hfw_max=self.levels[-1].hfw_max,
            bank_height=bank_height,
            water_level=self.get_attr_level("water_level"),
            ship_wave_max=self.get_attr_level("ship_wave_max"),
            ship_wave_min=self.get_attr_level("ship_wave_min"),
            velocity=self.get_attr_level("bank_velocity"),
            chezy=self.get_attr_level("chezy"),
            vol_per_discharge=self.get_attr_level("volume_per_discharge"),
        )

class BankLinesResultsError(Exception):
    """Custom exception for BankLine results errors."""

    pass
