"""Erosion-related data structures."""
from dataclasses import dataclass, field
from typing import (
    Any,
    ClassVar,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
from geopandas import GeoDataFrame
from geopandas.geoseries import GeoSeries
from shapely.geometry import Point

GenericType = TypeVar("GenericType")

__all__ = [
    "BaseBank",
    "SingleErosion",
    "ErosionInputs",
    "WaterLevelData",
    "MeshData",
    "SingleBank",
    "BankData",
    "FairwayData",
    "ErosionResults",
    "SingleParameters",
    "SingleLevelParameters",
    "SingleCalculation",
    "SingleDischargeLevel",
    "DischargeLevels",
]

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

    # bank height is calculated at the first discharge level only.
    height: Optional[np.ndarray] = field(default=lambda : np.array([]))

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

    def get_mid_points(self, as_geo_series: bool = False, crs: str = None) -> Union[GeoSeries, np.ndarray]:
        """Band line midpoints.

        Args:
            as_geo_series (bool):
                bool indicating if the output should be a GeoSeries or not.
            crs (str):
                coordinate reference system.
        Returns:
            the midpoints of the bank line coordinates as a GeoSeries or numpy array.
        """
        bank_coords = self.bank_line_coords
        bank_coords_mind = (bank_coords[:-1] + bank_coords[1:]) / 2

        if as_geo_series:
            bank_coords_mind = [Point(xy) for xy in bank_coords_mind]
            bank_coords_mind = GeoSeries(bank_coords_mind, crs=crs)
        return bank_coords_mind


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

    @property
    def height(self) -> List[np.ndarray]:
        """Get the bank height."""
        return [self.left.height, self.right.height]


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
        >>> from dfastbe.bank_erosion.data_models.calculation import ErosionResults
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
        >>> from dfastbe.bank_erosion.data_models.calculation import ErosionResults
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


@dataclass
class SingleParameters:
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
class SingleLevelParameters(BaseBank[SingleParameters]):
    pass


@dataclass
class SingleCalculation:
    bank_velocity: np.ndarray = field(default=lambda : np.array([]))
    water_level: np.ndarray = field(default=lambda : np.array([]))
    water_depth: np.ndarray = field(default=lambda : np.array([]))
    chezy: np.ndarray = field(default=lambda : np.array([]))
    ship_wave_max: np.ndarray = field(default=lambda : np.array([]))
    ship_wave_min: np.ndarray = field(default=lambda : np.array([]))
    volume_per_discharge: np.ndarray = field(default=lambda : np.array([]))
    erosion_distance_flow: np.ndarray = field(default=lambda : np.array([]))
    erosion_distance_shipping: np.ndarray = field(default=lambda : np.array([]))
    erosion_distance_tot: np.ndarray = field(default=lambda : np.array([]))
    erosion_volume_tot: np.ndarray = field(default=lambda : np.array([]))
    # the erosion distance and erosion volume at equilibrium is calculated at the last discharge level only.
    erosion_distance_eq: Optional[np.ndarray] = field(default=lambda : np.array([]))
    erosion_volume_eq: Optional[np.ndarray] = field(default=lambda : np.array([]))


@dataclass
class SingleDischargeLevel(BaseBank[SingleCalculation]):
    hfw_max: float = field(default=0.0)

    @classmethod
    def from_column_arrays(
        cls, data: dict, bank_cls: Type["SingleCalculation"], hfw_max: float,
        bank_order: Tuple[str, str] = ("left", "right")
    ) -> "SingleDischargeLevel":
        # Only include fields that belong to the bank-specific data
        base = BaseBank.from_column_arrays(data, bank_cls, bank_order=bank_order)

        return cls(
            id=base.id,
            left=base.left,
            right=base.right,
            hfw_max=hfw_max,
        )


class DischargeLevels:

    def __init__(self, levels: List[SingleDischargeLevel]):
        self.levels = levels

    def __getitem__(self, index: int) -> SingleDischargeLevel:
        return self.levels[index]

    def __len__(self) -> int:
        return len(self.levels)

    def append(self, level_calc: SingleDischargeLevel):
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

    def get_water_level_data(self) -> WaterLevelData:
        return WaterLevelData(
            hfw_max=self.levels[-1].hfw_max,
            water_level=self.get_attr_level("water_level"),
            ship_wave_max=self.get_attr_level("ship_wave_max"),
            ship_wave_min=self.get_attr_level("ship_wave_min"),
            velocity=self.get_attr_level("bank_velocity"),
            chezy=self.get_attr_level("chezy"),
            vol_per_discharge=self.get_attr_level("volume_per_discharge"),
        )
