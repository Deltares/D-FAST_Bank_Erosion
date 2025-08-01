import os
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from dfastio.xyc.models import XYCModel
from shapely.geometry import LineString

from dfastbe.bank_erosion.data_models.calculation import (
    SingleBank,
    SingleLevelParameters,
    SingleParameters,
)
from dfastbe.bank_erosion.mesh.data_models import MeshData
from dfastbe.io.config import ConfigFile
from dfastbe.io.data_models import BaseRiverData, BaseSimulationData, LineGeometry
from dfastbe.io.logger import log_text

__all__ = [
    "ErosionSimulationData",
    "ErosionRiverData",
    "BankLinesResultsError",
    "ShipsParameters",
    "Parameters",
]

Parameters = namedtuple("Parameters", "name default valid onefile positive ext")


class ErosionSimulationData(BaseSimulationData):

    def compute_mesh_topology(self, verbose: bool = False) -> MeshData:
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

        x_face_coords = self.apply_masked_indexing(self.x_node, self.face_node)
        y_face_coords = self.apply_masked_indexing(self.y_node, self.face_node)
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
            verbose=verbose,
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

    def calculate_bank_velocity(self, single_bank: "SingleBank", vel_dx) -> np.ndarray:
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
            vel_bank = moving_avg(single_bank.bank_chainage_midpoints, vel_bank, vel_dx)

        return vel_bank

    def calculate_bank_height(self, single_bank: SingleBank, zb_dx):
        # bank height = maximum bed elevation per cell
        from dfastbe.bank_erosion.utils import moving_avg

        bank_index = single_bank.bank_face_indices
        if self.bed_elevation_location == "node":
            zb_nodes = self.bed_elevation_values
            zb_all = self.apply_masked_indexing(zb_nodes, self.face_node[bank_index, :])
            zb_bank = zb_all.max(axis=1)
            if zb_dx > 0.0:
                zb_bank = moving_avg(
                    single_bank.bank_chainage_midpoints,
                    zb_bank,
                    zb_dx,
                )
        else:
            # don't know ... need to check neighbouring cells ...
            zb_bank = None

        return zb_bank

    def get_fairway_data(self, fairway_face_indices):
        # get fairway face indices
        # fairway_face_indices = fairway_face_indices

        # get water depth along the fair-way
        water_depth_fairway = self.water_depth_face[fairway_face_indices]
        water_level = self.water_level_face[fairway_face_indices]
        chez_face = self.chezy_face[fairway_face_indices]
        chezy = 0 * chez_face + chez_face.mean()

        data = {
            "water_depth": water_depth_fairway,
            "water_level": water_level,
            "chezy": chezy,
        }
        return data


class ErosionRiverData(BaseRiverData):

    def __init__(self, config_file: ConfigFile):
        super().__init__(config_file)
        self.bank_dir = self._get_bank_line_dir()
        self.output_dir = config_file.get_output_dir("erosion")
        self.debug = config_file.debug
        # set plotting flags
        self.plot_flags = config_file.get_plotting_flags(config_file.root_dir)
        # get filter settings for bank levels and flow velocities along banks
        self.zb_dx = config_file.get_float(
            "Erosion", "BedFilterDist", 0.0, positive=True
        )
        self.vel_dx = config_file.get_float(
            "Erosion", "VelFilterDist", 0.0, positive=True
        )
        log_text("get_levels")
        self.num_discharge_levels = config_file.get_int("Erosion", "NLevel")
        self.output_intervals = config_file.get_float("Erosion", "OutputInterval", 1.0)
        self.bank_lines = config_file.read_bank_lines(str(self.bank_dir))
        self.river_axis = self._read_river_axis()
        self.erosion_time = config_file.get_int("Erosion", "TErosion", positive=True)

    def simulation_data(self) -> ErosionSimulationData:
        """Simulation Data."""
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

    def process_river_axis_by_center_line(self) -> LineGeometry:
        """Process the river axis by the center line.

        Intersect the river center line with the river axis to map the stations from the first to the latter
        then clip the river axis by the first and last station of the centerline.
        """
        river_axis = LineGeometry(self.river_axis, crs=self.config_file.crs)
        river_axis_numpy = river_axis.as_array()
        # optional sorting --> see 04_Waal_D3D example
        # check: sum all distances and determine maximum distance ...
        # if maximum > alpha * sum then perform sort
        # Waal OK: 0.0082 ratio max/sum, Waal NotOK: 0.13 - Waal: 2500 points,
        # so even when OK still some 21 times more than 1/2500 = 0.0004
        dist2 = (np.diff(river_axis_numpy, axis=0) ** 2).sum(axis=1)
        alpha = dist2.max() / dist2.sum()
        if alpha > 0.03:
            print("The river axis needs sorting!!")

        # map km to axis points, further using axis
        log_text("chainage_to_axis")
        river_center_line_arr = self.river_center_line.as_array()
        river_axis_km = river_axis.intersect_with_line(river_center_line_arr)

        # clip river axis to reach of interest (get the closest point to the first and last station)
        i1 = np.argmin(
            ((river_center_line_arr[0, :2] - river_axis_numpy) ** 2).sum(axis=1)
        )
        i2 = np.argmin(
            ((river_center_line_arr[-1, :2] - river_axis_numpy) ** 2).sum(axis=1)
        )
        if i1 < i2:
            river_axis_km = river_axis_km[i1 : i2 + 1]
            river_axis_numpy = river_axis_numpy[i1 : i2 + 1]
        else:
            # reverse river axis
            river_axis_km = river_axis_km[i2 : i1 + 1][::-1]
            river_axis_numpy = river_axis_numpy[i2 : i1 + 1][::-1]

        river_axis = LineGeometry(river_axis_numpy, crs=self.config_file.crs)
        river_axis.add_data(data={"stations": river_axis_km})
        return river_axis


@dataclass
class ShipsParameters:
    """Data for ships going through the fairway for bank erosion simulation.

    Args:
        config_file (ConfigFile):
            Configuration file containing parameters.
        velocity (List[np.ndarray]):
            Ship velocities for each bank.
        number (List[np.ndarray]):
            Number of ships for each bank.
        num_waves (List[np.ndarray]):
            Number of waves per ship for each bank.
        draught (List[np.ndarray]):
            Draught of ships for each bank.
        type (List[np.ndarray]):
            Type of ships for each bank.
        slope (List[np.ndarray]):
            Slope values for each bank.
        reed (List[np.ndarray]):
            Reed values for each bank.
    """

    config_file: ConfigFile
    velocity: List[np.ndarray]
    number: List[np.ndarray]
    num_waves: List[np.ndarray]
    draught: List[np.ndarray]
    type: List[np.ndarray]
    slope: List[np.ndarray]
    reed: List[np.ndarray]

    REED_DAMPING_COEFFICIENT = 8.5e-4
    REED_DAMPING_EXPONENT = 0.8

    @classmethod
    def get_ship_data(
        cls, num_stations_per_bank: List[int], config_file: ConfigFile
    ) -> "ShipsParameters":
        """Get ship parameters from the configuration file.

        Args:
            num_stations_per_bank (List[int]):
                The number of stations per bank.
            config_file (ConfigFile):
                Configuration file containing parameters.

        Returns:
            ShipsParameters: An instance of ShipsParameters with parameters read from the config file.
        """

        param_defs = cls._get_initial_parameter_definitions()
        param_resolved = cls._get_parameters(
            config_file, num_stations_per_bank, param_defs
        )

        param_dict = {
            "velocity": param_resolved["vship"],
            "number": param_resolved["nship"],
            "num_waves": param_resolved["nwave"],
            "draught": param_resolved["draught"],
            "type": param_resolved["shiptype"],
            "slope": param_resolved["slope"],
            "reed": param_resolved["reed"],
        }

        return cls(config_file, **param_dict)

    @staticmethod
    def _get_parameters(
        config_file: ConfigFile,
        num_stations_per_bank: List[int],
        param_defs: List[Parameters],
        level_i_str: str = "",
    ) -> Dict[str, np.ndarray]:
        """Resolve a list of Parameters with values from the configuration file."""
        param_resolved = {}

        for param in param_defs:
            param_resolved[f"{param.name.lower()}"] = config_file.get_parameter(
                "Erosion",
                f"{param.name}{level_i_str}",
                num_stations_per_bank,
                default=param.default,
                valid=param.valid,
                onefile=param.onefile,
                positive=param.positive,
                ext=param.ext,
            )
        return param_resolved

    @classmethod
    def _get_initial_parameter_definitions(cls) -> List[Parameters]:
        """Get parameter definitions for discharge parameters.

        Returns:
            List[namedtuple]: List of parameter definitions.
        """
        return [
            Parameters("VShip", None, None, True, True, None),
            Parameters("NShip", None, None, True, True, None),
            Parameters("NWave", 5, None, True, True, None),
            Parameters("Draught", None, None, True, True, None),
            Parameters("ShipType", None, [1, 2, 3], True, None, None),
            Parameters("Slope", 20, None, None, True, "slp"),
            Parameters("Reed", 0, None, None, True, "rdd"),
        ]

    def _get_discharge_parameter_definitions(self) -> List[Parameters]:
        """Get parameter definitions for discharge parameters.

        Returns:
            List[namedtuple]: List of parameter definitions.
        """
        return [
            Parameters("VShip", self.velocity, None, None, None, None),
            Parameters("NShip", self.number, None, None, None, None),
            Parameters("NWave", self.num_waves, None, None, None, None),
            Parameters("Draught", self.draught, None, None, None, None),
            Parameters("ShipType", self.type, [1, 2, 3], True, None, None),
            Parameters("Slope", self.slope, None, None, True, "slp"),
            Parameters("Reed", self.reed, None, None, True, "rdd"),
        ]

    @staticmethod
    def _calculate_ship_derived_parameters(
        slope_values: List[np.ndarray], reed_values: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Calculate derived parameters from slope and reed values.

        Args:
            slope_values (List[np.ndarray]): Slope values for each bank.
            reed_values (List[np.ndarray]): Reed values for each bank.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: Calculated mu_slope and mu_reed values.
        """
        mu_slope = []
        mu_reed = []

        for ps, pr in zip(slope_values, reed_values):
            # Calculate mu_slope (inverse of slope for non-zero values)
            mus = ps.copy()
            mus[mus > 0] = 1.0 / mus[mus > 0]
            mu_slope.append(mus)

            # Calculate mu_reed (empirical damping coefficient)
            mu_reed.append(
                ShipsParameters.REED_DAMPING_COEFFICIENT
                * pr**ShipsParameters.REED_DAMPING_EXPONENT
            )

        return mu_slope, mu_reed

    def read_discharge_parameters(
        self,
        level_i: int,
        num_stations_per_bank: List[int],
    ) -> SingleLevelParameters:
        """Read Discharge level parameters.

        Read all discharge-specific input arrays for level_i.

        Args:
            level_i (int):
                The index of the discharge level.
            num_stations_per_bank (List[int]):
                The number of stations per bank.

        Returns:
            SingleLevelParameters: The discharge level parameters.
        """
        param_defs = self._get_discharge_parameter_definitions()
        param_resolved = self._get_parameters(
            self.config_file, num_stations_per_bank, param_defs, f"{level_i + 1}"
        )

        mu_slope, mu_reed = self._calculate_ship_derived_parameters(
            param_resolved["slope"], param_resolved["reed"]
        )

        return SingleLevelParameters.from_column_arrays(
            {
                "id": level_i,
                "ship_velocity": param_resolved["vship"],
                "num_ship": param_resolved["nship"],
                "num_waves_per_ship": param_resolved["nwave"],
                "ship_draught": param_resolved["draught"],
                "ship_type": param_resolved["shiptype"],
                "par_slope": param_resolved["slope"],
                "par_reed": param_resolved["reed"],
                "mu_slope": mu_slope,
                "mu_reed": mu_reed,
            },
            SingleParameters,
        )


class BankLinesResultsError(Exception):
    """Custom exception for BankLine results errors."""

    pass
