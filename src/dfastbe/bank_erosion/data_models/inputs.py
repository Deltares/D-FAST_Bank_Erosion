import os
import numpy as np
from pathlib import Path
from shapely.geometry import LineString
from dfastio.xyc.models import XYCModel
from dfastbe.io.data_models import BaseRiverData, BaseSimulationData
from dfastbe.io.logger import log_text
from dfastbe.io.config import ConfigFile
from dfastbe.bank_erosion.data_models.calculation import MeshData, SingleBank


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

class BankLinesResultsError(Exception):
    """Custom exception for BankLine results errors."""

    pass