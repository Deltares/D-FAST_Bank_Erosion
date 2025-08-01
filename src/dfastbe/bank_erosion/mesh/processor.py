"""module for processing mesh-related operations."""
import math
from typing import Tuple, List
import numpy as np
from dataclasses import dataclass
from shapely.geometry import Point
from dfastbe.bank_erosion.mesh.data_models import RiverSegment
from dfastbe.bank_erosion.utils import enlarge

from dfastbe.bank_erosion.mesh.data_models import MeshData
from dfastbe.bank_erosion.data_models.calculation import (
    BankData,
    SingleBank,
    FairwayData
)
from dfastbe.bank_erosion.data_models.inputs import ErosionRiverData
from dfastbe.io.data_models import LineGeometry
from dfastbe.io.logger import log_text
from dfastbe.utils import on_right_side

__all__ = ["MeshProcessor"]

ATOL = 1e-8
RTOL = 1e-8
SHAPE_MULTIPLIER = 2


@dataclass
class IntersectionState:
    coords = np.ndarray
    face_indexes = np.ndarray
    point_index = 0
    current_face_index: int
    vertex_index: np.ndarray | int

    def __init__(self, shape: Tuple[int, int] = (100, 2), verbose: bool = False):
        """Initialize the intersection results with given shape."""
        self.coords = np.zeros(shape, dtype=np.float64)
        self.face_indexes = np.zeros(shape[0], dtype=np.int64)
        self.point_index = 0
        self.current_face_index: int = -1
        self.verbose = verbose

    def update(self, current_bank_point, shape_length: bool = None):
        """Finalize a segment

        Enlarge arrays if needed, set coordinates and index, and increment ind.
        """
        shape_length = self.point_index + 1 if shape_length is None else shape_length

        if self.point_index == self.coords.shape[0]:
            # last coordinate reached, so enlarge arrays
            self.coords = enlarge(self.coords, (shape_length, 2))
            self.face_indexes = enlarge(self.face_indexes, (shape_length,))

        self.coords[self.point_index] = current_bank_point

        if self.current_face_index == -2:
            self.face_indexes[self.point_index] = self.vertex_index[0]
        else:
            self.face_indexes[self.point_index] = self.current_face_index
        self.point_index += 1

    def _log_mesh_transition(
        self, step, transition_type, transition_index, face_index, prev_b
    ):
        """Helper to print mesh transition information for debugging.

        Args:
            step (int):
                The current step or iteration.
            transition_type (str):
                The type of transition (e.g., "node", "edge").
            transition_index (int):
                The index of the transition (e.g., the node or edge index).
            face_index (int):
                The target mesh face index.
            prev_b (float):
                The previous value of b.
        """
        index_str = "outside" if self.current_face_index == -1 else self.current_face_index
        if self.current_face_index == -2:
            index_str = f"edge between {self.vertex_index}"
        print(
            f"{step}: moving from {index_str} via {transition_type} {transition_index} "
            f"to {face_index} at b = {prev_b}"
        )

    def _update_main_attributes(self, face_indexes, node, prev_b, step):
        if self.verbose:
            self._log_mesh_transition(step, "node", node, face_indexes, prev_b)

        if isinstance(face_indexes, (int, np.integer)):
            self.current_face_index = face_indexes
        elif hasattr(face_indexes, "__len__") and len(face_indexes) == 1:
            self.current_face_index = face_indexes[0]
        else:
            self.current_face_index = -2
            self.vertex_index = face_indexes

    def _update_mesh_index_and_log(self, step, node, edge, faces, face_indexes, prev_b):
        """
        Helper to update mesh index and log transitions for intersect_line_mesh.
        """
        if face_indexes is not None:
            self._update_main_attributes(face_indexes, node, prev_b, step)
            return

        for i, face in enumerate(faces):
            if face == self.current_face_index:
                other_face = faces[1 - i]
                if self.verbose:
                    self._log_mesh_transition(step, "edge", edge, other_face, prev_b)
                self.current_face_index = other_face
                return

        raise ValueError(
            f"Shouldn't come here .... index {self.current_face_index} differs from both faces "
            f"{faces[0]} and {faces[1]} associated with slicing edge {edge}"
        )

    
class MeshWrapper:
    """A class for processing mesh-related operations."""

    def __init__(
        self, mesh_data: MeshData, d_thresh: float = 0.001, verbose: bool = False
    ):
        self.mesh_data = mesh_data
        self.d_thresh = d_thresh
        self.verbose = verbose

    def _read_target_object(self, xy_coords):
        self.given_coords = xy_coords
        self.intersection_state = IntersectionState(xy_coords.shape, self.verbose)

    def _handle_first_point(self, current_bank_point: np.ndarray):
        # get the point on the mesh face that is closest to the first bank point
        dx = self.mesh_data.x_face_coords - current_bank_point[0]
        dy = self.mesh_data.y_face_coords - current_bank_point[1]
        closest_cell_ind = np.nonzero(
            ~(
                (dx < 0).all(axis=1)
                | (dx > 0).all(axis=1)
                | (dy < 0).all(axis=1)
                | (dy > 0).all(axis=1)
            )
        )[0]
        index, vindex = self._find_starting_face(closest_cell_ind)
        self.intersection_state.current_face_index = index
        self.intersection_state.vertex_index = vindex
        self.intersection_state.update(current_bank_point)

    def _find_starting_face(self, face_indexes: np.ndarray):
        """Find the starting face for a bank line segment.

        This function determines the face index and vertex indices of the mesh
        that the first point of a bank line segment is associated with.

        Args:
            face_indexes (np.ndarray): Array of possible cell indices.
        """
        if len(face_indexes) == 0 and self.verbose:
            print("starting outside mesh")

        edges_indexes = self.mesh_data.locate_point(self.given_coords[0], face_indexes)
        if not isinstance(edges_indexes, list):
            # A single index was returned
            index = edges_indexes
            vindex = None
        else:
            # A list of indices is expected
            if edges_indexes:
                if self.verbose:
                    print(f"starting on edge of {edges_indexes}")

                index = -2 if len(edges_indexes) > 1 else edges_indexes[0]
                vindex = edges_indexes if len(edges_indexes) > 1 else None
            else:
                if self.verbose:
                    print("starting outside mesh")
                index = -1
                vindex = None

        return index, vindex

    def _log_slice_status(self, j, prev_b, bpj):
        if prev_b > 0:
            print(f"{j}: -- no further slices along this segment --")
        else:
            print(f"{j}: -- no slices along this segment --")

        if self.intersection_state.current_face_index >= 0:
            pnt = Point(bpj)
            polygon_k = self.mesh_data.get_face_by_index(self.intersection_state.current_face_index, as_polygon=True)

            if not polygon_k.contains(pnt):
                raise ValueError(
                    f"{j}: ERROR: point actually not contained within {self.intersection_state.current_face_index}!"
                )

    def _process_node_transition(self, segment: RiverSegment, node):
        """Process the transition at a node when a segment ends or continues."""
        finished = False

        if self.verbose:
            print(
                f"{segment.index}: moving via node {node} on edges {segment.edges} at {segment.distances[0]}"
            )
        
        # figure out where we will be heading afterwards ...
        if segment.distances[0] < 1.0:
            # segment passes through node and enter non-neighbouring cell ...
            # direction of current segment from bpj1 to bpj
            theta = segment.theta
        else:
            if (
                np.isclose(segment.distances[0], 1.0, rtol=RTOL, atol=ATOL)
                and segment.index == len(self.given_coords) - 1
            ):
                # catch case of last segment
                if self.verbose:
                    print(f"{segment.index}: last point ends in a node")

                self.intersection_state.update(segment.current_point)
                theta = 0.0
                finished = True
            else:
                # this segment ends in the node, so check next segment ...
                # direction of next segment from bpj to bp[j+1]
                theta = math.atan2(
                    self.given_coords[segment.index + 1][1]
                    - segment.current_point[1],
                    self.given_coords[segment.index + 1][0]
                    - segment.current_point[0],
                )
        next_face_index = None
        if not finished:
            next_face_index = self.mesh_data.resolve_next_face_by_direction(
                theta, node, segment.index
            )
        return False, next_face_index

    def _slice_by_node_or_edge(
        self, segment: RiverSegment, node, edge, faces
    ):
        finished = False
        next_face_index = None
        if node >= 0:
            # if we slice at a node ...
            finished, next_face_index = self._process_node_transition(segment, node)

        elif segment.distances[0] == 1:
            # ending at slice point, so ending on an edge ...
            if self.verbose:
                print(
                    f"{segment.index}: ending on edge {edge} at {segment.distances[0]}"
                )
            # figure out where we will be heading afterwards ...
            if segment.index == len(self.given_coords) - 1:
                # catch case of last segment
                if self.verbose:
                    print(f"{segment.index}: last point ends on an edge")
                self.intersection_state.update(segment.current_point)
                finished = True
            else:
                next_point = [self.given_coords[segment.index + 1][0], self.given_coords[segment.index + 1][1]]
                next_face_index = self.determine_next_face_on_edge(segment, next_point, edge, faces)

        return finished, next_face_index

    def _process_bank_segment(self, segment: RiverSegment):

        while True:
            if self.intersection_state.current_face_index == -2:
                index_src = self.mesh_data.resolve_ambiguous_edge_transition(segment, self.intersection_state.vertex_index)

                if len(index_src) == 1:
                    self.intersection_state.current_face_index = index_src[0]
                    self.intersection_state.vertex_index = index_src[0:1]
                else:
                    self.intersection_state.current_face_index = -2
            elif segment.is_length_zero():
                # segment has zero length
                break
            else:
                segment.distances, segment.edges, segment.nodes = (
                    self.mesh_data.find_segment_intersections(
                        self.intersection_state.current_face_index,
                        segment,
                    )
                )

            if len(segment.edges) == 0:
                # rest of segment associated with same face
                shape_length = self.intersection_state.point_index * SHAPE_MULTIPLIER
                self.intersection_state.update(
                    segment.current_point, shape_length=shape_length
                )
                break

            if len(segment.edges) > 1:
                segment.select_first_intersection()

            # slice location identified ...   (number of edges should be 1)
            node = segment.nodes[0]
            edge = segment.edges[0]
            faces = self.mesh_data.edge_face_connectivity[edge]
            segment.min_relative_distance = segment.distances[0]

            finished, index0 = self._slice_by_node_or_edge(
                segment,
                node,
                edge,
                faces,
            )
            if finished:
                break

            self.intersection_state._update_mesh_index_and_log(
                segment.index,
                node,
                edge,
                faces,
                index0,
                segment.min_relative_distance,
            )
            segment_x = (
                    segment.previous_point
                    + segment.min_relative_distance
                    * (segment.current_point - segment.previous_point)
            )
            shape_length = self.intersection_state.point_index * SHAPE_MULTIPLIER
            self.intersection_state.update(segment_x, shape_length=shape_length)
            if segment.min_relative_distance == 1:
                break

    def resolve_next_face_by_direction(
            self, theta: float, node, verbose_index: int = None
    ):
        """Helper to resolve the next face index based on the direction theta at a node."""

        if self.verbose:
            print(f"{verbose_index}: moving in direction theta = {theta}")

        edges = self.mesh_data.find_edges(theta, node, verbose_index)

        if self.verbose:
            print(f"{verbose_index}: the edge to the left is edge {edges.left}")
            print(f"{verbose_index}: the edge to the right is edge {edges.right}")

        if edges.left == edges.right:
            if self.verbose:
                print(f"{verbose_index}: continue along edge {edges.left}")

            next_face_index = self.mesh_data.edge_face_connectivity[edges.left, :]
        else:
            if self.verbose:
                print(
                    f"{verbose_index}: continue between edges {edges.left}"
                    f" on the left and {edges.right} on the right"
                )
            next_face_index = self.mesh_data.resolve_next_face_from_edges(
                node, edges, verbose_index
            )
        return next_face_index

    def determine_next_face_on_edge(
        self, segment: RiverSegment, next_point: List[float], edge, faces,
    ):
        """Determine the next face to continue along an edge based on the segment direction."""
        theta = math.atan2(
            next_point[1] - segment.current_point[1],
            next_point[0] - segment.current_point[0],
            )
        if self.verbose:
            print(f"{segment.index}: moving in direction theta = {theta}")

        theta_edge = self.mesh_data.calculate_edge_angle(edge)
        if theta == theta_edge or theta == -theta_edge:
            if self.verbose:
                print(f"{segment.index}: continue along edge {edge}")
            next_face_index = faces
        else:
            # check whether the (extended) segment slices any edge of faces[0]
            fe1 = self.mesh_data.face_edge_connectivity[faces[0]]
            reversed_segment = RiverSegment(
                index=segment.index,
                previous_point=segment.current_point,
                current_point=next_point,
                min_relative_distance=0,
            )
            _, _, edges = self.mesh_data.calculate_edge_intersections(
                fe1,
                reversed_segment,
                False,
            )
            # yes, a slice (typically 1, but could be 2 if it slices at a node
            # but that doesn't matter) ... so, we continue towards faces[0]
            # if there are no slices for faces[0], we continue towards faces[1]
            next_face_index = faces[0] if len(edges) > 0 else faces[1]
        return next_face_index

    def intersect_with_coords(self, given_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Intersects a coords with an unstructured mesh and returns the intersection coordinates and mesh face indices.

        This function determines where a given line (e.g., a bank line) intersects the faces of an unstructured mesh.
        It calculates the intersection points and identifies the mesh faces corresponding to each segment of the line.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                A tuple containing:
                - `coords` (np.ndarray):
                    A 2D array of shape (M, 2) containing the x, y coordinates of the intersection points.
                - `idx` (np.ndarray):
                    A 1D array of shape (M-1,) containing the indices of the mesh faces corresponding to each segment
                    of the intersected line.

        Raises:
            Exception:
                If the line starts outside the mesh and cannot be associated with any mesh face, or if the line crosses
                ambiguous regions (e.g., edges shared by multiple faces).

        Notes:
            - The function uses Shapely geometry operations to determine whether points are inside polygons or on edges.
            - The function handles cases where the line starts outside the mesh, crosses multiple edges, or ends on a node.
            - Tiny segments shorter than `d_thresh` are removed from the output.
        """
        self._read_target_object(given_coords)
        for point_index, current_bank_point in enumerate(given_coords):
            if self.verbose:
                print(
                    f"Current location: {current_bank_point[0]}, {current_bank_point[1]}"
                )
            if point_index == 0:
                self._handle_first_point(current_bank_point)
            else:
                segment = RiverSegment(
                    index=point_index,
                    min_relative_distance=0,
                    current_point=current_bank_point,
                    previous_point=given_coords[point_index - 1],
                )
                self._process_bank_segment(segment)

        # clip to actual length (idx refers to segments, so we can ignore the last value)
        self.intersection_state.coords = self.intersection_state.coords[: self.intersection_state.point_index]
        self.intersection_state.face_indexes = self.intersection_state.face_indexes[: self.intersection_state.point_index - 1]

        # remove tiny segments
        d = np.sqrt((np.diff(self.intersection_state.coords, axis=0) ** 2).sum(axis=1))
        mask = np.concatenate((np.ones((1), dtype="bool"), d > self.d_thresh))
        self.intersection_state.coords = self.intersection_state.coords[mask, :]
        self.intersection_state.face_indexes = self.intersection_state.face_indexes[mask[1:]]

        # since index refers to segments, don't return the first one
        return self.intersection_state.coords, self.intersection_state.face_indexes


class MeshProcessor:
    """Class to process bank lines and intersect them with a mesh."""

    def __init__(self, river_data: ErosionRiverData, mesh_data: MeshData):
        """Constructor for MeshProcessor."""
        self.bank_lines = river_data.bank_lines
        self.mesh_data = mesh_data
        self.river_data = river_data
        self.wrapper = MeshWrapper(mesh_data)

    def get_fairway_data(self, river_axis: LineGeometry) -> FairwayData:
        log_text("chainage_to_fairway")
        # intersect fairway and mesh
        fairway_intersection_coords, fairway_face_indices = self.wrapper.intersect_with_coords(river_axis.as_array())

        if self.river_data.debug:
            arr = (fairway_intersection_coords[:-1] + fairway_intersection_coords[1:]) / 2
            line_geom = LineGeometry(arr, crs=river_axis.crs)
            line_geom.to_file(
                file_name=f"{str(self.river_data.output_dir)}/fairway_face_indices.shp",
                data={"iface": fairway_face_indices},
            )

        return FairwayData(fairway_face_indices, fairway_intersection_coords)

    def get_bank_data(self) -> BankData:
        """Intersect bank lines with a mesh and return bank data.

        Returns:
            BankData object containing bank line coordinates, face indices, and other bank-related data.
        """
        n_bank_lines = len(self.bank_lines)

        bank_line_coords = []
        bank_face_indices = []
        for bank_index in range(n_bank_lines):
            line_coords = np.array(self.bank_lines.geometry[bank_index].coords)
            log_text("bank_nodes", data={"ib": bank_index + 1, "n": len(line_coords)})

            coords_along_bank, face_indices = self.wrapper.intersect_with_coords(line_coords)
            bank_line_coords.append(coords_along_bank)
            bank_face_indices.append(face_indices)

        bank_order, data = self._link_lines_to_stations(bank_line_coords, bank_face_indices)

        return BankData.from_column_arrays(
            data,
            SingleBank,
            bank_lines=self.bank_lines,
            n_bank_lines=n_bank_lines,
            bank_order=bank_order,
        )

    def _link_lines_to_stations(self, bank_line_coords, bank_face_indices):
        # linking bank lines to chainage
        log_text("chainage_to_banks")
        river_center_line = self.river_data.river_center_line.as_array()
        n_bank_lines = len(self.bank_lines)

        bank_chainage_midpoints = [None] * n_bank_lines
        is_right_bank = [True] * n_bank_lines
        for bank_index, coords in enumerate(bank_line_coords):
            segment_mid_points = LineGeometry((coords[:-1, :] + coords[1:, :]) / 2)
            chainage_mid_points = segment_mid_points.intersect_with_line(
                river_center_line
            )

            # check if the bank line is defined from low chainage to high chainage
            if chainage_mid_points[0] > chainage_mid_points[-1]:
                # if not, flip the bank line and all associated data
                chainage_mid_points = chainage_mid_points[::-1]
                bank_line_coords[bank_index] = bank_line_coords[bank_index][::-1, :]
                bank_face_indices[bank_index] = bank_face_indices[bank_index][::-1]

            bank_chainage_midpoints[bank_index] = chainage_mid_points

            # check if the bank line is a left or right bank
            # when looking from low-to-high chainage
            is_right_bank[bank_index] = on_right_side(
                coords, river_center_line[:, :2]
            )
            if is_right_bank[bank_index]:
                log_text("right_side_bank", data={"ib": bank_index + 1})
            else:
                log_text("left_side_bank", data={"ib": bank_index + 1})

        bank_order = tuple("right" if val else "left" for val in is_right_bank)
        data = {
            'is_right_bank': is_right_bank,
            'bank_line_coords': bank_line_coords,
            'bank_face_indices': bank_face_indices,
            'bank_chainage_midpoints': bank_chainage_midpoints
        }
        return bank_order, data