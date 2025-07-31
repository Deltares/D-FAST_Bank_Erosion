"""module for processing mesh-related operations."""
import math
from typing import Tuple
import numpy as np
from shapely.geometry import Point
from dfastbe.bank_erosion.mesh.data_models import MeshData, RiverSegment
from dfastbe.bank_erosion.utils import enlarge

__all__ = ["MeshProcessor"]

ATOL = 1e-8
RTOL = 1e-8
SHAPE_MULTIPLIER = 2


class MeshProcessor:
    """A class for processing mesh-related operations."""

    def __init__(
        self, bank_points: np.ndarray, mesh_data: MeshData, d_thresh: float = 0.001
    ):
        self.bank_points = bank_points
        self.mesh_data = mesh_data
        self.d_thresh = d_thresh
        self.coords = np.zeros((len(bank_points), 2))
        self.face_indexes = np.zeros(len(bank_points), dtype=np.int64)
        self.verbose = False
        self.point_index = 0
        self.index: int
        self.vindex: np.ndarray | int

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
        self.index, self.vindex = self._find_starting_face(closest_cell_ind)
        self._store_segment_point(current_bank_point)

    def _find_starting_face(self, face_indexes: np.ndarray):
        """Find the starting face for a bank line segment.

        This function determines the face index and vertex indices of the mesh
        that the first point of a bank line segment is associated with.

        Args:
            face_indexes (np.ndarray): Array of possible cell indices.
        """
        if len(face_indexes) == 0 and self.verbose:
            print("starting outside mesh")

        edges_indexes = self.mesh_data.locate_point(self.bank_points[0], face_indexes)
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

    def _store_segment_point(self, current_bank_point, shape_length: bool = None):
        """Finalize a segment

        Enlarge arrays if needed, set coordinates and index, and increment ind.
        """
        shape_length = self.point_index + 1 if shape_length is None else shape_length

        if self.point_index == self.coords.shape[0]:
            # last coordinate reached, so enlarge arrays
            self.coords = enlarge(self.coords, (shape_length, 2))
            self.face_indexes = enlarge(self.face_indexes, (shape_length,))

        self.coords[self.point_index] = current_bank_point

        if self.index == -2:
            self.face_indexes[self.point_index] = self.vindex[0]
        else:
            self.face_indexes[self.point_index] = self.index
        self.point_index += 1

    def _log_mesh_transition(
        self, step, transition_type, transition_index, index0, prev_b
    ):
        """Helper to print mesh transition information for debugging.

        Args:
            step (int): The current step or iteration.
            index (int): The current mesh face index.
            vindex (int): The vertex index.
            transition_type (str): The type of transition (e.g., "node", "edge").
            transition_index (int): The index of the transition (e.g., the node or edge index).
            index0 (int): The target mesh face index.
            prev_b (float): The previous value of b.
        """
        index_str = "outside" if self.index == -1 else self.index
        if self.index == -2:
            index_str = f"edge between {self.vindex}"
        print(
            f"{step}: moving from {index_str} via {transition_type} {transition_index} "
            f"to {index0} at b = {prev_b}"
        )
    def _update_main_attributes(self, face_indexes, node, prev_b, j):
        if self.verbose:
            self._log_mesh_transition(j, "node", node, face_indexes, prev_b)

        if isinstance(face_indexes, (int, np.integer)):
            self.index = face_indexes
        elif hasattr(face_indexes, "__len__") and len(face_indexes) == 1:
            self.index = face_indexes[0]
        else:
            self.index = -2
            self.vindex = face_indexes

    def _update_mesh_index_and_log(self, j, node, edge, faces, face_indexes, prev_b):
        """
        Helper to update mesh index and log transitions for intersect_line_mesh.
        """
        if face_indexes is not None:
            self._update_main_attributes(face_indexes, node, prev_b, j)
            return

        for i, face in enumerate(faces):
            if face == self.index:
                other_face = faces[1 - i]
                if self.verbose:
                    self._log_mesh_transition(j, "edge", edge, other_face, prev_b)
                self.index = other_face
                return

        raise ValueError(
            f"Shouldn't come here .... index {self.index} differs from both faces "
            f"{faces[0]} and {faces[1]} associated with slicing edge {edge}"
        )

    def _log_slice_status(self, j, prev_b, bpj):
        if prev_b > 0:
            print(f"{j}: -- no further slices along this segment --")
        else:
            print(f"{j}: -- no slices along this segment --")

        if self.index >= 0:
            pnt = Point(bpj)
            polygon_k = self.mesh_data.get_face_by_index(self.index, as_polygon=True)
            if not polygon_k.contains(pnt):
                raise ValueError(
                    f"{j}: ERROR: point actually not contained within {self.index}!"
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
                and segment.index == len(self.bank_points) - 1
            ):
                # catch case of last segment
                if self.verbose:
                    print(f"{segment.index}: last point ends in a node")

                self._store_segment_point(segment.current_point)
                theta = 0.0
                finished = True
            else:
                # this segment ends in the node, so check next segment ...
                # direction of next segment from bpj to bp[j+1]
                theta = math.atan2(
                    self.bank_points[segment.index + 1][1]
                    - segment.current_point[1],
                    self.bank_points[segment.index + 1][0]
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
            if segment.index == len(self.bank_points) - 1:
                # catch case of last segment
                if self.verbose:
                    print(f"{segment.index}: last point ends on an edge")
                self._store_segment_point(segment.current_point)
                finished = True
            else:
                next_point = [self.bank_points[segment.index + 1][0], self.bank_points[segment.index + 1][1]]
                next_face_index = self.mesh_data.determine_next_face_on_edge(segment, next_point, edge, faces)

        return finished, next_face_index

    def _process_bank_segment(self, segment: RiverSegment):

        while True:
            if self.index == -2:
                index_src = self.mesh_data.resolve_ambiguous_edge_transition(segment, self.vindex)

                if len(index_src) == 1:
                    self.index = index_src[0]
                    self.vindex = index_src[0:1]
                else:
                    self.index = -2
            elif segment.is_length_zero():
                # segment has zero length
                break
            else:
                segment.distances, segment.edges, segment.nodes = (
                    self.mesh_data.find_segment_intersections(
                        self.index,
                        segment,
                    )
                )

            if len(segment.edges) == 0:
                # rest of segment associated with same face
                shape_length = self.point_index * SHAPE_MULTIPLIER
                self._store_segment_point(
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

            self._update_mesh_index_and_log(
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
            shape_length = self.point_index * SHAPE_MULTIPLIER
            self._store_segment_point(segment_x, shape_length=shape_length)
            if segment.min_relative_distance == 1:
                break

    def intersect_line_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Intersects a line with an unstructured mesh and returns the intersection coordinates and mesh face indices.

        This function determines where a given line (e.g., a bank line) intersects the faces of an unstructured mesh.
        It calculates the intersection points and identifies the mesh faces corresponding to each segment of the line.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                A tuple containing:
                - `coords` (np.ndarray): A 2D array of shape (M, 2) containing the x, y coordinates of the intersection points.
                - `idx` (np.ndarray): A 1D array of shape (M-1,) containing the indices of the mesh faces corresponding to
                each segment of the intersected line.

        Raises:
            Exception:
                If the line starts outside the mesh and cannot be associated with any mesh face, or if the line crosses
                ambiguous regions (e.g., edges shared by multiple faces).

        Notes:
            - The function uses Shapely geometry operations to determine whether points are inside polygons or on edges.
            - The function handles cases where the line starts outside the mesh, crosses multiple edges, or ends on a node.
            - Tiny segments shorter than `d_thresh` are removed from the output.
        """
        for point_index, current_bank_point in enumerate(self.bank_points):
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
                    previous_point=self.bank_points[point_index - 1],
                )
                self._process_bank_segment(segment)

        # clip to actual length (idx refers to segments, so we can ignore the last value)
        self.coords = self.coords[: self.point_index]
        self.face_indexes = self.face_indexes[: self.point_index - 1]

        # remove tiny segments
        d = np.sqrt((np.diff(self.coords, axis=0) ** 2).sum(axis=1))
        mask = np.concatenate((np.ones((1), dtype="bool"), d > self.d_thresh))
        self.coords = self.coords[mask, :]
        self.face_indexes = self.face_indexes[mask[1:]]

        # since index refers to segments, don't return the first one
        return self.coords, self.face_indexes
