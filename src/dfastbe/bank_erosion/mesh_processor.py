"""module for processing mesh-related operations."""
import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from dfastbe.bank_erosion.data_models.calculation import MeshData
__all__ = ["enlarge", "MeshProcessor"]

ATOL = 1e-8
RTOL = 1e-8
SHAPE_MULTIPLIER = 2
TWO_PI = 2 * math.pi


@dataclass
class EdgeCandidates:
    """Dataclass to hold edge candidates for left and right edges.

    Args:
        left_edge (int):
            Index of the left edge.
        left_dtheta (float):
            Angle of the left edge in radians.
        right_edge (int):
            Index of the right edge.
        right_dtheta (float):
            Angle of the right edge in radians.
        found (bool):
            Flag indicating whether a valid edge pair was found.
    """
    left_edge: int
    left_dtheta: float
    right_edge: int
    right_dtheta: float
    found: bool = False

    def update_edges_by_angle(
        self, edge_index: int, dtheta: float, j, verbose=False
    ):
        """Update the left and right edges based on the angle difference."""
        twopi = 2 * math.pi
        if dtheta > 0:
            if dtheta < self.left_dtheta:
                self.left_edge = edge_index
                self.left_dtheta = dtheta
            if twopi - dtheta < self.right_dtheta:
                self.right_edge = edge_index
                self.right_dtheta = twopi - dtheta
        elif dtheta < 0:
            dtheta = -dtheta
            if twopi - dtheta < self.left_dtheta:
                self.left_edge = edge_index
                self.left_dtheta = twopi - dtheta
            if dtheta < self.right_dtheta:
                self.right_edge = edge_index
                self.right_dtheta = dtheta
        else:
            # aligned with edge
            if verbose and j is not None:
                print(f"{j}: line is aligned with edge {edge_index}")
            self.left_edge = edge_index
            self.right_edge = edge_index
            self.found = True
        return self


@dataclass
class RiverSegment:
    """
    Args:
        min_relative_distance (float):
            The relative distance along the previous segment where the last intersection occurred. Used to filter
            intersections along the current segment.
        current_point (np.ndarray):
            A 1D array of shape (2,) containing the x, y coordinates of the current point of the line segment.
        previous_point (np.ndarray):
            A 1D array of shape (2,) containing the x, y coordinates of the previous point of the line segment.
    """
    index: int
    min_relative_distance: float
    current_point: np.ndarray
    previous_point: np.ndarray
    distances: np.ndarray = field(default_factory=lambda: np.zeros(0))
    edges: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    nodes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def is_length_zero(self) -> bool:
        """Check if the segment has zero length."""
        return np.array_equal(self.current_point, self.previous_point)

    @property
    def theta(self):
        theta = math.atan2(
            self.current_point[1]- self.previous_point[1],
            self.current_point[0] - self.previous_point[0]
        )
        return theta


def enlarge(
    old_array: np.ndarray,
    new_shape: Tuple
):
    """
    Copy the values of the old array to a new, larger array of specified shape.

    Arguments
    ---------
    old_array : numpy.ndarray
        Array containing the values.
    new_shape : Tuple
        New shape of the array.

    Returns
    -------
    new_array : numpy.ndarray
        Array of shape "new_shape" with the 'first entries filled by the same
        values as contained in "old_array". The data type of the new array is
        equal to that of the old array.
    """
    old_shape = old_array.shape
    print("old: ", old_shape)
    print("new: ", new_shape)
    new_array = np.zeros(new_shape, dtype=old_array.dtype)
    if len(new_shape)==1:
        new_array[:old_shape[0]] = old_array
    elif len(new_shape)==2:
        new_array[:old_shape[0], :old_shape[1]] = old_array
    return new_array


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
        self._find_starting_face(closest_cell_ind)
        self._store_segment_point(current_bank_point)


    def _find_starting_face(self, face_indexes: np.ndarray):
        """Find the starting face for a bank line segment.

        This function determines the face index and vertex indices of the mesh
        that the first point of a bank line segment is associated with.

        Args:
            face_indexes (np.ndarray): Array of possible cell indices.
        """
        self.index = -1
        if len(face_indexes) == 0 and self.verbose:
            print("starting outside mesh")

        on_edge = self.mesh_data.locate_point(self.bank_points[0], face_indexes)
        if not isinstance(on_edge, list):
            self.index = on_edge

        if self.index == -1:
            self._handle_starting_face_on_edge(on_edge)

    def _handle_starting_face_on_edge(self, on_edge):
        """Handle the case where the first bank point is on the edge of a mesh face.

        Args:
            on_edge (List[int]): List of face indices where the first bank point is located on the edge.
        """
        if on_edge:
            if self.verbose:
                print(f"starting on edge of {on_edge}")
            self.index = -2 if len(on_edge) > 1 else on_edge[0]
            self.vindex = on_edge if len(on_edge) > 1 else None
        else:
            if self.verbose:
                print("starting outside mesh")

    def _find_left_right_edges(self, theta, node, j=None) -> EdgeCandidates:
        """
        Helper to find the left and right edges at a node based on the direction theta.

        Args:
            theta (float): Direction angle of the segment.
            node (int): The node index.
            j (int, optional): Step index for verbose output.

        Returns:
            EdgeCandidates: A dataclass containing the left and right edge indices,
                            their angle differences, and a found flag.
        """
        candidates = EdgeCandidates(
            left_edge=-1, left_dtheta=TWO_PI, right_edge=-1, right_dtheta=TWO_PI
        )
        all_node_edges = np.nonzero((self.mesh_data.edge_node == node).any(axis=1))[0]

        if self.verbose and j is not None:
            print(f"{j}: the edges connected to node {node} are {all_node_edges}")

        for ie in all_node_edges:
            reverse = self.mesh_data.edge_node[ie, 0] != node
            theta_edge = self.mesh_data.calculate_edge_angle(ie, reverse=reverse)

            if self.verbose and j is not None:
                print(f"{j}: edge {ie} connects {self.mesh_data.edge_node[ie, :]}")
                print(f"{j}: edge {ie} theta is {theta_edge}")

            dtheta = theta_edge - theta

            candidates.update_edges_by_angle(ie, dtheta, j)
            if candidates.found:
                break

        if self.verbose and j is not None:
            print(f"{j}: the edge to the left is edge {candidates.left_edge}")
            print(f"{j}: the edge to the right is edge {candidates.right_edge}")

        return candidates

    def _resolve_next_face_from_edges(
        self, node, candidates: EdgeCandidates, j=None
    ) -> int:
        """
        Helper to resolve the next face index when traversing between two edges at a node.

        Args:
            node (int): The node index.
            left_edge (int): The edge index to the left.
            right_edge (int): The edge index to the right.
            j (int, optional): Step index for verbose output.

        Returns:
            int: The next face index.
        """
        left_faces = self.mesh_data.edge_face_connectivity[candidates.left_edge, :]
        right_faces = self.mesh_data.edge_face_connectivity[candidates.right_edge, :]

        if left_faces[0] in right_faces and left_faces[1] in right_faces:
            fn1 = self.mesh_data.face_node[left_faces[0]]
            fe1 = self.mesh_data.face_edge_connectivity[left_faces[0]]
            if self.verbose and j is not None:
                print(f"{j}: those edges are shared by two faces: {left_faces}")
                print(f"{j}: face {left_faces[0]} has nodes: {fn1}")
                print(f"{j}: face {left_faces[0]} has edges: {fe1}")
            # nodes of the face should be listed in clockwise order
            # edges[i] is the edge connecting node[i-1] with node[i]
            # the latter is guaranteed by batch.derive_topology_arrays
            if fe1[fn1 == node] == candidates.right_edge:
                index = left_faces[0]
            else:
                index = left_faces[1]
        elif left_faces[0] in right_faces:
            index = left_faces[0]
        elif left_faces[1] in right_faces:
            index = left_faces[1]
        else:
            raise ValueError(
                f"Shouldn't come here .... left edge {candidates.left_edge}"
                f" and right edge {candidates.right_edge} don't share any face"
            )
        return index

    def _store_segment_point(self, current_bank_point, shape_length=None):
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

    def _determine_next_face_on_edge(
        self, segment: RiverSegment, edge, faces
    ):
        """Determine the next face to continue along an edge based on the segment direction."""
        theta = math.atan2(
            self.bank_points[segment.index + 1][1]
            - segment.current_point[1],
            self.bank_points[segment.index + 1][0]
            - segment.current_point[0],
        )
        if self.verbose:
            print(f"{segment.index}: moving in direction theta = {theta}")
        theta_edge = self.mesh_data.calculate_edge_angle(edge)
        if theta == theta_edge or theta == -theta_edge:
            if self.verbose:
                print(f"{segment.index}: continue along edge {edge}")
            index0 = faces
        else:
            # check whether the (extended) segment slices any edge of faces[0]
            fe1 = self.mesh_data.face_edge_connectivity[faces[0]]
            reversed_segment = RiverSegment(
                index=segment.index,
                previous_point=segment.current_point,
                current_point=self.bank_points[segment.index + 1],
                min_relative_distance=0
            )
            _, _, edges = self.mesh_data.calculate_edge_intersections(
                fe1,
                reversed_segment,
                False,
            )
            # yes, a slice (typically 1, but could be 2 if it slices at a node
            # but that doesn't matter) ... so, we continue towards faces[0]
            # if there are no slices for faces[0], we continue towards faces[1]
            index0 = faces[0] if len(edges) > 0 else faces[1]
        return index0

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

    def _update_mesh_index_and_log(self, j, node, edge, faces, index0, prev_b):
        """
        Helper to update mesh index and log transitions for intersect_line_mesh.
        """
        if index0 is not None:
            if self.verbose:
                self._log_mesh_transition(j, "node", node, index0, prev_b)
            if isinstance(index0, (int, np.integer)):
                self.index = index0
            elif hasattr(index0, "__len__") and len(index0) == 1:
                self.index = index0[0]
            else:
                self.index = -2
                self.vindex = index0
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
            polygon_k = Polygon(self.mesh_data.get_face_by_index(self.index))
            if not polygon_k.contains(pnt):
                raise ValueError(
                    f"{j}: ERROR: point actually not contained within {self.index}!"
                )

    def _select_first_crossing(self, segment_state: RiverSegment):
        """Select the first crossing from a set of edges and their associated distances.

        line segment crosses the edge list multiple times
        - moving out of a cell at a corner node
        - moving into and out of the mesh from outside
        """
        bmin = segment_state.distances == np.amin(segment_state.distances)
        segment_state.distances = segment_state.distances[bmin]
        segment_state.edges = segment_state.edges[bmin]
        segment_state.nodes = segment_state.nodes[bmin]

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
        index0 = None
        if not finished:
            index0 = self._resolve_next_face_by_direction(
                theta, node, segment.index
            )
        return False, index0

    def _resolve_next_face_by_direction(self, theta, node, j):
        if self.verbose:
            print(f"{j}: moving in direction theta = {theta}")

        candidates = self._find_left_right_edges(theta, node, j)

        if self.verbose:
            print(f"{j}: the edge to the left is edge {candidates.left_edge}")
            print(f"{j}: the edge to the right is edge {candidates.right_edge}")

        if candidates.left_edge == candidates.right_edge:
            if self.verbose:
                print(f"{j}: continue along edge {candidates.left_edge}")
            index0 = self.mesh_data.edge_face_connectivity[candidates.left_edge, :]
        else:
            if self.verbose:
                print(
                    f"{j}: continue between edges {candidates.left_edge}"
                    f" on the left and {candidates.right_edge} on the right"
                )
            index0 = self._resolve_next_face_from_edges(node, candidates, j)
        return index0

    def _slice_by_node_or_edge(
        self, segment: RiverSegment, node, edge, faces
    ):
        finished = False
        index0 = None
        if node >= 0:
            # if we slice at a node ...
            finished, index0 = self._process_node_transition(segment, node)

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
                index0 = self._determine_next_face_on_edge(segment, edge, faces)
        return finished, index0

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
                self._select_first_crossing(segment)

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
