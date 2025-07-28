"""module for processing mesh-related operations."""
import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from dfastbe.bank_erosion.data_models.calculation import MeshData

__all__ = ["get_slices_ab", "enlarge", "MeshProcessor"]

ATOL = 1e-8
RTOL = 1e-8


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


@dataclass
class SegmentTraversalState:
    index: int
    prev_distance: float
    current_bank_point: np.ndarray
    previous_bank_point: np.ndarray
    distances: np.ndarray = field(default_factory=lambda: np.zeros(0))
    edges: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    nodes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


def _get_slices(
    index: int,
    prev_b: float,
    bpj: np.ndarray,
    bpj1: np.ndarray,
    mesh_data: MeshData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the intersection of a line segment with the edges of a mesh face.

    This function determines where a line segment (defined by two points) intersects the edges of a mesh face.
    It returns the relative distances along the segment and the edges where the intersections occur, as well as
    flags indicating whether the intersections occur at nodes.

    Args:
        index (int):
            Index of the current mesh face. If `index` is negative, the function assumes the segment intersects
            the boundary edges of the mesh.
        prev_b (float):
            The relative distance along the previous segment where the last intersection occurred. Used to filter
            intersections along the current segment.
        bpj (np.ndarray):
            A 1D array of shape (2,) containing the x, y coordinates of the current point of the line segment.
        bpj1 (np.ndarray):
            A 1D array of shape (2,) containing the x, y coordinates of the previous point of the line segment.
        mesh_data (MeshData):
            An instance of the `MeshData` class containing mesh-related data, such as edge coordinates, face-edge
            connectivity, and edge-node connectivity.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            A tuple containing:
            - `b` (np.ndarray): Relative distances along the segment `bpj1-bpj` where the intersections occur.
            - `edges` (np.ndarray): Indices of the edges that are intersected by the segment.
            - `nodes` (np.ndarray): Flags indicating whether the intersections occur at nodes. A value of `-1`
              indicates no intersection at a node, while other values correspond to node indices.

    Raises:
        ValueError:
            If the input data is invalid or inconsistent.

    Notes:
        - If `index` is negative, the function assumes the segment intersects the boundary edges of the mesh.
        - The function uses the `get_slices_core` helper function to calculate the intersections.
        - Intersections at nodes are flagged in the `nodes` array, with the corresponding node indices.

    """
    if index < 0:
        edges = mesh_data.boundary_edge_nrs
    else:
        edges = mesh_data.face_edge_connectivity[index, : mesh_data.n_nodes[index]]
    a, b, edges = _get_slices_core(edges, mesh_data, bpj1, bpj, prev_b, True)
    nodes = -np.ones(a.shape, dtype=np.int64)
    nodes[a == 0] = mesh_data.edge_node[edges[a == 0], 0]
    nodes[a == 1] = mesh_data.edge_node[edges[a == 1], 1]
    return b, edges, nodes

def _get_slices_core(
    edges: np.ndarray,
    mesh_data: MeshData,
    bpj1: np.ndarray,
    bpj: np.ndarray,
    bmin: float,
    bmax1: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the intersection of a line segment with multiple mesh edges.

    This function determines where a line segment intersects a set of mesh edges.
    It calculates the relative distances along the segment and the edges where
    the intersections occur, and returns the indices of the intersected edges.

    Args:
        edges (np.ndarray):
            Array containing the indices of the edges to check for intersections.
        mesh_data (MeshData):
            An instance of the `MeshData` class containing mesh-related data,
            such as edge coordinates and connectivity information.
        bpj1 (np.ndarray):
            A 1D array of shape (2,) containing the x, y coordinates of the
            starting point of the line segment.
        bpj (np.ndarray):
            A 1D array of shape (2,) containing the x, y coordinates of the
            ending point of the line segment.
        bmin (float):
            Minimum relative distance along the segment `bpj1-bpj` at which
            intersections should be considered valid.
        bmax1 (bool, optional):
            If True, limits the relative distance along the segment `bpj1-bpj`
            to a maximum of 1. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            A tuple containing:
            - `a` (np.ndarray): Relative distances along the edges where the
              intersections occur.
            - `b` (np.ndarray): Relative distances along the segment `bpj1-bpj`
              where the intersections occur.
            - `edges` (np.ndarray): Indices of the edges that are intersected
              by the segment.

    Raises:
        ValueError:
            If the input data is invalid or inconsistent.

    Notes:
        - The function uses the `get_slices_ab` helper function to calculate the
          relative distances `a` and `b` for each edge.
        - The `bmin` parameter is used to filter out intersections that occur
          too close to the starting point of the segment.
        - If `bmax1` is True, intersections beyond the endpoint of the segment
          are ignored.
    """
    a, b, slices = get_slices_ab(
        mesh_data.x_edge_coords[edges, 0],
        mesh_data.y_edge_coords[edges, 0],
        mesh_data.x_edge_coords[edges, 1],
        mesh_data.y_edge_coords[edges, 1],
        bpj1[0],
        bpj1[1],
        bpj[0],
        bpj[1],
        bmin,
        bmax1,
    )
    edges = edges[slices]
    return a, b, edges


def get_slices_ab(
    X0: np.ndarray,
    Y0: np.ndarray,
    X1: np.ndarray,
    Y1: np.ndarray,
    xi0: float,
    yi0: float,
    xi1: float,
    yi1: float,
    bmin: float,
    bmax1: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the relative locations a and b at which a segment intersects/slices a number of edges.

    Arguments
    ---------
    X0 : np.ndarray
        Array containing the x-coordinates of the start point of each edge.
    X1 : np.ndarray
        Array containing the x-coordinates of the end point of each edge.
    Y0 : np.ndarray
        Array containing the y-coordinates of the start point of each edge.
    Y1 : np.ndarray
        Array containing the y-coordinates of the end point of each edge.
    xi0 : float
        x-coordinate of start point of the segment.
    xi1 : float
        x-coordinate of end point of the segment.
    yi0 : float
        y-coordinate of start point of the segment.
    yi1 : float
        y-coordinate of end point of the segment.
    bmin : float
        Minimum relative distance from bpj1 at which slice should occur.
    bmax1 : bool
        Flag indicating whether the the relative distance along the segment bpj1-bpj should be limited to 1.

    Returns
    -------
    a : np.ndarray
        Array containing relative distance along each edge.
    b : np.ndarray
        Array containing relative distance along the segment for each edge.
    slices : np.ndarray
        Array containing a flag indicating whether the edge is sliced at a valid location.
    """
    dX = X1 - X0
    dY = Y1 - Y0
    dxi = xi1 - xi0
    dyi = yi1 - yi0
    det = dX * dyi - dY * dxi
    det[det == 0] = 1e-10
    a = (dyi * (xi0 - X0) - dxi * (yi0 - Y0)) / det  # along mesh edge
    b = (dY * (xi0 - X0) - dX * (yi0 - Y0)) / det  # along bank line
    if bmax1:
        slices = np.nonzero((b > bmin) & (b <= 1) & (a >= 0) & (a <= 1))[0]
    else:
        slices = np.nonzero((b > bmin) & (a >= 0) & (a <= 1))[0]
    a = a[slices]
    b = b[slices]
    return a, b, slices


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
        self.ind = 0
        self.index: int
        self.vindex: np.ndarray | int

    def _handle_first_point(self, current_bank_point: np.ndarray):
        dx = self.mesh_data.x_face_coords - current_bank_point[0]
        dy = self.mesh_data.y_face_coords - current_bank_point[1]
        possible_cells = np.nonzero(
            ~(
                (dx < 0).all(axis=1)
                | (dx > 0).all(axis=1)
                | (dy < 0).all(axis=1)
                | (dy > 0).all(axis=1)
            )
        )[0]
        self._find_starting_face(possible_cells)
        self._store_segment_point(current_bank_point)

    def _get_face_coordinates(self, index: int) -> np.ndarray:
        """Returns the coordinates of the index-th mesh face as an (N, 2) array.

        Args:
            index (int): The face index.

        Returns:
            np.ndarray: Array of shape (n_nodes, 2) with x, y coordinates.
        """
        x = self.mesh_data.x_face_coords[
            index : index + 1, : self.mesh_data.n_nodes[index]
        ]
        y = self.mesh_data.y_face_coords[
            index : index + 1, : self.mesh_data.n_nodes[index]
        ]
        return np.concatenate((x, y), axis=0).T

    def _edge_angle(self, edge: int, reverse: bool = False) -> float:
        """Calculate the angle of a mesh edge in radians.

        Args:
            edge (int):
                The edge index.
            reverse (bool):
                If True, computes the angle from end to start.

        Returns:
            float: The angle of the edge in radians.
        """
        start, end = (1, 0) if reverse else (0, 1)
        dx = (
            self.mesh_data.x_edge_coords[edge, end]
            - self.mesh_data.x_edge_coords[edge, start]
        )
        dy = (
            self.mesh_data.y_edge_coords[edge, end]
            - self.mesh_data.y_edge_coords[edge, start]
        )
        return math.atan2(dy, dx)

    def _find_faces_containing_point_on_edge(self, possible_cells) -> List[int]:
        """Get the faces that contain the first bank point on the edge of the mesh.

        This function checks if the first bank point is inside or on the edge of any mesh face.

        Args:
            possible_cells (np.ndarray):
                Array of possible cell indices where the first bank point might be located.

        Returns:
            List[int]: A list of face indices where the first bank point is located.
        """
        pnt = Point(self.bank_points[0])
        on_edge = []
        for k in possible_cells:
            polygon_k = Polygon(self._get_face_coordinates(k))
            if polygon_k.contains(pnt):
                if self.verbose:
                    print(f"starting in {k}")
                self.index = k
                break
            nd = self._get_face_coordinates(k)
            line_k = LineString(np.vstack([nd, nd[0]]))
            if line_k.contains(pnt):
                on_edge.append(k)
        return on_edge

    def _find_starting_face(self, possible_cells: np.ndarray):
        """Find the starting face for a bank line segment.

        This function determines the face index and vertex indices of the mesh
        that the first point of a bank line segment is associated with.

        Args:
            possible_cells (np.ndarray): Array of possible cell indices.
        """
        self.index = -1
        if len(possible_cells) == 0:
            if self.verbose:
                print("starting outside mesh")
            self.index = -1

        on_edge = self._find_faces_containing_point_on_edge(possible_cells)

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

    def _update_candidate_edges_by_angle(
        self, edge_index: int, dtheta: float, candidates: EdgeCandidates, j
    ):
        """Update the left and right edges based on the angle difference."""
        twopi = 2 * math.pi
        if dtheta > 0:
            if dtheta < candidates.left_dtheta:
                candidates.left_edge = edge_index
                candidates.left_dtheta = dtheta
            if twopi - dtheta < candidates.right_dtheta:
                candidates.right_edge = edge_index
                candidates.right_dtheta = twopi - dtheta
        elif dtheta < 0:
            dtheta = -dtheta
            if twopi - dtheta < candidates.left_dtheta:
                candidates.left_edge = edge_index
                candidates.left_dtheta = twopi - dtheta
            if dtheta < candidates.right_dtheta:
                candidates.right_edge = edge_index
                candidates.right_dtheta = dtheta
        else:
            # aligned with edge
            if self.verbose and j is not None:
                print(f"{j}: line is aligned with edge {edge_index}")
            candidates.left_edge = edge_index
            candidates.right_edge = edge_index
            candidates.found = True
        return candidates

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
        two_pi = 2 * math.pi
        candidates = EdgeCandidates(
            left_edge=-1, left_dtheta=two_pi, right_edge=-1, right_dtheta=two_pi
        )
        all_node_edges = np.nonzero((self.mesh_data.edge_node == node).any(axis=1))[0]

        if self.verbose and j is not None:
            print(f"{j}: the edges connected to node {node} are {all_node_edges}")

        for ie in all_node_edges:
            reverse = self.mesh_data.edge_node[ie, 0] != node
            theta_edge = self._edge_angle(ie, reverse=reverse)
            if self.verbose and j is not None:
                print(f"{j}: edge {ie} connects {self.mesh_data.edge_node[ie, :]}")
                print(f"{j}: edge {ie} theta is {theta_edge}")
            dtheta = theta_edge - theta
            self._update_candidate_edges_by_angle(ie, dtheta, candidates, j)
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

    def _resolve_ambiguous_edge_transition(self, segment_state: SegmentTraversalState):
        """Resolve ambiguous edge transitions when a line segment is on the edge of multiple mesh faces."""
        b = np.zeros(0)
        edges = np.zeros(0, dtype=np.int64)
        nodes = np.zeros(0, dtype=np.int64)
        index_src = np.zeros(0, dtype=np.int64)
        for i in self.vindex:
            b1, edges1, nodes1 = _get_slices(
                i,
                segment_state.prev_distance,
                segment_state.current_bank_point,
                segment_state.previous_bank_point,
                self.mesh_data,
            )
            b = np.concatenate((b, b1), axis=0)
            edges = np.concatenate((edges, edges1), axis=0)
            nodes = np.concatenate((nodes, nodes1), axis=0)
            index_src = np.concatenate((index_src, i + 0 * edges1), axis=0)
        segment_state.edges, id_edges = np.unique(edges, return_index=True)
        segment_state.distances = b[id_edges]
        segment_state.nodes = nodes[id_edges]
        index_src = index_src[id_edges]
        if len(index_src) == 1:
            self.index = index_src[0]
            self.vindex = index_src[0:1]
        else:
            self.index = -2

    def _store_segment_point(self, current_bank_point, shape_length=None):
        """Finalize a segment

        Enlarge arrays if needed, set coordinates and index, and increment ind.
        """
        if shape_length is None:
            shape_length = self.ind + 1
        if self.ind == self.coords.shape[0]:
            self.coords = enlarge(self.coords, (shape_length, 2))
            self.face_indexes = enlarge(self.face_indexes, (shape_length,))
        self.coords[self.ind] = current_bank_point
        if self.index == -2:
            self.face_indexes[self.ind] = self.vindex[0]
        else:
            self.face_indexes[self.ind] = self.index
        self.ind += 1

    def _determine_next_face_on_edge(self, bpj, j, edge, faces):
        """Determine the next face to continue along an edge based on the segment direction."""
        theta = math.atan2(
            self.bank_points[j + 1][1] - bpj[1], self.bank_points[j + 1][0] - bpj[0]
        )
        if self.verbose:
            print(f"{j}: moving in direction theta = {theta}")
        theta_edge = self._edge_angle(edge)
        if theta == theta_edge or theta == -theta_edge:
            if self.verbose:
                print(f"{j}: continue along edge {edge}")
            index0 = faces
        else:
            # check whether the (extended) segment slices any edge of faces[0]
            fe1 = self.mesh_data.face_edge_connectivity[faces[0]]
            a, b, edges = _get_slices_core(
                fe1, self.mesh_data, bpj, self.bank_points[j + 1], 0.0, False
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
            polygon_k = Polygon(self._get_face_coordinates(self.index))
            if not polygon_k.contains(pnt):
                raise ValueError(
                    f"{j}: ERROR: point actually not contained within {self.index}!"
                )

    def _select_first_crossing(self, segment_state: SegmentTraversalState):
        """Select the first crossing from a set of edges and their associated distances.

        line segment crosses the edge list multiple times
        - moving out of a cell at a corner node
        - moving into and out of the mesh from outside
        """
        bmin = segment_state.distances == np.amin(segment_state.distances)
        segment_state.distances = segment_state.distances[bmin]
        segment_state.edges = segment_state.edges[bmin]
        segment_state.nodes = segment_state.nodes[bmin]

    def _process_node_transition(self, segment_state: SegmentTraversalState, node):
        """Process the transition at a node when a segment ends or continues."""
        finished = False
        if self.verbose:
            print(
                f"{segment_state.index}: moving via node {node} on edges {segment_state.edges} at {segment_state.distances[0]}"
            )
        # figure out where we will be heading afterwards ...
        if segment_state.distances[0] < 1.0:
            # segment passes through node and enter non-neighbouring cell ...
            # direction of current segment from bpj1 to bpj
            theta = math.atan2(
                segment_state.current_bank_point[1]
                - segment_state.previous_bank_point[1],
                segment_state.current_bank_point[0]
                - segment_state.previous_bank_point[0],
            )
        else:
            if (
                np.isclose(segment_state.distances[0], 1.0, rtol=RTOL, atol=ATOL)
                and segment_state.index == len(self.bank_points) - 1
            ):
                # catch case of last segment
                if self.verbose:
                    print(f"{segment_state.index}: last point ends in a node")
                self._store_segment_point(segment_state.current_bank_point)
                theta = 0.0
                finished = True
            else:
                # this segment ends in the node, so check next segment ...
                # direction of next segment from bpj to bp[j+1]
                theta = math.atan2(
                    self.bank_points[segment_state.index + 1][1]
                    - segment_state.current_bank_point[1],
                    self.bank_points[segment_state.index + 1][0]
                    - segment_state.current_bank_point[0],
                )
        index0 = None
        if not finished:
            index0 = self._resolve_next_face_by_direction(
                theta, node, segment_state.index
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
        self, segment_state: SegmentTraversalState, node, edge, faces
    ):
        finished = False
        index0 = None
        if node >= 0:
            # if we slice at a node ...
            finished, index0 = self._process_node_transition(segment_state, node)

        elif segment_state.distances[0] == 1:
            # ending at slice point, so ending on an edge ...
            if self.verbose:
                print(
                    f"{segment_state.index}: ending on edge {edge} at {segment_state.distances[0]}"
                )
            # figure out where we will be heading afterwards ...
            if segment_state.index == len(self.bank_points) - 1:
                # catch case of last segment
                if self.verbose:
                    print(f"{segment_state.index}: last point ends on an edge")
                self._store_segment_point(segment_state.current_bank_point)
                finished = True
            else:
                index0 = self._determine_next_face_on_edge(
                    segment_state.current_bank_point, segment_state.index, edge, faces
                )
        return finished, index0

    def _process_bank_segment(self, j, bpj):
        shape_multiplier = 2
        segment_state = SegmentTraversalState(
            index=j,
            prev_distance=0,
            current_bank_point=bpj,
            previous_bank_point=self.bank_points[j - 1],
        )
        while True:
            if self.index == -2:
                self._resolve_ambiguous_edge_transition(segment_state)
            elif (
                segment_state.current_bank_point == segment_state.previous_bank_point
            ).all():
                # this is a segment of length 0, skip it since it takes us nowhere
                break
            else:
                segment_state.distances, segment_state.edges, segment_state.nodes = (
                    _get_slices(
                        self.index,
                        segment_state.prev_distance,
                        segment_state.current_bank_point,
                        segment_state.previous_bank_point,
                        self.mesh_data,
                    )
                )

            if len(segment_state.edges) == 0:
                # rest of segment associated with same face
                shape_length = self.ind * shape_multiplier
                self._store_segment_point(
                    segment_state.current_bank_point, shape_length=shape_length
                )
                break
            index0 = None
            if len(segment_state.edges) > 1:
                self._select_first_crossing(segment_state)

            # slice location identified ...
            node = segment_state.nodes[0]
            edge = segment_state.edges[0]
            faces = self.mesh_data.edge_face_connectivity[edge]
            segment_state.prev_distance = segment_state.distances[0]

            finished, index0 = self._slice_by_node_or_edge(
                segment_state,
                node,
                edge,
                faces,
            )
            if finished:
                break

            self._update_mesh_index_and_log(
                j,
                node,
                edge,
                faces,
                index0,
                segment_state.prev_distance,
            )
            segment = (
                segment_state.previous_bank_point
                + segment_state.prev_distance
                * (bpj - segment_state.previous_bank_point)
            )
            shape_length = self.ind * shape_multiplier
            self._store_segment_point(segment, shape_length=shape_length)
            if segment_state.prev_distance == 1:
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
        for j, current_bank_point in enumerate(self.bank_points):
            if self.verbose:
                print(
                    f"Current location: {current_bank_point[0]}, {current_bank_point[1]}"
                )
            if j == 0:
                self._handle_first_point(current_bank_point)
            else:
                self._process_bank_segment(j, current_bank_point)

        # clip to actual length (idx refers to segments, so we can ignore the last value)
        self.coords = self.coords[: self.ind]
        self.face_indexes = self.face_indexes[: self.ind - 1]

        # remove tiny segments
        d = np.sqrt((np.diff(self.coords, axis=0) ** 2).sum(axis=1))
        mask = np.concatenate((np.ones((1), dtype="bool"), d > self.d_thresh))
        self.coords = self.coords[mask, :]
        self.face_indexes = self.face_indexes[mask[1:]]

        # since index refers to segments, don't return the first one
        return self.coords, self.face_indexes
