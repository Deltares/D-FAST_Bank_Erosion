"""This module defines data structures and methods for handling mesh data and river segments."""

from typing import List, Tuple
import numpy as np
import math
from dataclasses import dataclass, field
from shapely.geometry import Point, Polygon, LineString

TWO_PI = 2 * math.pi

__all__ = ["MeshData", "RiverSegment"]


@dataclass
class RiverSegment:
    """Represents a segment of a river line.

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
        """Calculate the angle of the segment in radians."""
        theta = math.atan2(
            self.current_point[1] - self.previous_point[1],
            self.current_point[0] - self.previous_point[0],
        )
        return theta

    def select_first_intersection(self):
        """Select the first crossing from a set of edges and their associated distances.

        line segment crosses the edge list multiple times
            - moving out of a cell at a corner node
            - moving into and out of the mesh from outside
        """
        # get the minimum distance from the distances array
        min_distance_indices = self.distances == np.amin(self.distances)
        self.distances = self.distances[min_distance_indices]
        self.edges = self.edges[min_distance_indices]
        self.nodes = self.nodes[min_distance_indices]


@dataclass
class Edges:
    """Dataclass to hold edge candidates for left and right edges.

    Args:
        left (int):
            Index of the left edge.
        left_theta (float):
            Angle of the left edge in radians.
        right (int):
            Index of the right edge.
        right_theta (float):
            Angle of the right edge in radians.
        found (bool):
            Flag indicating whether a valid edge pair was found.
    """

    left: int
    left_theta: float
    right: int
    right_theta: float
    found: bool = False

    def update_edges_by_angle(
        self, edge_index: int, dtheta: float, j, verbose: bool = False
    ):
        """Update the left and right edges based on the angle difference."""

        if dtheta > 0:
            if dtheta < self.left_theta:
                self.left = edge_index
                self.left_theta = dtheta
            if TWO_PI - dtheta < self.right_theta:
                self.right = edge_index
                self.right_theta = TWO_PI - dtheta
        elif dtheta < 0:
            dtheta = -dtheta
            if TWO_PI - dtheta < self.left_theta:
                self.left = edge_index
                self.left_theta = TWO_PI - dtheta
            if dtheta < self.right_theta:
                self.right = edge_index
                self.right_theta = dtheta
        else:
            # aligned with edge
            if verbose and j is not None:
                print(f"{j}: line is aligned with edge {edge_index}")

            self.left = edge_index
            self.right = edge_index
            self.found = True


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
    verbose: bool = False

    def get_face_by_index(self, index: int, as_polygon: bool = False) -> np.ndarray | Polygon:
        """Returns the coordinates of the index-th mesh face as an (N, 2) array.

        Args:
            index:
                The face index.
            as_polygon:
                whither to return the face as a shapely polygon or not. Default is False

        Returns:
            np.ndarray:
                Array of shape (n_nodes, 2) with x, y coordinates.
        """
        x = self.x_face_coords[index : index + 1, : self.n_nodes[index]]
        y = self.y_face_coords[index : index + 1, : self.n_nodes[index]]
        face = np.concatenate((x, y), axis=0).T

        if as_polygon:
            face = Polygon(face)
        return face

    def locate_point(
        self, point: Point | np.ndarray | list | Tuple, face_index: int | np.ndarray
    ) -> int | List[int]:
        """Locate a point in the mesh faces.

        Args:
            point:
                The point to check.
            face_index:
                The index of the mesh face.

        Returns:
            indexes (int|list[int]):
                index if the face that the point is located in, or a list of indexes if the point is on the edge of
                multiple faces.
        """
        if not isinstance(point, Point) and isinstance(
            point, (list, tuple, np.ndarray)
        ):
            point = Point(point)
        else:
            raise TypeError(
                "point must be a Point object, a list, a tuple, or an np.ndarray of coordinates"
            )

        index_list = []
        for ind in face_index:
            face = self.get_face_by_index(ind)
            face_polygon = Polygon(face)
            if face_polygon.contains(point):
                return ind
            else:
                # create a closed line string from the face coordinates
                closed_line = LineString(np.vstack([face, face[0]]))
                if closed_line.contains(point):
                    index_list.append(ind)

        return index_list

    def find_segment_intersections(
        self,
        index: int,
        segment: RiverSegment,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the intersection of a line segment with the edges of a mesh face.

        This function determines where a line segment (defined by two points) intersects the edges of a mesh face.
        It returns the relative distances along the segment and the edges where the intersections occur, as well as
        flags indicating whether the intersections occur at nodes.

        Args:
            index (int):
                Index of the current mesh face. If `index` is negative, the function assumes the segment intersects
                the boundary edges of the mesh.
            segment (RiverSegment):
                A `RiverSegment` object containing the previous and current points of the segment, as well as the
                minimum relative distance along the segment where the last intersection occurred.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - b (np.ndarray):
                    Relative distances along the segment `bpj1-bpj` where the intersections occur.
                - edges (np.ndarray):
                    Indices of the edges that are intersected by the segment.
                - nodes (np.ndarray):
                    Flags indicating whether the intersections occur at nodes. A value of `-1` indicates no
                    intersection at a node, while other values correspond to node indices.

        Raises:
            ValueError:
                If the input data is invalid or inconsistent.

        Notes:
            - If `index` is negative, the function assumes the segment intersects the boundary edges of the mesh.
            - The function uses the `get_slices_core` helper function to calculate the intersections.
            - Intersections at nodes are flagged in the `nodes` array, with the corresponding node indices.

        """
        if index < 0:
            edges = self.boundary_edge_nrs
        else:
            edges = self.face_edge_connectivity[index, : self.n_nodes[index]]
        edge_relative_dist, segment_relative_dist, edges = (
            self.calculate_edge_intersections(edges, segment, True)
        )
        is_intersected_at_node = -np.ones(edge_relative_dist.shape, dtype=np.int64)
        is_intersected_at_node[edge_relative_dist == 0] = self.edge_node[
            edges[edge_relative_dist == 0], 0
        ]
        is_intersected_at_node[edge_relative_dist == 1] = self.edge_node[
            edges[edge_relative_dist == 1], 1
        ]

        return segment_relative_dist, edges, is_intersected_at_node

    def calculate_edge_intersections(
        self,
        edges: np.ndarray,
        segment: RiverSegment,
        limit_relative_distance: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the intersection of a line segment with multiple mesh edges.

        This function determines where a line segment intersects a set of mesh edges.
        It calculates the relative distances along the segment and the edges where
        the intersections occur, and returns the indices of the intersected edges.

        Args:
            edges (np.ndarray):
                Array containing the indices of the edges to check for intersections.
            segment (RiverSegment):
                A `RiverSegment` object containing the previous and current points of the segment, as well as the
                minimum relative distance along the segment where the last intersection occurred.
            limit_relative_distance (bool, optional):
                If True, limits the relative distance along the segment `bpj1-bpj`
                to a maximum of 1. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - a (np.ndarray): Relative distances along the edges where the
                  intersections occur.
                - b (np.ndarray): Relative distances along the segment `bpj1-bpj`
                  where the intersections occur.
                - edges (np.ndarray): Indices of the edges that are intersected
                  by the segment.

        Raises:
            ValueError:
                If the input data is invalid or inconsistent.

        Notes:
            - The function uses the `get_slices_ab` helper function to calculate the
              relative distances `a` and `b` for each edge.
            - The `bmin` parameter is used to filter out intersections that occur
              too close to the starting point of the segment.
            - If `limit_relative_distance` is True, intersections beyond the endpoint of the segment
              are ignored.
        """
        from dfastbe.bank_erosion.utils import calculate_segment_edge_intersections

        edge_relative_dist, segment_relative_dist, valid_intersections = (
            calculate_segment_edge_intersections(
                self.x_edge_coords[edges, 0],
                self.y_edge_coords[edges, 0],
                self.x_edge_coords[edges, 1],
                self.y_edge_coords[edges, 1],
                segment.previous_point[0],
                segment.previous_point[1],
                segment.current_point[0],
                segment.current_point[1],
                segment.min_relative_distance,
                limit_relative_distance,
            )
        )
        edges = edges[valid_intersections]
        return edge_relative_dist, segment_relative_dist, edges

    def resolve_ambiguous_edge_transition(self, segment: RiverSegment, vindex):
        """Resolve ambiguous edge transitions when a line segment is on the edge of multiple mesh faces."""
        b = np.zeros(0)
        edges = np.zeros(0, dtype=np.int64)
        nodes = np.zeros(0, dtype=np.int64)
        index_src = np.zeros(0, dtype=np.int64)

        for i in vindex:
            b1, edges1, nodes1 = self.find_segment_intersections(i, segment)
            b = np.concatenate((b, b1), axis=0)
            edges = np.concatenate((edges, edges1), axis=0)
            nodes = np.concatenate((nodes, nodes1), axis=0)
            index_src = np.concatenate((index_src, i + 0 * edges1), axis=0)

        segment.edges, id_edges = np.unique(edges, return_index=True)
        segment.distances = b[id_edges]
        segment.nodes = nodes[id_edges]
        index_src = index_src[id_edges]

        return index_src

    def calculate_edge_angle(self, edge: int, reverse: bool = False) -> float:
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
        dx = self.x_edge_coords[edge, end] - self.x_edge_coords[edge, start]
        dy = self.y_edge_coords[edge, end] - self.y_edge_coords[edge, start]

        return math.atan2(dy, dx)

    def find_edges(self, theta, node, verbose_index: int = None) -> Edges:
        """
        Helper to find the left and right edges at a node based on the direction theta.

        Args:
            theta (float):
                Direction angle of the segment.
            node (int):
                The node index.
            verbose_index (int, optional):
                Step index for verbose output.

        Returns:
            Edges:
                A dataclass containing the left and right edge indices, their angle differences, and a found flag.
        """
        all_node_edges = np.nonzero((self.edge_node == node).any(axis=1))[0]

        if self.verbose and verbose_index is not None:
            print(
                f"{verbose_index}: the edges connected to node {node} are {all_node_edges}"
            )

        edges = Edges(left=-1, left_theta=TWO_PI, right=-1, right_theta=TWO_PI)

        for ie in all_node_edges:
            reverse = self.edge_node[ie, 0] != node
            theta_edge = self.calculate_edge_angle(ie, reverse=reverse)

            if self.verbose and verbose_index is not None:
                print(f"{verbose_index}: edge {ie} connects {self.edge_node[ie, :]}")
                print(f"{verbose_index}: edge {ie} theta is {theta_edge}")

            dtheta = theta_edge - theta

            edges.update_edges_by_angle(ie, dtheta, verbose_index)
            if edges.found:
                break

        if self.verbose and verbose_index is not None:
            print(f"{verbose_index}: the edge to the left is edge {edges.left}")
            print(f"{verbose_index}: the edge to the right is edge {edges.right}")

        return edges

    def resolve_next_face_from_edges(
        self, node, edges: Edges, verbose_index: int = None
    ) -> int:
        """
        Helper to resolve the next face index when traversing between two edges at a node.

        Args:
            node (int): The node index.
            edges (Edges):
                The edges connecting the node, containing left and right edge indices.
            verbose_index (int, optional):
                Step index for verbose output.

        Returns:
            next_face_index (int):
                The next face index.
        """
        left_faces = self.edge_face_connectivity[edges.left, :]
        right_faces = self.edge_face_connectivity[edges.right, :]

        if left_faces[0] in right_faces and left_faces[1] in right_faces:
            fn1 = self.face_node[left_faces[0]]
            fe1 = self.face_edge_connectivity[left_faces[0]]

            if self.verbose and verbose_index is not None:
                print(
                    f"{verbose_index}: those edges are shared by two faces: {left_faces}"
                )
                print(f"{verbose_index}: face {left_faces[0]} has nodes: {fn1}")
                print(f"{verbose_index}: face {left_faces[0]} has edges: {fe1}")

            # nodes of the face should be listed in clockwise order
            # edges[i] is the edge connecting node[i-1] with node[i]
            # the latter is guaranteed by batch.derive_topology_arrays
            if fe1[fn1 == node] == edges.right:
                next_face_index = left_faces[0]
            else:
                next_face_index = left_faces[1]

        elif left_faces[0] in right_faces:
            next_face_index = left_faces[0]
        elif left_faces[1] in right_faces:
            next_face_index = left_faces[1]
        else:
            raise ValueError(
                f"Shouldn't come here .... left edge {edges.left}"
                f" and right edge {edges.right} don't share any face"
            )
        return next_face_index