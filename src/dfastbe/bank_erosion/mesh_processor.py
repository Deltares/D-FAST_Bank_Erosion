"""module for processing mesh-related operations."""
import math
from typing import List, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from dfastbe.bank_erosion.data_models.calculation import MeshData

__all__ = ["get_slices_ab", "enlarge", "intersect_line_mesh"]


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


def _get_face_coordinates(mesh_data: MeshData, index: int) -> np.ndarray:
    """Returns the coordinates of the k-th mesh face as an (N, 2) array.

    Args:
        mesh_data (MeshData): The mesh data object.
        k (int): The face index.

    Returns:
        np.ndarray: Array of shape (n_nodes, 2) with x, y coordinates.
    """
    x = mesh_data.x_face_coords[index : index + 1, : mesh_data.n_nodes[index]]
    y = mesh_data.y_face_coords[index : index + 1, : mesh_data.n_nodes[index]]
    return np.concatenate((x, y), axis=0).T


def edge_angle(mesh_data: MeshData, edge: int, reverse: bool = False) -> float:
    """Calculate the angle of a mesh edge in radians.

    Args:
        mesh_data (MeshData): The mesh data object.
        edge (int): The edge index.
        reverse (bool): If True, computes the angle from end to start.

    Returns:
        float: The angle of the edge in radians.
    """
    start, end = (1, 0) if reverse else (0, 1)
    dx = mesh_data.x_edge_coords[edge, end] - mesh_data.x_edge_coords[edge, start]
    dy = mesh_data.y_edge_coords[edge, end] - mesh_data.y_edge_coords[edge, start]
    return math.atan2(dy, dx)


def log_mesh_transition(
    j, index, vindex, transition_type, transition_index, index0, prev_b
):
    """
    Helper to print mesh transition information for debugging.
    """
    index_str = "outside" if index == -1 else index
    if index == -2:
        index_str = f"edge between {vindex}"
    print(
        f"{j}: moving from {index_str} via {transition_type} {transition_index} to {index0} at b = {prev_b}"
    )


def intersect_line_mesh(
    bp: np.ndarray,
    mesh_data: MeshData,
    d_thresh: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray]:
    """Intersects a line with an unstructured mesh and returns the intersection coordinates and mesh face indices.

    This function determines where a given line (e.g., a bank line) intersects the faces of an unstructured mesh.
    It calculates the intersection points and identifies the mesh faces corresponding to each segment of the line.

    Args:
        bp (np.ndarray):
            A 2D array of shape (N, 2) containing the x, y coordinates of the line to be intersected with the mesh.
        mesh_data (MeshData):
            An instance of the `MeshData` class containing mesh-related data, such as face coordinates, edge coordinates,
            and connectivity information.
        d_thresh (float, optional):
            A distance threshold for filtering out very small segments. Segments shorter than this threshold will be removed.
            Defaults to 0.001.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            A tuple containing:
            - `crds` (np.ndarray): A 2D array of shape (M, 2) containing the x, y coordinates of the intersection points.
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
    crds = np.zeros((len(bp), 2))
    idx = np.zeros(len(bp), dtype=np.int64)
    verbose = False
    ind = 0
    index: int
    vindex: np.ndarray
    for j, bpj in enumerate(bp):
        if verbose:
            print(f"Current location: {bpj[0]}, {bpj[1]}")
        if j == 0:
            # first bp inside or outside?
            dx = mesh_data.x_face_coords - bpj[0]
            dy = mesh_data.y_face_coords - bpj[1]
            possible_cells = np.nonzero(
                ~(
                        (dx < 0).all(axis=1)
                        | (dx > 0).all(axis=1)
                        | (dy < 0).all(axis=1)
                        | (dy > 0).all(axis=1)
                )
            )[0]
            if len(possible_cells) == 0:
                # no cells found ... it must be outside
                index = -1
                if verbose:
                    print("starting outside mesh")
            else:
                # one or more possible cells, check whether it's really inside one of them
                # using np math might be faster, but since it's should only be for a few points let's use Shapely
                # a point on the edge of a polygon is not contained in the polygon.
                # a point on the edge of two polygons will thus be considered outside the mesh whereas it actually isn't.
                pnt = Point(bp[0])
                for k in possible_cells:
                    polygon_k = Polygon(_get_face_coordinates(mesh_data, k))
                    if polygon_k.contains(pnt):
                        index = k
                        if verbose:
                            print(f"starting in {index}")
                        break
                else:
                    on_edge: List[int] = []
                    for k in possible_cells:
                        nd = _get_face_coordinates(mesh_data, k)
                        line_k = LineString(np.concatenate(nd, nd[0:1], axis=0))
                        if line_k.contains(pnt):
                            on_edge.append(k)
                    if not on_edge:
                        index = -1
                        if verbose:
                            print("starting outside mesh")
                    else:
                        if len(on_edge) == 1:
                            index = on_edge[0]
                        else:
                            index = -2
                            vindex = on_edge
                        if verbose:
                            print(f"starting on edge of {on_edge}")
                        raise Exception("determine direction!")
            crds[ind] = bpj
            if index == -2:
                idx[ind] = vindex[0]
            else:
                idx[ind] = index
            ind += 1
        else:
            # second or later point
            bpj1 = bp[j - 1]
            prev_b = 0
            while True:
                if index == -2:
                    b = np.zeros(0)
                    edges = np.zeros(0, dtype=np.int64)
                    nodes = np.zeros(0, dtype=np.int64)
                    index_src = np.zeros(0, dtype=np.int64)
                    for i in vindex:
                        b1, edges1, nodes1 = _get_slices(
                            i,
                            prev_b,
                            bpj,
                            bpj1,
                            mesh_data,
                        )
                        b = np.concatenate((b, b1), axis=0)
                        edges = np.concatenate((edges, edges1), axis=0)
                        nodes = np.concatenate((nodes, nodes1), axis=0)
                        index_src = np.concatenate((index_src, i + 0 * edges1), axis=0)
                    edges, id_edges = np.unique(edges, return_index=True)
                    b = b[id_edges]
                    nodes = nodes[id_edges]
                    index_src = index_src[id_edges]
                    if len(index_src) == 1:
                        index = index_src[0]
                        vindex = index_src[0:1]
                elif (bpj == bpj1).all():
                    # this is a segment of length 0, skip it since it takes us nowhere
                    break
                else:
                    b, edges, nodes = _get_slices(
                        index,
                        prev_b,
                        bpj,
                        bpj1,
                        mesh_data,
                    )

                if len(edges) == 0:
                    # rest of segment associated with same face
                    if verbose:
                        if prev_b > 0:
                            print(f"{j}: -- no further slices along this segment --")
                        else:
                            print(f"{j}: -- no slices along this segment --")
                        if index >= 0:
                            pnt = Point(bpj)
                            polygon_k = Polygon(_get_face_coordinates(mesh_data, index))
                            if not polygon_k.contains(pnt):
                                raise Exception(
                                    f"{j}: ERROR: point actually not contained within {index}!"
                                )
                    if ind == crds.shape[0]:
                        crds = enlarge(crds, (2 * ind, 2))
                        idx = enlarge(idx, (2 * ind,))
                    crds[ind] = bpj
                    idx[ind] = index
                    ind += 1
                    break
                else:
                    index0 = None
                    if len(edges) > 1:
                        # line segment crosses the edge list multiple times
                        # - moving out of a cell at a corner node
                        # - moving into and out of the mesh from outside
                        # select first crossing ...
                        bmin = b == np.amin(b)
                        b = b[bmin]
                        edges = edges[bmin]
                        nodes = nodes[bmin]

                    # slice location identified ...
                    node = nodes[0]
                    edge = edges[0]
                    faces = mesh_data.edge_face_connectivity[edge]
                    prev_b = b[0]

                    if node >= 0:
                        # if we slice at a node ...
                        if verbose:
                            print(
                                f"{j}: moving via node {node} on edges {edges} at {b[0]}"
                            )
                        # figure out where we will be heading afterwards ...
                        all_node_edges = np.nonzero(
                            (mesh_data.edge_node == node).any(axis=1)
                        )[0]
                        if b[0] < 1.0:
                            # segment passes through node and enter non-neighbouring cell ...
                            # direction of current segment from bpj1 to bpj
                            theta = math.atan2(bpj[1] - bpj1[1], bpj[0] - bpj1[0])
                        else:
                            if b[0] == 1.0 and j == len(bp) - 1:
                                # catch case of last segment
                                if verbose:
                                    print(f"{j}: last point ends in a node")
                                if ind == crds.shape[0]:
                                    crds = enlarge(crds, (ind + 1, 2))
                                    idx = enlarge(idx, (ind + 1,))
                                crds[ind] = bpj
                                if index == -2:
                                    idx[ind] = vindex[0]
                                else:
                                    idx[ind] = index
                                ind += 1
                                break
                            else:
                                # this segment ends in the node, so check next segment ...
                                # direction of next segment from bpj to bp[j+1]
                                theta = math.atan2(
                                    bp[j + 1][1] - bpj[1], bp[j + 1][0] - bp[j][0]
                                )
                        if verbose:
                            print(f"{j}: moving in direction theta = {theta}")
                        twopi = 2 * math.pi
                        left_edge = -1
                        left_dtheta = twopi
                        right_edge = -1
                        right_dtheta = twopi
                        if verbose:
                            print(
                                f"{j}: the edges connected to node {node} are {all_node_edges}"
                            )
                        for ie in all_node_edges:
                            reverse = not mesh_data.edge_node[ie, 0] == node
                            theta_edge = edge_angle(mesh_data, ie, reverse=reverse)
                            if verbose:
                                print(
                                    f"{j}: edge {ie} connects {mesh_data.edge_node[ie, :]}"
                                )
                                print(f"{j}: edge {ie} theta is {theta_edge}")
                            dtheta = theta_edge - theta
                            if dtheta > 0:
                                if dtheta < left_dtheta:
                                    left_edge = ie
                                    left_dtheta = dtheta
                                if twopi - dtheta < right_dtheta:
                                    right_edge = ie
                                    right_dtheta = twopi - dtheta
                            elif dtheta < 0:
                                dtheta = -dtheta
                                if twopi - dtheta < left_dtheta:
                                    left_edge = ie
                                    left_dtheta = twopi - dtheta
                                if dtheta < right_dtheta:
                                    right_edge = ie
                                    right_dtheta = dtheta
                            else:
                                # aligned with edge
                                if verbose:
                                    print(f"{j}: line is aligned with edge {ie}")
                                left_edge = ie
                                right_edge = ie
                                break
                        if verbose:
                            print(f"{j}: the edge to the left is edge {left_edge}")
                            print(f"{j}: the edge to the right is edge {right_edge}")
                        if left_edge == right_edge:
                            if verbose:
                                print(f"{j}: continue along edge {left_edge}")
                            index0 = mesh_data.edge_face_connectivity[left_edge, :]
                        else:
                            if verbose:
                                print(
                                    f"{j}: continue between edges {left_edge} on the left and {right_edge} on the right"
                                )
                            left_faces = mesh_data.edge_face_connectivity[left_edge, :]
                            right_faces = mesh_data.edge_face_connectivity[
                                          right_edge, :
                                          ]
                            if (
                                    left_faces[0] in right_faces
                                    and left_faces[1] in right_faces
                            ):
                                # the two edges are shared by two faces ... check first face
                                fn1 = mesh_data.face_node[left_faces[0]]
                                fe1 = mesh_data.face_edge_connectivity[left_faces[0]]
                                if verbose:
                                    print(
                                        f"{j}: those edges are shared by two faces: {left_faces}"
                                    )
                                    print(f"{j}: face {left_faces[0]} has nodes: {fn1}")
                                    print(f"{j}: face {left_faces[0]} has edges: {fe1}")
                                # here we need that the nodes of the face are listed in clockwise order
                                # and that edges[i] is the edge connecting node[i-1] with node[i]
                                # the latter is guaranteed by batch.derive_topology_arrays
                                if fe1[fn1 == node] == right_edge:
                                    index0 = left_faces[0]
                                else:
                                    index0 = left_faces[1]
                            elif left_faces[0] in right_faces:
                                index0 = left_faces[0]
                            elif left_faces[1] in right_faces:
                                index0 = left_faces[1]
                            else:
                                raise Exception(
                                    f"Shouldn't come here .... left edge {left_edge} and right edge {right_edge} don't share any face"
                                )

                    elif b[0] == 1:
                        # ending at slice point, so ending on an edge ...
                        if verbose:
                            print(f"{j}: ending on edge {edge} at {b[0]}")
                        # figure out where we will be heading afterwards ...
                        if j == len(bp) - 1:
                            # catch case of last segment
                            if verbose:
                                print(f"{j}: last point ends on an edge")
                            if ind == crds.shape[0]:
                                crds = enlarge(crds, (ind + 1, 2))
                                idx = enlarge(idx, (ind + 1,))
                            crds[ind] = bpj
                            if index == -2:
                                idx[ind] = vindex[0]
                            else:
                                idx[ind] = index
                            ind += 1
                            break
                        else:
                            # this segment ends on the edge, so check next segment ...
                            # direction of next segment from bpj to bp[j+1]
                            theta = math.atan2(
                                bp[j + 1][1] - bpj[1], bp[j + 1][0] - bp[j][0]
                            )
                        if verbose:
                            print(f"{j}: moving in direction theta = {theta}")
                        theta_edge = edge_angle(mesh_data, edge)
                        if theta == theta_edge or theta == -theta_edge:
                            # aligned with edge
                            if verbose:
                                print(f"{j}: continue along edge {edge}")
                            index0 = faces
                        else:
                            # check whether the (extended) segment slices any edge of faces[0]
                            fe1 = mesh_data.face_edge_connectivity[faces[0]]
                            a, b, edges = _get_slices_core(
                                fe1, mesh_data, bpj, bp[j + 1], 0.0, False
                            )
                            if len(edges) > 0:
                                # yes, a slice (typically 1, but could be 2 if it slices at a node
                                # but that doesn't matter) ... so, we continue towards faces[0]
                                index0 = faces[0]
                            else:
                                # no slice for faces[0], so we must be going in the other direction
                                index0 = faces[1]

                    if index0 is not None:
                        if verbose:
                            log_mesh_transition(
                                j, index, vindex, "node", node, index0, prev_b
                            )

                        if (
                                isinstance(index0, int)
                                or isinstance(index0, np.int32)
                                or isinstance(index0, np.int64)
                        ):
                            index = index0
                        elif len(index0) == 1:
                            index = index0[0]
                        else:
                            index = -2
                            vindex = index0

                    elif faces[0] == index:
                        if verbose:
                            log_mesh_transition(
                                j, index, vindex, "edge", edge, faces[1], prev_b
                            )
                        index = faces[1]
                    elif faces[1] == index:
                        if verbose:
                            log_mesh_transition(
                                j, index, vindex, "edge", edge, faces[0], prev_b
                            )
                        index = faces[0]
                    else:
                        raise Exception(
                            f"Shouldn't come here .... index {index} differs from both faces "
                            f"{faces[0]} and {faces[1]} associated with slicing edge {edge}"
                        )
                    if ind == crds.shape[0]:
                        crds = enlarge(crds, (2 * ind, 2))
                        idx = enlarge(idx, (2 * ind,))
                    crds[ind] = bpj1 + prev_b * (bpj - bpj1)
                    if index == -2:
                        idx[ind] = vindex[0]
                    else:
                        idx[ind] = index
                    ind += 1
                    if prev_b == 1:
                        break

    # clip to actual length (idx refers to segments, so we can ignore the last value)
    crds = crds[:ind]
    idx = idx[: ind - 1]

    # remove tiny segments
    d = np.sqrt((np.diff(crds, axis=0) ** 2).sum(axis=1))
    mask = np.concatenate((np.ones((1), dtype="bool"), d > d_thresh))
    crds = crds[mask, :]
    idx = idx[mask[1:]]

    # since index refers to segments, don't return the first one
    return crds, idx
