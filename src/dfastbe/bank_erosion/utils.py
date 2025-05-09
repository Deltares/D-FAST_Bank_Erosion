"""Bank erosion utilities."""

import math
import sys
from typing import List, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from dfastbe.bank_erosion.data_models import (
    BankData,
    DischargeCalculationParameters,
    ErosionRiverData,
    MeshData,
    ParametersPerBank,
    SingleBank,
    SingleErosion,
)
from dfastbe.io.logger import log_text
from dfastbe.io.data_models import LineGeometry
from dfastbe.utils import on_right_side

EPS = sys.float_info.epsilon
water_density = 1000  # density of water [kg/m3]
g = 9.81  # gravitational acceleration [m/s2]


class BankLinesProcessor:
    """Class to process bank lines and intersect them with a mesh."""

    def __init__(self, river_data: ErosionRiverData):
        """Constructor for BankLinesProcessor."""
        self.bank_lines = river_data.bank_lines
        self.river_center_line = river_data.river_center_line.as_array()
        self.num_bank_lines = len(self.bank_lines)

    def intersect_with_mesh(self, mesh_data: MeshData) -> BankData:
        """Intersect with Mesh."""
        n_bank_lines = len(self.bank_lines)

        bank_line_coords = []
        bank_face_indices = []
        for bank_index in range(n_bank_lines):
            line_coords = np.array(self.bank_lines.geometry[bank_index].coords)
            log_text("bank_nodes", data={"ib": bank_index + 1, "n": len(line_coords)})

            coords_along_bank, face_indices = intersect_line_mesh(
                line_coords, mesh_data
            )
            bank_line_coords.append(coords_along_bank)
            bank_face_indices.append(face_indices)

        # linking bank lines to chainage
        log_text("chainage_to_banks")
        bank_chainage_midpoints = [None] * n_bank_lines
        is_right_bank = [True] * n_bank_lines
        for bank_index, coords in enumerate(bank_line_coords):
            segment_mid_points = LineGeometry((coords[:-1, :] + coords[1:, :]) / 2)
            chainage_mid_points = segment_mid_points.intersect_with_line(
                self.river_center_line
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
                coords, self.river_center_line[:, :2]
            )
            if is_right_bank[bank_index]:
                log_text("right_side_bank", data={"ib": bank_index + 1})
            else:
                log_text("left_side_bank", data={"ib": bank_index + 1})

        bank_order = tuple("right" if val else "left" for val in is_right_bank)
        data = dict(
            is_right_bank=is_right_bank,
            bank_line_coords=bank_line_coords,
            bank_face_indices=bank_face_indices,
            bank_chainage_midpoints=bank_chainage_midpoints,
        )
        return BankData.from_column_arrays(
            data,
            SingleBank,
            bank_lines=self.bank_lines,
            n_bank_lines=n_bank_lines,
            bank_order=bank_order,
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
            print("Current location: {}, {}".format(bpj[0], bpj[1]))
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
                    polygon_k = Polygon(
                        np.concatenate(
                            (
                                mesh_data.x_face_coords[
                                    k : k + 1, : mesh_data.n_nodes[k]
                                ],
                                mesh_data.y_face_coords[
                                    k : k + 1, : mesh_data.n_nodes[k]
                                ],
                            ),
                            axis=0,
                        ).T
                    )
                    if polygon_k.contains(pnt):
                        index = k
                        if verbose:
                            print("starting in {}".format(index))
                        break
                else:
                    on_edge: List[int] = []
                    for k in possible_cells:
                        nd = np.concatenate(
                            (
                                mesh_data.x_face_coords[
                                    k : k + 1, : mesh_data.n_nodes[k]
                                ],
                                mesh_data.y_face_coords[
                                    k : k + 1, : mesh_data.n_nodes[k]
                                ],
                            ),
                            axis=0,
                        ).T
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
                            print("starting on edge of {}".format(on_edge))
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
                        b1, edges1, nodes1 = get_slices(
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
                    b, edges, nodes = get_slices(
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
                            print(
                                "{}: -- no further slices along this segment --".format(
                                    j
                                )
                            )
                        else:
                            print("{}: -- no slices along this segment --".format(j))
                        if index >= 0:
                            pnt = Point(bpj)
                            polygon_k = Polygon(
                                np.concatenate(
                                    (
                                        mesh_data.x_face_coords[
                                            index : index + 1,
                                            : mesh_data.n_nodes[index],
                                        ],
                                        mesh_data.y_face_coords[
                                            index : index + 1,
                                            : mesh_data.n_nodes[index],
                                        ],
                                    ),
                                    axis=0,
                                ).T
                            )
                            if not polygon_k.contains(pnt):
                                raise Exception(
                                    "{}: ERROR: point actually not contained within {}!".format(
                                        j, index
                                    )
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
                                "{}: moving via node {} on edges {} at {}".format(
                                    j, node, edges, b[0]
                                )
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
                                    print("{}: last point ends in a node".format(j))
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
                            print("{}: moving in direction theta = {}".format(j, theta))
                        twopi = 2 * math.pi
                        left_edge = -1
                        left_dtheta = twopi
                        right_edge = -1
                        right_dtheta = twopi
                        if verbose:
                            print(
                                "{}: the edges connected to node {} are {}".format(
                                    j, node, all_node_edges
                                )
                            )
                        for ie in all_node_edges:
                            if mesh_data.edge_node[ie, 0] == node:
                                theta_edge = math.atan2(
                                    mesh_data.y_edge_coords[ie, 1]
                                    - mesh_data.y_edge_coords[ie, 0],
                                    mesh_data.x_edge_coords[ie, 1]
                                    - mesh_data.x_edge_coords[ie, 0],
                                )
                            else:
                                theta_edge = math.atan2(
                                    mesh_data.y_edge_coords[ie, 0]
                                    - mesh_data.y_edge_coords[ie, 1],
                                    mesh_data.x_edge_coords[ie, 0]
                                    - mesh_data.x_edge_coords[ie, 1],
                                )
                            if verbose:
                                print(
                                    "{}: edge {} connects {}".format(
                                        j, ie, mesh_data.edge_node[ie, :]
                                    )
                                )
                                print(
                                    "{}: edge {} theta is {}".format(j, ie, theta_edge)
                                )
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
                                    print(
                                        "{}: line is aligned with edge {}".format(j, ie)
                                    )
                                left_edge = ie
                                right_edge = ie
                                break
                        if verbose:
                            print(
                                "{}: the edge to the left is edge {}".format(
                                    j, left_edge
                                )
                            )
                            print(
                                "{}: the edge to the right is edge {}".format(
                                    j, left_edge
                                )
                            )
                        if left_edge == right_edge:
                            if verbose:
                                print("{}: continue along edge {}".format(j, left_edge))
                            index0 = mesh_data.edge_face_connectivity[left_edge, :]
                        else:
                            if verbose:
                                print(
                                    "{}: continue between edges {} on the left and {} on the right".format(
                                        j, left_edge, right_edge
                                    )
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
                                        "{}: those edges are shared by two faces: {}".format(
                                            j, left_faces
                                        )
                                    )
                                    print(
                                        "{}: face {} has nodes: {}".format(
                                            j, left_faces[0], fn1
                                        )
                                    )
                                    print(
                                        "{}: face {} has edges: {}".format(
                                            j, left_faces[0], fe1
                                        )
                                    )
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
                                    "Shouldn't come here .... left edge {} and right edge {} don't share any face".format(
                                        left_edge, right_edge
                                    )
                                )

                    elif b[0] == 1:
                        # ending at slice point, so ending on an edge ...
                        if verbose:
                            print("{}: ending on edge {} at {}".format(j, edge, b[0]))
                        # figure out where we will be heading afterwards ...
                        if j == len(bp) - 1:
                            # catch case of last segment
                            if verbose:
                                print("{}: last point ends on an edge".format(j))
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
                            print("{}: moving in direction theta = {}".format(j, theta))
                        theta_edge = math.atan2(
                            mesh_data.y_edge_coords[edge, 1]
                            - mesh_data.y_edge_coords[edge, 0],
                            mesh_data.x_edge_coords[edge, 1]
                            - mesh_data.x_edge_coords[edge, 0],
                        )
                        if theta == theta_edge or theta == -theta_edge:
                            # aligned with edge
                            if verbose:
                                print("{}: continue along edge {}".format(j, edge))
                            index0 = faces
                        else:
                            # check whether the (extended) segment slices any edge of faces[0]
                            fe1 = mesh_data.face_edge_connectivity[faces[0]]
                            a, b, edges = get_slices_core(
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
                            if index == -1:
                                print(
                                    "{}: moving from outside via node {} to {} at b = {}".format(
                                        j, node, index0, prev_b
                                    )
                                )
                            elif index == -2:
                                print(
                                    "{}: moving from edge between {} via node {} to {} at b = {}".format(
                                        j, vindex, node, index0, prev_b
                                    )
                                )
                            else:
                                print(
                                    "{}: moving from {} via node {} to {} at b = {}".format(
                                        j, index, node, index0, prev_b
                                    )
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
                            print(
                                "{}: moving from {} via edge {} to {} at b = {}".format(
                                    j, index, edge, faces[1], prev_b
                                )
                            )
                        index = faces[1]
                    elif faces[1] == index:
                        if verbose:
                            print(
                                "{}: moving from {} via edge {} to {} at b = {}".format(
                                    j, index, edge, faces[0], prev_b
                                )
                            )
                        index = faces[0]
                    else:
                        raise Exception(
                            "Shouldn't come here .... index {} differs from both faces {} and {} associated with slicing edge {}".format(
                                index, faces[0], faces[1], edge
                            )
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


def comp_erosion_eq(
    bank_height: np.ndarray,
    segment_length: np.ndarray,
    water_level_fairway_ref: np.ndarray,
    discharge_level_pars: ParametersPerBank,
    bank_fairway_dist: np.ndarray,
    water_depth_fairway: np.ndarray,
    erosion_inputs: SingleErosion,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the equilibrium bank erosion.

    Args:
        bank_height : np.ndarray
            Array containing bank height [m]
        segment_length : np.ndarray
            Array containing length of the segment [m]
        water_level_fairway_ref : np.ndarray
            Array containing water level at fairway [m]
        discharge_level_pars (ParametersPerBank):
            Discharge level parameters object containing the following attributes.
                ship_velocity : np.ndarray
                    Array containing ship velocity [m/s]
                ship_type : np.ndarray
                    Array containing ship type [-]
                ship_draught : np.ndarray
                    Array containing ship draught [m]
                mu_slope : np.ndarray
                    Array containing slope [-]
        bank_fairway_dist : np.ndarray
            Array containing distance from bank to fairway [m]
        water_depth_fairway : np.ndarray
            Array containing water depth at the fairway [m]
        erosion_inputs (ErosionInputs):
            ErosionInputs object.

    Returns:
        dn_eq : np.ndarray
             Equilibrium bank erosion distance [m]
        dv_eq : np.ndarray
             Equilibrium bank erosion volume [m]
    """
    # ship induced wave height at the beginning of the foreshore
    h0 = comp_hw_ship_at_bank(
        bank_fairway_dist,
        erosion_inputs.wave_fairway_distance_0,
        erosion_inputs.wave_fairway_distance_1,
        water_depth_fairway,
        discharge_level_pars.ship_type,
        discharge_level_pars.ship_draught,
        discharge_level_pars.ship_velocity,
    )
    h0 = np.maximum(h0, EPS)

    zup = np.minimum(bank_height, water_level_fairway_ref + 2 * h0)
    zdo = np.maximum(
        water_level_fairway_ref - 2 * h0, erosion_inputs.bank_protection_level
    )
    ht = np.maximum(zup - zdo, 0)
    hs = np.maximum(bank_height - water_level_fairway_ref + 2 * h0, 0)
    eq_erosion_distance = ht / discharge_level_pars.mu_slope
    eq_erosion_volume = (0.5 * ht + hs) * eq_erosion_distance * segment_length

    return eq_erosion_distance, eq_erosion_volume


def compute_bank_erosion_dynamics(
    parameters: DischargeCalculationParameters,
    bank_height: np.ndarray,
    segment_length: np.ndarray,
    bank_fairway_dist: np.ndarray,
    water_level_fairway_ref: np.ndarray,
    discharge_level_pars: ParametersPerBank,
    time_erosion: float,
    water_depth_fairway: np.ndarray,
    erosion_inputs: SingleErosion,
) -> DischargeCalculationParameters:
    """
    Compute the bank erosion during a specific discharge level.

    Args:
        parameters (DischargeCalculationParameters):
            velocity : np.ndarray
                Array containing flow velocity magnitude [m/s]
            water_level_fairway : np.ndarray
                Array containing water levels at fairway [m]
            chezy : np.ndarray
                Array containing Chezy values [m0.5/s]
        bank_height : np.ndarray
            Array containing bank height
        segment_length : np.ndarray
            Array containing length of line segment [m]
        water_level_fairway_ref : np.ndarray
            Array containing reference water levels at fairway [m]
        tauc : np.ndarray
            Array containing critical shear stress [N/m2]
        discharge_level_pars: DischargeLevelParameters,
            num_ship : np.ndarray
                Array containing number of ships [-]
            ship_velocity : np.ndarray
                Array containing ship velocity [m/s]
            num_waves_per_ship : np.ndarray
                Array containing number of waves per ship [-]
            ship_type : np.ndarray
                Array containing ship type [-]
            ship_draught : np.ndarray
                Array containing ship draught [m]
        time_erosion : float
            Erosion period [yr]
        bank_fairway_dist : np.ndarray
            Array containing distance from bank to fairway [m]
        fairway_wave_reduction_distance : np.ndarray
            Array containing distance from fairway at which wave reduction starts [m]
        fairway_wave_disappear_distance : np.ndarray
            Array containing distance from fairway at which all waves are gone [m]
        water_depth_fairway : np.ndarray
            Array containing water depth at fairway [m]
        dike_height : np.ndarray
            Array containing bank protection height [m]
        water_density : float
            Water density [kg/m3]

    Returns:
        parameters (CalculationParameters):
            erosion_distance : np.ndarray
                Total bank erosion distance [m]
            erosion_volume : np.ndarray
                Total bank erosion volume [m]
            erosion_distance_shipping : np.ndarray
                Bank erosion distance due to shipping [m]
            erosion_distance_flow : np.ndarray
                Bank erosion distance due to current [m]
            ship_wave_max : np.ndarray
                Maximum bank level subject to ship waves [m]
            ship_wave_min : np.ndarray
                Minimum bank level subject to ship waves [m]
    """
    sec_year = 3600 * 24 * 365

    # period of ship waves [s]
    ship_wave_period = 0.51 * discharge_level_pars.ship_velocity / g
    ts = (
        ship_wave_period
        * discharge_level_pars.num_ship
        * discharge_level_pars.num_waves_per_ship
    )
    vel = parameters.bank_velocity

    # the ship induced wave height at the beginning of the foreshore
    wave_height = comp_hw_ship_at_bank(
        bank_fairway_dist,
        erosion_inputs.wave_fairway_distance_0,
        erosion_inputs.wave_fairway_distance_1,
        water_depth_fairway,
        discharge_level_pars.ship_type,
        discharge_level_pars.ship_draught,
        discharge_level_pars.ship_velocity,
    )
    wave_height = np.maximum(wave_height, EPS)

    # compute erosion parameters for each line part
    # erosion coefficient
    erosion_coef = 0.2 * np.sqrt(erosion_inputs.tauc) * 1e-6

    # critical velocity
    critical_velocity = np.sqrt(
        erosion_inputs.tauc / water_density * parameters.chezy**2 / g
    )

    # strength
    cE = 1.85e-4 / erosion_inputs.tauc

    # total wave damping coefficient
    # mu_tot = (mu_slope / H0) + mu_reed
    # water level along bank line
    ho_line_ship = np.minimum(
        parameters.water_level - erosion_inputs.bank_protection_level, 2 * wave_height
    )
    ho_line_flow = np.minimum(
        parameters.water_level - erosion_inputs.bank_protection_level,
        water_depth_fairway,
    )
    h_line_ship = np.maximum(bank_height - parameters.water_level + ho_line_ship, 0)
    h_line_flow = np.maximum(bank_height - parameters.water_level + ho_line_flow, 0)

    # compute displacement due to flow
    crit_ratio = np.ones(critical_velocity.shape)
    mask = (vel > critical_velocity) & (
        parameters.water_level > erosion_inputs.bank_protection_level
    )
    crit_ratio[mask] = (vel[mask] / critical_velocity[mask]) ** 2
    erosion_distance_flow = erosion_coef * (crit_ratio - 1) * time_erosion * sec_year

    # compute displacement due to ship waves
    ship_wave_max = parameters.water_level + 0.5 * wave_height
    ship_wave_min = parameters.water_level - 2 * wave_height
    mask = (ship_wave_min < water_level_fairway_ref) & (
        water_level_fairway_ref < ship_wave_max
    )
    # limit mu -> 0

    erosion_distance_shipping = cE * wave_height**2 * ts * time_erosion
    erosion_distance_shipping[~mask] = 0

    # compute erosion volume
    mask = (h_line_ship > 0) & (
        parameters.water_level > erosion_inputs.bank_protection_level
    )
    dv_ship = erosion_distance_shipping * segment_length * h_line_ship
    dv_ship[~mask] = 0.0
    erosion_distance_shipping[~mask] = 0.0

    mask = (h_line_flow > 0) & (
        parameters.water_level > erosion_inputs.bank_protection_level
    )
    dv_flow = erosion_distance_flow * segment_length * h_line_flow
    dv_flow[~mask] = 0.0
    erosion_distance_flow[~mask] = 0.0

    erosion_distance = erosion_distance_shipping + erosion_distance_flow
    erosion_volume = dv_ship + dv_flow
    parameters.erosion_volume_tot = erosion_volume
    parameters.erosion_distance_tot = erosion_distance
    parameters.erosion_distance_shipping = erosion_distance_shipping
    parameters.erosion_distance_flow = erosion_distance_flow
    parameters.ship_wave_max = ship_wave_max
    parameters.ship_wave_min = ship_wave_min
    return parameters


def comp_hw_ship_at_bank(
    bank_fairway_dist: np.ndarray,
    fairway_wave_reduction_distance: np.ndarray,
    fairway_wave_disappear_distance: np.ndarray,
    water_depth_fairway: np.ndarray,
    ship_type: np.ndarray,
    ship_draught: np.ndarray,
    ship_velocity: np.ndarray,
) -> np.ndarray:
    """
    Compute wave heights at bank due to passing ships.

    Arguments
    ---------
    bank_fairway_dist : np.ndarray
        Array containing distance from bank to fairway [m]
    fairway_wave_reduction_distance : np.ndarray
        Array containing distance from fairway at which wave reduction starts [m]
    fairway_wave_disappear_distance : np.ndarray
        Array containing distance from fairway at which all waves are gone [m]
    water_depth_fairway : np.ndarray
        Array containing the water depth at the fairway [m]
    ship_type : np.ndarray
        Array containing the ship type [-]
    ship_draught : np.ndarray
        Array containing draught of the ships [m]
    ship_velocity : np.ndarray
        Array containing velocity of the ships [m/s]
    g : float
        Gravitational acceleration [m/s2]

    Returns
    -------
    h0 : np.ndarray
        Array containing wave height at the bank [m]
    """
    h = np.copy(water_depth_fairway)

    a1 = np.zeros(len(bank_fairway_dist))
    # multiple barge convoy set
    a1[ship_type == 1] = 0.5
    # RHK ship / motor ship
    a1[ship_type == 2] = 0.28 * ship_draught[ship_type == 2] ** 1.25
    # towboat
    a1[ship_type == 3] = 1

    froude = ship_velocity / np.sqrt(h * g)
    froude_limit = 0.8
    high_froude = froude > froude_limit
    h[high_froude] = ((ship_velocity[high_froude] / froude_limit) ** 2) / g
    froude[high_froude] = froude_limit

    A = 0.5 * (
        1
        + np.cos(
            (bank_fairway_dist - fairway_wave_disappear_distance)
            / (fairway_wave_reduction_distance - fairway_wave_disappear_distance)
            * np.pi
        )
    )
    A[bank_fairway_dist < fairway_wave_disappear_distance] = 1
    A[bank_fairway_dist > fairway_wave_reduction_distance] = 0

    h0 = a1 * h * (bank_fairway_dist / h) ** (-1 / 3) * froude**4 * A
    return h0


def get_km_bins(
    km_bin: Tuple[float, float, float], station_type: str = "upper", adjust: bool = False
) -> np.ndarray:
    """
    Get an array of representative chainage values.

    Args:
        km_bin (Tuple[float, float, float]):
            Tuple containing (start, end, step) for the chainage bins
        station_type (str, default="upper"):
            Type of characteristic chainage values returned
                all: all bounds (N+1 values)
                lower: lower bounds (N values)
                upper: upper bounds (N values)
                mid: mid-points (N values)
        adjust (bool):
            Flag indicating whether the step size should be adjusted to include an integer number of steps

    Returns:
        km (np.ndarray):
            Array containing the chainage bin upper bounds
    """
    stations_step = km_bin[2]
    num_bins = int(math.ceil((km_bin[1] - km_bin[0]) / stations_step))

    lb = 0
    ub = num_bins + 1
    dx = 0.0

    if adjust:
        stations_step = (km_bin[1] - km_bin[0]) / num_bins

    if station_type == "all":
        pass
    elif station_type == "lower":
        ub = ub - 1
    elif station_type == "upper":
        lb = lb + 1
    elif station_type == "mid":
        ub = ub - 1
        dx = km_bin[2] / 2

    stations = km_bin[0] + dx + np.arange(lb, ub) * stations_step

    return stations


def get_km_eroded_volume(
    bank_km_mid: np.ndarray,
    erosion_volume: np.ndarray,
    km_bin: Tuple[float, float, float],
) -> np.ndarray:
    """
    Accumulate the erosion volumes per chainage bin.

    Arguments
    ---------
    bank_km_mid : np.ndarray
        Array containing the chainage per bank segment [km]
    erosion_volume : np.ndarray
        Array containing the eroded volume per bank segment [m3]
    km_bin : Tuple[float, float, float]
        Tuple containing (start, end, step) for the chainage bins

    Returns
    -------
    dvol : np.ndarray
        Array containing the accumulated eroded volume per chainage bin.
    """
    km_step = km_bin[2]

    bin_idx = np.rint((bank_km_mid - km_bin[0] - km_step / 2.0) / km_step).astype(
        np.int64
    )
    dvol_temp = np.bincount(bin_idx, weights=erosion_volume)
    length = int((km_bin[1] - km_bin[0]) / km_bin[2])
    if len(dvol_temp) == length:
        dvol = dvol_temp
    else:
        dvol = np.zeros((length,))
        dvol[: len(dvol_temp)] = dvol_temp
    return dvol


def moving_avg(xi: np.ndarray, yi: np.ndarray, dx: float) -> np.ndarray:
    """
    Perform a moving average for given averaging distance.

    Arguments
    ---------
    xi : np.ndarray
        Array containing the distance - should be monotonically increasing or decreasing [m or equivalent]
    yi : np.ndarray
        Array containing the values to be average [arbitrary]
    dx: float
        Averaging distance [same unit as x]

    Returns
    -------
    yo : np.ndarray
        Array containing the averaged values [same unit as y].
    """
    dx2 = dx / 2.0
    nx = len(xi)
    if xi[0] < xi[-1]:
        x = xi
        y = yi
    else:
        x = xi[::-1]
        y = yi[::-1]
    ym = np.zeros(y.shape)
    di = np.zeros(y.shape)
    j0 = 1
    for i in range(nx):
        for j in range(j0, nx):
            dxj = x[j] - x[j - 1]
            if x[i] - x[j] > dx2:
                # point j is too far back for point i and further
                j0 = j + 1
            elif x[j] - x[i] > dx2:
                # point j is too far ahead; wrap up and continue
                d0 = (x[i] + dx2) - x[j - 1]
                ydx2 = y[j - 1] + (y[j] - y[j - 1]) * d0 / dxj
                ym[i] += (y[j - 1] + ydx2) / 2.0 * d0
                di[i] += d0
                break
            elif x[i] - x[j - 1] > dx2:
                # point j is ok, but j-1 is too far back, so let's start
                d0 = x[j] - (x[i] - dx2)
                ydx2 = y[j] + (y[j - 1] - y[j]) * d0 / dxj
                ym[i] += (y[j] + ydx2) / 2.0 * d0
                di[i] += d0
            else:
                # segment right in the middle
                ym[i] += (y[j] + y[j - 1]) / 2.0 * dxj
                di[i] += dxj
    yo = ym / di
    if xi[0] < xi[-1]:
        return yo
    else:
        return yo[::-1]


def write_km_eroded_volumes(stations: np.ndarray, volume: np.ndarray, file_name: str) -> None:
    """
    Write a text file with eroded volume data binned per kilometre.

    Arguments
    ---------
    stations :
        Array containing chainage values.
    volume :
        Array containing erosion volume values.
    file_name : str
        Name of the file to be written.

    Returns
    -------
    None
    """
    with open(file_name, "w") as file:
        for i in range(len(stations)):
            str_value = "\t".join(["{:.2f}".format(x) for x in volume[i, :]])
            file.write("{:.2f}\t".format(stations[i]) + str_value + "\n")


def get_slices(
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
    a, b, edges = get_slices_core(edges, mesh_data, bpj1, bpj, prev_b, True)
    nodes = -np.ones(a.shape, dtype=np.int64)
    nodes[a == 0] = mesh_data.edge_node[edges[a == 0], 0]
    nodes[a == 1] = mesh_data.edge_node[edges[a == 1], 1]
    return b, edges, nodes

def get_slices_core(
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
    a, b, slices = _get_slices_ab(
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


def _get_slices_ab(
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


def move_line(
    xylines: np.ndarray, erosion_distance: np.ndarray, right_bank: bool
) -> np.ndarray:
    """
    Shift a line of a variable distance sideways (positive shift away from centre line).

    Chainage must be increasing along all lines. For a bank on the right side a
    positive shift will move the line to the right. For a bank on the left side
    a positive shift will move the line to the left.

    Arguments
    ---------
    xylines : np.ndarray
        Nx2 array containing the x- and y-coordinates of the line to be moved.
    erosion_distance : np.ndarray
        Distance over which to move the line sideways. A positive shift is
        defined towards the right for the right bank, and towards the left for
        the left bank.
    right_bank : bool
        Flag indicating whether line is on the right (or not).

    Returns
    -------
    xylines_new : umpy.ndarray
        Nx2 array containing the x- and y-coordinates of the moved line.
    """
    if right_bank:
        xylines_new = _move_line_right(xylines, erosion_distance)
    else:
        xylines_rev = xylines[::-1, :]
        dn_rev = erosion_distance[::-1]
        xylines_new_rev = _move_line_right(xylines_rev, dn_rev)
        xylines_new = xylines_new_rev[::-1, :]
    return xylines_new


def _move_line_right(xylines: np.ndarray, erosion_distance: np.ndarray) -> np.ndarray:
    """
    Shift a line of a variable distance sideways (positive shift to the right).

    Arguments
    ---------
    xylines : np.ndarray
        Nx2 array containing the x- and y-coordinates of the line to be moved.
    dn0 : np.ndarray
        Distance over which to move the line sideways. A positive shift is
        defined towards the right when looking along the line.

    Returns
    -------
    xylines_new : umpy.ndarray
        Nx2 array containing the x- and y-coordinates of the moved line.
    """
    nsegments = len(erosion_distance)
    colvec = (nsegments, 1)

    # determine segment angle
    dxy = xylines[1:, :] - xylines[:-1, :]
    theta = np.arctan2(dxy[:, 1], dxy[:, 0])

    # determine shift vector nxy for each segment
    ds = np.sqrt((dxy ** 2).sum(axis=1))
    nxy = dxy[:, ::-1] * [1, -1] * (erosion_distance / ds).reshape(colvec)

    xylines_new = np.zeros((100, 2))
    xylines_new[0] = xylines[0] + nxy[0]
    ixy, xylines_new = _add_point(0, xylines_new, xylines[1] + nxy[0])
    ixy, xylines_new = _add_point(ixy, xylines_new, xylines[1])

    verbose = False
    prec = 0.000001
    ixy1: int
    for iseg in range(1, nsegments):
        dtheta = theta[iseg] - theta[iseg - 1]
        if dtheta > math.pi:
            dtheta = dtheta - 2 * math.pi
        if verbose:
            print("{}: current length of new bankline is {}".format(iseg, ixy))
            print(
                "{}: segment starting at {} to be shifted by {}".format(
                    iseg, xylines[iseg], erosion_distance[iseg]
                )
            )
            print("{}: change in direction quantified as {}".format(iseg, dtheta))

        # create a polyline for the outline of the new segment
        if erosion_distance[iseg] < prec:
            # no erosion, so just a linear extension
            if verbose:
                print("{}: no shifting, just linear extension".format(iseg))
            poly = np.row_stack([xylines[iseg + 1], xylines[iseg],])
        elif dtheta <= 0:
            # right bend
            if -0.001 * math.pi < dtheta:
                # almost straight
                if verbose:
                    print("{}: slight bend to right".format(iseg))
                if erosion_distance[iseg] > erosion_distance[iseg]:
                    poly = np.row_stack(
                        [
                            xylines[iseg + 1],
                            xylines[iseg + 1] + nxy[iseg],
                            xylines[iseg] + nxy[iseg],
                            xylines[iseg] + nxy[iseg - 1],
                            xylines[iseg - 1],
                            ]
                    )
                else:
                    poly = np.row_stack(
                        [
                            xylines[iseg + 1],
                            xylines[iseg + 1] + nxy[iseg],
                            xylines[iseg] + nxy[iseg],
                            xylines[iseg - 1],
                            ]
                    )
            else:
                # more significant bend
                if verbose:
                    print("{}: bend to right".format(iseg))
                poly = np.row_stack(
                    [
                        xylines[iseg + 1],
                        xylines[iseg + 1] + nxy[iseg],
                        xylines[iseg] + nxy[iseg],
                        xylines[iseg],
                        ]
                )
        elif erosion_distance[iseg - 1] < prec:
            # left bend: previous segment isn't eroded, so nothing to connect to
            if verbose:
                print("{}: bend to left".format(iseg))
            poly = np.row_stack(
                [
                    xylines[iseg + 1],
                    xylines[iseg + 1] + nxy[iseg],
                    xylines[iseg] + nxy[iseg],
                    xylines[iseg],
                    ]
            )
        else:
            # left bend: connect it to the previous segment to avoid non eroded wedges
            if verbose:
                print("{}: bend to left".format(iseg))
            poly = np.row_stack(
                [
                    xylines[iseg + 1],
                    xylines[iseg + 1] + nxy[iseg],
                    xylines[iseg] + nxy[iseg],
                    xylines[iseg] + nxy[iseg - 1],
                    xylines[iseg - 1],
                    ]
            )

        nedges = poly.shape[0] - 1

        # make a temporary copy of the last 20 nodes of the already shifted bankline
        if ixy > 20:
            X0 = xylines_new[(ixy - 20) : ixy, 0].copy()
            Y0 = xylines_new[(ixy - 20) : ixy, 1].copy()
            X1 = xylines_new[(ixy - 19) : (ixy + 1), 0].copy()
            Y1 = xylines_new[(ixy - 19) : (ixy + 1), 1].copy()
            ixy0 = ixy - 20
        else:
            X0 = xylines_new[:ixy, 0].copy()
            Y0 = xylines_new[:ixy, 1].copy()
            X1 = xylines_new[1 : ixy + 1, 0].copy()
            Y1 = xylines_new[1 : ixy + 1, 1].copy()
            ixy0 = 0

        a = []
        b = []
        slices = []
        n = []
        # for each edge of the new polyline collect all intersections with the
        # already shifted bankline ...
        for i in range(nedges):
            if (poly[i + 1] == poly[i]).all():
                # polyline segment has no actual length, so skip it
                pass
            else:
                # check for intersection
                a2, b2, slices2 = _get_slices_ab(
                    X0,
                    Y0,
                    X1,
                    Y1,
                    poly[i, 0],
                    poly[i, 1],
                    poly[i + 1, 0],
                    poly[i + 1, 1],
                    0,
                    True,
                )
                # exclude the intersection if it's only at the very last point
                # of the last segment
                if i == nedges - 1:
                    keep_mask = a2 < 1 - prec
                    a2 = a2[keep_mask]
                    b2 = b2[keep_mask]
                    slices2 = slices2[keep_mask]
                a.append(a2)
                b.append(b2)
                slices.append(slices2)
                n.append(slices2 * 0 + i)

        s = np.concatenate(slices)
        if verbose:
            print("{}: {} intersections detected".format(iseg, len(s)))
        if len(s) == 0:
            # no intersections found
            if dtheta < 0:
                # right bend (not straight)
                if erosion_distance[iseg] > 0:
                    cross = (xylines_new[ixy, 0] - xylines_new[ixy - 1, 0]) * nxy[
                        iseg, 1
                    ] - (xylines_new[ixy, 1] - xylines_new[ixy - 1, 1]) * nxy[iseg, 0]
                else:
                    cross = (xylines_new[ixy, 0] - xylines_new[ixy - 1, 0]) * dxy[
                        iseg, 1
                    ] - (xylines_new[ixy, 1] - xylines_new[ixy - 1, 1]) * dxy[iseg, 0]
                if cross <= 0.0:
                    # extended path turns right ... always add
                    pass
                else:
                    # extended path turns left
                    # we can probably ignore it, let's do so...
                    # the only exception would be an eroded patch encompassing
                    # all of the eroded bank line
                    if verbose:
                        print("{}: ignoring segment".format(iseg))
                    continue
            else:
                # left bend or straight: always add ... just the rectangle of eroded material
                pass
            ixy1 = ixy
            for n2 in range(min(nedges, 2), -1, -1):
                if verbose:
                    print("  adding point {}".format(poly[n2]))
                ixy1, xylines_new = _add_point(ixy1, xylines_new, poly[n2])
            ixy = ixy1

        else:
            # one or more intersections found
            a = np.concatenate(a)
            b = np.concatenate(b)
            n = np.concatenate(n)

            # sort the intersections by distance along the already shifted bank line
            d = s + a
            sorted = np.argsort(d)
            s = s[sorted] + ixy0
            a = a[sorted]
            b = b[sorted]
            d = d[sorted]
            n = n[sorted]

            ixy1 = s[0]
            if verbose:
                print("{}: continuing new path at point {}".format(iseg, ixy1))
            xytmp = xylines_new[ixy1 : ixy + 1].copy()
            ixytmp = ixy1

            inside = False
            s_last = s[0]
            n_last = nedges
            for i in range(len(s)):
                if verbose:
                    print(
                        "- intersection {}: new polyline edge {} crosses segment {} at {}".format(
                            i, n[i], s[i], a[i]
                        )
                    )
                if i == 0 or n[i] != nedges - 1:
                    if inside:
                        if verbose:
                            print("  existing line is inside the new polygon")
                        for n2 in range(n_last, n[i], -1):
                            if verbose:
                                print("  adding new point {}".format(poly[n2]))
                            ixy1, xylines_new = _add_point(ixy1, xylines_new, poly[n2])
                    else:
                        if verbose:
                            print("  existing line is outside the new polygon")
                        for s2 in range(s_last, s[i]):
                            if verbose:
                                print(
                                    "  re-adding old point {}".format(
                                        xytmp[s2 - ixytmp + 1]
                                    )
                                )
                            ixy1, xylines_new = _add_point(
                                ixy1, xylines_new, xytmp[s2 - ixytmp + 1]
                            )
                    pnt_intersect = poly[n[i]] + b[i] * (poly[n[i] + 1] - poly[n[i]])
                    if verbose:
                        print("  adding intersection point {}".format(pnt_intersect))
                    ixy1, xylines_new = _add_point(ixy1, xylines_new, pnt_intersect, )
                    n_last = n[i]
                    s_last = s[i]
                    if a[i] < prec:
                        dPy = poly[n[i] + 1, 1] - poly[n[i], 1]
                        dPx = poly[n[i] + 1, 0] - poly[n[i], 0]
                        s2 = s[i] - ixy0
                        dBy = Y1[s2] - Y0[s2]
                        dBx = X1[s2] - X0[s2]
                        inside = dPy * dBx - dPx * dBy > 0
                    elif a[i] > 1 - prec:
                        dPy = poly[n[i] + 1, 1] - poly[n[i], 1]
                        dPx = poly[n[i] + 1, 0] - poly[n[i], 0]
                        s2 = s[i] - ixy0 + 1
                        if s2 > len(X0) - 1:
                            inside = True
                        else:
                            dBy = Y1[s2] - Y0[s2]
                            dBx = X1[s2] - X0[s2]
                            inside = dPy * dBx - dPx * dBy > 0
                    else:
                        # line segment slices the edge somewhere in the middle
                        inside = not inside
                    if verbose:
                        if inside:
                            print("  existing line continues inside")
                        else:
                            print("  existing line continues outside")

            if verbose:
                print("- wrapping up after last intersection")
            if inside:
                if verbose:
                    print("  existing line is inside the new polygon")
                for n2 in range(n_last, -1, -1):
                    if verbose:
                        print("  adding new point {}".format(poly[n2]))
                    ixy1, xylines_new = _add_point(ixy1, xylines_new, poly[n2])
            else:
                if verbose:
                    print("  existing line is inside the new polygon")
                for s2 in range(s_last, len(xytmp) + ixytmp - 1):
                    if verbose:
                        print("  re-adding old point {}".format(xytmp[s2 - ixytmp + 1]))
                    ixy1, xylines_new = _add_point(
                        ixy1, xylines_new, xytmp[s2 - ixytmp + 1]
                    )
            ixy = ixy1
        # if iseg == isegstop:
        #     break
    xylines_new = xylines_new[:ixy, :]

    return xylines_new


def _add_point(
    ixy1: int, xy_in: np.ndarray, point: np.ndarray
) -> Tuple[int, np.ndarray]:
    """
    Add the x,y-coordinates of a point to an array of x,y-coordinates if it differs from the last point.

    Arguments
    ---------
    ixy1 : int
        Index of last point in xy_in array
    xy_in : np.ndarray
        N x 2 array containing the x- and y-coordinates of points (partially filled)
    point : np.ndarray
        1 x 2 array containing the x- and y-coordinates of one point

    Results
    -------
    ixy1 : int
        Index of the new point in the xy_out array
    xy_out : np.ndarray
        Possibly extended copy of xy_in that includes the coordinates of point at ixy1
    """
    if (xy_in[ixy1] - point != 0).any():
        ixy1 = ixy1 + 1
        if ixy1 >= len(xy_in):
            xy_out = enlarge(xy_in, (2 * ixy1, 2))
        else:
            xy_out = xy_in
        xy_out[ixy1] = point
    else:
        xy_out = xy_in
    return ixy1, xy_out
