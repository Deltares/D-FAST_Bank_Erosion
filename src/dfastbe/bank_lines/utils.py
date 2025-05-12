import numpy as np
from shapely.geometry import LineString, MultiLineString, Point
from shapely import line_merge
from dfastbe.utils import on_right_side

__all__ = ["sort_connect_bank_lines", "poly_to_line", "tri_to_line"]


def sort_connect_bank_lines(
    banklines: MultiLineString,
    xykm: LineString,
    right_bank: bool,
) -> LineString:
    """
    Connect the bank line segments to bank lines.

    Arguments
    ---------
    banklines : MultiLineString
        Unordered set of bank line segments.
    xykm : LineString
        Array containing x,y,chainage values.
    right_bank : bool
        Flag indicating whether line is on the right (or not).

    Returns
    -------
    bank : LineString
        The detected bank line.
    """
    # convert MultiLineString into list of LineStrings that can be modified later
    banklines_list = [line for line in banklines.geoms]

    # loop over banklines and determine minimum/maximum projected length
    min_locs = np.zeros(len(banklines_list))
    max_locs = np.zeros(len(banklines_list))
    lengths = np.zeros(len(banklines_list))
    keep = lengths == 1

    for i, bl in enumerate(banklines_list):
        min_loc = 1e20
        max_loc = -1
        for j, p in enumerate(bl.coords):
            loc = xykm.project(Point(p))
            if loc < min_loc:
                min_loc = loc
                min_j = j
            if loc > max_loc:
                max_loc = loc
                max_j = j
        min_locs[i] = min_loc  # at minj
        max_locs[i] = max_loc  # at maxj
        if min_j == max_j:
            pass
        elif bl.coords[0] == bl.coords[-1]:

            crd_numpy = np.array(bl.coords)
            num_coord = len(crd_numpy)
            if min_j < max_j:
                op1 = np.array(bl.coords[min_j : max_j + 1])
                op2 = np.zeros((num_coord + min_j - max_j, 2))
                op2[0: num_coord - max_j - 1] = crd_numpy[max_j:-1]
                op2[num_coord - max_j - 1: num_coord - max_j + min_j] = crd_numpy[: min_j + 1]
                op2 = op2[::-1]
            else:  # minj > maxj
                op1 = np.array(bl.coords[max_j : min_j + 1][::-1])
                op2 = np.zeros((num_coord + max_j - min_j, 2))
                op2[0: num_coord - min_j - 1] = crd_numpy[min_j:-1]
                op2[num_coord - min_j - 1: num_coord - min_j + max_j] = crd_numpy[: max_j + 1]
            op1_right_of_op2 = on_right_side(op1, op2)
            if (right_bank and op1_right_of_op2) or (
                    (not right_bank) and (not op1_right_of_op2)
            ):
                op = op2
            else:
                op = op1
            banklines_list[i] = LineString(op)
        else:
            if min_j < max_j:
                banklines_list[i] = LineString(bl.coords[min_j : max_j + 1])
            else:  # minj > maxj
                banklines_list[i] = LineString(bl.coords[max_j : min_j + 1][::-1])
        lengths[i] = max_loc - min_loc

    while True:
        maxl = lengths.max()
        if maxl == 0:
            break
        iarray = np.nonzero(lengths == maxl)
        i = iarray[0][0]

        keep[i] = True
        # remove lines that are a subset
        lengths[(min_locs >= min_locs[i]) & (max_locs <= max_locs[i])] = 0

        # if line partially overlaps ... but stick out on the high side
        jarray = np.nonzero(
            (min_locs > min_locs[i]) & (min_locs < max_locs[i]) & (max_locs > max_locs[i])
        )[0]
        if jarray.size > 0:
            for j in jarray:
                bl = banklines_list[j]
                kmax = len(bl.coords) - 1
                for k, p in enumerate(bl.coords):
                    if k == kmax:
                        # a line string of a single point would remain
                        lengths[j] = 0
                        break
                    loc = xykm.project(Point(p))
                    if loc >= max_locs[i]:
                        banklines_list[j] = LineString(bl.coords[k:])
                        min_locs[j] = loc
                        break
        # if line partially overlaps ... but stick out on the low side
        jarray = np.nonzero(
            (min_locs < min_locs[i]) & (max_locs > min_locs[i]) & (max_locs < max_locs[i])
        )[0]
        if jarray.size > 0:
            for j in jarray:
                bl = banklines_list[j]
                kmax = len(bl.coords) - 1
                for k, p in zip(range(-1, -kmax, -1), bl.coords[:-1][::-1]):
                    if k == kmax + 1:
                        # a line string of a single point would remain
                        lengths[j] = 0
                        break
                    loc = xykm.project(Point(p))
                    if loc <= min_locs[i]:
                        banklines_list[j] = LineString(bl.coords[:k])
                        max_locs[j] = loc
                        break

    # select banks in order of projected length
    idx = np.argsort(min_locs[keep])
    idx2 = np.nonzero(keep)[0]
    new_bank_coords = []
    for i in idx2[idx]:
        new_bank_coords.extend(banklines_list[i].coords)
    bank = LineString(new_bank_coords)

    return bank


def poly_to_line(
    nnodes: int,
    x: np.ndarray,
    y: np.ndarray,
    wet_node: np.ndarray,
    h_node: np.ndarray,
    h0: float,
):
    """
    Detect the bank line segments inside an individual face of arbitrary (convex) polygonal shape.

    Arguments
    ---------
    nnode : int
        Number of nodes of mesh face.
    x : np.ndarray
        Array of x-coordinates of the nodes making up the mesh face.
    y : np.ndarray
        Array of y-coordinates of the nodes making up the mesh face.
    wet_node : np.ndarray
        Array of booleans indicating whether nodes are wet.
    h_node : np.ndarray
        Array of water depths (negative for dry) at the mesh nodes.
    h0 : float
        Critical water depth for determining the banks.

    Results
    -------
    lines : Optional[...]
        Optional bank line segments detected within the mesh face.
    """
    Lines = [None] * (nnodes - 2)
    for i in range(nnodes - 2):
        iv = [0, i + 1, i + 2]
        nwet = sum(wet_node[iv])
        if nwet == 1 or nwet == 2:
            # print("x: ",x[iv]," y: ",y[iv], " w: ", wet_node[iv], " d: ", h_node[iv])
            Lines[i] = tri_to_line(x[iv], y[iv], wet_node[iv], h_node[iv], h0)
    Lines = [line for line in Lines if not line is None]
    if len(Lines) == 0:
        return None
    else:
        multi_line = MultiLineString(Lines)
        merged_line = line_merge(multi_line)
        return merged_line


def tri_to_line(
    x: np.ndarray,
    y: np.ndarray,
    wet_node: np.ndarray,
    h_node: np.ndarray,
    h0: float,
):
    """
    Detect the bank line segments inside an individual triangle.

    Arguments
    ---------
    x : np.ndarray
        Array of x-coordinates of the nodes making up the mesh face.
    y : np.ndarray
        Array of y-coordinates of the nodes making up the mesh face.
    wet_node : np.ndarray
        Array of booleans indicating whether nodes are wet.
    h_node : np.ndarray
        Array of water depths (negative for dry) at the mesh nodes.
    h0 : float
        Critical water depth for determining the banks.

    Returns
    -------
    Line : Optional[]
        Optional bank line segment detected within the triangle.
    """
    if wet_node[0] and wet_node[1]:
        A = 0
        B = 2
        C = 1
        D = 2
    elif wet_node[0] and wet_node[2]:
        A = 0
        B = 1
        C = 2
        D = 1
    elif wet_node[0]:
        A = 0
        B = 1
        C = 0
        D = 2
    elif wet_node[1] and wet_node[2]:
        A = 2
        B = 0
        C = 1
        D = 0
    elif wet_node[1]:
        A = 1
        B = 0
        C = 1
        D = 2
    else:  # wet_node[2]
        A = 2
        B = 0
        C = 2
        D = 1
    facAB = (h_node[A] - h0) / (h_node[A] - h_node[B])  # large facAB -> close to B
    xl = x[A] + facAB * (x[B] - x[A])
    yl = y[A] + facAB * (y[B] - y[A])
    facCD = (h_node[C] - h0) / (h_node[C] - h_node[D])  # large facCD -> close to D
    xr = x[C] + facCD * (x[D] - x[C])
    yr = y[C] + facCD * (y[D] - y[C])
    if xl == xr and yl == yr:
        Line = None
    else:
        Line = LineString([[xl, yl], [xr, yr]])
    return Line
