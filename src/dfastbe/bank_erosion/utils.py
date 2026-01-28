"""Bank erosion utilities."""

import math
from typing import Tuple

import numpy as np


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
            poly = np.vstack([xylines[iseg + 1], xylines[iseg],])
        elif dtheta <= 0:
            # right bend
            if -0.001 * math.pi < dtheta:
                # almost straight
                if verbose:
                    print("{}: slight bend to right".format(iseg))
                if erosion_distance[iseg] > erosion_distance[iseg]:
                    poly = np.vstack(
                        [
                            xylines[iseg + 1],
                            xylines[iseg + 1] + nxy[iseg],
                            xylines[iseg] + nxy[iseg],
                            xylines[iseg] + nxy[iseg - 1],
                            xylines[iseg - 1],
                            ]
                    )
                else:
                    poly = np.vstack(
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
                poly = np.vstack(
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
            poly = np.vstack(
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
            poly = np.vstack(
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
                a2, b2, slices2 = calculate_segment_edge_intersections(
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


def calculate_segment_edge_intersections(
    x_edge_coords_prev_point: np.ndarray,
    y_edge_coords_prev_point: np.ndarray,
    x_edge_coords_current_point: np.ndarray,
    y_edge_coords_current_point: np.ndarray,
    prev_point_x: float,
    prev_point_y: float,
    current_point_x: float,
    current_point_y: float,
    min_relative_dist: float,
    limit_relative_distance: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the relative locations a and b at which a segment intersects/slices a number of edges.

    Arguments
    ---------
    x_edge_coords_prev_point : np.ndarray
        Array containing the x-coordinates of the start point of each edge.
    x_edge_coords_current_point : np.ndarray
        Array containing the x-coordinates of the end point of each edge.
    y_edge_coords_prev_point : np.ndarray
        Array containing the y-coordinates of the start point of each edge.
    y_edge_coords_current_point : np.ndarray
        Array containing the y-coordinates of the end point of each edge.
    prev_point_x : float
        x-coordinate of start point of the segment.
    current_point_x : float
        x-coordinate of end point of the segment.
    prev_point_y : float
        y-coordinate of start point of the segment.
    current_point_y : float
        y-coordinate of end point of the segment.
    min_relative_dist : float
        Minimum relative distance from bpj1 at which slice should occur.
    limit_relative_distance : bool
        Flag indicating whether the the relative distance along the segment bpj1-bpj should be limited to 1.

    Returns
    -------
    edge_relative_dist : np.ndarray
        Array containing relative distance along each edge.
    segment_relative_dist : np.ndarray
        Array containing relative distance along the segment for each edge.
    valid_intersections : np.ndarray
        Array containing a flag indicating whether the edge is sliced at a valid location.
    """
    # difference between edges
    dx_edge = x_edge_coords_current_point - x_edge_coords_prev_point
    dy_edge = y_edge_coords_current_point - y_edge_coords_prev_point
    # difference between the two points themselves
    dx_segment = current_point_x - prev_point_x
    dy_segment = current_point_y - prev_point_y
    # check if the line and the edge are parallel
    determinant = dx_edge * dy_segment - dy_edge * dx_segment
    # if determinant is zero, the line and edge are parallel, so we set it to a small value
    determinant[determinant == 0] = 1e-10

    # calculate the relative distances along the edge where the intersection occur
    edge_relative_dist = (dy_segment * (prev_point_x - x_edge_coords_prev_point) - dx_segment * (prev_point_y - y_edge_coords_prev_point)) / determinant  # along mesh edge
    # calculate the relative distances along the segment where the intersection occur
    segment_relative_dist = (dy_edge * (prev_point_x - x_edge_coords_prev_point) - dx_edge * (prev_point_y - y_edge_coords_prev_point)) / determinant  # along bank line

    if limit_relative_distance:
        valid_intersections = np.nonzero((segment_relative_dist > min_relative_dist) & (segment_relative_dist <= 1) & (edge_relative_dist >= 0) & (edge_relative_dist <= 1))[0]
    else:
        valid_intersections = np.nonzero((segment_relative_dist > min_relative_dist) & (edge_relative_dist >= 0) & (edge_relative_dist <= 1))[0]

    edge_relative_dist = edge_relative_dist[valid_intersections]
    segment_relative_dist = segment_relative_dist[valid_intersections]
    return edge_relative_dist, segment_relative_dist, valid_intersections