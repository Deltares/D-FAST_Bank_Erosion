"""Bank erosion utilities."""

import math
from dataclasses import dataclass
from typing import List, Tuple

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
        eroded_bank_line = ErodedBankLine(xylines, erosion_distance)
        xylines_new = eroded_bank_line.move_line_right()
    else:
        xylines_rev = xylines[::-1, :]
        dn_rev = erosion_distance[::-1]
        eroded_bank_line = ErodedBankLine(xylines_rev, dn_rev)
        xylines_new_rev = eroded_bank_line.move_line_right()
        xylines_new = xylines_new_rev[::-1, :]
    return xylines_new


@dataclass
class ErodedBankLineSegment:
    """Class to represent a segment of an eroded bank line.

    Args:
        x_start_segments (np.ndarray): x-coordinates of the segment start.
        y_start_segments (np.ndarray): y-coordinates of the segment start.
        x_end_segments (np.ndarray): x-coordinates of the segment end.
        y_end_segments (np.ndarray): y-coordinates of the segment end.
        ixy0 (int): Index of the segment start in the eroded bank line.
    """
    x_start_segments: np.ndarray
    y_start_segments: np.ndarray
    x_end_segments: np.ndarray
    y_end_segments: np.ndarray
    ixy0: int


@dataclass
class PolylineIntersections:
    """Class to hold the results of polyline intersections.

    Args:
        intersection_alphas (List[np.ndarray]): List of alphas for intersections with segments.
        polygon_alphas (List[np.ndarray]): List of alphas for intersections with polygons.
        polygon_edge_indices (List[np.ndarray]): List of indices for polygon edges.
        segment_indices (List[np.ndarray]): List of indices for segments intersected.
    """

    intersection_alphas: List[np.ndarray]
    polygon_alphas: List[np.ndarray]
    polygon_edge_indices: List[np.ndarray]
    segment_indices: List[np.ndarray]


@dataclass
class IntersectionContext(PolylineIntersections):
    """Describes part of the polyline with intersections and the polygon it intersects.

    Args:
        poly (np.ndarray):
            Polygon coordinates.
        recent_bankline_points (np.ndarray):
            Points of the recent bank line.
        bankline_start_index (int):
            Start index of the recent bank line.
        num_edges (int):
            Number of edges in the polygon.
    """

    poly: np.ndarray
    recent_bankline_points: np.ndarray
    bankline_start_index: int
    num_edges: int

    def get_delta(self, current_intersection: int) -> Tuple[float, float]:
        """Get the x, y-delta for the current intersection."""
        return (
            self.poly[self.polygon_edge_indices[current_intersection] + 1, 0]
            - self.poly[self.polygon_edge_indices[current_intersection], 0],
            self.poly[self.polygon_edge_indices[current_intersection] + 1, 1]
            - self.poly[self.polygon_edge_indices[current_intersection], 1],
        )

    def get_intersection_point(self, current_intersection: int) -> np.ndarray:
        """Get the intersection point for the current index."""
        return self.poly[
            self.polygon_edge_indices[current_intersection]
        ] + self.polygon_alphas[current_intersection] * (
            self.poly[self.polygon_edge_indices[current_intersection] + 1]
            - self.poly[self.polygon_edge_indices[current_intersection]]
        )


class ErodedBankLine:
    """Class to calculate an eroded bank line with its segments and erosion data."""

    def __init__(
        self, xylines: np.ndarray, erosion_distance: np.ndarray, verbose: bool = False
    ):
        self.xylines = xylines
        self.erosion_distance = erosion_distance
        colvec = (len(self.erosion_distance), 1)

        # determine segment angle
        self.dxy = xylines[1:, :] - xylines[:-1, :]
        self.theta = np.arctan2(self.dxy[:, 1], self.dxy[:, 0])

        # determine shift vector nxy for each segment
        ds = np.sqrt((self.dxy**2).sum(axis=1))
        self.nxy = (
            self.dxy[:, ::-1] * [1, -1] * (self.erosion_distance / ds).reshape(colvec)
        )

        self.xylines_new = np.zeros((100, 2))
        self.xylines_new[0] = xylines[0] + self.nxy[0]
        self._add_point(1, self.xylines[1] + self.nxy[0])
        point_is_new, self.point_index = self._point_in_bankline(1, self.xylines[1])
        if point_is_new:
            self._add_point(self.point_index, self.xylines[1])

        self.verbose = verbose
        self.prec = 0.000001

    def _construct_bend_polygon(
        self, index: int, include_wedge: bool = False, include_current: bool = False
    ) -> np.ndarray:
        """
        Construct a polygon for a bank bend segment.

        Args:
            index: Current segment index.
            include_wedge: If True, include the wedge (shifted previous point).
            include_current: If True, include the current point (not shifted).

        Returns:
            np.ndarray: Polygon coordinates.
        """
        points = [
            self.xylines[index + 1],
            self.xylines[index + 1] + self.nxy[index],
            self.xylines[index] + self.nxy[index],
        ]
        if include_wedge:
            points.append(self.xylines[index] + self.nxy[index - 1])
        if include_current:
            points.append(self.xylines[index])
        else:
            points.append(self.xylines[index - 1])
        return np.row_stack(points)

    def _create_segment_outline_polygon(
        self, erosion_index: int, dtheta: float
    ) -> np.ndarray:
        """Create a polyline for the outline of the new segment based on bend type and erosion distance.

        Args:
            erosion_index: Current segment index.
            dtheta: Change in direction at the segment.

        Returns:
            np.ndarray: Polygon coordinates for the segment outline.
        """
        if self.erosion_distance[erosion_index] < self.prec:
            # no erosion, so just a linear extension
            poly = np.row_stack(
                [
                    self.xylines[erosion_index + 1],
                    self.xylines[erosion_index],
                ]
            )
        elif dtheta <= 0:
            # right bend
            if -0.001 * math.pi < dtheta:
                # slight bend to the right (almost straight)
                # TODO: check if this is still needed and if it is fix the expression.
                # if (
                #     self.erosion_distance[erosion_index]
                #     > self.erosion_distance[erosion_index]
                # ):
                #     poly = self._construct_bend_polygon(
                #         erosion_index, include_wedge=True
                #     )
                # else:
                poly = self._construct_bend_polygon(erosion_index)
            else:
                # more significant bend to the right
                poly = self._construct_bend_polygon(erosion_index, include_current=True)
        elif self.erosion_distance[erosion_index - 1] < self.prec:
            # left bend: previous segment isn't eroded, so nothing to connect to
            poly = self._construct_bend_polygon(erosion_index, include_current=True)
        else:
            # left bend: connect it to the previous segment to avoid non eroded wedges
            poly = self._construct_bend_polygon(erosion_index, include_wedge=True)
        return poly

    def _get_recent_shifted_bankline_segments(self) -> ErodedBankLineSegment:
        """Get the recent segments of the shifted bankline for intersection checks.

        Returns:
            ErodedBankLineSegment:
                A dataclass containing the coordinates of the segments
                and the index of the first segment.
        """
        if self.point_index > 20:
            x_start_segments = self.xylines_new[
                (self.point_index - 20) : self.point_index, 0
            ].copy()
            y_start_segments = self.xylines_new[
                (self.point_index - 20) : self.point_index, 1
            ].copy()
            x_end_segments = self.xylines_new[
                (self.point_index - 19) : (self.point_index + 1), 0
            ].copy()
            y_end_segments = self.xylines_new[
                (self.point_index - 19) : (self.point_index + 1), 1
            ].copy()
            ixy0 = self.point_index - 20
        else:
            x_start_segments = self.xylines_new[: self.point_index, 0].copy()
            y_start_segments = self.xylines_new[: self.point_index, 1].copy()
            x_end_segments = self.xylines_new[1 : self.point_index + 1, 0].copy()
            y_end_segments = self.xylines_new[1 : self.point_index + 1, 1].copy()
            ixy0 = 0
        return ErodedBankLineSegment(
            x_start_segments=x_start_segments,
            y_start_segments=y_start_segments,
            x_end_segments=x_end_segments,
            y_end_segments=y_end_segments,
            ixy0=ixy0,
        )

    def _collect_polyline_intersections(
        self,
        eroded_segment: ErodedBankLineSegment,
        poly: np.ndarray,
        num_edges: int,
    ) -> PolylineIntersections:
        """For each edge of the new polyline, collect all intersections with the already shifted bankline.

        Args:
            eroded_segment: ErodedBankLineSegment containing x0, y0, x1, y1, ixy0.
            poly: Nx2 array of polygon coordinates.
            num_edges: Number of edges in the polygon (poly.shape[0] - 1).

        Returns:
            PolylineIntersections: A dataclass containing lists of intersection data.
        """
        intersection_alphas = []
        polygon_alphas = []
        segment_indices = []
        polygon_edge_indices = []
        for i in range(num_edges):
            if (poly[i + 1] == poly[i]).all():
                # polyline segment has no actual length, so skip it
                continue
            # check for intersection
            a2, b2, slices2 = calculate_segment_edge_intersections(
                eroded_segment.x_start_segments,
                eroded_segment.y_start_segments,
                eroded_segment.x_end_segments,
                eroded_segment.y_end_segments,
                poly[i, 0],
                poly[i, 1],
                poly[i + 1, 0],
                poly[i + 1, 1],
                0,
                True,
            )
            # exclude the intersection if it's only at the very last point of the last segment
            if i == num_edges - 1:
                keep_mask = a2 < 1 - self.prec
                a2 = a2[keep_mask]
                b2 = b2[keep_mask]
                slices2 = slices2[keep_mask]
            intersection_alphas.append(a2)
            polygon_alphas.append(b2)
            segment_indices.append(slices2)
            polygon_edge_indices.append(slices2 * 0 + i)
        return PolylineIntersections(
            intersection_alphas=intersection_alphas,
            polygon_alphas=polygon_alphas,
            segment_indices=segment_indices,
            polygon_edge_indices=polygon_edge_indices,
        )

    def _calculate_cross_product(
        self, erosion_index: int, shifted: bool = False
    ) -> float:
        """Calculate the cross product for the current erosion segment.

        Args:
            erosion_index (int): Index of the current erosion segment.
            shifted (bool): If True, use the shifted coordinates for the calculation.

        Returns:
            float: The cross product value.
        """
        multiply_array = self.nxy if shifted else self.dxy
        return (
            self.xylines_new[self.point_index, 0]
            - self.xylines_new[self.point_index - 1, 0]
        ) * multiply_array[erosion_index, 1] - (
            self.xylines_new[self.point_index, 1]
            - self.xylines_new[self.point_index - 1, 1]
        ) * multiply_array[
            erosion_index, 0
        ]

    def _should_add_right_bend_segment(self, erosion_index: int) -> bool:
        """
        Determine if the right bend segment should be added based on the cross product.

        Args:
            erosion_index (int): Index of the current erosion segment.

        Returns:
            bool: True if the segment should be added, False if it should be ignored.
        """
        add = True
        if self.erosion_distance[erosion_index] > 0:
            cross = self._calculate_cross_product(erosion_index, shifted=True)
        else:
            cross = self._calculate_cross_product(erosion_index)
        if cross > 0.0:
            if self.verbose:
                print(f"{erosion_index}: ignoring segment")
            add = False
        return add

    def _add_right_bend_segment_points(self, poly: np.ndarray, num_edges: int) -> None:
        """
        Add the first three points of the polygon to the shifted bankline for a right bend segment.

        Args:
            poly (np.ndarray): Polygon coordinates for the current segment.
            num_edges (int): Number of edges in the polygon.
        """
        ixy1 = self.point_index
        for n2 in range(min(num_edges, 2), -1, -1):
            if self.verbose:
                print(f"  adding point {poly[n2]}")
            point_is_new, ixy1 = self._point_in_bankline(ixy1, poly[n2])
            if point_is_new:
                self._add_point(ixy1, poly[n2])
        self.point_index = ixy1

    def _update_points_between_segments(
        self,
        ixy1: int,
        start: int,
        end: int,
        intersection_context: IntersectionContext,
        inside: bool = True,
    ) -> int:
        """
        Add or re-add points between segments, using either poly or recent_bankline_points from intersection_context.

        Args:
            ixy1 (int): Current index in the shifted bankline.
            start (int): Start index for the points to add.
            end (int): End index for the points to add.
            intersection_context (IntersectionContext): Context containing poly and recent_bankline_points.
            inside (bool): If True, use poly; if False, use recent_bankline_points.

        Returns:
            int: Updated index in the shifted bankline.
        """
        modifier = -1 if inside else 1
        if self.verbose:
            print(
                f"  existing line is {'inside' if inside else 'outside'} the new polygon"
            )
        for idx in range(start, end, modifier):
            if inside:
                point = intersection_context.poly[idx]
                if self.verbose:
                    print(f"  adding new point {point}")
            else:
                point = intersection_context.recent_bankline_points[
                    idx - intersection_context.bankline_start_index + 1
                ]
                if self.verbose:
                    print(f"  re-adding old point {point}")
            point_is_new, ixy1 = self._point_in_bankline(ixy1, point)
            if point_is_new:
                self._add_point(ixy1, point)
        return ixy1

    def _add_intersection_point_to_bankline(
        self,
        ixy1: int,
        intersection_context: IntersectionContext,
        intersection_index: int,
    ) -> int:
        """
        Add the intersection point for the given index to the shifted bankline.

        Args:
            ixy1 (int): Current index in the shifted bankline.
            intersection_context (IntersectionContext): Context containing intersection data.
            intersection_index (int): Index of the current intersection.

        Returns:
            int: Updated index in the shifted bankline.
        """
        pnt_intersect = intersection_context.get_intersection_point(intersection_index)
        if self.verbose:
            print(f"  adding intersection point {pnt_intersect}")
        point_is_new, ixy1 = self._point_in_bankline(ixy1, pnt_intersect)
        if point_is_new:
            self._add_point(ixy1, pnt_intersect)
        return ixy1

    def _update_inside_flag_for_intersection(
        self,
        intersection_context: IntersectionContext,
        intersection_index: int,
        current_segment: int,
        eroded_segment: ErodedBankLineSegment,
        inside: bool,
    ) -> bool:
        """
        Update the 'inside' flag after processing an intersection point.

        Args:
            intersection_context (IntersectionContext): Context containing intersection data.
            i (int): Index of the current intersection.
            current_segment (int): Current segment index.
            eroded_segment (ErodedBankLineSegment): Segment data for the eroded bankline.

        Returns:
            bool: Updated inside flag.
        """
        is_start = (
            intersection_context.intersection_alphas[intersection_index] < self.prec
        )
        is_end = (
            intersection_context.intersection_alphas[intersection_index] > 1 - self.prec
        )
        offset = 1 if is_end else 0
        if is_start or is_end:
            # intersection is at the start or end of the segment
            delta_intersection_x, delta_intersection_y = intersection_context.get_delta(
                intersection_index
            )
            s2 = current_segment - eroded_segment.ixy0 + offset
            if is_end and s2 > len(eroded_segment.x_start_segments) - 1:
                # if the end is beyond the last segment, consider it inside
                inside = True
            else:
                delta_bankline_y = (
                    eroded_segment.y_end_segments[s2]
                    - eroded_segment.y_start_segments[s2]
                )
                delta_bankline_x = (
                    eroded_segment.x_end_segments[s2]
                    - eroded_segment.x_start_segments[s2]
                )
                inside = (
                    delta_intersection_y * delta_bankline_x
                    - delta_intersection_x * delta_bankline_y
                    > 0
                )
        else:
            # line segment slices the edge somewhere in the middle
            inside = not inside
        return inside

    def _process_single_intersection(
        self,
        intersection_index: int,
        current_segment: int,
        eroded_segment: ErodedBankLineSegment,
        intersection_context: IntersectionContext,
        inside: bool,
        last_segment: int,
        last_edge_index: int,
        ixy1: int,
    ) -> Tuple[bool, int, int, int]:
        """
        Process a single intersection and update the shifted bankline accordingly.

        Returns:
            inside (bool): Updated inside flag.
            s_last (int): Updated last segment index.
            n_last (int): Updated last edge index.
            ixy1 (int): Updated index in shifted bankline.
        """
        if self.verbose:
            print(
                f"- intersection {intersection_index}: new polyline edge {intersection_context.polygon_edge_indices[intersection_index]} crosses segment {current_segment} at {intersection_context.intersection_alphas[intersection_index]}"
            )
        if (
            intersection_index == 0
            or intersection_context.polygon_edge_indices[intersection_index]
            != intersection_context.num_edges - 1
        ):
            if inside:
                start = last_edge_index
                end = intersection_context.polygon_edge_indices[intersection_index]
            else:
                start = last_segment
                end = current_segment
            ixy1 = self._update_points_between_segments(
                ixy1,
                start,
                end,
                intersection_context,
                inside=inside,
            )
            ixy1 = self._add_intersection_point_to_bankline(
                ixy1, intersection_context, intersection_index
            )
            last_edge_index = intersection_context.polygon_edge_indices[
                intersection_index
            ]
            last_segment = current_segment
            inside = self._update_inside_flag_for_intersection(
                intersection_context,
                intersection_index,
                current_segment,
                eroded_segment,
                inside,
            )
            if self.verbose:
                if inside:
                    print("  existing line continues inside")
                else:
                    print("  existing line continues outside")
        return inside, last_segment, last_edge_index, ixy1

    def _finalize_bankline_after_intersections(
        self,
        inside: bool,
        last_edge_index: int,
        ixy1: int,
        last_segment: int,
        intersection_context: IntersectionContext,
    ) -> int:
        """
        Finalize the shifted bankline after processing all intersections.

        Args:
            inside (bool): Whether the current path is inside the new polygon.
            n_last (int): Last polygon edge index processed.
            poly (np.ndarray): Polygon coordinates for the current segment.
            ixy1 (int): Current index in the shifted bankline.
            recent_bankline_points (np.ndarray): Temporary array of old bankline points.
            bankline_start_index (int): Starting index for recent_bankline_points.
            s_last (int): Last segment index processed.

        Returns:
            int: Updated index in the shifted bankline.
        """
        if inside:
            start = last_edge_index
            end = -1
        else:
            start = last_segment
            end = (
                len(intersection_context.recent_bankline_points)
                + intersection_context.bankline_start_index
                - 1
            )
        ixy1 = self._update_points_between_segments(
            ixy1, start, end, intersection_context, inside=inside
        )
        return ixy1

    def _process_intersections_and_update_bankline(
        self,
        intersections: PolylineIntersections,
        poly: np.ndarray,
        num_edges: int,
        eroded_segment: ErodedBankLineSegment,
    ) -> None:
        """
        Process sorted intersections and update the shifted bankline accordingly.

        Args:
            intersections: PolylineIntersections object with intersection data.
            poly: Polygon coordinates for the current segment.
            num_edges: Number of edges in the polygon.
            eroded_segment: ErodedBankLineSegment with recent shifted bankline segments.

        Modifies:
            self.xylines_new and self.point_index in place.
        """
        segment_indices = np.concatenate(intersections.segment_indices)
        intersection_alphas = np.concatenate(intersections.intersection_alphas)
        polygon_alphas = np.concatenate(intersections.polygon_alphas)
        polygon_edge_indices = np.concatenate(intersections.polygon_edge_indices)

        # sort the intersections by distance along the already shifted bank line
        segment_intersection_distance = segment_indices + intersection_alphas
        sorted_idx = np.argsort(segment_intersection_distance)
        sorted_segment_indices = segment_indices[sorted_idx] + eroded_segment.ixy0
        sorted_intersection_alphas = intersection_alphas[sorted_idx]
        sorted_polygon_alphas = polygon_alphas[sorted_idx]
        sorted_polygon_edge_indices = polygon_edge_indices[sorted_idx]

        segment_index = sorted_segment_indices[0]
        if self.verbose:
            print(f"continuing new path at point {segment_index}")
        recent_bankline_points = self.xylines_new[
            segment_index : self.point_index + 1
        ].copy()
        bankline_start_index = segment_index

        inside = False
        last_segment = sorted_segment_indices[0]
        last_edge_index = num_edges
        intersection_context = IntersectionContext(
            intersection_alphas=sorted_intersection_alphas,
            polygon_alphas=sorted_polygon_alphas,
            segment_indices=sorted_segment_indices,
            polygon_edge_indices=sorted_polygon_edge_indices,
            poly=poly,
            recent_bankline_points=recent_bankline_points,
            bankline_start_index=bankline_start_index,
            num_edges=num_edges,
        )
        for i, current_segment in enumerate(sorted_segment_indices):
            inside, last_segment, last_edge_index, segment_index = (
                self._process_single_intersection(
                    i,
                    current_segment,
                    eroded_segment,
                    intersection_context,
                    inside,
                    last_segment,
                    last_edge_index,
                    segment_index,
                )
            )

        self.point_index = self._finalize_bankline_after_intersections(
            inside, last_edge_index, segment_index, last_segment, intersection_context
        )

    def move_line_right(self) -> np.ndarray:
        """Shift a line using the erosion distance.

        Returns
            np.ndarray: Nx2 array containing the x- and y-coordinates of the moved line.
        """
        for erosion_index, eroded_distance in enumerate(self.erosion_distance):
            dtheta = self.theta[erosion_index] - self.theta[erosion_index - 1]
            if dtheta > math.pi:
                dtheta = dtheta - 2 * math.pi
            if self.verbose:
                print(
                    f"{erosion_index}: current length of new bankline is {self.point_index}"
                )
                print(
                    f"{erosion_index}: segment starting at {self.xylines[erosion_index]} to be shifted by {eroded_distance}"
                )
                print(f"{erosion_index}: change in direction quantified as {dtheta}")
            poly = self._create_segment_outline_polygon(erosion_index, dtheta)
            num_edges = poly.shape[0] - 1

            eroded_segment = self._get_recent_shifted_bankline_segments()

            intersections = self._collect_polyline_intersections(
                eroded_segment, poly, num_edges
            )

            s = np.concatenate(intersections.segment_indices)
            if self.verbose:
                print(f"{erosion_index}: {len(s)} intersections detected")
            if len(s) == 0:
                if dtheta < 0 and not self._should_add_right_bend_segment(
                    erosion_index
                ):
                    continue
                self._add_right_bend_segment_points(poly, num_edges)
            else:
                self._process_intersections_and_update_bankline(
                    intersections, poly, num_edges, eroded_segment
                )
            # if iseg == isegstop:
            #     break
        self.xylines_new = self.xylines_new[: self.point_index, :]

        return self.xylines_new

    def _point_in_bankline(self, ixy1, point: np.ndarray) -> Tuple[bool, int]:
        """Check if a point is within the bankline.

        Args:
            point (np.ndarray): The point to check.

        Returns:
            bool: True if the point is within the bankline, False otherwise.
        """
        # Check if the point is within the x and y bounds of the bankline
        point_is_new = (self.xylines_new[ixy1] - point != 0).any()
        if point_is_new:
            ixy1 = ixy1 + 1
        return point_is_new, ixy1

    def _add_point(self, ixy1: int, point: np.ndarray):
        """Add the x,y-coordinates of a point to the xylines_new array.

        Args:
            ixy1 (int):
                Index of last point in xy_in array
            point (np.ndarray):
                1 x 2 array containing the x- and y-coordinates of one point
        """
        if ixy1 >= len(self.xylines_new):
            self.xylines_new = enlarge(self.xylines_new, (2 * ixy1, 2))
        self.xylines_new[ixy1] = point


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
