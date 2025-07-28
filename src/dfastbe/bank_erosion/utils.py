"""Bank erosion utilities."""

import math
from dataclasses import dataclass, field
from typing import Any, List, Tuple

import numpy as np

from dfastbe.bank_erosion.data_models.calculation import BankData, MeshData, SingleBank
from dfastbe.bank_erosion.data_models.inputs import ErosionRiverData
from dfastbe.bank_erosion.mesh_processor import (
    enlarge,
    get_slices_ab,
    intersect_line_mesh,
)
from dfastbe.io.data_models import LineGeometry
from dfastbe.io.logger import log_text
from dfastbe.utils import on_right_side


class BankLinesProcessor:
    """Class to process bank lines and intersect them with a mesh."""

    def __init__(self, river_data: ErosionRiverData):
        """Constructor for BankLinesProcessor."""
        self.bank_lines = river_data.bank_lines
        self.river_center_line = river_data.river_center_line.as_array()
        self.num_bank_lines = len(self.bank_lines)

    def intersect_with_mesh(self, mesh_data: MeshData) -> BankData:
        """Intersect bank lines with a mesh and return bank data.

        Args:
            mesh_data: Mesh data containing face coordinates and connectivity information.

        Returns:
            BankData object containing bank line coordinates, face indices, and other bank-related data.
        """
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
        data = {
            'is_right_bank': is_right_bank,
            'bank_line_coords': bank_line_coords,
            'bank_face_indices': bank_face_indices,
            'bank_chainage_midpoints': bank_chainage_midpoints
        }
        return BankData.from_column_arrays(
            data,
            SingleBank,
            bank_lines=self.bank_lines,
            n_bank_lines=n_bank_lines,
            bank_order=bank_order,
        )


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
        x0 (np.ndarray): x-coordinates of the segment start.
        y0 (np.ndarray): y-coordinates of the segment start.
        x1 (np.ndarray): x-coordinates of the segment end.
        y1 (np.ndarray): y-coordinates of the segment end.
        ixy0 (int): Index of the segment start in the eroded bank line.
    """
    x0: np.ndarray
    y0: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    ixy0: int


@dataclass
class PolylineIntersections:
    """Class to hold the results of polyline intersections.

    Args:
        a (List[np.ndarray]): List of intersection x-coordinates.
        b (List[np.ndarray]): List of intersection y-coordinates.
        slices (List[np.ndarray]): List of slices for each intersection.
        n (List[np.ndarray]): List of segment indices for each intersection.
    """

    a: List[np.ndarray]
    b: List[np.ndarray]
    n: List[np.ndarray]
    slices: List[np.ndarray]


@dataclass
class IntersectionContext(PolylineIntersections):
    """Context for polyline intersection processing.

    Args:
        intersections (PolylineIntersections): Intersections data.
        poly (np.ndarray): Polygon coordinates.
        xytmp (np.ndarray): Temporary coordinates for intersection processing.
        ixytmp (int): Temporary index for intersection processing.
        nedges (int): Number of edges in the polygon.
        s_last (int): Last segment index processed.
    """

    poly: np.ndarray
    xytmp: np.ndarray
    ixytmp: int
    nedges: int
    s_last: int


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
        self.ixy, self.xylines_new = _add_point(
            0, self.xylines_new, self.xylines[1] + self.nxy[0]
        )
        self.ixy, self.xylines_new = _add_point(
            self.ixy, self.xylines_new, self.xylines[1]
        )

        self.verbose = verbose
        self.prec = 0.000001
        self.ixy1: int

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

    def _create_segment_outline_polygon(self, index: int, dtheta: float) -> np.ndarray:
        """
        Create a polyline for the outline of the new segment based on bend type and erosion distance.

        Args:
            index: Current segment index.
            dtheta: Change in direction at the segment.

        Returns:
            np.ndarray: Polygon coordinates for the segment outline.
        """
        # create a polyline for the outline of the new segment
        if self.erosion_distance[index] < self.prec:
            # no erosion, so just a linear extension
            if self.verbose:
                print(f"{index}: no shifting, just linear extension")
            poly = np.row_stack(
                [
                    self.xylines[index + 1],
                    self.xylines[index],
                ]
            )
        elif dtheta <= 0:
            # right bend
            if -0.001 * math.pi < dtheta:
                # almost straight
                if self.verbose:
                    print(f"{index}: slight bend to right")
                if self.erosion_distance[index] > self.erosion_distance[index]:
                    poly = self._construct_bend_polygon(index, include_wedge=True)
                else:
                    poly = self._construct_bend_polygon(index)
            else:
                # more significant bend
                if self.verbose:
                    print(f"{index}: bend to right")
                poly = self._construct_bend_polygon(index, include_current=True)
        elif self.erosion_distance[index - 1] < self.prec:
            # left bend: previous segment isn't eroded, so nothing to connect to
            if self.verbose:
                print(f"{index}: bend to left")
            poly = self._construct_bend_polygon(index, include_current=True)
        else:
            # left bend: connect it to the previous segment to avoid non eroded wedges
            if self.verbose:
                print(f"{index}: bend to left")
            poly = self._construct_bend_polygon(index, include_wedge=True)
        return poly

    def _get_recent_shifted_bankline_segments(self) -> ErodedBankLineSegment:
        """Get the recent segments of the shifted bankline for intersection checks.

        Returns:
            ErodedBankLineSegment:
                A dataclass containing the coordinates of the segments
                and the index of the first segment.
        """
        if self.ixy > 20:
            X0 = self.xylines_new[(self.ixy - 20) : self.ixy, 0].copy()
            Y0 = self.xylines_new[(self.ixy - 20) : self.ixy, 1].copy()
            X1 = self.xylines_new[(self.ixy - 19) : (self.ixy + 1), 0].copy()
            Y1 = self.xylines_new[(self.ixy - 19) : (self.ixy + 1), 1].copy()
            ixy0 = self.ixy - 20
        else:
            X0 = self.xylines_new[: self.ixy, 0].copy()
            Y0 = self.xylines_new[: self.ixy, 1].copy()
            X1 = self.xylines_new[1 : self.ixy + 1, 0].copy()
            Y1 = self.xylines_new[1 : self.ixy + 1, 1].copy()
            ixy0 = 0
        return ErodedBankLineSegment(x0=X0, y0=Y0, x1=X1, y1=Y1, ixy0=ixy0)

    def _collect_polyline_intersections(
        self,
        eroded_segment: ErodedBankLineSegment,
        poly: np.ndarray,
        nedges: int,
    ) -> PolylineIntersections:
        """
        For each edge of the new polyline, collect all intersections with the already shifted bankline.

        Args:
            eroded_segment: ErodedBankLineSegment containing x0, y0, x1, y1, ixy0.
            poly: Nx2 array of polygon coordinates.
            nedges: Number of edges in the polygon (poly.shape[0] - 1).

        Returns:
            PolylineIntersections: A dataclass containing lists of intersection data.
        """
        intersections = PolylineIntersections(a=[], b=[], slices=[], n=[])
        for i in range(nedges):
            if (poly[i + 1] == poly[i]).all():
                # polyline segment has no actual length, so skip it
                continue
            # check for intersection
            a2, b2, slices2 = get_slices_ab(
                eroded_segment.x0,
                eroded_segment.y0,
                eroded_segment.x1,
                eroded_segment.y1,
                poly[i, 0],
                poly[i, 1],
                poly[i + 1, 0],
                poly[i + 1, 1],
                0,
                True,
            )
            # exclude the intersection if it's only at the very last point of the last segment
            if i == nedges - 1:
                keep_mask = a2 < 1 - self.prec
                a2 = a2[keep_mask]
                b2 = b2[keep_mask]
                slices2 = slices2[keep_mask]
            intersections.a.append(a2)
            intersections.b.append(b2)
            intersections.slices.append(slices2)
            intersections.n.append(slices2 * 0 + i)
        return intersections

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
            self.xylines_new[self.ixy, 0] - self.xylines_new[self.ixy - 1, 0]
        ) * multiply_array[erosion_index, 1] - (
            self.xylines_new[self.ixy, 1] - self.xylines_new[self.ixy - 1, 1]
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

    def _add_right_bend_segment_points(self, poly: np.ndarray, nedges: int) -> None:
        """
        Add the first three points of the polygon to the shifted bankline for a right bend segment.

        Args:
            poly (np.ndarray): Polygon coordinates for the current segment.
            nedges (int): Number of edges in the polygon.
        """
        ixy1 = self.ixy
        for n2 in range(min(nedges, 2), -1, -1):
            if self.verbose:
                print(f"  adding point {poly[n2]}")
            ixy1, self.xylines_new = _add_point(ixy1, self.xylines_new, poly[n2])
        self.ixy = ixy1

    def _process_single_intersection(
        self,
        i: int,
        s_current: int,
        a: np.ndarray,
        b: np.ndarray,
        n: np.ndarray,
        nedges: int,
        poly: np.ndarray,
        xytmp: np.ndarray,
        ixytmp: int,
        eroded_segment: ErodedBankLineSegment,
        inside: bool,
        s_last: int,
        n_last: int,
        ixy1: int,
    ) -> tuple[bool, int, int, int]:
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
                f"- intersection {i}: new polyline edge {n[i]} crosses segment {s_current} at {a[i]}"
            )
        if i == 0 or n[i] != nedges - 1:
            if inside:
                if self.verbose:
                    print("  existing line is inside the new polygon")
                for n2 in range(n_last, n[i], -1):
                    if self.verbose:
                        print(f"  adding new point {poly[n2]}")
                    ixy1, self.xylines_new = _add_point(
                        ixy1, self.xylines_new, poly[n2]
                    )
            else:
                if self.verbose:
                    print("  existing line is outside the new polygon")
                for s2 in range(s_last, s_current):
                    if self.verbose:
                        print(f"  re-adding old point {xytmp[s2 - ixytmp + 1]}")
                    ixy1, self.xylines_new = _add_point(
                        ixy1, self.xylines_new, xytmp[s2 - ixytmp + 1]
                    )
            pnt_intersect = poly[n[i]] + b[i] * (poly[n[i] + 1] - poly[n[i]])
            if self.verbose:
                print(f"  adding intersection point {pnt_intersect}")
            ixy1, self.xylines_new = _add_point(ixy1, self.xylines_new, pnt_intersect)
            n_last = n[i]
            s_last = s_current
            if a[i] < self.prec:
                dPy = poly[n[i] + 1, 1] - poly[n[i], 1]
                dPx = poly[n[i] + 1, 0] - poly[n[i], 0]
                s2 = s_current - eroded_segment.ixy0
                dBy = eroded_segment.y1[s2] - eroded_segment.y0[s2]
                dBx = eroded_segment.x1[s2] - eroded_segment.x0[s2]
                inside = dPy * dBx - dPx * dBy > 0
            elif a[i] > 1 - self.prec:
                dPy = poly[n[i] + 1, 1] - poly[n[i], 1]
                dPx = poly[n[i] + 1, 0] - poly[n[i], 0]
                s2 = s_current - eroded_segment.ixy0 + 1
                if s2 > len(eroded_segment.x0) - 1:
                    inside = True
                else:
                    dBy = eroded_segment.y1[s2] - eroded_segment.y0[s2]
                    dBx = eroded_segment.x1[s2] - eroded_segment.x0[s2]
                    inside = dPy * dBx - dPx * dBy > 0
            else:
                # line segment slices the edge somewhere in the middle
                inside = not inside
            if self.verbose:
                if inside:
                    print("  existing line continues inside")
                else:
                    print("  existing line continues outside")
        return inside, s_last, n_last, ixy1

    def _finalize_bankline_after_intersections(
        self,
        inside: bool,
        n_last: int,
        poly: np.ndarray,
        ixy1: int,
        xytmp: np.ndarray,
        ixytmp: int,
        s_last: int,
    ) -> int:
        """
        Finalize the shifted bankline after processing all intersections.

        Args:
            inside (bool): Whether the current path is inside the new polygon.
            n_last (int): Last polygon edge index processed.
            poly (np.ndarray): Polygon coordinates for the current segment.
            ixy1 (int): Current index in the shifted bankline.
            xytmp (np.ndarray): Temporary array of old bankline points.
            ixytmp (int): Starting index for xytmp.
            s_last (int): Last segment index processed.

        Returns:
            int: Updated index in the shifted bankline.
        """
        if self.verbose:
            print("- wrapping up after last intersection")
        if inside:
            if self.verbose:
                print("  existing line is inside the new polygon")
            for n2 in range(n_last, -1, -1):
                if self.verbose:
                    print(f"  adding new point {poly[n2]}")
                ixy1, self.xylines_new = _add_point(ixy1, self.xylines_new, poly[n2])
        else:
            if self.verbose:
                print("  existing line is inside the new polygon")
            for s2 in range(s_last, len(xytmp) + ixytmp - 1):
                if self.verbose:
                    print(f"  re-adding old point {xytmp[s2 - ixytmp + 1]}")
                ixy1, self.xylines_new = _add_point(
                    ixy1, self.xylines_new, xytmp[s2 - ixytmp + 1]
                )
        return ixy1

    def _process_intersections_and_update_bankline(
        self,
        intersections: PolylineIntersections,
        poly: np.ndarray,
        nedges: int,
        eroded_segment: ErodedBankLineSegment,
    ) -> None:
        """
        Process sorted intersections and update the shifted bankline accordingly.

        Args:
            intersections: PolylineIntersections object with intersection data.
            poly: Polygon coordinates for the current segment.
            nedges: Number of edges in the polygon.
            eroded_segment: ErodedBankLineSegment with recent shifted bankline segments.

        Modifies:
            self.xylines_new and self.ixy in place.
        """
        s = np.concatenate(intersections.slices)
        a = np.concatenate(intersections.a)
        b = np.concatenate(intersections.b)
        n = np.concatenate(intersections.n)

        # sort the intersections by distance along the already shifted bank line
        d = s + a
        sorted_idx = np.argsort(d)
        s = s[sorted_idx] + eroded_segment.ixy0
        a = a[sorted_idx]
        b = b[sorted_idx]
        d = d[sorted_idx]
        n = n[sorted_idx]

        ixy1 = s[0]
        if self.verbose:
            print(f"continuing new path at point {ixy1}")
        xytmp = self.xylines_new[ixy1 : self.ixy + 1].copy()
        ixytmp = ixy1

        inside = False
        s_last = s[0]
        n_last = nedges
        intersection_context = IntersectionContext(
            a=a,
            b=b,
            n=n,
            slices=None,
            poly=poly,
            xytmp=xytmp,
            ixytmp=ixytmp,
            nedges=nedges,
            s_last=s_last,
        )
        for i, s_current in enumerate(s):
            inside, s_last, n_last, ixy1 = self._process_single_intersection(
                i,
                s_current,
                a,
                b,
                n,
                nedges,
                poly,
                xytmp,
                ixytmp,
                eroded_segment,
                inside,
                s_last,
                n_last,
                ixy1,
            )

        self.ixy = self._finalize_bankline_after_intersections(
            inside, n_last, poly, ixy1, xytmp, ixytmp, s_last
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
                print(f"{erosion_index}: current length of new bankline is {self.ixy}")
                print(
                    f"{erosion_index}: segment starting at {self.xylines[erosion_index]} to be shifted by {eroded_distance}"
                )
                print(f"{erosion_index}: change in direction quantified as {dtheta}")
            poly = self._create_segment_outline_polygon(erosion_index, dtheta)
            nedges = poly.shape[0] - 1

            eroded_segment = self._get_recent_shifted_bankline_segments()

            intersections = self._collect_polyline_intersections(
                eroded_segment, poly, nedges
            )

            s = np.concatenate(intersections.slices)
            if self.verbose:
                print(f"{erosion_index}: {len(s)} intersections detected")
            if len(s) == 0:
                if dtheta < 0 and not self._should_add_right_bend_segment(
                    erosion_index
                ):
                    continue
                self._add_right_bend_segment_points(poly, nedges)
            else:
                self._process_intersections_and_update_bankline(
                    intersections, poly, nedges, eroded_segment
                )
            # if iseg == isegstop:
            #     break
        self.xylines_new = self.xylines_new[: self.ixy, :]

        return self.xylines_new


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


def calculate_alpha(coords: np.ndarray, ind_1: int, ind_2: int, bp: Tuple[int, Any]):
    """Calculate the alpha value for the bank erosion model."""
    alpha = (
                    (coords[ind_1, 0] - coords[ind_2, 0]) * (bp[0] - coords[ind_2, 0])
                    + (coords[ind_1, 1] - coords[ind_2, 1]) * (bp[1] - coords[ind_2, 1])
            ) / (
                    (coords[ind_1, 0] - coords[ind_2, 0]) ** 2
                    + (coords[ind_1, 1] - coords[ind_2, 1]) ** 2
            )

    return alpha
