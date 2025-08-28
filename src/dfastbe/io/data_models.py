"""
Copyright (C) 2020 Stichting Deltares.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation version 2.1.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, see <http://www.gnu.org/licenses/>.

contact: delft3d.support@deltares.nl
Stichting Deltares
P.O. Box 177
2600 MH Delft, The Netherlands

All indications and logos of, and references to, "Delft3D" and "Deltares"
are registered trademarks of Stichting Deltares, and remain the property of
Stichting Deltares. All rights reserved.

INFORMATION
This file is part of D-FAST Bank Erosion: https://github.com/Deltares/D-FAST_Bank_Erosion
"""
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import numpy as np
import netCDF4
import math
from shapely.geometry import LineString, Point
from shapely import prepare
from geopandas.geodataframe import GeoDataFrame
from dfastbe.io.logger import log_text
from dfastbe.io.config import SimulationFilesError, ConfigFile, get_bbox


__all__ = ["BaseSimulationData", "BaseRiverData", "LineGeometry"]


class LineGeometry:
    """Center line class."""

    def __init__(self, line: LineString | np.ndarray, mask: Tuple[float, float] = None, crs: str = None):
        """Geometry Line initialization.

        Args:
            line (LineString):
                River center line as a linestring.
            mask (Tuple[float, float], optional):
                Lower and upper limit for the chainage. Defaults to None.
            crs (str, Optional):
                the coordinate reference system number as a string.

        Examples:
            ```python
            >>> line_string = LineString([(0, 0, 0), (1, 1, 1), (2, 2, 2)])
            >>> mask = (0.5, 1.5)
            >>> center_line = LineGeometry(line_string, mask)
            No message found for clip_chainage
            >>> np.array(center_line.values.coords)
            array([[0.5, 0.5, 0.5],
                   [1. , 1. , 1. ],
                   [1.5, 1.5, 1.5]])

            ```
        """
        self.station_bounds = mask
        self.crs = crs
        self._data = {}
        if isinstance(line, np.ndarray):
            line = LineString(line)
        if mask is None:
            self.values = line
        else:
            self.values: LineString = self.mask(line, mask)

            log_text(
                "clip_chainage", data={"low": self.station_bounds[0], "high": self.station_bounds[1]}
            )

    @property
    def data(self) -> Dict[str, np.ndarray]:
        """any data assigned to the line using the `add_data` method."""
        return self._data

    def as_array(self) -> np.ndarray:
        return np.array(self.values.coords)

    def add_data(self, data: Dict[str, np.ndarray]) -> None:
        """
        Add data to the LineGeometry object.

        Args:
            data (Dict[str, np.ndarray]):
                Dictionary of quantities to be added, each np array should have length k.
        """
        self._data = self.data | data

    def to_file(
            self, file_name: str, data: Dict[str, np.ndarray] = None,
    ) -> None:
        """
        Write a shape point file with x, y, and values.

        Args:
            file_name : str
                Name of the file to be written.
            data : Dict[str, np.ndarray]
                Dictionary of quantities to be written, each np array should have length k.
        """
        xy = self.as_array()
        geom = [Point(xy_i) for xy_i in xy]
        if data is None:
            data = self.data
        else:
            data = data | self.data
        GeoDataFrame(data, geometry=geom, crs=self.crs).to_file(file_name)

    @staticmethod
    def mask(line_string: LineString, bounds: Tuple[float, float]) -> LineString:
        """Clip a LineGeometry to the relevant reach.

        Args:
            line_string (LineString):
                river center line as a linestring.
            bounds (Tuple[float, float]):
                Lower and upper limit for the chainage.

        Returns:
            LineString: Clipped river chainage line.

        Examples:
            ```python
            >>> line_string = LineString([(0, 0, 0), (1, 1, 1), (2, 2, 2)])
            >>> bounds = (0.5, 1.5)
            >>> center_line = LineGeometry.mask(line_string, bounds)
            >>> np.array(center_line.coords)
            array([[0.5, 0.5, 0.5],
                   [1. , 1. , 1. ],
                   [1.5, 1.5, 1.5]])

            ```
        """
        line_string_coords = line_string.coords
        start_index = LineGeometry._find_mask_index(bounds[0], line_string_coords)
        end_index = LineGeometry._find_mask_index(bounds[1], line_string_coords)
        lower_bound_point, start_index = LineGeometry._handle_bound(
            start_index, bounds[0], True, line_string_coords
        )
        upper_bound_point, end_index = LineGeometry._handle_bound(
            end_index, bounds[1], False, line_string_coords
        )

        if lower_bound_point is None and upper_bound_point is None:
            return line_string
        elif lower_bound_point is None:
            return LineString(line_string_coords[: end_index + 1] + [upper_bound_point])
        elif upper_bound_point is None:
            return LineString([lower_bound_point] + line_string_coords[start_index:])
        else:
            return LineString(
                [lower_bound_point]
                + line_string_coords[start_index:end_index]
                + [upper_bound_point]
            )

    @staticmethod
    def _find_mask_index(
            station_bound: float, line_string_coords: np.ndarray
    ) -> Optional[int]:
        """Find the start and end indices for clipping the chainage line.

        Args:
            station_bound (float):
                Station bound for clipping.
            line_string_coords (np.ndarray):
                Coordinates of the line string.

        Returns:
            Optional[int]: index for clipping.
        """
        mask_index = next(
            (
                index
                for index, coord in enumerate(line_string_coords)
                if coord[2] >= station_bound
            ),
            None,
        )
        return mask_index

    @staticmethod
    def _handle_bound(
        index: Optional[int],
        station_bound: float,
        is_lower: bool,
        line_string_coords: np.ndarray,
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[int]]:
        """Handle the clipping of the stations line for a given bound.

        Args:
            index (Optional[int]):
                Index for clipping (start or end).
            station_bound (float):
                Station bound for clipping.
            is_lower (bool):
                True if handling the lower bound, False for the upper bound.
            line_string_coords (np.ndarray):
                Coordinates of the line string.

        Returns:
            Tuple[Optional[Tuple[float, float, float]], Optional[int]]:
                Adjusted bound point and updated index.
        """
        if index is None:
            bound_type = "Lower" if is_lower else "Upper"
            end_station = line_string_coords[-1][2]
            if is_lower or (station_bound - end_station > 0.1):
                raise ValueError(
                    f"{bound_type} chainage bound {station_bound} "
                    f"is larger than the maximum chainage {end_station} available"
                )
            return None, index

        if index == 0:
            bound_type = "Lower" if is_lower else "Upper"
            start_station = line_string_coords[0][2]
            if not is_lower or (start_station - station_bound > 0.1):
                raise ValueError(
                    f"{bound_type} chainage bound {station_bound} "
                    f"is smaller than the minimum chainage {start_station} available"
                )
            return None, index

        # Interpolate the point
        alpha, interpolated_point = LineGeometry._interpolate_point(
            index, station_bound, line_string_coords
        )

        # Adjust the index based on the interpolation factor
        if is_lower and alpha > 0.9:
            index += 1
        elif not is_lower and alpha < 0.1:
            index -= 1

        return interpolated_point, index

    @staticmethod
    def _interpolate_point(
        index: int, station_bound: float, line_string_coords: np.ndarray
    ) -> Tuple[float, Tuple[float, float, float]]:
        """Interpolate a point between two coordinates.

        Args:
            index (int):
                Index of the coordinate to interpolate.
            station_bound (float):
                Station bound for interpolation.
            line_string_coords (np.ndarray):
                Coordinates of the line string.

        Returns:
            float: Interpolation factor.
            Tuple[float, float, float]: Interpolated point.
        """
        alpha = (station_bound - line_string_coords[index - 1][2]) / (
                line_string_coords[index][2] - line_string_coords[index - 1][2]
        )
        interpolated_point = tuple(
            prev_coord + alpha * (next_coord - prev_coord)
            for prev_coord, next_coord in zip(
                line_string_coords[index - 1], line_string_coords[index]
            )
        )
        return alpha, interpolated_point

    def intersect_with_line(
        self, reference_line_with_stations: np.ndarray
    ) -> np.ndarray:
        """
        Project chainage(stations) values from a reference line onto a target line by spatial proximity and interpolation.

        Project chainage values from source line L1 onto another line L2.

        The chainage values are giving along a line L1 (xykm_numpy). For each node
        of the line L2 (line_xy) on which we would like to know the chainage, first
        the closest node (discrete set of nodes) on L1 is determined and
        subsequently the exact chainage is obtained by determining the closest point
        (continuous line) on L1 for which the chainage is determined using by means
        of interpolation.

        Args:
            reference_line_with_stations (np.ndarray):
                Mx3 array with x, y, and chainage values for the reference line.

        Returns:
            line_km (np.ndarray):
                1D Array containing the chainage(stations in km) for every coordinate specified in line_xy.
        """
        coords = self.as_array()
        # pre-allocates the array for the mapped chainage values
        projected_stations = np.zeros(coords.shape[0])

        # get an array with only the x,y coordinates of line L1
        ref_coords = reference_line_with_stations[:, :2]
        last_index = reference_line_with_stations.shape[0] - 1

        # for each node rp on line L2 get the chainage ...
        for i, station_i in enumerate(coords):
            # find the node on L1 closest to rp
            # get the distance to all the nodes on the reference line, and find the closest one
            closest_ind = np.argmin(((station_i - ref_coords) ** 2).sum(axis=1))
            closest_coords = ref_coords[closest_ind]

            # determine the distance between that node and rp
            squared_distance = ((station_i - closest_coords) ** 2).sum()

            # chainage value of that node
            station = reference_line_with_stations[closest_ind, 2]

            # if we didn't get the first node
            if closest_ind > 0:
                # project rp onto the line segment before this node
                closest_coord_minus_1 = ref_coords[closest_ind - 1]
                alpha = (
                                (closest_coord_minus_1[0] - closest_coords[0]) * (station_i[0] - closest_coords[0])
                                + (closest_coord_minus_1[1] - closest_coords[1]) * (station_i[1] - closest_coords[1])
                        ) / ((closest_coord_minus_1[0] - closest_coords[0]) ** 2 + (closest_coord_minus_1[1] - closest_coords[1]) ** 2)
                # if there is a closest point not coinciding with the nodes ...
                if 0 < alpha < 1:
                    dist2link = (station_i[0] - closest_coords[0] - alpha * (closest_coord_minus_1[0] - closest_coords[0])) ** 2 + (
                            station_i[1] - closest_coords[1] - alpha * (closest_coord_minus_1[1] - closest_coords[1])
                    ) ** 2
                    # if it's actually closer than the node ...
                    if dist2link < squared_distance:
                        # update the closest point information
                        squared_distance = dist2link
                        station = reference_line_with_stations[closest_ind, 2] + alpha * (
                                reference_line_with_stations[closest_ind - 1, 2] - reference_line_with_stations[closest_ind, 2]
                        )

            # if we didn't get the last node
            if closest_ind < last_index:
                # project rp onto the line segment after this node
                closest_coord_minus_1 = ref_coords[closest_ind + 1]
                alpha = (
                                (closest_coord_minus_1[0] - closest_coords[0]) * (station_i[0] - closest_coords[0])
                                + (closest_coord_minus_1[1] - closest_coords[1]) * (station_i[1] - closest_coords[1])
                        ) / ((closest_coord_minus_1[0] - closest_coords[0]) ** 2 + (closest_coord_minus_1[1] - closest_coords[1]) ** 2)
                # if there is a closest point not coinciding with the nodes ...
                if alpha > 0 and alpha < 1:
                    dist2link = (station_i[0] - closest_coords[0] - alpha * (closest_coord_minus_1[0] - closest_coords[0])) ** 2 + (
                            station_i[1] - closest_coords[1] - alpha * (closest_coord_minus_1[1] - closest_coords[1])
                    ) ** 2
                    # if it's actually closer than the previous value ...
                    if dist2link < squared_distance:
                        # update the closest point information
                        # squared_distance = dist2link
                        station = reference_line_with_stations[closest_ind, 2] + alpha * (
                                reference_line_with_stations[closest_ind + 1, 2] - reference_line_with_stations[closest_ind, 2]
                        )
            # store the chainage value, loop ... and return
            projected_stations[i] = station
        return projected_stations

    def get_bbox(
        self, buffer: float = 0.1
    ) -> Tuple[float, float, float, float]:
        """
        Derive the bounding box from a line.

        Args:
            buffer : float
                Buffer fraction surrounding the tight bounding box

        Returns:
            bbox (Tuple[float, float, float, float]):
                bounding box consisting of [min x, min y, max x, max y)
        """
        coords = self.as_array()
        x = coords[:, 0]
        y = coords[:, 1]
        x_min = x.min()
        y_min = y.min()
        x_max = x.max()
        y_max = y.max()
        d = buffer * max(x_max - x_min, y_max - y_min)
        bbox = (x_min - d, y_min - d, x_max + d, y_max + d)

        return bbox

class BaseSimulationData:
    """Class to hold simulation data.

    This class contains the simulation data read from a UGRID netCDF file.
    It includes methods to read the data from the file and clip the simulation
    mesh to a specified area of interest.
    """

    def __init__(
        self,
        x_node: np.ndarray,
        y_node: np.ndarray,
        n_nodes: np.ndarray,
        face_node: np.ma.masked_array,
        bed_elevation_location: np.ndarray,
        bed_elevation_values: np.ndarray,
        water_level_face: np.ndarray,
        water_depth_face: np.ndarray,
        velocity_x_face: np.ndarray,
        velocity_y_face: np.ndarray,
        chezy_face: np.ndarray,
        dry_wet_threshold: float,
    ):
        """
        Initialize the SimulationData object.

        Args:
        x_node (np.ndarray):
            X-coordinates of the nodes.
        y_node (np.ndarray):
            Y-coordinates of the nodes.
        n_nodes (np.ndarray):
            Number of nodes in each face.
        face_node (np.ma.masked_array):
            Face-node connectivity array.
        bed_elevation_location (np.ndarray):
            Determines whether the bed elevation is associated with nodes
            or faces in the computational mesh.
        bed_elevation_values (np.ndarray):
            Bed elevation values for each node in the simulation data.
        water_level_face (np.ndarray):
            Water levels at the faces.
        water_depth_face (np.ndarray):
            Water depths at the faces.
        velocity_x_face (np.ndarray):
            X-component of the velocity at the faces.
        velocity_y_face (np.ndarray):
            Y-component of the velocity at the faces.
        chezy_face (np.ndarray):
            Chezy roughness values at the faces.
        dry_wet_threshold (float):
            Threshold depth for detecting drying and flooding.
        """
        self.x_node = x_node
        self.y_node = y_node
        self.n_nodes = n_nodes
        self.face_node = face_node
        self.bed_elevation_location = bed_elevation_location
        self.bed_elevation_values = bed_elevation_values
        self.water_level_face = water_level_face
        self.water_depth_face = water_depth_face
        self.velocity_x_face = velocity_x_face
        self.velocity_y_face = velocity_y_face
        self.chezy_face = chezy_face
        self.dry_wet_threshold = dry_wet_threshold

    @classmethod
    def read(cls, file_name: str, indent: str = "") -> "BaseSimulationData":
        """Read a default set of quantities from a UGRID netCDF file.

        Supported files are coming from D-Flow FM (or similar).

        Args:
            file_name (str):
                Name of the simulation output file to be read.
            indent (str):
                String to use for each line as indentation (default empty).

        Raises:
            SimulationFilesError:
                If the file is not recognized as a D-Flow FM map-file.

        Returns:
            BaseSimulationData (Tuple[BaseSimulationData, float]):
                Dictionary containing the data read from the simulation output file.

        Examples:
            ```python
            >>> from dfastbe.io.data_models import BaseSimulationData
            >>> sim_data = BaseSimulationData.read("tests/data/erosion/inputs/sim0075/SDS-j19_map.nc") # doctest: +ELLIPSIS
            No message ... read_drywet
            >>> print(sim_data.x_node[0:3])
            [194949.796875 194966.515625 194982.8125  ]

            ```
        """
        name = Path(file_name).name
        if name.endswith("map.nc"):
            log_text("read_grid", indent=indent)

            # read the node coordinates
            x_node = _read_fm_map(file_name, "x", location="node")
            y_node = _read_fm_map(file_name, "y", location="node")

            # read the face node connectivity and make sure it's a masked array
            f_nc_read = _read_fm_map(file_name, "face_node_connectivity")
            if isinstance(f_nc_read, np.ma.MaskedArray):
                f_nc = f_nc_read
                # make sure the mask is a full array
                if f_nc.mask.size == 1:
                    f_nc.mask = np.full(f_nc.shape, False)
            else:
                f_nc = np.ma.MaskedArray(f_nc_read, np.full(f_nc_read.shape, False))

            # remove invalid node indices ... this happens typically if _FillValue is not correctly set or applied
            f_nc.mask[f_nc.data < 0] = True
            f_nc.mask[f_nc.data > x_node.size-1] = True
            # consider checking for invalid indices and applying mask
            f_nc.data[f_nc.mask] = 0
            n_nodes_per_face = f_nc.mask.shape[1] - f_nc.mask.sum(axis=1)

            face_node = f_nc
            log_text("read_bathymetry", indent=indent)
            bed_elevation_location = "node"
            bed_elevation_values = _read_fm_map(file_name, "altitude", location="node")
            log_text("read_water_level", indent=indent)
            water_level_face = _read_fm_map(file_name, "Water level")
            log_text("read_water_depth", indent=indent)
            water_depth_face = np.maximum(
                _read_fm_map(file_name, "sea_floor_depth_below_sea_surface"), 0.0
            )
            log_text("read_velocity", indent=indent)
            velocity_x_face = _read_fm_map(file_name, "sea_water_x_velocity")
            velocity_y_face = _read_fm_map(file_name, "sea_water_y_velocity")
            log_text("read_chezy", indent=indent)
            chezy_face = _read_fm_map(file_name, "Chezy roughness")

            log_text("read_drywet", indent=indent)
            root_group = netCDF4.Dataset(file_name)
            try:
                file_source = root_group.converted_from
                if file_source == "SIMONA":
                    dry_wet_threshold = 0.1
                else:
                    dry_wet_threshold = 0.01
            except AttributeError:
                dry_wet_threshold = 0.01

        elif name.startswith("SDS"):
            raise SimulationFilesError(
                f"WAQUA output files not yet supported. Unable to process {name}"
            )
        elif name.startswith("trim"):
            raise SimulationFilesError(
                f"Delft3D map files not yet supported. Unable to process {name}"
            )
        else:
            raise SimulationFilesError(f"Unable to determine file type for {name}")

        return cls(
            x_node=x_node,
            y_node=y_node,
            n_nodes=n_nodes_per_face,
            face_node=face_node,
            bed_elevation_location=bed_elevation_location,
            bed_elevation_values=bed_elevation_values,
            water_level_face=water_level_face,
            water_depth_face=water_depth_face,
            velocity_x_face=velocity_x_face,
            velocity_y_face=velocity_y_face,
            chezy_face=chezy_face,
            dry_wet_threshold=dry_wet_threshold,
        )

    def clip(self, river_center_line: LineString, max_distance: float):
        """Clip the simulation mesh.

        Clipping data to the area of interest,
        that is sufficiently close to the reference line.

        Args:
            river_center_line (np.ndarray):
                Reference line.
            max_distance (float):
                Maximum distance between the reference line and a point in the area of
                interest defined based on the search lines for the banks and the search
                distance.

        Notes:
            The function uses the Shapely library to create a buffer around the river
            profile and checks if the nodes are within that buffer. If they are not,
            they are removed from the simulation data.

        Examples:
            ```python
            >>> from dfastbe.io.data_models import BaseSimulationData
            >>> sim_data = BaseSimulationData.read("tests/data/erosion/inputs/sim0075/SDS-j19_map.nc")
            No message found for read_grid
            No message found for read_bathymetry
            No message found for read_water_level
            No message found for read_water_depth
            No message found for read_velocity
            No message found for read_chezy
            No message found for read_drywet
            >>> river_center_line = LineString([
            ...     [194949.796875, 361366.90625],
            ...     [194966.515625, 361399.46875],
            ...     [194982.8125, 361431.03125]
            ... ])
            >>> max_distance = 10.0
            >>> sim_data.clip(river_center_line, max_distance)
            >>> print(sim_data.x_node)
            [194949.796875 194966.515625 194982.8125  ]

            ```
        """
        xy_buffer = river_center_line.buffer(max_distance + max_distance)
        bbox = xy_buffer.envelope.exterior
        x_min = bbox.coords[0][0]
        x_max = bbox.coords[1][0]
        y_min = bbox.coords[0][1]
        y_max = bbox.coords[2][1]

        # mark which nodes to keep
        prepare(xy_buffer)
        x = self.x_node
        y = self.y_node
        nnodes = x.shape
        keep = (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)
        for i in range(x.size):
            if keep[i] and not xy_buffer.contains(Point((x[i], y[i]))):
                keep[i] = False

        # mark which faces to keep
        fnc = self.face_node
        keep_face_nodes = np.ma.masked_array(keep[fnc], fnc.mask)
        keep_face = keep_face_nodes.all(axis=1)
        renum = np.zeros(nnodes, dtype=int)
        renum[keep] = range(sum(keep))
        self.face_node = np.ma.masked_array(renum[fnc[keep_face]], fnc.mask[keep_face])

        self.x_node = x[keep]
        self.y_node = y[keep]
        if self.bed_elevation_location == "node":
            self.bed_elevation_values = self.bed_elevation_values[keep]
        else:
            self.bed_elevation_values = self.bed_elevation_values[keep_face]

        self.n_nodes = self.n_nodes[keep_face]
        self.water_level_face = self.water_level_face[keep_face]
        self.water_depth_face = self.water_depth_face[keep_face]
        self.velocity_x_face = self.velocity_x_face[keep_face]
        self.velocity_y_face = self.velocity_y_face[keep_face]
        self.chezy_face = self.chezy_face[keep_face]


class BaseRiverData:
    """River data class."""

    def __init__(self, config_file: ConfigFile):
        """River Data initialization.

        Args:
            config_file (ConfigFile):
                Configuration file with settings for the analysis.

        Examples:
            ```python
            >>> from dfastbe.io.data_models import ConfigFile, BaseRiverData
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> river_data = BaseRiverData(config_file)
            No message found for read_chainage
            No message found for clip_chainage

            ```
        """
        self.config_file = config_file
        center_line = config_file.get_river_center_line()
        bounds = config_file.get_start_end_stations()
        self._river_center_line = LineGeometry(center_line, bounds, crs=config_file.crs)
        self._station_bounds: Tuple = config_file.get_start_end_stations()

    @property
    def river_center_line(self) -> LineGeometry:
        """LineGeometry: the clipped river center line."""
        return self._river_center_line

    @property
    def station_bounds(self) -> Tuple[float, float]:
        """Tuple: the lower and upper bounds of the river center line."""
        return self._station_bounds

    @staticmethod
    def get_bbox(
        coords: np.ndarray, buffer: float = 0.1
    ) -> Tuple[float, float, float, float]:
        """
        Derive the bounding box from an array of coordinates.

        Args:
            coords (np.ndarray):
                An N x M array containing x- and y-coordinates as first two M entries
            buffer : float
                Buffer fraction surrounding the tight bounding box

        Returns:
            bbox (Tuple[float, float, float, float]):
                Tuple bounding box consisting of [min x, min y, max x, max y)
        """
        return get_bbox(coords, buffer)

    def get_erosion_sim_data(self, num_discharge_levels: int) -> Tuple[List[str], List[float]]:
        # get p discharges
        sim_files = []
        p_discharge = []
        for iq in range(num_discharge_levels):
            iq_str = str(iq + 1)
            sim_files.append(self.config_file.get_sim_file("Erosion", iq_str))
            p_discharge.append(
                self.config_file.get_float("Erosion", f"PDischarge{iq_str}")
            )
        return sim_files, p_discharge

def _read_fm_map(filename: str, varname: str, location: str = "face") -> np.ndarray:
    """
    Read the last time step of any quantity defined at faces from a D-Flow FM map-file.

    Arguments
    ---------
    filename : str
        Name of the D-Flow FM map.nc file to read the data.
    varname : str
        Name of the netCDF variable to be read.
    location : str
        Name of the stagger location at which the data should be located
        (default is "face")

    Raises
    ------
    Exception
        If the data file doesn't include a 2D mesh.
        If it cannot uniquely identify the variable to be read.

    Returns
    -------
    data
        Data of the requested variable (for the last time step only if the variable is
        time dependent).
    """
    # open file
    root_group = netCDF4.Dataset(filename)
    remove_mask = True

    # locate 2d mesh variable
    mesh2d = root_group.get_variables_by_attributes(
        cf_role="mesh_topology", topology_dimension=2
    )
    if len(mesh2d) != 1:
        raise ValueError(
            f"Currently only one 2D mesh supported ... this file contains {len(mesh2d)} 2D meshes."
        )
    mesh_name = mesh2d[0].name

    # define a default start_index
    start_index = 0

    # locate the requested variable ... start with some special cases
    if varname == "x":
        # the x-coordinate or longitude
        coords_names = mesh2d[0].getncattr(location + "_coordinates").split()
        for n in coords_names:
            std_name = root_group.variables[n].standard_name
            if std_name == "projection_x_coordinate" or std_name == "longitude":
                var = root_group.variables[n]
                break

    elif varname == "y":
        # the y-coordinate or latitude
        coords_names = mesh2d[0].getncattr(location + "_coordinates").split()
        for n in coords_names:
            std_name = root_group.variables[n].standard_name
            if std_name == "projection_y_coordinate" or std_name == "latitude":
                var = root_group.variables[n]
                break

    elif varname.endswith("connectivity"):
        # a mesh connectivity variable with corrected index
        varname = mesh2d[0].getncattr(varname)
        var = root_group.variables[varname]
        if "start_index" in var.ncattrs():
            start_index = var.getncattr("start_index")
        remove_mask = False

    else:
        # find any other variable by standard_name or long_name
        var = root_group.get_variables_by_attributes(
            standard_name=varname, mesh=mesh_name, location=location
        )
        if len(var) == 0:
            var = root_group.get_variables_by_attributes(
                long_name=varname, mesh=mesh_name, location=location
            )
        if len(var) != 1:
            raise ValueError(
                f'Expected one variable for "{varname}", but obtained {len(var)}.'
            )
        var = var[0]

    # read data checking for time dimension
    if var.get_dims()[0].isunlimited():
        # assume that time dimension is unlimited and is the first dimension
        # slice to obtain last time step
        data_read = var[-1, :]
    else:
        data_read = var[...] - start_index

    if remove_mask and isinstance(data_read, np.ma.MaskedArray):
        data = data_read.data
        data[data_read.mask] = math.nan
    else:
        data = data_read

    root_group.close()

    return data