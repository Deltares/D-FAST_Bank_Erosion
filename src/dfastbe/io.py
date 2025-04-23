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

from configparser import ConfigParser
from configparser import Error as ConfigparserError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple, Union

import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd
from dfastio.xyc.models import XYCModel
from geopandas.geodataframe import GeoDataFrame
from geopandas.geoseries import GeoSeries
from shapely import prepare
from shapely.geometry import LineString, Point

from dfastbe.structures import MeshData

MAX_RIVER_WIDTH = 1000


PROGTEXTS: Dict[str, List[str]]


class SimulationData:
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
    def read(cls, file_name: str, indent: str = "") -> "SimulationData":
        """Read a default set of quantities from a UGRID netCDF file.

        Supported files are coming from D-Flow FM (or similar).

        Args:
            file_name (str):
                Name of the simulation output file to be read.
            indent (str):
                String to use for each line as indentation (default empty).

        Raises:
            SimulationFilesError
                If the file is not recognized as a D-Flow FM map-file.

        Returns:
            SimulationData: Dictionary containing the data read from the simulation output file.
            float: Threshold depth for detecting drying and flooding.

        Examples:
            ```python
            >>> from dfastbe.io import SimulationData
            >>> sim_data = SimulationData.read("tests/data/erosion/inputs/sim0075/SDS-j19_map.nc")
            No message found for read_grid
            No message found for read_bathymetry
            No message found for read_water_level
            No message found for read_water_depth
            No message found for read_velocity
            No message found for read_chezy
            No message found for read_drywet
            >>> print(sim_data.x_node[0:3])
            [194949.796875 194966.515625 194982.8125  ]

            ```
        """
        name = Path(file_name).name
        if name.endswith("map.nc"):
            log_text("read_grid", indent=indent)
            x_node = _read_fm_map(file_name, "x", location="node")
            y_node = _read_fm_map(file_name, "y", location="node")
            f_nc = _read_fm_map(file_name, "face_node_connectivity")
            if f_nc.mask.shape == ():
                # all faces have the same number of nodes
                n_nodes = np.ones(f_nc.data.shape[0], dtype=int) * f_nc.data.shape[1]
            else:
                # varying number of nodes
                n_nodes = f_nc.mask.shape[1] - f_nc.mask.sum(axis=1)
            f_nc.data[f_nc.mask] = 0

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
            except:
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
            n_nodes=n_nodes,
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
            >>> from dfastbe.io import SimulationData
            >>> sim_data = SimulationData.read("tests/data/erosion/inputs/sim0075/SDS-j19_map.nc")
            No message found for read_grid
            No message found for read_bathymetry
            No message found for read_water_level
            No message found for read_water_depth
            No message found for read_velocity
            No message found for read_chezy
            No message found for read_drywet
            >>> river_profile = LineString([
            ... [194949.796875, 361366.90625],
            ... [194966.515625, 361399.46875],
            ... [194982.8125, 361431.03125]
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

        prepare(xy_buffer)
        x = self.x_node
        y = self.y_node
        nnodes = x.shape
        keep = (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)
        for i in range(x.size):
            if keep[i] and not xy_buffer.contains(Point((x[i], y[i]))):
                keep[i] = False

        fnc = self.face_node
        keep_face = keep[fnc].all(axis=1)
        renum = np.zeros(nnodes, dtype=int)
        renum[keep] = range(sum(keep))
        self.face_node = renum[fnc[keep_face]]

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

    def compute_mesh_topology(self) -> MeshData:
        """Derive secondary topology arrays from the face-node connectivity of the mesh.

        This function computes the edge-node, edge-face, and face-edge connectivity arrays,
        as well as the boundary edges of the mesh, based on the face-node connectivity provided
        in the simulation data.

        Returns:
            MeshData: a dataclass containing the following attributes:
                - `x_face_coords`: x-coordinates of face nodes
                - `y_face_coords`: y-coordinates of face nodes
                - `x_edge_coords`: x-coordinates of edge nodes
                - `y_edge_coords`: y-coordinates of edge nodes
                - `face_node`: the node indices for each of the mesh faces.
                - `n_nodes`: number of nodes per face
                - `edge_node`: the node indices for each of the mesh edges.
                - `edge_face_connectivity`: the face indices for each of the mesh edge
                - `face_edge_connectivity`: the edge indices for each of the mesh face
                - `boundary_edge_nrs`: indices of boundary edges

        Raises:
            KeyError:
                If required keys (e.g., `face_node`, `nnodes`, `x_node`, `y_node`) are missing from the `sim` object.

        Notes:
            - The function identifies unique edges by sorting and comparing node indices.
            - Boundary edges are identified as edges that belong to only one face.
            - The function assumes that the mesh is well-formed, with consistent face-node connectivity.
        """

        # get a sorted list of edge node connections (shared edges occur twice)
        # face_nr contains the face index to which the edge belongs
        n_faces = self.face_node.shape[0]
        n_edges = sum(self.n_nodes)
        edge_node = np.zeros((n_edges, 2), dtype=int)
        face_nr = np.zeros((n_edges,), dtype=int)
        i = 0
        for face_i in range(n_faces):
            num_edges = self.n_nodes[face_i]  # note: nEdges = nNodes
            for edge_i in range(num_edges):
                if edge_i == 0:
                    edge_node[i, 1] = self.face_node[face_i, num_edges - 1]
                else:
                    edge_node[i, 1] = self.face_node[face_i, edge_i - 1]
                edge_node[i, 0] = self.face_node[face_i, edge_i]
                face_nr[i] = face_i
                i = i + 1
        edge_node.sort(axis=1)
        i2 = np.argsort(edge_node[:, 1], kind="stable")
        i1 = np.argsort(edge_node[i2, 0], kind="stable")
        i12 = i2[i1]
        edge_node = edge_node[i12, :]
        face_nr = face_nr[i12]

        # detect which edges are equal to the previous edge, and get a list of all unique edges
        numpy_true = np.array([True])
        equal_to_previous = np.concatenate(
            (~numpy_true, (np.diff(edge_node, axis=0) == 0).all(axis=1))
        )
        unique_edge = ~equal_to_previous
        n_unique_edges = np.sum(unique_edge)
        # reduce the edge node connections to only the unique edges
        edge_node = edge_node[unique_edge, :]

        # number the edges
        edge_nr = np.zeros(n_edges, dtype=int)
        edge_nr[unique_edge] = np.arange(n_unique_edges, dtype=int)
        edge_nr[equal_to_previous] = edge_nr[
            np.concatenate((equal_to_previous[1:], equal_to_previous[:1]))
        ]

        # if two consecutive edges are unique, the first one occurs only once and represents a boundary edge
        is_boundary_edge = unique_edge & np.concatenate((unique_edge[1:], numpy_true))
        boundary_edge_nrs = edge_nr[is_boundary_edge]

        # go back to the original face order
        edge_nr_in_face_order = np.zeros(n_edges, dtype=int)
        edge_nr_in_face_order[i12] = edge_nr
        # create the face edge connectivity array
        face_edge_connectivity = np.zeros(self.face_node.shape, dtype=int)

        i = 0
        for face_i in range(n_faces):
            num_edges = self.n_nodes[face_i]  # note: num_edges = n_nodes
            for edge_i in range(num_edges):
                face_edge_connectivity[face_i, edge_i] = edge_nr_in_face_order[i]
                i = i + 1

        # determine the edge face connectivity
        edge_face = -np.ones((n_unique_edges, 2), dtype=int)
        edge_face[edge_nr[unique_edge], 0] = face_nr[unique_edge]
        edge_face[edge_nr[equal_to_previous], 1] = face_nr[equal_to_previous]

        x_face_coords = SimulationData.apply_masked_indexing(
            self.x_node, self.face_node
        )
        y_face_coords = SimulationData.apply_masked_indexing(
            self.y_node, self.face_node
        )
        x_edge_coords = self.x_node[edge_node]
        y_edge_coords = self.y_node[edge_node]

        return MeshData(
            x_face_coords=x_face_coords,
            y_face_coords=y_face_coords,
            x_edge_coords=x_edge_coords,
            y_edge_coords=y_edge_coords,
            face_node=self.face_node,
            n_nodes=self.n_nodes,
            edge_node=edge_node,
            edge_face_connectivity=edge_face,
            face_edge_connectivity=face_edge_connectivity,
            boundary_edge_nrs=boundary_edge_nrs,
        )

    @staticmethod
    def apply_masked_indexing(
        x0: np.array, idx: np.ma.masked_array
    ) -> np.ma.masked_array:
        """
        Index one array by another transferring the mask.

        Args:
            x0 : np.ndarray
                A linear array.
            idx : np.ma.masked_array
                An index array with possibly masked indices.

        returns:
            x1: np.ma.masked_array
                An array with same shape as idx, with mask.
        """
        idx_safe = idx.copy()
        idx_safe.data[np.ma.getmask(idx)] = 0
        x1 = np.ma.masked_where(np.ma.getmask(idx), x0[idx_safe])
        return x1


class ConfigFile:
    """Class to read configuration files for D-FAST Bank Erosion.

    This class provides methods to read, write, and manage configuration files
    for the D-FAST Bank Erosion analysis. It also allows access to configuration
    settings and supports upgrading older configuration formats.
    """

    def __init__(self, config: ConfigParser, path: Union[Path, str] = None):
        """
        Initialize the ConfigFile object.

        Args:
            config (ConfigParser):
                Settings for the D-FAST Bank Erosion analysis.
            path (Union[Path, str]):
                Path to the configuration file.

        Examples:
            Reading a configuration file:
                ```python
                >>> import tempfile
                >>> from dfastbe.io import ConfigFile
                >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
                >>> print(config_file.config["General"]["Version"])
                1.0

                ```
            Writing a configuration file:
                ```python
                >>> from dfastbe.io import ConfigFile
                >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
                >>> with tempfile.TemporaryDirectory() as tmpdirname:
                ...     config_file.write(f"{tmpdirname}/meuse_manual_out.cfg")

                ```
        """
        self._config = config
        self.crs = "EPSG:28992"
        if path:
            self.path = Path(path)
            self.root_dir = self.path.parent
            self.make_paths_absolute()

    @property
    def config(self) -> ConfigParser:
        """ConfigParser: Get the configuration settings."""
        return self._config

    @config.setter
    def config(self, value: ConfigParser):
        self._config = value

    @property
    def version(self) -> str:
        """str: Get the version of the configuration file."""
        return self.get_str("General", "Version")

    @property
    def root_dir(self) -> Path:
        """Path: Get the root directory of the configuration file."""
        return self._root_dir

    @root_dir.setter
    def root_dir(self, value: str):
        self._root_dir = value

    @classmethod
    def read(cls, path: Union[str, Path]) -> "ConfigFile":
        """Read a configParser object (configuration file).

        Reads the config file using the standard `configparser`. Falls back to a
        dedicated reader compatible with old waqbank files.

        Args:
            path (Union[str, Path]): Path to the configuration file.

        Returns:
            ConfigFile: Settings for the D-FAST Bank Erosion analysis.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            Exception: If there is an error reading the config file.

        Examples:
            Read a config file:
            ```python
            >>> from dfastbe.io import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")

            ```
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"The Config-File: {path} does not exist")

        try:
            config = ConfigParser(comment_prefixes="%")
            with open(path, "r") as configfile:
                config.read_file(configfile)
        except ConfigparserError as e:
            print(f"Error during reading the config file: {e}")
            config = cls.config_file_callback_parser(path)

        # if version != "1.0":
        config = cls._upgrade(config)
        return cls(config, path=path)

    @staticmethod
    def config_file_callback_parser(path: str) -> ConfigParser:
        """Parse a configuration file as fallback to the read method.

        Args:
            path (str): Path to the configuration file.

        Returns:
            ConfigParser: Parsed configuration file.
        """
        config = ConfigParser()
        config["General"] = {}
        all_lines = open(path, "r").read().splitlines()
        for line in all_lines:
            perc = line.find("%")
            if perc >= 0:
                line = line[:perc]
            data = line.split()
            if len(data) >= 3:
                config["General"][data[0]] = data[2]
        return config

    @staticmethod
    def _upgrade(config: ConfigParser) -> ConfigParser:
        """Upgrade the configuration data structure to version 1.0 format.

        Args:
            config (ConfigParser): D-FAST Bank Erosion settings in 0.1 format.

        Returns:
            ConfigParser: D-FAST Bank Erosion settings in 1.0 format.

        Examples:
            ```python
            >>> from dfastbe.io import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> result = config_file._upgrade(config_file.config)
            >>> isinstance(result, ConfigParser)
            True

            ```
        """
        try:
            version = config["General"]["Version"]
        except KeyError:
            version = "0.1"

        if version == "0.1":
            config["General"]["Version"] = "1.0"

            config["Detect"] = {}
            config = _move_parameter_location(
                config, "General", "Delft3Dfile", "Detect", "SimFile", convert=_sim2nc
            )
            config = _move_parameter_location(
                config, "General", "SDSfile", "Detect", "SimFile", convert=_sim2nc
            )
            config = _move_parameter_location(config, "General", "SimFile", "Detect")
            config = _move_parameter_location(config, "General", "NBank", "Detect")
            config_file = ConfigFile(config)
            n_bank = config_file.get_int("Detect", "NBank", default=0, positive=True)
            for i in range(1, n_bank + 1):
                config = _move_parameter_location(
                    config, "General", f"Line{i}", "Detect"
                )

            config = _move_parameter_location(config, "General", "WaterDepth", "Detect")
            config = _move_parameter_location(config, "General", "DLines", "Detect")

            config["Erosion"] = {}
            config = _move_parameter_location(config, "General", "TErosion", "Erosion")
            config = _move_parameter_location(config, "General", "RiverAxis", "Erosion")
            config = _move_parameter_location(config, "General", "Fairway", "Erosion")
            config = _move_parameter_location(config, "General", "RefLevel", "Erosion")
            config = _move_parameter_location(
                config, "General", "OutputInterval", "Erosion"
            )
            config = _move_parameter_location(config, "General", "OutputDir", "Erosion")
            config = _move_parameter_location(config, "General", "BankNew", "Erosion")
            config = _move_parameter_location(config, "General", "BankEq", "Erosion")
            config = _move_parameter_location(config, "General", "EroVol", "Erosion")
            config = _move_parameter_location(
                config, "General", "EroVolEqui", "Erosion"
            )
            config = _move_parameter_location(config, "General", "NLevel", "Erosion")
            config_file = ConfigFile(config)
            n_level = config_file.get_int("Erosion", "NLevel", default=0, positive=True)

            for i in range(1, n_level + 1):
                config = _move_parameter_location(
                    config,
                    "General",
                    f"Delft3Dfile{i}",
                    "Erosion",
                    f"SimFile{i}",
                    convert=_sim2nc,
                )
                config = _move_parameter_location(
                    config,
                    "General",
                    f"SDSfile{i}",
                    "Erosion",
                    f"SimFile{i}",
                    convert=_sim2nc,
                )
                config = _move_parameter_location(
                    config, "General", f"SimFile{i}", "Erosion"
                )
                config = _move_parameter_location(
                    config, "General", f"PDischarge{i}", "Erosion"
                )

            config = _move_parameter_location(config, "General", "ShipType", "Erosion")
            config = _move_parameter_location(config, "General", "VShip", "Erosion")
            config = _move_parameter_location(config, "General", "NShip", "Erosion")
            config = _move_parameter_location(config, "General", "NWave", "Erosion")
            config = _move_parameter_location(config, "General", "Draught", "Erosion")
            config = _move_parameter_location(config, "General", "Wave0", "Erosion")
            config = _move_parameter_location(config, "General", "Wave1", "Erosion")

            config = _move_parameter_location(config, "General", "Classes", "Erosion")
            config = _move_parameter_location(config, "General", "BankType", "Erosion")
            config = _move_parameter_location(
                config, "General", "ProtectLevel", "Erosion", "ProtectionLevel"
            )
            config = _move_parameter_location(config, "General", "Slope", "Erosion")
            config = _move_parameter_location(config, "General", "Reed", "Erosion")
            config = _move_parameter_location(config, "General", "VelFilter", "Erosion")

            for i in range(1, n_level + 1):
                config = _move_parameter_location(
                    config, "General", f"ShipType{i}", "Erosion"
                )
                config = _move_parameter_location(
                    config, "General", f"VShip{i}", "Erosion"
                )
                config = _move_parameter_location(
                    config, "General", f"NShip{i}", "Erosion"
                )
                config = _move_parameter_location(
                    config, "General", f"NWave{i}", "Erosion"
                )
                config = _move_parameter_location(
                    config, "General", f"Draught{i}", "Erosion"
                )
                config = _move_parameter_location(
                    config, "General", f"Slope{i}", "Erosion"
                )
                config = _move_parameter_location(
                    config, "General", f"Reed{i}", "Erosion"
                )
                config = _move_parameter_location(
                    config, "General", f"EroVol{i}", "Erosion"
                )

        return config

    def write(self, filename: str) -> None:
        """Pretty print a configParser object (configuration file) to file.

        Pretty prints a `configparser` object to a file. Aligns the equal signs for
        all keyword/value pairs, adds a two-space indentation to all keyword lines,
        and adds an empty line before the start of a new block.

        Args:
            filename (str): Name of the configuration file to be written.

        Examples:
            ```python
            >>> import tempfile
            >>> from dfastbe.io import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> with tempfile.TemporaryDirectory() as tmpdirname:
            ...     config_file.write(f"{tmpdirname}/meuse_manual_out.cfg")

            ```
        """
        sections = self.config.sections()
        max_length = 0
        for section in sections:
            options = self.config.options(section)
            max_length = max(max_length, *[len(option) for option in options])

        with open(filename, "w") as configfile:
            for index, section in enumerate(sections):
                if index > 0:
                    configfile.write("\n")
                configfile.write(f"[{section}]\n")

                for option in self.config.options(section):
                    configfile.write(
                        f"  {option:<{max_length}} = {self.config[section][option]}\n"
                    )

    def make_paths_absolute(self) -> str:
        """Convert all relative paths in the configuration to absolute paths.

        Returns:
            str: Absolute path of the configuration file's root directory.
        """
        self.resolve(self.root_dir)

        return self.root_dir

    def get_str(
        self,
        group: str,
        key: str,
        default: Optional[str] = None,
    ) -> str:
        """Get a string from a selected group and keyword in the analysis settings.

        Args:
            group (str): Name of the group from which to read.
            key (str): Name of the keyword from which to read.
            default (Optional[str]): Optional default value.

        Raises:
            ConfigFileError: If the keyword isn't specified and no default value is given.

        Returns:
            str: value of the keyword as string.

        Examples:
            ```python
            >>> from dfastbe.io import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> result = config_file.get_str("General", "BankDir")
            >>> expected = Path("tests/data/erosion/output/banklines").resolve()
            >>> str(expected) == result
            True

            ```
        """
        try:
            val = self.config[group][key]
        except KeyError as e:
            if default is not None:
                val = default
            else:
                raise ConfigFileError(
                    f"No value specified for required keyword {key} in block {group}."
                ) from e
        return val

    def get_bool(
        self,
        group: str,
        key: str,
        default: Optional[bool] = None,
    ) -> bool:
        """Get a boolean from a selected group and keyword in the analysis settings.

        Args:
            group (str): Name of the group from which to read.
            key (str): Name of the keyword from which to read.
            default (Optional[bool]): Optional default value.

        Raises:
            ConfigFileError: If the keyword isn't specified and no default value is given.

        Returns:
            bool: value of the keyword as boolean.

        Examples:
            ```python
            >>> from dfastbe.io import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> config_file.get_bool("General", "Plotting")
            True

            ```
        """
        try:
            str_val = self.config[group][key].lower()
            val = (
                (str_val == "yes")
                or (str_val == "y")
                or (str_val == "true")
                or (str_val == "t")
                or (str_val == "1")
            )
        except KeyError as e:
            if default is not None:
                val = default
            else:
                raise ConfigFileError(
                    f"No boolean value specified for required keyword {key} in block {group}."
                ) from e

        return val

    def get_float(
        self,
        group: str,
        key: str,
        default: Optional[float] = None,
        positive: bool = False,
    ) -> float:
        """Get a floating point value from a selected group and keyword in the analysis settings.

        Args:
            group (str): Name of the group from which to read.
            key (str): Name of the keyword from which to read.
            default (Optional[float]): Optional default value.
            positive (bool): Flag specifying which floats are accepted.
                All floats are accepted (if False), or only positive floats (if True).

        Raises:
            ConfigFileError: If the keyword isn't specified and no default value is given.
            ConfigFileError: If a negative value is specified when a positive value is required.


        Returns:
            float: value of the keyword as float.

        Examples:
            ```python
            >>> from dfastbe.io import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> config_file.get_float("General", "ZoomStepKM")
            1.0

            ```
        """
        try:
            val = float(self.config[group][key])
        except (KeyError, ValueError) as e:
            if default is not None:
                val = default
            else:
                raise ConfigFileError(
                    f"No floating point value specified for required keyword {key} in block {group}."
                ) from e
        if positive and val < 0.0:
            raise ConfigFileError(
                f"Value for {key} in block {group} must be positive, not {val}."
            )
        return val

    def get_int(
        self,
        group: str,
        key: str,
        default: Optional[int] = None,
        positive: bool = False,
    ) -> int:
        """Get an integer from a selected group and keyword in the analysis settings.

        Args:
            group (str): Name of the group from which to read.
            key (str): Name of the keyword from which to read.
            default (Optional[int]): Optional default value.
            positive (bool): Flag specifying which floats are accepted.
                All floats are accepted (if False), or only positive floats (if True).

        Raises:
            ConfigFileError: If the keyword isn't specified and no default value is given.
            ConfigFileError: If a negative or zero value is specified when a positive value is required.


        Returns:
            int: value of the keyword as int.

        Examples:
            ```python
            >>> from dfastbe.io import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> config_file.get_int("Detect", "NBank")
            2

            ```
        """
        try:
            val = int(self.config[group][key])
        except (KeyError, ValueError) as e:
            if default is not None:
                val = default
            else:
                raise ConfigFileError(
                    f"No integer value specified for required keyword {key} in block {group}."
                ) from e
        if positive and val <= 0:
            raise ConfigFileError(
                f"Value for {key} in block {group} must be positive, not {val}."
            )
        return val

    def get_sim_file(self, group: str, istr: str) -> str:
        """Get the name of the simulation file from the analysis settings.

        Args:
            group (str): Name of the group in which to search for the simulation file name.
            istr (str): Postfix for the simulation file name keyword;
                typically a string representation of the index.

        Returns:
            str: Name of the simulation file (empty string if keywords are not found).

        Examples:
            ```python
            >>> from dfastbe.io import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> result = config_file.get_sim_file("Erosion", "1")
            >>> expected = Path("tests/data/erosion/inputs/sim0075/SDS-j19_map.nc").resolve()
            >>> str(expected) == result
            True

            ```
        """
        sim_file = self.config[group].get(f"SimFile{istr}", "")
        return sim_file

    def get_start_end_stations(self) -> Tuple[float, float]:
        """Get the start and end station for the river.

        Returns:
            Tuple[float, float]: start and end station.

        Examples:
            ```python
            >>> from dfastbe.io import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> config_file.get_start_end_stations()
            (123.0, 128.0)

            ```
        """
        stations = self.get_range("General", "Boundaries")

        return stations

    def get_search_lines(self) -> List[LineString]:
        """Get the search lines for the bank lines from the analysis settings.

        Returns:
            List[np.ndarray]: List of arrays containing the x,y-coordinates of a bank search lines.
        """
        # read guiding bank line
        n_bank = self.get_int("Detect", "NBank")
        line = [None] * n_bank
        for b in range(n_bank):
            bankfile = self.config["Detect"][f"Line{b + 1}"]
            log_text("read_search_line", data={"nr": b + 1, "file": bankfile})
            line[b] = XYCModel.read(bankfile)
        return line

    def read_bank_lines(self, bank_dir: str) -> List[np.ndarray]:
        """Get the bank lines from the detection step.

        Args:
            bank_dir (str): Name of directory in which the bank lines files are located.

        Returns:
            List[np.ndarray]: List of arrays containing the x,y-coordinates of a bank lines.
        """
        bank_name = self.get_str("General", "BankFile", "bankfile")
        bankfile = Path(bank_dir) / f"{bank_name}.shp"
        if bankfile.exists():
            log_text("read_banklines", data={"file": str(bankfile)})
            return gpd.read_file(bankfile)

        bankfile = Path(bank_dir) / f"{bank_name}_#.xyc"
        log_text("read_banklines", data={"file": str(bankfile)})
        bankline_list = []
        b = 1
        while True:
            xyc_file = Path(bank_dir) / f"{bank_name}_{b}.xyc"
            if not xyc_file.exists():
                break

            xy_bank = XYCModel.read(xyc_file)
            bankline_list.append(LineString(xy_bank))
            b += 1
        bankline_series = GeoSeries(bankline_list, crs=self.crs)
        banklines = GeoDataFrame(geometry=bankline_series)
        return banklines

    def get_parameter(
        self,
        group: str,
        key: str,
        bank_km: List[np.ndarray],
        default=None,
        ext: str = "",
        positive: bool = False,
        valid: Optional[List[float]] = None,
        onefile: bool = False,
    ) -> List[np.ndarray]:
        """Get a parameter field from a selected group and keyword in the analysis settings.

        Args:
            group (str): Name of the group from which to read.
            key (str): Name of the keyword from which to read.
            bank_km (List[np.ndarray]): For each bank a listing of the bank points (bank chainage locations).
            default (Optional[Union[float, List[np.ndarray]]]): Optional default value or default parameter field; default None.
            ext (str): File name extension; default empty string.
            positive (bool): Flag specifying which boolean values are accepted.
                All values are accepted (if False), or only strictly positive values (if True); default False.
            valid (Optional[List[float]]): Optional list of valid values; default None.
            onefile (bool): Flag indicating whether parameters are read from one file.
                One file should be used for all bank lines (True) or one file per bank line (False; default).

        Raises:
            Exception:
                If a parameter isn't provided in the configuration, but no default value provided either.
                If the value is negative while a positive value is required (positive = True).
                If the value doesn't match one of the value values (valid is not None).

        Returns:
            List[np.ndarray]: Parameter field
                For each bank a parameter value per bank point (bank chainage location).

        Examples:
            ```python
            >>> from dfastbe.io import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> bank_km = [np.array([0, 1, 2]), np.array([3, 4, 5])]
            >>> config_file.get_parameter("General", "ZoomStepKM", bank_km)
            [array([1., 1., 1.]), array([1., 1., 1.])]

            ```
        """
        try:
            filename = self.config[group][key]
            use_default = False
        except (KeyError, TypeError) as e:
            if default is None:
                raise ConfigFileError(
                    f'No value specified for required keyword "{key}" in block "{group}".'
                ) from e
            use_default = True

        # if val is value then use that value globally
        parfield = [None] * len(bank_km)
        try:
            if use_default:
                if isinstance(default, list):
                    return default
                rval = default
            else:
                rval = float(filename)
                if positive and rval < 0:
                    raise ValueError(
                        f'Value of "{key}" should be positive, not {rval}.'
                    )
                if valid is not None and valid.count(rval) == 0:
                    raise ValueError(
                        f'Value of "{key}" should be in {valid}, not {rval}.'
                    )
            for ib, bkm in enumerate(bank_km):
                parfield[ib] = np.zeros(len(bkm)) + rval
        except (ValueError, TypeError):
            if onefile:
                log_text("read_param", data={"param": key, "file": filename})
                km_thr, val = _get_kmval(filename, key, positive, valid)
            for ib, bkm in enumerate(bank_km):
                if not onefile:
                    filename_i = filename + f"_{ib + 1}" + ext
                    log_text(
                        "read_param_one_bank",
                        data={"param": key, "i": ib + 1, "file": filename_i},
                    )
                    km_thr, val = _get_kmval(filename_i, key, positive, valid)
                if km_thr is None:
                    parfield[ib] = np.zeros(len(bkm)) + val[0]
                else:
                    idx = np.zeros(len(bkm), dtype=int)
                    for thr in km_thr:
                        idx[bkm >= thr] += 1
                    parfield[ib] = val[idx]
                # print("Min/max of data: ", parfield[ib].min(), parfield[ib].max())
        return parfield

    def get_bank_search_distances(self, nbank: int) -> List[float]:
        """Get the search distance per bank line from the analysis settings.

        Args:
            nbank (int): Number of bank search lines.

        Returns:
            List[float]: Array of length nbank containing the search distance value per bank line (default value: 50).

        Examples:
            ```python
            >>> from dfastbe.io import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> config_file.get_bank_search_distances(2)
            [50.0, 50.0]

            ```
        """
        dlines_key = self.config["Detect"].get("DLines", None)
        if dlines_key is None:
            dlines = [50] * nbank
        elif dlines_key[0] == "[" and dlines_key[-1] == "]":
            dlines_split = dlines_key[1:-1].split(",")
            dlines = [float(d) for d in dlines_split]
            if not all([d > 0 for d in dlines]):
                raise ValueError(
                    "keyword DLINES should contain positive values in configuration file."
                )
            if len(dlines) != nbank:
                raise ConfigFileError(
                    "keyword DLINES should contain NBANK values in configuration file."
                )
        return dlines

    def get_range(self, group: str, key: str) -> Tuple[float, float]:
        """Get a start and end value from a selected group and keyword in the analysis settings.

        Args:
            group (str): Name of the group from which to read.
            key (str): Name of the keyword from which to read.

        Returns:
            Tuple[float,float]: Lower and upper limit of the range.

        Examples:
            ```python
            >>> from dfastbe.io import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> config_file.get_range("General", "Boundaries")
            (123.0, 128.0)

            ```
        """
        str_val = self.get_str(group, key)
        try:
            obrack = str_val.find("[")
            cbrack = str_val.find("]")
            if obrack >= 0 and cbrack >= 0:
                str_val = str_val[obrack + 1 : cbrack - 1]
            val_list = [float(fstr) for fstr in str_val.split(":")]
            if val_list[0] > val_list[1]:
                val = (val_list[1], val_list[0])
            else:
                val = (val_list[0], val_list[1])
        except ValueError as e:
            raise ValueError(
                f'Invalid range specification "{str_val}" for required keyword "{key}" in block "{group}".'
            ) from e
        return val

    def get_river_center_line(self) -> LineString:
        """Get the river center line from the xyc file as a linestring g.

        Returns:
            LineString: Chainage line.
        """
        # get the chainage file
        river_center_line_file = self.get_str("General", "RiverKM")
        log_text("read_chainage", data={"file": river_center_line_file})
        river_center_line = XYCModel.read(river_center_line_file, num_columns=3)

        # make sure that chainage is increasing with node index
        if river_center_line.coords[0][2] > river_center_line.coords[1][2]:
            river_center_line = LineString(river_center_line.coords[::-1])

        return river_center_line

    def resolve(self, rootdir: str):
        """Convert a configuration object to contain absolute paths (for editing).

        Args:
            rootdir (str): The path to be used as base for the absolute paths.

        Examples:
            ```python
            >>> from dfastbe.io import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> config_file.resolve("tests/data/erosion")

            ```
        """
        if "General" in self.config:
            self.resolve_parameter("General", "RiverKM", rootdir)
            self.resolve_parameter("General", "BankDir", rootdir)
            self.resolve_parameter("General", "FigureDir", rootdir)

        if "Detect" in self.config:
            self.resolve_parameter("Detect", "SimFile", rootdir)
            i = 0
            while True:
                i = i + 1
                line_i = "Line" + str(i)
                if line_i in self.config["Detect"]:
                    self.resolve_parameter("Detect", line_i, rootdir)
                else:
                    break

        if "Erosion" in self.config:
            self.resolve_parameter("Erosion", "RiverAxis", rootdir)
            self.resolve_parameter("Erosion", "Fairway", rootdir)
            self.resolve_parameter("Erosion", "OutputDir", rootdir)

            self.resolve_parameter("Erosion", "ShipType", rootdir)
            self.resolve_parameter("Erosion", "VShip", rootdir)
            self.resolve_parameter("Erosion", "NShip", rootdir)
            self.resolve_parameter("Erosion", "NWave", rootdir)
            self.resolve_parameter("Erosion", "Draught", rootdir)
            self.resolve_parameter("Erosion", "Wave0", rootdir)
            self.resolve_parameter("Erosion", "Wave1", rootdir)

            self.resolve_parameter("Erosion", "BankType", rootdir)
            self.resolve_parameter("Erosion", "ProtectionLevel", rootdir)
            self.resolve_parameter("Erosion", "Slope", rootdir)
            self.resolve_parameter("Erosion", "Reed", rootdir)

            n_level = self.get_int("Erosion", "NLevel", default=0)
            for i in range(1, n_level + 1):
                self.resolve_parameter("Erosion", f"SimFile{i}", rootdir)
                self.resolve_parameter("Erosion", f"ShipType{i}", rootdir)
                self.resolve_parameter("Erosion", f"VShip{i}", rootdir)
                self.resolve_parameter("Erosion", f"NShip{i}", rootdir)
                self.resolve_parameter("Erosion", f"NWave{i}", rootdir)
                self.resolve_parameter("Erosion", f"Draught{i}", rootdir)
                self.resolve_parameter("Erosion", f"Slope{i}", rootdir)
                self.resolve_parameter("Erosion", f"Reed{i}", rootdir)

    def relative_to(self, rootdir: str) -> None:
        """Convert a configuration object to contain relative paths (for saving).

        Args:
            rootdir (str): The path to be used as base for the relative paths.

        Examples:
            ```python
            >>> from dfastbe.io import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> config_file.relative_to("testing/data/erosion")

            ```
        """
        if "General" in self.config:
            self.parameter_relative_to("General", "RiverKM", rootdir)
            self.parameter_relative_to("General", "BankDir", rootdir)
            self.parameter_relative_to("General", "FigureDir", rootdir)

        if "Detect" in self.config:
            self.parameter_relative_to("Detect", "SimFile", rootdir)

            i = 0
            while True:
                i = i + 1
                line_i = f"Line{i}"
                if line_i in self.config["Detect"]:
                    self.parameter_relative_to("Detect", line_i, rootdir)
                else:
                    break

        if "Erosion" in self.config:
            self.parameter_relative_to("Erosion", "RiverAxis", rootdir)
            self.parameter_relative_to("Erosion", "Fairway", rootdir)
            self.parameter_relative_to("Erosion", "OutputDir", rootdir)

            self.parameter_relative_to("Erosion", "ShipType", rootdir)
            self.parameter_relative_to("Erosion", "VShip", rootdir)
            self.parameter_relative_to("Erosion", "NShip", rootdir)
            self.parameter_relative_to("Erosion", "NWave", rootdir)
            self.parameter_relative_to("Erosion", "Draught", rootdir)
            self.parameter_relative_to("Erosion", "Wave0", rootdir)
            self.parameter_relative_to("Erosion", "Wave1", rootdir)

            self.parameter_relative_to("Erosion", "BankType", rootdir)
            self.parameter_relative_to("Erosion", "ProtectionLevel", rootdir)
            self.parameter_relative_to("Erosion", "Slope", rootdir)
            self.parameter_relative_to("Erosion", "Reed", rootdir)

            n_level = self.get_int("Erosion", "NLevel", default=0)
            for i in range(1, n_level + 1):
                self.parameter_relative_to("Erosion", f"SimFile{i}", rootdir)
                self.parameter_relative_to("Erosion", f"ShipType{i}", rootdir)
                self.parameter_relative_to("Erosion", f"VShip{i}", rootdir)
                self.parameter_relative_to("Erosion", f"NShip{i}", rootdir)
                self.parameter_relative_to("Erosion", f"NWave{i}", rootdir)
                self.parameter_relative_to("Erosion", f"Draught{i}", rootdir)
                self.parameter_relative_to("Erosion", f"Slope{i}", rootdir)
                self.parameter_relative_to("Erosion", f"Reed{i}", rootdir)

    def resolve_parameter(self, group: str, key: str, rootdir: str):
        """Convert a parameter value to contain an absolute path.

        Determine whether the string represents a number.
        If not, try to convert to an absolute path.

        Args:
            group (str): Name of the group in the configuration.
            key (str): Name of the key in the configuration.
            rootdir (str): The path to be used as base for the absolute paths.

        Examples:
            ```python
            >>> from dfastbe.io import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> config_file.resolve_parameter("General", "RiverKM", "tests/data/erosion")

            ```
        """
        if key in self.config[group]:
            val_str = self.config[group][key]
            try:
                float(val_str)
            except ValueError:
                self.config[group][key] = absolute_path(rootdir, val_str)

    def parameter_relative_to(self, group: str, key: str, rootdir: str):
        """Convert a parameter value to contain a relative path.

        Determine whether the string represents a number.
        If not, try to convert to a relative path.

        Args:
            group (str): Name of the group in the configuration.
            key (str): Name of the key in the configuration.
            rootdir (str): The path to be used as base for the relative paths.

        Examples:
            ```python
            >>> from dfastbe.io import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
            >>> config_file.parameter_relative_to("General", "RiverKM", "tests/data/erosion")

            ```
        """
        if key in self.config[group]:
            val_str = self.config[group][key]

            try:
                float(val_str)
            except ValueError:
                self.config[group][key] = relative_path(rootdir, val_str)

    def get_plotting_flags(self, root_dir: str) -> Dict[str, bool]:
        """Get the plotting flags from the configuration file.

        Returns:
            data (Dict[str, bool]):
                Dictionary containing the plotting flags.
                save_plot (bool): Flag indicating whether to save the plot.
                save_plot_zoomed (bool): Flag indicating whether to save the zoomed plot.
                zoom_km_step (float): Step size for zooming in on the plot.
                close_plot (bool): Flag indicating whether to close the plot.
        """
        plot_data = self.get_bool("General", "Plotting", True)

        if plot_data:
            save_plot = self.get_bool("General", "SavePlots", True)
            save_plot_zoomed = self.get_bool("General", "SaveZoomPlots", True)
            zoom_km_step = self.get_float("General", "ZoomStepKM", 1.0)
            if zoom_km_step < 0.01:
                save_plot_zoomed = False
            close_plot = self.get_bool("General", "ClosePlots", False)
        else:
            save_plot = False
            save_plot_zoomed = False
            close_plot = False

        data = {
            "plot_data": plot_data,
            "save_plot": save_plot,
            "save_plot_zoomed": save_plot_zoomed,
            "zoom_km_step": zoom_km_step,
            "close_plot": close_plot,
        }

        # as appropriate, check output dir for figures and file format
        if save_plot:
            fig_dir = self.get_str("General", "FigureDir", Path(root_dir) / "figure")
            log_text("figure_dir", data={"dir": fig_dir})
            path_fig_dir = Path(fig_dir)
            if path_fig_dir.exists():
                log_text("overwrite_dir", data={"dir": fig_dir})
            path_fig_dir.mkdir(parents=True, exist_ok=True)
            plot_ext = self.get_str("General", "FigureExt", ".png")
            data = data | {
                "fig_dir": fig_dir,
                "plot_ext": plot_ext,
            }

        return data

    def get_output_dir(self, option: str) -> Path:
        """Get the output directory for the analysis.

        Args:
            option (str):
                Option for which to get the output directory. "banklines" for bank lines, else the erosion output
                directory will be returned.
        Returns:
            output_dir (Path):
                Path to the output directory.
        """
        if option == "banklines":
            output_dir = self.get_str("General", "BankDir")
        else:
            output_dir = self.get_str("Erosion", "OutputDir")

        output_dir = Path(output_dir)
        log_text(f"{option}_out", data={"dir": output_dir})
        if output_dir.exists():
            log_text("overwrite_dir", data={"dir": output_dir})
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir

class CenterLine:
    """Center line class."""

    def __init__(self, line_string: LineString, mask: Tuple[float, float] = None):
        """Center Line initialization.

        Args:
            config_file : ConfigFile
                Configuration file with settings for the analysis.

        Examples:
            ```python
            >>> from dfastbe.io import ConfigFile, CenterLine
            >>> config_file = ConfigFile("tests/data/erosion/meuse_manual.cfg")
            >>> center_line = CenterLine(config_file)
            ```
        """
        self.station_bounds = mask
        if mask is None:
            self.values = line_string
        else:
            self.values: LineString = self.mask(line_string, mask)

        log_text(
            "clip_chainage", data={"low": self.station_bounds[0], "high": self.station_bounds[1]}
        )

    def as_array(self):
        # self.masked_profile_arr =
        return np.array(self.values.coords)

    @staticmethod
    def mask(line_string, bounds: Tuple[float, float]) -> LineString:
        """
        Clip a chainage line to the relevant reach.

        Args:
            line_string (LineString):
                river center line as a linestring.
            bounds (Tuple[float, float]):
                Lower and upper limit for the chainage.

        Returns:
            LineString: Clipped river chainage line.
        """
        start_i = None
        end_i = None
        for i, c in enumerate(line_string.coords):
            if start_i is None and c[2] >= bounds[0]:
                start_i = i
            if c[2] >= bounds[1]:
                end_i = i
                break

        if start_i is None:
            raise Exception(
                "Lower chainage bound {} is larger than the maximum chainage {} available".format(
                    bounds[0], line_string.coords[-1][2]
                )
            )
        elif start_i == 0:
            # lower bound (potentially) clipped to available reach
            if line_string.coords[0][2] - bounds[0] > 0.1:
                raise Exception(
                    "Lower chainage bound {} is smaller than the minimum chainage {} available".format(
                        bounds[0], line_string.coords[0][2]
                    )
                )
            x0 = None
        else:
            alpha = (bounds[0] - line_string.coords[start_i - 1][2]) / (
                    line_string.coords[start_i][2] - line_string.coords[start_i - 1][2]
            )
            x0 = tuple(
                (c1 + alpha * (c2 - c1))
                for c1, c2 in zip(line_string.coords[start_i - 1], line_string.coords[start_i])
            )
            if alpha > 0.9:
                # value close to the first node (start_i), so let's skip that one
                start_i = start_i + 1

        if end_i is None:
            if bounds[1] - line_string.coords[-1][2] > 0.1:
                raise Exception(
                    "Upper chainage bound {} is larger than the maximum chainage {} available".format(
                        bounds[1], line_string.coords[-1][2]
                    )
                )
            # else kmbounds[1] matches chainage of last point
            if x0 is None:
                # whole range available selected
                pass
            else:
                line_string = LineString([x0] + line_string.coords[start_i:])
        elif end_i == 0:
            raise Exception(
                "Upper chainage bound {} is smaller than the minimum chainage {} available".format(
                    bounds[1], line_string.coords[0][2]
                )
            )
        else:
            alpha = (bounds[1] - line_string.coords[end_i - 1][2]) / (
                    line_string.coords[end_i][2] - line_string.coords[end_i - 1][2]
            )
            x1 = tuple(
                (c1 + alpha * (c2 - c1))
                for c1, c2 in zip(line_string.coords[end_i - 1], line_string.coords[end_i])
            )
            if alpha < 0.1:
                # value close to the previous point (end_i - 1), so let's skip that one
                end_i = end_i - 1
            if x0 is None:
                line_string = LineString(line_string.coords[:end_i] + [x1])
            else:
                line_string = LineString([x0] + line_string.coords[start_i:end_i] + [x1])
        return line_string

class SearchLines:

    def __init__(self, lines: List[LineString], mask: CenterLine = None):
        """Search lines initialization.

        Args:
            lines (List[LineString]):
                List of search lines.
        """
        if mask is None:
            self.values = lines
            self.max_distance = None
        else:
            self.values, self.max_distance = self.mask(lines, mask.values)

        self.size = len(lines)

    @staticmethod
    def mask(
        search_lines: List[LineString],
        river_center_line: LineString,
        max_river_width: float = MAX_RIVER_WIDTH,
    ) -> Tuple[List[LineString], float]:
        """
        Clip the list of lines to the envelope of a certain size surrounding a reference line.

        Arg:
            max_river_width: float
                Maximum distance away from river_profile.

        Returns:
            List[LineString]: List of clipped search lines.
            float: Maximum distance from any point within line to reference line.
        """
        num = len(search_lines)
        profile_buffer = river_center_line.buffer(max_river_width, cap_style=2)

        # The algorithm uses simplified geometries for determining the distance between lines for speed.
        # Stay accurate to within about 1 m
        profile_simplified = river_center_line.simplify(1)

        max_distance = 0
        for ind in range(num):
            # Clip the bank search lines to the reach of interest (indicated by the reference line).
            search_lines[ind] = search_lines[ind].intersection(profile_buffer)

            # If the bank search line breaks into multiple parts, select the part closest to the reference line.
            if search_lines[ind].geom_type == "MultiLineString":
                distance_min = max_river_width
                i_min = 0
                for i in range(len(search_lines[ind])):
                    line_simplified = search_lines[ind][i].simplify(1)
                    distance_min_i = line_simplified.distance(profile_simplified)
                    if distance_min_i < distance_min:
                        distance_min = distance_min_i
                        i_min = i
                search_lines[ind] = search_lines[ind][i_min]

            # Determine the maximum distance from a point on this line to the reference line.
            line_simplified = search_lines[ind].simplify(1)
            max_distance = max(
                [Point(c).distance(profile_simplified) for c in line_simplified.coords]
            )

            # Increase the value of max_distance by 2 to account for error introduced by using simplified lines.
            max_distance = max(max_distance, max_distance + 2)

        return search_lines, max_distance

class RiverData:
    """River data class."""

    def __init__(self, config_file: ConfigFile):
        """River Data initialization.

        Args:
            config_file : ConfigFile
                Configuration file with settings for the analysis.

        Examples:
            ```python
            >>> from dfastbe.io import ConfigFile, RiverData
            >>> config_file = ConfigFile("tests/data/erosion/meuse_manual.cfg")
            >>> river_data = RiverData(config_file)
            ```
        """
        self.config_file = config_file
        center_line = config_file.get_river_center_line()
        bounds = config_file.get_start_end_stations()
        self.river_center_line: CenterLine = CenterLine(center_line, bounds)
        self.station_bounds: Tuple = config_file.get_start_end_stations()

    @property
    def search_lines(self) -> SearchLines:
        return SearchLines(self.config_file.get_search_lines(), self.river_center_line)

    def read_river_axis(self):
        """Get the river axis from the analysis settings."""
        river_axis_file = self.config_file.get_str("Erosion", "RiverAxis")
        log_text("read_river_axis", data={"file": river_axis_file})
        river_axis = XYCModel.read(river_axis_file)
        return river_axis

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


def get_bbox(
        coords: np.ndarray, buffer: float = 0.1
) -> Tuple[float, float, float, float]:
    """
    Derive the bounding box from a line.

    Args:
        coords (np.ndarray):
            An N x M array containing x- and y-coordinates as first two M entries
        buffer : float
            Buffer fraction surrounding the tight bounding box

    Returns:
        bbox (Tuple[float, float, float, float]):
            Tuple bounding box consisting of [min x, min y, max x, max y)
    """
    x = coords[:, 0]
    y = coords[:, 1]
    x_min = x.min()
    y_min = y.min()
    x_max = x.max()
    y_max = y.max()
    d = buffer * max(x_max - x_min, y_max - y_min)
    bbox = (x_min - d, y_min - d, x_max + d, y_max + d)

    return bbox

def load_program_texts(file_name: Union[str, Path]) -> None:
    """Load texts from a configuration file, and store globally for access.

    This routine reads the text file "file_name", and detects the keywords
    indicated by lines starting with [ and ending with ]. The content is
    placed in a global dictionary PROGTEXTS which may be queried using the
    routine "get_text". These routines are used to implement multi-language support.

    Arguments
    ---------
    file_name : str
        The name of the file to be read and parsed.
    """
    global PROGTEXTS

    all_lines = open(file_name, "r").read().splitlines()
    data: Dict[str, List[str]] = {}
    text: List[str] = []
    key = None
    for line in all_lines:
        r_line = line.strip()
        if r_line.startswith("[") and r_line.endswith("]"):
            if key is not None:
                data[key] = text
            key = r_line[1:-1]
            text = []
        else:
            text.append(line)
    if key in data.keys():
        raise ValueError(f"Duplicate entry for {key} in {file_name}.")

    if key is not None:
        data[key] = text

    PROGTEXTS = data


def log_text(
    key: str,
    file: Optional[TextIO] = None,
    data: Dict[str, Any] = {},
    repeat: int = 1,
    indent: str = "",
) -> None:
    """
    Write a text to standard out or file.

    Arguments
    ---------
    key : str
        The key for the text to show to the user.
    file : Optional[TextIO]
        The file to write to (None for writing to standard out).
    data : Dict[str, Any]
        A dictionary used for placeholder expansions (default empty).
    repeat : int
        The number of times that the same text should be repeated (default 1).
    indent : str
        String to use for each line as indentation (default empty).

    Returns
    -------
    None
    """
    str_value = get_text(key)
    for _ in range(repeat):
        for s in str_value:
            sexp = s.format(**data)
            if file is None:
                print(indent + sexp)
            else:
                file.write(indent + sexp + "\n")


def get_filename(key: str) -> str:
    """
    Query the global dictionary of texts for a file name.

    The file name entries in the global dictionary have a prefix "filename_"
    which will be added to the key by this routine.

    Arguments
    ---------
    key : str
        The key string used to query the dictionary.

    Results
    -------
    filename : str
        File name.
    """
    filename = get_text("filename_" + key)[0]
    return filename


def get_text(key: str) -> List[str]:
    """
    Query the global dictionary of texts via a string key.

    Query the global dictionary PROGTEXTS by means of a string key and return
    the list of strings contained in the dictionary. If the dictionary doesn't
    include the key, a default string is returned.

    Parameters
    ----------
    key : str
        The key string used to query the dictionary.

    Returns
    -------
    text : List[str]
        The list of strings returned contain the text stored in the dictionary
        for the key. If the key isn't available in the dictionary, the routine
        returns the default string "No message found for <key>"
    """

    global PROGTEXTS

    try:
        str_value = PROGTEXTS[key]
    except:
        str_value = ["No message found for " + key]
    return str_value


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
    rootgrp = netCDF4.Dataset(filename)

    # locate 2d mesh variable
    mesh2d = rootgrp.get_variables_by_attributes(
        cf_role="mesh_topology", topology_dimension=2
    )
    if len(mesh2d) != 1:
        raise Exception(
            "Currently only one 2D mesh supported ... this file contains {} 2D meshes.".format(
                len(mesh2d)
            )
        )
    meshname = mesh2d[0].name

    # define a default start_index
    start_index = 0

    # locate the requested variable ... start with some special cases
    if varname == "x":
        # the x-coordinate or longitude
        crdnames = mesh2d[0].getncattr(location + "_coordinates").split()
        for n in crdnames:
            stdname = rootgrp.variables[n].standard_name
            if stdname == "projection_x_coordinate" or stdname == "longitude":
                var = rootgrp.variables[n]
                break

    elif varname == "y":
        # the y-coordinate or latitude
        crdnames = mesh2d[0].getncattr(location + "_coordinates").split()
        for n in crdnames:
            stdname = rootgrp.variables[n].standard_name
            if stdname == "projection_y_coordinate" or stdname == "latitude":
                var = rootgrp.variables[n]
                break

    elif varname.endswith("connectivity"):
        # a mesh connectivity variable with corrected index
        varname = mesh2d[0].getncattr(varname)
        var = rootgrp.variables[varname]
        if "start_index" in var.ncattrs():
            start_index = var.getncattr("start_index")

    else:
        # find any other variable by standard_name or long_name
        var = rootgrp.get_variables_by_attributes(
            standard_name=varname, mesh=meshname, location=location
        )
        if len(var) == 0:
            var = rootgrp.get_variables_by_attributes(
                long_name=varname, mesh=meshname, location=location
            )
        if len(var) != 1:
            raise Exception(
                'Expected one variable for "{}", but obtained {}.'.format(
                    varname, len(var)
                )
            )
        var = var[0]

    # read data checking for time dimension
    if var.get_dims()[0].isunlimited():
        # assume that time dimension is unlimited and is the first dimension
        # slice to obtain last time step
        data = var[-1, :]
    else:
        data = var[...] - start_index

    rootgrp.close()

    return data


def absolute_path(rootdir: str, path: str) -> str:
    """
    Convert a relative path to an absolute path.

    Args:
        rootdir (str): Any relative paths should be given relative to this location.
        path (str): A relative or absolute location.

    Returns:
        str: An absolute location.
    """
    if not path:
        return path
    root_path = Path(rootdir).resolve()
    target_path = Path(path)

    if target_path.is_absolute():
        return str(target_path)

    resolved_path = (root_path / target_path).resolve()
    return str(resolved_path)


def relative_path(rootdir: str, file: str) -> str:
    """
    Convert an absolute path to a relative path.

    Args:
        rootdir (str): Any relative paths will be given relative to this location.
        file (str): An absolute location.

    Returns:
        str: A relative location if possible, otherwise the absolute location.
    """
    if not file:
        return file

    root_path = Path(rootdir).resolve()
    file_path = Path(file).resolve()

    try:
        return str(file_path.relative_to(root_path))
    except ValueError:
        return str(file_path)


def write_shp_pnt(
    xy: np.ndarray, data: Dict[str, np.ndarray], filename: str, config_file: ConfigFile
) -> None:
    """
    Write a shape point file with x, y, and values.

    Arguments
    ---------
    xy : np.ndarray
        N x 2 array containing x and y coordinates.
    data : Dict[str, np.ndarray]
        Dictionary of quantities to be written, each np array should have length k.
    filename : str
        Name of the file to be written.

    Returns
    -------
    None
    """
    xy_points = [Point(xy1) for xy1 in xy]
    geom = GeoSeries(xy_points, crs=config_file.crs)
    write_shp(geom, data, filename)


def write_shp(geom: GeoSeries, data: Dict[str, np.ndarray], filename: str) -> None:
    """Write a shape file.

    Write a shape file for a given GeoSeries and dictionary of np arrays.
    The GeoSeries and all np should have equal length.

    Arguments
    ---------
    geom : geopandas.geoseries.GeoSeries
        geopandas GeoSeries containing k geometries.
    data : Dict[str, np.ndarray]
        Dictionary of quantities to be written, each np array should have length k.
    filename : str
        Name of the file to be written.

    Returns
    -------
    None
    """
    df = pd.DataFrame(data)
    GeoDataFrame(df, geometry=geom).to_file(filename)


def write_csv(data: Dict[str, np.ndarray], filename: str) -> None:
    """
    Write a data to csv file.

    Arguments
    ---------
    data : Dict[str, np.ndarray]
        Value(s) to be written.
    filename : str
        Name of the file to be written.

    Returns
    -------
    None
    """
    keys = [key for key in data.keys()]
    header = ""
    for i in range(len(keys)):
        if i < len(keys) - 1:
            header = header + '"' + keys[i] + '", '
        else:
            header = header + '"' + keys[i] + '"'

    data = np.column_stack([array for array in data.values()])
    np.savetxt(filename, data, delimiter=", ", header=header, comments="")


def write_km_eroded_volumes(km: np.ndarray, vol: np.ndarray, filename: str) -> None:
    """
    Write a text file with eroded volume data binned per kilometre.

    Arguments
    ---------
    km :
        Array containing chainage values.
    vol :
        Array containing erosion volume values.
    filename : str
        Name of the file to be written.

    Returns
    -------
    None
    """
    with open(filename, "w") as erofile:
        for i in range(len(km)):
            valstr = "\t".join(["{:.2f}".format(x) for x in vol[i, :]])
            erofile.write("{:.2f}\t".format(km[i]) + valstr + "\n")


def _move_parameter_location(
    config: ConfigParser,
    group1: str,
    key1: str,
    group2: str,
    key2: Optional[str] = None,
    convert: Optional[Callable[[str], str]] = None,
) -> ConfigParser:
    """
    Move a parameter from one group/keyword to another.

    Args:
        config : ConfigParser
            Original settings for the D-FAST Bank Erosion analysis.
        group1 : str
            Name of the group in the original configuration.
        key1 : str
            Name of the keyword in the original configuration.
        group2 : str
            Name of the group in the target configuration.
        key2 : Optional[str]
            Name of the keyword in the target configuration (can be None if equal to the keyword in the original file).
        convert: Optional[Callable[[str], str]]
            Function to convert the original value into new value.

    Returns:
        config : ConfigParser
            Updated settings for the D-FAST Bank Erosion analysis.
    """
    val2: str
    if group1 in config.sections() and key1 in config[group1]:
        if key2 is None:
            key2 = key1
        val1 = config[group1][key1]
        if convert is None:
            val2 = val1
        else:
            val2 = convert(val1)
        config[group2][key2] = val2
        config[group1].pop(key1)
    return config


def _sim2nc(oldfile: str) -> str:
    """
    Convert an SDS file name to an NC file (mirrors sim2ugrid.m).

    Args:
        oldfile (str):Name of the original SIMONA SDS or Delft3D-FLOW TRIM file.

    Returns:
        str: Name of the netCDF file as created by sim2ugrid.m.
    """
    name = Path(oldfile).name
    if name.startswith("SDS"):
        # SDS-case_map.nc
        nc_file = f"{oldfile}_map.nc"
    elif name.startswith("trim"):
        # trim-case_map.nc
        nc_file = f"{Path(oldfile).stem}_map.nc"
    else:
        raise SimulationFilesError(f'Unable to determine file type for "{oldfile}"')
    return nc_file


def _get_kmval(filename: str, key: str, positive: bool, valid: Optional[List[float]]):
    """
    Read a parameter file, check its contents and return arrays of chainages and values.

    Arguments
    ---------
    filename : str
        Name of the parameter file to be read.
    key : str
        Name of the quantity that we're reading.
    positive : bool
        Flag specifying whether all values are accepted (if False), or only positive values (if True).
    valid : Optional[List[float]]
        Optional list of valid values.

    Raises
    ------
    Exception
        If negative values are read while values are required to be positive (positive = True).
        If some values are not valid when cross validated against the valid list (valid is not None).
        If the chainage values in the file are not strictly increasing.

    Returns
    -------
    km_thr : Optional[np.ndarray]
        Array containing the chainage of the midpoints between the values.
    val : np.ndarray
        Array containing the values.
    """
    # print("Trying to read: ",filename)
    points = pd.read_csv(
        filename,
        names=["Chainage", "Val"],
        skipinitialspace=True,
        delim_whitespace=True,
    )
    # nPnts = len(P.Chainage)
    km = points.Chainage.to_numpy()
    val = points.Val.to_numpy()

    if len(km.shape) == 0:
        km = km[None]
        val = val[None]

    if positive and (val < 0).any():
        raise ValueError(
            f'Values of "{key}" in {filename} should be positive. Negative value read for chainage(s): {km[val < 0]}'
        )

    if len(km) == 1:
        km_thr = None
    else:
        if not (km[1:] > km[:-1]).all():
            raise ValueError(
                f"Chainage values are not increasing in the file {filename} read for {key}."
            )
        km_thr = km[1:]

    return km_thr, val


class ConfigFileError(Exception):
    """Custom exception for configuration file errors."""

    pass


class SimulationFilesError(Exception):
    """Custom exception for configuration file errors."""

    pass
