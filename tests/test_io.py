import os
import platform
import sys
from configparser import ConfigParser
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Dict
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from geopandas import GeoDataFrame
from pyfakefs.fake_filesystem import FakeFilesystem
from shapely.geometry import LineString

from dfastbe.io import (
    SimulationData,
    SimulationFilesError,
    ConfigFile,
    RiverData,
    absolute_path,
    get_filename,
    get_text,
    load_program_texts,
    log_text,
    _read_fm_map,
    relative_path,
)
from dfastbe.erosion.structures import MeshData


@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def test_load_program_texts_01():
    """
    Testing load_program_texts.
    """
    print("current work directory: ", os.getcwd())
    assert load_program_texts("tests/files/messages.UK.ini") is None


class TestSimulationData:

    def test_read(self):
        file_name = "test_map.nc"
        mock_x_node = np.array([0.0, 1.0, 2.0])
        mock_y_node = np.array([0.0, 1.0, 2.0])
        mock_face_node = MagicMock()
        mock_face_node.data = np.array([[0, 1, 2], [2, 3, 4]])
        mock_face_node.mask = np.array([[False, False, False], [False, False, False]])
        mock_bed_level_values = np.array([10.0, 20.0, 30.0])
        mock_water_level_face = np.array([1.0, 2.0, 3.0])
        mock_water_depth_face = np.array([0.5, 1.0, 1.5])
        mock_velocity_x_face = np.array([0.1, 0.2, 0.3])
        mock_velocity_y_face = np.array([0.4, 0.5, 0.6])
        mock_chezy_face = np.array([30.0, 40.0, 50.0])

        with patch("dfastbe.io._read_fm_map") as mock_read_fm_map, patch(
            "netCDF4.Dataset"
        ) as mock_dataset:
            mock_read_fm_map.side_effect = [
                mock_x_node,
                mock_y_node,
                mock_face_node,
                mock_bed_level_values,
                mock_water_level_face,
                mock_water_depth_face,
                mock_velocity_x_face,
                mock_velocity_y_face,
                mock_chezy_face,
            ]

            mock_root_group = MagicMock()
            mock_root_group.converted_from = "SIMONA"
            mock_dataset.return_value = mock_root_group

            sim_object = SimulationData.read(file_name)

            assert isinstance(sim_object, SimulationData)
            assert np.array_equal(sim_object.x_node, mock_x_node)
            assert np.array_equal(sim_object.y_node, mock_y_node)
            assert np.array_equal(sim_object.face_node.data, mock_face_node.data)
            assert np.array_equal(
                sim_object.bed_elevation_values, mock_bed_level_values
            )
            assert np.array_equal(sim_object.water_level_face, mock_water_level_face)
            assert np.array_equal(sim_object.water_depth_face, mock_water_depth_face)
            assert np.array_equal(sim_object.velocity_x_face, mock_velocity_x_face)
            assert np.array_equal(sim_object.velocity_y_face, mock_velocity_y_face)
            assert np.array_equal(sim_object.chezy_face, mock_chezy_face)
            assert sim_object.dry_wet_threshold == 0.1

            mock_read_fm_map.assert_any_call(file_name, "x", location="node")
            mock_read_fm_map.assert_any_call(file_name, "y", location="node")
            mock_read_fm_map.assert_any_call(file_name, "face_node_connectivity")
            mock_read_fm_map.assert_any_call(file_name, "altitude", location="node")
            mock_read_fm_map.assert_any_call(file_name, "Water level")
            mock_read_fm_map.assert_any_call(
                file_name, "sea_floor_depth_below_sea_surface"
            )
            mock_read_fm_map.assert_any_call(file_name, "sea_water_x_velocity")
            mock_read_fm_map.assert_any_call(file_name, "sea_water_y_velocity")
            mock_read_fm_map.assert_any_call(file_name, "Chezy roughness")

    def test_read_invalid_file(self):
        invalid_file_name = "invalid_file.nc"

        with pytest.raises(SimulationFilesError):
            SimulationData.read(invalid_file_name)

    @pytest.fixture
    def simulation_data(self) -> SimulationData:
        x_node = np.array([194949.796875, 194966.515625, 194982.8125, 195000.0])
        y_node = np.array([361366.90625, 361399.46875, 361431.03125, 361450.0])
        n_nodes = np.array([4, 4])
        face_node = np.ma.masked_array(
            data=[[0, 1, 2, 3], [1, 2, 3, 0]],
            mask=[[False, False, False, False], [False, False, False, False]],
        )
        bed_elevation_location = "node"
        bed_elevation_values = np.array([10.0, 20.0, 30.0, 40.0])
        water_level_face = np.array([1.0, 2.0])
        water_depth_face = np.array([0.5, 1.0])
        velocity_x_face = np.array([0.1, 0.2])
        velocity_y_face = np.array([0.4, 0.5])
        chezy_face = np.array([30.0, 40.0])
        dry_wet_threshold = 0.1

        sim_data = SimulationData(
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
        return sim_data

    def test_clip(self, simulation_data: SimulationData):
        river_profile = LineString(
            [
                [194949.796875, 361366.90625],
                [194966.515625, 361399.46875],
                [194982.8125, 361431.03125],
            ]
        )
        max_distance = 10.0
        simulation_data.clip(river_profile, max_distance)

        assert np.array_equal(
            simulation_data.x_node,
            np.array([194949.796875, 194966.515625, 194982.8125]),
        )
        assert np.array_equal(
            simulation_data.y_node, np.array([361366.90625, 361399.46875, 361431.03125])
        )
        assert np.array_equal(
            simulation_data.bed_elevation_values, np.array([10.0, 20.0, 30.0])
        )
        assert simulation_data.n_nodes.size == 0
        assert simulation_data.water_level_face.size == 0
        assert simulation_data.water_depth_face.size == 0
        assert simulation_data.velocity_x_face.size == 0
        assert simulation_data.velocity_y_face.size == 0
        assert simulation_data.chezy_face.size == 0

    def test_clip_no_nodes_in_buffer(self, simulation_data: SimulationData):
        river_profile = LineString(
            [
                [194900.0, 361300.0],
                [194910.0, 361310.0],
                [194920.0, 361320.0],
            ]
        )
        max_distance = 10.0

        simulation_data.clip(river_profile, max_distance)

        assert simulation_data.x_node.size == 0
        assert simulation_data.y_node.size == 0
        assert simulation_data.bed_elevation_values.size == 0
        assert simulation_data.n_nodes.size == 0
        assert simulation_data.water_level_face.size == 0
        assert simulation_data.water_depth_face.size == 0
        assert simulation_data.velocity_x_face.size == 0
        assert simulation_data.velocity_y_face.size == 0
        assert simulation_data.chezy_face.size == 0

    def test_compute_mesh_topology(self, simulation_data: SimulationData):
        """
        Test the compute_mesh_topology method of SimulationData.
        """
        # Call the method to compute the mesh topology
        mesh_data = simulation_data.compute_mesh_topology()

        assert isinstance(mesh_data, MeshData)

        assert np.array_equal(
            mesh_data.edge_face_connectivity, np.array([[0, 1], [0, 1], [0, 1], [0, 1]])
        )
        assert np.array_equal(
            mesh_data.face_edge_connectivity, np.array([[1, 0, 2, 3], [0, 2, 3, 1]])
        )
        assert np.allclose(
            mesh_data.x_edge_coords,
            np.array(
                [
                    [194949.796875, 194966.515625],
                    [194949.796875, 195000.0],
                    [194966.515625, 194982.8125],
                    [194982.8125, 195000.0],
                ]
            ),
        )
        assert np.allclose(
            mesh_data.x_face_coords.data,
            np.array(
                [
                    [194949.796875, 194966.515625, 194982.8125, 195000.0],
                    [194966.515625, 194982.8125, 195000.0, 194949.796875],
                ]
            ),
        )
        assert np.allclose(
            mesh_data.y_edge_coords,
            np.array(
                [
                    [361366.90625, 361399.46875],
                    [361366.90625, 361450.0],
                    [361399.46875, 361431.03125],
                    [361431.03125, 361450.0],
                ]
            ),
        )
        assert np.allclose(
            mesh_data.y_face_coords.data,
            np.array(
                [
                    [361366.90625, 361399.46875, 361431.03125, 361450.0],
                    [361399.46875, 361431.03125, 361450.0, 361366.90625],
                ]
            ),
        )


class TestLogText:
    def test_log_text_01(self):
        """
        Testing standard output of a single text without expansion.
        """
        key = "confirm"
        with captured_output() as (out, err):
            log_text(key)
        outstr = out.getvalue().splitlines()
        strref = ['Confirm using "y" ...', '']
        assert outstr == strref

    def test_log_text_02(self):
        """
        Testing standard output of a repeated text without expansion.
        """
        key = ""
        nr = 3
        with captured_output() as (out, err):
            log_text(key, repeat=nr)
        outstr = out.getvalue().splitlines()
        strref = ['', '', '']
        assert outstr == strref

    def test_log_text_03(self):
        """
        Testing standard output of a text with expansion.
        """
        key = "reach"
        data = {"reach": "ABC"}
        with captured_output() as (out, err):
            log_text(key, data=data)
        outstr = out.getvalue().splitlines()
        strref = ['The measure is located on reach ABC']
        assert outstr == strref

    def test_log_text_04(self):
        """
        Testing file output of a text with expansion.
        """
        key = "reach"
        data = {"reach": "ABC"}
        filename = "test.log"
        with open(filename, "w") as f:
            log_text(key, data=data, file=f)
        all_lines = open(filename, "r").read().splitlines()
        strref = ['The measure is located on reach ABC']
        assert all_lines == strref


def test_get_filename_01():
    """
    Testing get_filename wrapper for get_text.
    """
    assert get_filename("report.out") == "report.txt"


class TestGetText:
    def test_get_text_01(self):
        """
        Testing get_text: key not found.
        """
        assert get_text("@") == ["No message found for @"]

    def test_get_text_02(self):
        """
        Testing get_text: empty line key.
        """
        assert get_text("") == [""]

    def test_get_text_03(self):
        """
        Testing get_text: "confirm" key.
        """
        assert get_text("confirm") == ['Confirm using "y" ...', '']


class TestReadFMMap:
    def test_read_fm_map_01(self):
        """
        Testing read_fm_map: x coordinates of the faces.
        """
        filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        varname = "x"
        datac = _read_fm_map(filename, varname)
        dataref = 41.24417604888325
        assert datac[1] == pytest.approx(dataref)

    def test_read_fm_map_02(self):
        """
        Testing read_fm_map: y coordinates of the edges.
        """
        filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        varname = "y"
        location = "edge"
        datac = _read_fm_map(filename, varname, location)
        dataref = 7059.853000358055
        assert datac[1] == dataref

    def test_read_fm_map_03(self):
        """
        Testing read_fm_map: face node connectivity.
        """
        filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        varname = "face_node_connectivity"
        datac = _read_fm_map(filename, varname)
        dataref = 2352
        assert datac[-1][1] == dataref

    def test_read_fm_map_04(self):
        """
        Testing read_fm_map: variable by standard name.
        """
        filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        varname = "sea_floor_depth_below_sea_surface"
        datac = _read_fm_map(filename, varname)
        dataref = 3.894498393076889
        assert datac[1] == pytest.approx(dataref)

    def test_read_fm_map_05(self):
        """
        Testing read_fm_map: variable by long name.
        """
        filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        varname = "Water level"
        datac = _read_fm_map(filename, varname)
        dataref = 3.8871328177527262
        assert datac[1] == pytest.approx(dataref)

    def test_read_fm_map_06(self):
        """
        Testing read_fm_map: variable by long name.
        """
        filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        varname = "water level"
        with pytest.raises(Exception) as cm:
            _read_fm_map(filename, varname)
        assert (
            str(cm.value) == 'Expected one variable for "water level", but obtained 0.'
        )


@pytest.mark.skipif(
    platform.system() != "Windows", reason="it will be completely changed"
)
class TestAbsolutePath:
    def test_absolute_path_01(self):
        """Convert absolute path into relative path using relative_path (Windows)."""
        rootdir = "q:" + os.sep + "some" + os.sep + "dir"
        afile = (
            "q:"
            + os.sep
            + "some"
            + os.sep
            + "other"
            + os.sep
            + "dir"
            + os.sep
            + "file.ext"
        )
        rfile = ".." + os.sep + "other" + os.sep + "dir" + os.sep + "file.ext"
        assert absolute_path(rootdir, rfile) == afile

    def test_absolute_path_02(self):
        """Empty string should not be adjusted by relative_path."""
        rootdir = "d:" + os.sep + "some" + os.sep + "dir"
        file = ""
        assert absolute_path(rootdir, file) == file

    def test_absolute_path_03(self):
        """If path on different drive, it shouldn't be adjusted by relative_path (Windows)."""
        rootdir = "d:" + os.sep + "some" + os.sep + "dir"
        file = (
            "e:"
            + os.sep
            + "some"
            + os.sep
            + "other"
            + os.sep
            + "dir"
            + os.sep
            + "file.ext"
        )
        assert absolute_path(rootdir, file) == file


class TestRelativePath:
    def test_relative_path_02(self):
        """Empty string should not be adjusted by relative_path."""
        rootdir = "d:" + os.sep + "some" + os.sep + "dir"
        file = ""
        assert relative_path(rootdir, file) == file

    @pytest.mark.skipif(
        platform.system() != "Windows", reason="it will be completely changed"
    )
    def test_relative_path_03(self):
        """If path on different drive, it shouldn't be adjusted by relative_path (Windows)."""
        rootdir = "d:" + os.sep + "some" + os.sep + "dir"
        file = (
            "e:"
            + os.sep
            + "some"
            + os.sep
            + "other"
            + os.sep
            + "dir"
            + os.sep
            + "file.ext"
        )
        assert relative_path(rootdir, file) == file


class TestConfigFile:
    """Test cases for the ConfigFile class."""

    @pytest.fixture
    def config(self) -> ConfigParser:
        """Fixture to create a ConfigFile instance."""
        config = ConfigParser()
        config.read_dict(
            {
                "General": {
                    "Version": "1.0",
                    "plotting": "yes",
                    "ZoomStepKM": "0.1",
                    "Boundaries": "123.0:128.0",
                },
                "Detect": {
                    "SimFile": "test_sim.nc",
                    "NBank": "2",
                    "DLines": "[ 50.0, 50.0 ]",
                },
                "Erosion": {"OutputDir": "./output"},
            }
        )
        return config

    @pytest.fixture
    def config_data(self) -> str:
        """Fixture to create a sample configuration file string."""
        content = (
            "[General]\n"
            "  version    = 1.0\n"
            "  plotting   = yes\n"
            "  zoomstepkm = 0.1\n"
            "  boundaries = 123.0:128.0\n\n"
            "[Detect]\n"
            "  simfile    = test_sim.nc\n"
            "  nbank      = 2\n"
            "  dlines     = [ 50.0, 50.0 ]\n\n"
            "[Erosion]\n"
            "  outputdir  = ./output\n"
        )
        return content

    def test_init(self, config: ConfigParser):
        """Test initialization of ConfigFile."""
        config_file = ConfigFile(config=config)
        assert isinstance(config_file, ConfigFile)

    def test_read(self, config_data: str, fs: FakeFilesystem):
        """Test reading a configuration file."""
        fs.create_file("dummy_path.cfg", contents=config_data)
        config_file = ConfigFile.read("dummy_path.cfg")
        assert isinstance(config_file, ConfigFile)
        assert config_file.config["General"]["Version"] == "1.0"
        assert config_file.config["Detect"]["NBank"] == "2"

    def test_write(self, config: ConfigParser, config_data: str, fs: FakeFilesystem):
        """Test writing a configuration file."""
        config_file = ConfigFile(config=config)
        config_file.write("test_output.cfg")
        with open("test_output.cfg", "r") as file:
            assert file.read() == config_data

    def test_get_str(self, config: ConfigParser):
        """Test retrieving a string value."""
        config_file = ConfigFile(config, "tests/data/erosion/test.cfg")
        assert config_file.get_str("General", "Version") == "1.0"

    def test_get_int(self, config: ConfigParser):
        """Test retrieving an integer value."""
        config_file = ConfigFile(config, "tests/data/erosion/test.cfg")
        assert config_file.get_int("Detect", "NBank") == 2

    def test_get_bool(self, config: ConfigParser):
        """Test retrieving a boolean value."""
        config_file = ConfigFile(config, "tests/data/erosion/test.cfg")
        assert config_file.get_bool("General", "plotting") is True

    def test_get_float(self, config: ConfigParser):
        """Test retrieving a float value."""
        config_file = ConfigFile(config, "tests/data/erosion/test.cfg")
        assert config_file.get_float("General", "ZoomStepKM") == pytest.approx(
            0.1, rel=1e-6
        )

    def test_get_sim_file(self, config: ConfigParser):
        """Test retrieving a simulation file."""
        path = Path("tests/data/erosion")
        config_file = ConfigFile(config, str(path / "test.cfg"))
        assert config_file.get_sim_file("Detect", "") == str(
            path.resolve() / "test_sim.nc"
        )

    def test_get_start_end_stations(self, config: ConfigParser):
        """Test retrieving km bounds."""
        config_file = ConfigFile(config, "tests/data/erosion/test.cfg")
        start, end = config_file.get_start_end_stations()
        assert start == pytest.approx(123.0, rel=1e-6)
        assert end == pytest.approx(128.0, rel=1e-6)

    def test_get_search_lines(self):
        """Test retrieving search lines."""
        config = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
        mock_linestring = LineString([(0, 0), (1, 1), (2, 2)])

        with patch("dfastio.xyc.models.XYCModel.read", return_value=mock_linestring):
            search_lines = config.get_search_lines()

        assert len(search_lines) == 2
        assert list(search_lines[0].coords) == [(0, 0), (1, 1), (2, 2)]

    def test_read_bank_lines(self, config: ConfigParser, fs: FakeFilesystem):
        """Test retrieving bank lines."""
        config["General"]["BankLine"] = "bankfile"
        config_file = ConfigFile(config, "tests/data/erosion/test.cfg")

        fs.create_file(
            "inputs/bankfile_1.xyc",
            contents="0.0 0.0\n1.0 1.0\n2.0 2.0\n3.0 3.0\n4.0 4.0\n",
        )

        bank_lines = config_file.read_bank_lines("inputs")

        assert isinstance(bank_lines, GeoDataFrame)
        assert len(bank_lines) == 1
        assert list(bank_lines.geometry[0].coords) == [
            (0.0, 0.0),
            (1.0, 1.0),
            (2.0, 2.0),
            (3.0, 3.0),
            (4.0, 4.0),
        ]

    @pytest.mark.parametrize(
        "key, value, default, valid, expected",
        [
            (
                "ZoomStepKM",
                1.0,
                None,
                None,
                [np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0])],
            ),
            (
                "NonExistentKey",
                None,
                2.0,
                None,
                [np.array([2.0, 2.0, 2.0]), np.array([2.0, 2.0, 2.0])],
            ),
            (
                "ValidValue",
                3.0,
                None,
                [1.0, 2.0, 3.0],
                [np.array([3.0, 3.0, 3.0]), np.array([3.0, 3.0, 3.0])],
            ),
        ],
        ids=[
            "Valid parameter with value 1.0",
            "Missing parameter with default value 2.0",
            "Valid parameter with restricted valid values",
        ],
    )
    def test_get_parameter(
        self, key, value, default, valid, expected, config: ConfigParser
    ):
        """Test retrieving a parameter field."""
        config_file = ConfigFile(config, "tests/data/erosion/test.cfg")
        bank_km = [np.array([0, 1, 2]), np.array([3, 4, 5])]

        if value:
            config["General"] = {key: str(value)}
        result = config_file.get_parameter(
            "General", key, bank_km, default=default, valid=valid
        )
        assert all(np.array_equal(r, e) for r, e in zip(result, expected))

    @pytest.mark.parametrize(
        "key, value, positive, valid, expected",
        [
            ("NegativeValue", -1.0, True, None, "No such file or directory"),
            ("InvalidValue", 4.0, False, [1.0, 2.0, 3.0], "No such file or directory"),
        ],
        ids=[
            "Negative value with positive=True",
            "Invalid value not in valid list",
        ],
    )
    def test_get_parameter_exception(
        self, key, value, positive, valid, expected, config: ConfigParser
    ):
        """Test retrieving a parameter field."""
        config_file = ConfigFile(config, "tests/data/erosion/test.cfg")
        bank_km = [np.array([0, 1, 2]), np.array([3, 4, 5])]

        # Case 5: Parameter does not match valid values
        config["General"] = {key: str(value)}
        with pytest.raises(Exception, match=expected):
            config_file.get_parameter(
                "General", key, bank_km, positive=positive, valid=valid
            )

    def test_get_bank_search_distances(self, config: ConfigParser):
        """Test retrieving bank search distances."""
        config_file = ConfigFile(config, "tests/data/erosion/test.cfg")

        # Case 1: Bank search distances exist in the configuration
        result = config_file.get_bank_search_distances(2)
        assert all(pytest.approx(item, rel=1e-6) == 50.0 for item in result)

        # Case 2: Bank search distances do not exist, use default value
        result = config_file.get_bank_search_distances(2)
        assert all(pytest.approx(item, rel=1e-6) == 50.0 for item in result)

    def test_get_river_center_line(self):
        """Test retrieving x and y coordinates."""
        config = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
        mock_linestring = LineString([(0, 0, 1), (1, 1, 2), (2, 2, 3)])

        with patch("dfastio.xyc.models.XYCModel.read", return_value=mock_linestring):
            river_center_line = config.get_river_center_line()

        assert river_center_line.wkt == 'LINESTRING Z (0 0 1, 1 1 2, 2 2 3)'

    @pytest.fixture
    def path_dict(self) -> Dict:
        """Fixture to create a dictionary for path resolution."""
        return {
            "General": {
                "RiverKM": "inputs/rivkm_20m.xyc",
                "BankDir": "output/banklines",
                "FigureDir": "output/figures",
            }
        }

    def test_resolve(self, path_dict: Dict):
        """Test resolving paths in the configuration."""
        config = ConfigParser()
        config.read_dict(path_dict)
        config_file = ConfigFile(config, "tests/data/erosion/test.cfg")
        config_file.resolve("tests/data/erosion")
        assert config_file.config["General"]["RiverKM"] == str(
            Path("tests/data/erosion").resolve() / "inputs/rivkm_20m.xyc"
        )

    def test_relative_to(self, path_dict: Dict):
        """Test converting paths to relative paths."""
        config = ConfigParser()
        config.read_dict(path_dict)
        config_file = ConfigFile(config, "tests/data/erosion/test.cfg")
        config_file.relative_to("tests/data")
        assert config_file.config["General"]["RiverKM"] == str(
            Path("erosion") / "inputs/rivkm_20m.xyc"
        )

    def test_make_paths_absolute(self, path_dict: Dict):
        """Test converting filenames to be relative to the current working directory."""
        cwd = Path("tests/data/erosion").resolve()
        config = ConfigParser()
        config.read_dict(path_dict)
        config_file = ConfigFile(config, cwd / "test.cfg")

        with patch("dfastbe.io.Path.cwd", return_value=str(cwd)):
            rootdir = config_file.make_paths_absolute()

        assert rootdir == cwd
        assert config_file.config["General"]["RiverKM"] == str(
            cwd / "inputs/rivkm_20m.xyc"
        )
        assert config_file.config["General"]["BankDir"] == str(cwd / "output/banklines")
        assert config_file.config["General"]["FigureDir"] == str(cwd / "output/figures")

    def test__upgrade(self):
        """Test upgrading the configuration."""
        config = ConfigParser()
        config.read_dict(
            {
                "General": {
                    "Version": "0.1",
                    "RiverKM": "inputs/rivkm_20m.xyc",
                    "Boundaries": "123.0:128.0",
                    "BankDir": "output/banklines",
                    "BankFile": "bankfile",
                    "Plotting": "yes",
                    "SavePlots": "True",
                    "SaveZoomPlots": "False",
                    "ZoomStepKM": "1.0",
                    "FigureDir": "output/figures",
                    "ClosePlots": "False",
                    "DebugOutput": "False",
                    "SimFile": "inputs/sim0270/SDS-j19_map.nc",
                    "WaterDepth": "0.0",
                    "NBank": "2",
                    "Line1": "inputs/oeverlijn_links_mod.xyc",
                    "Line2": "inputs/oeverlijn_rechts_mod.xyc",
                    "DLines": "[ 50.0, 50.0 ]",
                    "TErosion": "1",
                    "RiverAxis": "inputs/maas_rivieras_mod.xyc",
                    "Fairway": "inputs/maas_rivieras_mod.xyc",
                    "OutputInterval": "0.1",
                    "OutputDir": "output/bankerosion",
                    "BankNew": "banknew",
                    "BankEq": "bankeq",
                    "EroVol": "erovol_standard.evo",
                    "EroVolEqui": "erovol_eq.evo",
                    "ShipType": "2",
                    "VShip": "5.0",
                    "NShip": "inputs/nships_totaal",
                    "NWaves": "5",
                    "Draught": "1.2",
                    "Wave0": "150.0",
                    "Wave1": "110.0",
                    "Classes": "false",
                    "BankType": "inputs/bankstrength_tauc",
                    "ProtectionLevel": "inputs/stortsteen",
                    "Slope": "20.0",
                    "Reed": "0.0",
                    "VelFilterDist": "0.3",
                    "BedFilterDist": "0.3",
                    "NLevel": "1",
                    "SimFile1": "inputs/sim0270/SDS-j19_map.nc",
                    "RefLevel": "3",
                }
            }
        )
        config_file = ConfigFile(config=config)
        config_result = config_file._upgrade(config_file.config)
        assert config_result["General"]["plotting"] == "yes"
        assert config_result["Detect"]["SimFile"] == "inputs/sim0270/SDS-j19_map.nc"

    @pytest.fixture
    def plotting_data(self) -> Dict:
        """Fixture to create a dictionary for plotting flags."""
        return {
            "General": {
                "Plotting": "yes",
                "SavePlots": "yes",
                "SaveZoomPlots": "no",
                "ZoomStepKM": "0.5",
                "ClosePlots": "no",
                "FigureDir": "output/figures",
                "FigureExt": ".png",
            }
        }

    def test_get_plotting_flags(self, plotting_data: Dict, fs: FakeFilesystem):
        """Test the get_plotting_flags method."""
        config = ConfigParser()
        config.read_dict(plotting_data)
        config_file = ConfigFile(config, "tests/data/erosion/test.cfg")
        root_dir = Path("tests/data/erosion").resolve()
        plotting_flags = config_file.get_plotting_flags(str(root_dir))

        assert plotting_flags["plot_data"] is True
        assert plotting_flags["save_plot"] is True
        assert plotting_flags["save_plot_zoomed"] is False
        assert plotting_flags["zoom_km_step"] == 0.5
        assert plotting_flags["close_plot"] is False
        assert plotting_flags["fig_dir"] == str(root_dir / "output/figures")
        assert plotting_flags["plot_ext"] == ".png"


class TestConfigFileE2E:
    def test_initialization(self):
        path = "tests/data/erosion/meuse_manual.cfg"
        config_file = ConfigFile.read(path)
        river_km = config_file.config["General"]["riverkm"]
        assert Path(river_km).exists()

    def test_write_config_01(self):
        """
        Testing write_config.
        """
        filename = "test.cfg"
        config = ConfigParser()
        config.add_section("G 1")
        config["G 1"]["K 1"] = "V 1"
        config.add_section("Group 2")
        config["Group 2"]["K1"] = "1.0 0.1 0.0 0.01"
        config["Group 2"]["K2"] = "2.0 0.2 0.02 0.0"
        config.add_section("Group 3")
        config["Group 3"]["LongKey"] = "3"

        config = ConfigFile(config)
        config.write(filename)
        all_lines = open(filename, "r").read().splitlines()
        all_lines_ref = [
            '[G 1]',
            '  k 1     = V 1',
            '',
            '[Group 2]',
            '  k1      = 1.0 0.1 0.0 0.01',
            '  k2      = 2.0 0.2 0.02 0.0',
            '',
            '[Group 3]',
            '  longkey = 3',
        ]
        assert all_lines == all_lines_ref
        Path(filename).unlink()


class TestRiverData:
    def test_initialization(self):
        path = "tests/data/erosion/meuse_manual.cfg"
        config_file = ConfigFile.read(path)
        river_data = RiverData(config_file)
        assert isinstance(river_data.config_file, ConfigFile)
        search_lines = river_data.search_lines
        assert search_lines.size == 2
        center_line = river_data.river_center_line
        assert center_line.station_bounds[0] == 123.0
        assert center_line.station_bounds[1] == 128.0
        assert isinstance(center_line.values, LineString)
        center_line_arr = center_line.as_array()
        assert isinstance(center_line_arr, np.ndarray)
        assert center_line_arr.shape == (251, 3)
