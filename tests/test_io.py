from typing import Dict
import unittest
from unittest.mock import patch

import numpy as np

from dfastbe.io import ConfigFile, load_program_texts, log_text, get_text, read_fm_map,\
    ugrid_add, copy_ugrid, copy_var, read_waqua_xyz, write_simona_box, \
    get_mesh_and_facedim_names, absolute_path, relative_path, get_filename, \
    RiverData

import configparser
import os
from pathlib import Path
import platform
import numpy
import netCDF4
from shapely.geometry.linestring import LineString
import pytest
from pyfakefs.fake_filesystem import FakeFilesystem

import sys
from contextlib import contextmanager
from io import StringIO


@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


class Test_load_program_texts:
    def test_load_program_texts_01(self):
        """
        Testing load_program_texts.
        """
        print("current work directory: ", os.getcwd())
        assert load_program_texts("tests/files/messages.UK.ini") == None

class Test_log_text:
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
        dict = {"reach": "ABC"}
        with captured_output() as (out, err):
            log_text(key, dict=dict)
        outstr = out.getvalue().splitlines()
        strref = ['The measure is located on reach ABC']
        assert outstr == strref

    def test_log_text_04(self):
        """
        Testing file output of a text with expansion.
        """
        key = "reach"
        dict = {"reach": "ABC"}
        filename = "test.log"
        with open(filename, "w") as f:
            log_text(key, dict=dict, file=f)
        all_lines = open(filename, "r").read().splitlines()
        strref = ['The measure is located on reach ABC']
        assert all_lines == strref

class Test_get_filename:
    def test_get_filename_01(self):
        """
        Testing get_filename wrapper for get_text.
        """
        assert get_filename("report.out") == "report.txt"

class Test_get_text:
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
        assert get_text("confirm") == ['Confirm using "y" ...','']

class Test_read_fm_map:
    def test_read_fm_map_01(self):
        """
        Testing read_fm_map: x coordinates of the faces.
        """
        filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        varname = "x"
        # location = "face"
        datac = read_fm_map(filename, varname)
        dataref = 41.24417604888325
        assert datac[1] == dataref

    def test_read_fm_map_02(self):
        """
        Testing read_fm_map: y coordinates of the edges.
        """
        filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        varname = "y"
        location = "edge"
        datac = read_fm_map(filename, varname, location)
        dataref = 7059.853000358055
        assert datac[1] == dataref

    def test_read_fm_map_03(self):
        """
        Testing read_fm_map: face node connectivity.
        """
        filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        varname = "face_node_connectivity"
        datac = read_fm_map(filename, varname)
        dataref = 2352
        assert datac[-1][1] == dataref

    def test_read_fm_map_04(self):
        """
        Testing read_fm_map: variable by standard name.
        """
        filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        varname = "sea_floor_depth_below_sea_surface"
        datac = read_fm_map(filename, varname)
        dataref = 3.894498393076889
        assert datac[1] == dataref

    def test_read_fm_map_05(self):
        """
        Testing read_fm_map: variable by long name.
        """
        filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        varname = "Water level"
        datac = read_fm_map(filename, varname)
        dataref = 3.8871328177527262
        assert datac[1] == dataref

    def test_read_fm_map_06(self):
        """
        Testing read_fm_map: variable by long name.
        """
        filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        varname = "water level"
        with pytest.raises(Exception) as cm:
            datac = read_fm_map(filename, varname)
        assert str(cm.value) == 'Expected one variable for "water level", but obtained 0.'

class Test_get_mesh_and_facedim_names():
    def test_get_mesh_and_facedim_names_01(self):
        """
        Testing get_mesh_and_facedim_names.
        """
        filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        name_and_dim = get_mesh_and_facedim_names(filename)
        assert name_and_dim == ("mesh2d", "mesh2d_nFaces")

class Test_copy_ugrid():
    def test_copy_ugrid_01(self):
        """
        Testing copy_ugrid (depends on copy_var).
        """
        src_filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        dst_filename = "test.nc"
        meshname, facedim = get_mesh_and_facedim_names(src_filename)
        copy_ugrid(src_filename, meshname, dst_filename)
        #
        varname = "face_node_connectivity"
        datac = read_fm_map(dst_filename, varname)
        dataref = 2352
        assert datac[-1][1] == dataref


class Test_copy_var():
    def test_copy_var_01(self):
        """
        Testing copy_var.
        """
        src_filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        dst_filename = "test.nc"
        #
        with netCDF4.Dataset(src_filename) as src:
            with netCDF4.Dataset(dst_filename, "a") as dst:
                copy_var(src, "mesh2d_s1", dst)
        #
        varname = "sea_surface_height"
        datac = read_fm_map(dst_filename, varname)
        dataref = 3.8871328177527262
        assert datac[1] == dataref

class Test_ugrid_add():
    def test_ugrid_add_01(self):
        """
        Testing ugrid_add.
        """
        dst_filename = "test.nc"
        meshname = "mesh2d"
        facedim = "mesh2d_nFaces"
        #
        varname = "xxx"
        ldata = numpy.zeros((4132))
        ldata[1] = 3.14159
        long_name = "added_variable"
        #
        ugrid_add(dst_filename, varname, ldata, meshname, facedim, long_name)
        #
        datac = read_fm_map(dst_filename, long_name)
        assert datac[1] == ldata[1]

class Test_read_waqua_xyz():
    def test_read_waqua_xyz_01(self):
        """
        Read WAQUA xyz file default column 2.
        """
        filename = "tests/files/read_waqua_xyz_test.xyc"
        data = read_waqua_xyz(filename)
        datar = numpy.array([3., 6., 9., 12.])
        print("data reference: ", datar)
        print("data read     : ", data)
        assert numpy.shape(data) == (4,)
        assert (data == datar).all() == True

    def test_read_waqua_xyz_02(self):
        """
        Read WAQUA xyz file columns 1 and 2.
        """
        filename = "tests/files/read_waqua_xyz_test.xyc"
        col = (1,2)
        data = read_waqua_xyz(filename, col)
        datar = numpy.array([[ 2., 3.], [ 5., 6.], [ 8., 9.], [11., 12.]])
        print("data reference: ", datar)
        print("data read     : ", data)
        assert numpy.shape(data) == (4,2)
        assert (data == datar).all() == True

class Test_write_simona_box():
    def test_write_simona_box_01(self):
        """
        Write small SIMONA BOX file.
        """
        filename = "test.box"
        data = numpy.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
        firstm = 0
        firstn = 0
        write_simona_box(filename, data, firstm, firstn)
        all_lines = open(filename, "r").read().splitlines()
        all_lines_ref = ['      BOX MNMN=(   1,    1,    3,    3), VARIABLE_VAL=',
                         '          1.000       2.000       3.000',
                         '          4.000       5.000       6.000',
                         '          7.000       8.000       9.000']
        assert all_lines == all_lines_ref

    def test_write_simona_box_02(self):
        """
        Write small SIMONA BOX file with offset.
        """
        filename = "test.box"
        data = numpy.array([[0, 0, 0, 0, 0], [0, 0, 1, 2, 3],[0, 0, 4, 5, 6],[0, 0, 7, 8, 9]])
        firstm = 1
        firstn = 2
        write_simona_box(filename, data, firstm, firstn)
        all_lines = open(filename, "r").read().splitlines()
        all_lines_ref = ['      BOX MNMN=(   2,    3,    4,    5), VARIABLE_VAL=',
                         '          1.000       2.000       3.000',
                         '          4.000       5.000       6.000',
                         '          7.000       8.000       9.000']
        assert all_lines == all_lines_ref

    def test_write_simona_box_03(self):
        """
        Write large SIMONA BOX file.
        """
        filename = "test.box"
        data = numpy.zeros((15,15))
        firstm = 0
        firstn = 0
        write_simona_box(filename, data, firstm, firstn)
        all_lines = open(filename, "r").read().splitlines()
        all_lines_ref = ['      BOX MNMN=(   1,    1,   15,   10), VARIABLE_VAL=']
        all_lines_ref.extend(['          0.000       0.000       0.000       0.000       0.000       0.000       0.000       0.000       0.000       0.000']*15)
        all_lines_ref.extend(['      BOX MNMN=(   1,   11,   15,   15), VARIABLE_VAL='])
        all_lines_ref.extend(['          0.000       0.000       0.000       0.000       0.000']*15)
        self.maxDiff = None
        assert all_lines == all_lines_ref


@pytest.mark.skipif(
    platform.system() != "Windows", reason="it will be completely changed"
)
class TestAbsolutePath:

    def test_absolute_path_01(self):
        """
        Convert absolute path into relative path using relative_path (Windows).
        """
        rootdir = "g:" + os.sep + "some" + os.sep + "dir"
        afile = "g:" + os.sep + "some" + os.sep + "other" + os.sep + "dir" + os.sep + "file.ext"
        rfile = ".." + os.sep + "other" + os.sep + "dir" + os.sep + "file.ext"
        assert absolute_path(rootdir, rfile) == afile

    def test_absolute_path_02(self):
        """
        Empty string should not be adjusted by relative_path.
        """
        rootdir = "d:" + os.sep + "some" + os.sep + "dir"
        file = ""
        assert absolute_path(rootdir, file) == file

    def test_absolute_path_03(self):
        """
        If path on different drive, it shouldn't be adjusted by relative_path (Windows).
        """
        rootdir = "d:" + os.sep + "some" + os.sep + "dir"
        file = "e:" + os.sep + "some" + os.sep + "other" + os.sep + "dir" + os.sep + "file.ext"
        assert absolute_path(rootdir, file) == file

    def test_absolute_path_04(self):
        """
        Convert absolute path into relative path using relative_path (Linux).
        """
        rootdir = os.sep + "some" + os.sep + "dir"
        afile = os.sep + "some" + os.sep + "other" + os.sep + "dir" + os.sep + "file.ext"
        rfile = ".." + os.sep + "other" + os.sep + "dir" + os.sep + "file.ext"
        assert absolute_path(rootdir, rfile) == afile

class Test_relative_path():
    def test_relative_path_01(self):
        """
        Convert absolute path into relative path using relative_path (Windows).
        """
        rootdir = "d:" + os.sep + "some" + os.sep + "dir"
        afile = "d:" + os.sep + "some" + os.sep + "other" + os.sep + "dir" + os.sep + "file.ext"
        rfile = ".." + os.sep + "other" + os.sep + "dir" + os.sep + "file.ext"
        assert relative_path(rootdir, afile) == rfile

    def test_relative_path_02(self):
        """
        Empty string should not be adjusted by relative_path.
        """
        rootdir = "d:" + os.sep + "some" + os.sep + "dir"
        file = ""
        assert relative_path(rootdir, file) == file

    @pytest.mark.skipif(
        platform.system() != "Windows", reason="it will be completely changed"
    )
    def test_relative_path_03(self):
        """
        If path on different drive, it shouldn't be adjusted by relative_path (Windows).
        """
        rootdir = "d:" + os.sep + "some" + os.sep + "dir"
        file = "e:" + os.sep + "some" + os.sep + "other" + os.sep + "dir" + os.sep + "file.ext"
        assert relative_path(rootdir, file) == file

    def test_relative_path_04(self):
        """
        Convert absolute path into relative path using relative_path (Linux).
        """
        rootdir = os.sep + "some" + os.sep + "dir"
        afile = os.sep + "some" + os.sep + "other" + os.sep + "dir" + os.sep + "file.ext"
        rfile = ".." + os.sep + "other" + os.sep + "dir" + os.sep + "file.ext"
        assert relative_path(rootdir, afile) == rfile


class Test_ConfigFile:
    """Test cases for the ConfigFile class."""

    @pytest.fixture
    def config(self) -> configparser.ConfigParser:
        """Fixture to create a ConfigFile instance."""
        config = configparser.ConfigParser()
        config.read_dict(
            {
                "General": {
                    "Version": "1.0",
                    "plotting": "yes",
                    "ZoomStepKM": "0.1",
                },
                "Detect": {"SimFile": "test_sim.nc", "NBank": "2"},
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
            "  zoomstepkm = 0.1\n\n"
            "[Detect]\n"
            "  simfile    = test_sim.nc\n"
            "  nbank      = 2\n\n"
            "[Erosion]\n"
            "  outputdir  = ./output\n"
        )
        return content

    def test_init(self, config: configparser.ConfigParser):
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

    def test_write(
        self, config: configparser.ConfigParser, config_data: str, fs: FakeFilesystem
    ):
        """Test writing a configuration file."""
        config_file = ConfigFile(config=config)
        config_file.write("test_output.cfg")
        with open("test_output.cfg", "r") as file:
            assert file.read() == config_data

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
        config = configparser.ConfigParser()
        config.read_dict(path_dict)
        config_file = ConfigFile(config, "tests/data/erosion/test.cfg")
        config_file.resolve("tests/data")
        assert config_file.config["General"]["RiverKM"] == str(
            Path("tests/data") / "tests/data/erosion" / "inputs/rivkm_20m.xyc"
        )

    def test_relative_to(self, path_dict: Dict):
        """Test converting paths to relative paths."""
        config = configparser.ConfigParser()
        config.read_dict(path_dict)
        config_file = ConfigFile(config, "tests/data/erosion/test.cfg")
        config_file.relative_to("tests/data")
        assert config_file.config["General"]["RiverKM"] == str(
            Path("erosion") / "inputs/rivkm_20m.xyc"
        )

    def test_adjust_filenames(self, path_dict: Dict):
        """Test adjusting filenames to be relative to the current working directory."""
        cwd = Path("tests/data/erosion")
        config = configparser.ConfigParser()
        config.read_dict(path_dict)
        config_file = ConfigFile(config, cwd / "test.cfg")

        with patch("os.getcwd", return_value=str(cwd)):
            rootdir = config_file.adjust_filenames()

        assert rootdir == "."
        assert config_file.config["General"]["RiverKM"] == str(
            cwd / "inputs/rivkm_20m.xyc"
        )
        assert config_file.config["General"]["BankDir"] == str(cwd / "output/banklines")
        assert config_file.config["General"]["FigureDir"] == str(cwd / "output/figures")

    def test__upgrade(self):
        """Test upgrading the configuration."""
        config = configparser.ConfigParser()
        config.read_dict(
            {
                "General": {
                    "Version": "1.0",
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
                    "NLevel": "0",
                    "RefLevel": "3",
                }
            }
        )
        config_file = ConfigFile(config=config)
        config_result = config_file._upgrade(config_file.config)
        assert config_result["General"]["plotting"] == "yes"
        # TODO: Check why the _upgrade function is not working as expected
        # assert config_result["Detect"]["SimFile"] == "test_sim.nc"


class TestConfigFile(unittest.TestCase):

    def setUp(self):
        """Set up a sample configuration for testing."""
        self.config = configparser.ConfigParser()
        self.config.read_dict({
            "General": {"Version": "1.0", "TestParam": "42"},
            "Detect": {"SimFile": "test_sim.nc"},
            "Erosion": {"OutputDir": "./output"}
        })
        self.config_file = ConfigFile(self.config)

    def test_init(self):
        """Test initialization of ConfigFile."""
        self.assertIsInstance(self.config_file, ConfigFile)

    def test_config_property(self):
        """Test getting and setting the config property."""
        new_config = configparser.ConfigParser()
        self.config_file.config = new_config
        self.assertEqual(self.config_file.config, new_config)

    def test_read(self):
        """Test reading a configuration file."""
        config_str = """[General]\nVersion = 1.0\nTestParam = 42\n"""
        with patch("builtins.open", return_value=StringIO(config_str)), patch("pathlib.Path.exists", return_value=True):
            config_obj = ConfigFile.read("dummy_path.cfg")
            version = config_obj.version
        self.assertEqual(config_obj.config["General"]["Version"], "1.0")
        self.assertEqual(version, "1.0")

    def test_get_str(self):
        """Test retrieving a string value."""
        self.assertEqual(self.config_file.get_str("General", "Version"), "1.0")

    def test_get_int(self):
        """Test retrieving an integer value."""
        self.assertEqual(self.config_file.get_int("General", "TestParam"), 42)

    def test_get_bool(self):
        """Test retrieving a boolean value."""
        self.config_file.config["General"]["Enabled"] = "yes"
        self.assertTrue(self.config_file.get_bool("General", "Enabled"))

    def test_write(self):
        """Test writing a configuration file."""
        with patch("builtins.open", unittest.mock.mock_open()) as mock_file:
            self.config_file.write("test_output.cfg")
            mock_file.assert_called_with("test_output.cfg", "w")

    def test_adjust_filenames(self):
        """Test adjusting filenames."""
        self.config_file.path = "/home/user/config.cfg"
        self.config_file.root_dir = "/home/user"
        with patch("os.getcwd", return_value="/home/user"):
            rootdir = self.config_file.adjust_filenames()
        self.assertEqual(rootdir, ".")


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
        config = configparser.ConfigParser()
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
        all_lines_ref = ['[G 1]',
                         '  k 1     = V 1',
                         '',
                         '[Group 2]',
                         '  k1      = 1.0 0.1 0.0 0.01',
                         '  k2      = 2.0 0.2 0.02 0.0',
                         '',
                         '[Group 3]',
                         '  longkey = 3']
        assert all_lines == all_lines_ref
        Path(filename).unlink()


class TestRiverData:
    def test_initialization(self):
        path = "tests/data/erosion/meuse_manual.cfg"
        config_file = ConfigFile.read(path)
        river_data = RiverData(config_file)
        assert isinstance(river_data.config_file, ConfigFile)
        assert river_data.num_search_lines == 2
        assert river_data.start_station == 123.0
        assert river_data.end_station == 128.0
        assert isinstance(river_data.masked_profile, LineString)
        assert isinstance(river_data.profile, LineString)
        assert isinstance(river_data.masked_profile_arr, np.ndarray)
        assert river_data.masked_profile_arr.shape == (251, 3)
