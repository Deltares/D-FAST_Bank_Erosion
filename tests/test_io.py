import os
import platform
import sys
from configparser import ConfigParser
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import netCDF4
import numpy as np
import pytest
from geopandas import GeoDataFrame
from pyfakefs.fake_filesystem import FakeFilesystem
from shapely.geometry import LineString

from dfastbe.io import (
    ConfigFile,
    RiverData,
    absolute_path,
    copy_ugrid,
    copy_var,
    get_filename,
    get_mesh_and_facedim_names,
    get_text,
    load_program_texts,
    log_text,
    read_fm_map,
    read_waqua_xyz,
    relative_path,
    ugrid_add,
    write_simona_box,
)


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
        datac = read_fm_map(filename, varname)
        dataref = 41.24417604888325
        assert datac[1] == pytest.approx(dataref)

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
        assert datac[1] == pytest.approx(dataref)

    def test_read_fm_map_05(self):
        """
        Testing read_fm_map: variable by long name.
        """
        filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        varname = "Water level"
        datac = read_fm_map(filename, varname)
        dataref = 3.8871328177527262
        assert datac[1] == pytest.approx(dataref)

    def test_read_fm_map_06(self):
        """
        Testing read_fm_map: variable by long name.
        """
        filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        varname = "water level"
        with pytest.raises(Exception) as cm:
            read_fm_map(filename, varname)
        assert (
            str(cm.value) == 'Expected one variable for "water level", but obtained 0.'
        )


def test_get_mesh_and_facedim_names_01():
    """
    Testing get_mesh_and_facedim_names.
    """
    filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
    name_and_dim = get_mesh_and_facedim_names(filename)
    assert name_and_dim == ("mesh2d", "mesh2d_nFaces")


class TestCopyUgrid:
    def test_copy_ugrid_01(self):
        """
        Testing copy_ugrid (depends on copy_var).
        """
        src_filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        dst_filename = "test.nc"
        meshname, _ = get_mesh_and_facedim_names(src_filename)
        copy_ugrid(src_filename, meshname, dst_filename)
        #
        varname = "face_node_connectivity"
        datac = read_fm_map(dst_filename, varname)
        dataref = 2352
        assert datac[-1][1] == dataref


class TestCopyVar:

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
        assert datac[1] == pytest.approx(dataref)


def test_ugrid_add_01():
    """
    Testing ugrid_add.
    """
    dst_filename = "test.nc"
    meshname = "mesh2d"
    facedim = "mesh2d_nFaces"
    #
    varname = "xxx"
    ldata = np.zeros((4132))
    ldata[1] = 3.14159
    long_name = "added_variable"
    #
    ugrid_add(dst_filename, varname, ldata, meshname, facedim, long_name)
    #
    datac = read_fm_map(dst_filename, long_name)
    assert datac[1] == ldata[1]


class TestReadWaquaXYZ:
    def test_read_waqua_xyz_01(self):
        """
        Read WAQUA xyz file default column 2.
        """
        filename = "tests/files/read_waqua_xyz_test.xyc"
        data = read_waqua_xyz(filename)
        datar = np.array([3.0, 6.0, 9.0, 12.0])
        print("data reference: ", datar)
        print("data read     : ", data)
        assert np.shape(data) == (4,)
        assert (data == datar).all() == True

    def test_read_waqua_xyz_02(self):
        """
        Read WAQUA xyz file columns 1 and 2.
        """
        filename = "tests/files/read_waqua_xyz_test.xyc"
        col = (1, 2)
        data = read_waqua_xyz(filename, col)
        datar = np.array([[2.0, 3.0], [5.0, 6.0], [8.0, 9.0], [11.0, 12.0]])
        print("data reference: ", datar)
        print("data read     : ", data)
        assert np.shape(data) == (4, 2)
        assert (data == datar).all() == True


class TestWriteSimonaBox:
    def test_write_simona_box_01(self):
        """
        Write small SIMONA BOX file.
        """
        filename = "test.box"
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        firstm = 0
        firstn = 0
        write_simona_box(filename, data, firstm, firstn)
        all_lines = open(filename, "r").read().splitlines()
        all_lines_ref = [
            '      BOX MNMN=(   1,    1,    3,    3), VARIABLE_VAL=',
            '          1.000       2.000       3.000',
            '          4.000       5.000       6.000',
            '          7.000       8.000       9.000',
        ]
        assert all_lines == all_lines_ref

    def test_write_simona_box_02(self):
        """
        Write small SIMONA BOX file with offset.
        """
        filename = "test.box"
        data = np.array(
            [[0, 0, 0, 0, 0], [0, 0, 1, 2, 3], [0, 0, 4, 5, 6], [0, 0, 7, 8, 9]]
        )
        firstm = 1
        firstn = 2
        write_simona_box(filename, data, firstm, firstn)
        all_lines = open(filename, "r").read().splitlines()
        all_lines_ref = [
            '      BOX MNMN=(   2,    3,    4,    5), VARIABLE_VAL=',
            '          1.000       2.000       3.000',
            '          4.000       5.000       6.000',
            '          7.000       8.000       9.000',
        ]
        assert all_lines == all_lines_ref

    def test_write_simona_box_03(self):
        """
        Write large SIMONA BOX file.
        """
        filename = "test.box"
        data = np.zeros((15, 15))
        firstm = 0
        firstn = 0
        write_simona_box(filename, data, firstm, firstn)
        all_lines = open(filename, "r").read().splitlines()
        all_lines_ref = ['      BOX MNMN=(   1,    1,   15,   10), VARIABLE_VAL=']
        all_lines_ref.extend(
            [
                '          0.000       0.000       0.000       0.000       0.000       0.000       0.000       0.000       0.000       0.000'
            ]
            * 15
        )
        all_lines_ref.extend(['      BOX MNMN=(   1,   11,   15,   15), VARIABLE_VAL='])
        all_lines_ref.extend(
            ['          0.000       0.000       0.000       0.000       0.000'] * 15
        )
        self.maxDiff = None
        assert all_lines == all_lines_ref


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

    def test_get_km_bounds(self, config: ConfigParser):
        """Test retrieving km bounds."""
        config_file = ConfigFile(config, "tests/data/erosion/test.cfg")
        start, end = config_file.get_km_bounds()
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

    def test_get_bank_lines(self, config: ConfigParser, fs: FakeFilesystem):
        """Test retrieving bank lines."""
        config["General"]["BankLine"] = "bankfile"
        config = ConfigFile(config, "tests/data/erosion/test.cfg")

        fs.create_file(
            "inputs/bankfile_1.xyc",
            contents="0.0 0.0\n1.0 1.0\n2.0 2.0\n3.0 3.0\n4.0 4.0\n",
        )

        bank_lines = config.get_bank_lines("inputs")

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

    def test_get_xy_km(self):
        """Test retrieving x and y coordinates."""
        config = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
        mock_linestring = LineString([(0, 0, 1), (1, 1, 2), (2, 2, 3)])

        with patch("dfastio.xyc.models.XYCModel.read", return_value=mock_linestring):
            xykm = config.get_xy_km()

        assert xykm.wkt == 'LINESTRING Z (0 0 1, 1 1 2, 2 2 3)'

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
    # def test_initialization(self):
    #     path = "tests/data/erosion/meuse_manual.cfg"
    #     config_file = ConfigFile.read(path)
    #     river_data = RiverData(config_file)
    #     assert isinstance(river_data.config_file, ConfigFile)
    #     assert river_data.num_search_lines == 2
    #     assert river_data.start_station == 123.0
    #     assert river_data.end_station == 128.0
    #     assert isinstance(river_data.masked_profile, LineString)
    #     assert isinstance(river_data.profile, LineString)
    #     assert isinstance(river_data.masked_profile_coords, np.ndarray)
    #     assert river_data.masked_profile_coords.shape == (251, 3)

    @pytest.fixture
    def river_data(self):
        """Fixture to create a RiverData instance with mock data."""
        config_file = ConfigFile.read("tests/data/erosion/meuse_manual.cfg")
        river_data = RiverData(config_file)

        river_data._profile = LineString(
            [
                (0, 0, 0),
                (1, 1, 1),
                (2, 2, 2),
                (3, 3, 3),
                (4, 4, 4),
            ]
        )

        return river_data

    def test_mask_profile(self, river_data: RiverData):
        """Test the mask_profile method of the RiverData class."""
        river_data._station_bounds = (1.5, 3.5)
        masked_profile = river_data.mask_profile()

        expected_profile = LineString(
            [
                (1.5, 1.5, 1.5),
                (2, 2, 2),
                (3, 3, 3),
                (3.5, 3.5, 3.5),
            ]
        )

        assert isinstance(masked_profile, LineString)
        assert masked_profile.equals(expected_profile)

    def test_mask_profile_out_of_bounds(self, river_data: RiverData):
        """Test the mask_profile method for out-of-bounds station bounds."""
        # Case 1: Lower bound is larger than the maximum chainage
        river_data._station_bounds = (5.0, 6.0)
        with pytest.raises(
            ValueError,
            match="Lower chainage bound 5.0 is larger than the maximum chainage 4.0 available",
        ):
            river_data.mask_profile()

        # Case 2: Lower bound is slightly smaller than the minimum chainage
        river_data._station_bounds = (-0.2, 3.0)
        with pytest.raises(
            ValueError,
            match="Lower chainage bound -0.2 is smaller than the minimum chainage 0.0 available",
        ):
            river_data.mask_profile()

        # Case 3: Upper bound is smaller than the minimum chainage
        river_data._station_bounds = (0.0, -0.5)
        with pytest.raises(
            ValueError,
            match="Upper chainage bound -0.5 is smaller than the minimum chainage 0.0 available",
        ):
            river_data.mask_profile()

        # Case 4: Upper bound is larger than the maximum chainage
        river_data._station_bounds = (0.0, 5.0)
        with pytest.raises(
            ValueError,
            match="Upper chainage bound 5.0 is larger than the maximum chainage 4.0 available",
        ):
            river_data.mask_profile()

    def test_mask_profile_end_i_none(self, river_data: RiverData):
        """Test the mask_profile method where end_i is None."""
        # Mock the profile with a simple LineString
        river_data._profile = LineString(
            [
                (0, 0, 0.0),  # First chainage value
                (1, 1, 1.0),  # Second chainage value
                (2, 2, 2.0),
                (3, 3, 3.0),
            ]
        )

        # Set station_bounds such that:
        # - Lower bound (station_bounds[0]) is between two chainage values (triggers x0 interpolation).
        # - Upper bound (station_bounds[1]) is greater than the maximum chainage (end_i = None).
        river_data._station_bounds = (0.5, 3.05)

        # Call the mask_profile method
        masked_profile = river_data.mask_profile()

        # Expected result: The start point (0.5, 0.5, 0.5) is interpolated, and the upper bound is ignored.
        expected_profile = LineString(
            [
                (0.5, 0.5, 0.5),  # Interpolated start point
                (1, 1, 1.0),
                (2, 2, 2.0),
                (3, 3, 3.0),
            ]
        )

        # Assertions
        assert isinstance(masked_profile, LineString)
        assert masked_profile.equals(
            expected_profile
        ), "Masked profile does not match the expected result."
