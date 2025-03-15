import configparser
import os
import platform
import sys
from contextlib import contextmanager
from io import StringIO

import netCDF4 as nc
import numpy as np
import pytest

import dfastbe.io as df_io


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
        assert df_io.load_program_texts("tests/files/messages.UK.ini") == None


class Test_log_text:
    def test_log_text_01(self):
        """
        Testing standard output of a single text without expansion.
        """
        key = "confirm"
        with captured_output() as (out, err):
            df_io.log_text(key)
        outstr = out.getvalue().splitlines()
        strref = ['Confirm using "y" ...', ""]
        assert outstr == strref

    def test_log_text_02(self):
        """
        Testing standard output of a repeated text without expansion.
        """
        key = ""
        nr = 3
        with captured_output() as (out, err):
            df_io.log_text(key, repeat=nr)
        outstr = out.getvalue().splitlines()
        strref = ["", "", ""]
        assert outstr == strref

    def test_log_text_03(self):
        """
        Testing standard output of a text with expansion.
        """
        key = "reach"
        dict = {"reach": "ABC"}
        with captured_output() as (out, err):
            df_io.log_text(key, dict=dict)
        outstr = out.getvalue().splitlines()
        strref = ["The measure is located on reach ABC"]
        assert outstr == strref

    def test_log_text_04(self):
        """
        Testing file output of a text with expansion.
        """
        key = "reach"
        dict = {"reach": "ABC"}
        filename = "test.log"
        with open(filename, "w") as f:
            df_io.log_text(key, dict=dict, file=f)
        all_lines = open(filename, "r").read().splitlines()
        strref = ["The measure is located on reach ABC"]
        assert all_lines == strref


class Test_get_filename:
    def test_get_filename_01(self):
        """
        Testing get_filename wrapper for get_text.
        """
        assert df_io.get_filename("report.out") == "report.txt"


class Test_get_text:
    def test_get_text_01(self):
        """
        Testing get_text: key not found.
        """
        assert df_io.get_text("@") == ["No message found for @"]

    def test_get_text_02(self):
        """
        Testing get_text: empty line key.
        """
        assert df_io.get_text("") == [""]

    def test_get_text_03(self):
        """
        Testing get_text: "confirm" key.
        """
        assert df_io.get_text("confirm") == ['Confirm using "y" ...', ""]


class Test_write_config:
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
        df_io.write_config(filename, config)
        all_lines = open(filename, "r").read().splitlines()
        all_lines_ref = [
            "[G 1]",
            "  k 1     = V 1",
            "",
            "[Group 2]",
            "  k1      = 1.0 0.1 0.0 0.01",
            "  k2      = 2.0 0.2 0.02 0.0",
            "",
            "[Group 3]",
            "  longkey = 3",
        ]
        assert all_lines == all_lines_ref


class Test_read_fm_map:
    def test_read_fm_map_01(self):
        """
        Testing read_fm_map: x coordinates of the faces.
        """
        filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        varname = "x"
        # location = "face"
        datac = df_io.read_fm_map(filename, varname)
        dataref = 41.24417604888325
        assert datac[1] == dataref

    def test_read_fm_map_02(self):
        """
        Testing read_fm_map: y coordinates of the edges.
        """
        filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        varname = "y"
        location = "edge"
        datac = df_io.read_fm_map(filename, varname, location)
        dataref = 7059.853000358055
        assert datac[1] == dataref

    def test_read_fm_map_03(self):
        """
        Testing read_fm_map: face node connectivity.
        """
        filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        varname = "face_node_connectivity"
        datac = df_io.read_fm_map(filename, varname)
        dataref = 2352
        assert datac[-1][1] == dataref

    def test_read_fm_map_04(self):
        """
        Testing read_fm_map: variable by standard name.
        """
        filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        varname = "sea_floor_depth_below_sea_surface"
        datac = df_io.read_fm_map(filename, varname)
        dataref = 3.894498393076889
        assert datac[1] == dataref

    def test_read_fm_map_05(self):
        """
        Testing read_fm_map: variable by long name.
        """
        filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        varname = "Water level"
        datac = df_io.read_fm_map(filename, varname)
        dataref = 3.8871328177527262
        assert datac[1] == dataref

    def test_read_fm_map_06(self):
        """
        Testing read_fm_map: variable by long name.
        """
        filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        varname = "water level"
        with pytest.raises(Exception) as cm:
            datac = df_io.read_fm_map(filename, varname)
        assert (
            str(cm.value) == 'Expected one variable for "water level", but obtained 0.'
        )


class Test_get_mesh_and_facedim_names:
    def test_get_mesh_and_facedim_names_01(self):
        """
        Testing get_mesh_and_facedim_names.
        """
        filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        name_and_dim = df_io.get_mesh_and_facedim_names(filename)
        assert name_and_dim == ("mesh2d", "mesh2d_nFaces")


class Test_copy_ugrid:
    def test_copy_ugrid_01(self):
        """
        Testing copy_ugrid (depends on copy_var).
        """
        src_filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        dst_filename = "test.nc"
        meshname, facedim = df_io.get_mesh_and_facedim_names(src_filename)
        df_io.copy_ugrid(src_filename, meshname, dst_filename)
        #
        varname = "face_node_connectivity"
        datac = df_io.read_fm_map(dst_filename, varname)
        dataref = 2352
        assert datac[-1][1] == dataref


class Test_copy_var:
    def test_copy_var_01(self):
        """
        Testing copy_var.
        """
        src_filename = "tests/files/e02_f001_c011_simplechannel_map.nc"
        dst_filename = "test.nc"

        with nc.Dataset(src_filename) as src:
            with nc.Dataset(dst_filename, "a") as dst:
                df_io.copy_var(src, "mesh2d_s1", dst)

        varname = "sea_surface_height"
        datac = df_io.read_fm_map(dst_filename, varname)
        dataref = 3.8871328177527262
        assert datac[1] == dataref


class Test_ugrid_add:
    def test_ugrid_add_01(self):
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
        df_io.ugrid_add(dst_filename, varname, ldata, meshname, facedim, long_name)
        #
        datac = df_io.read_fm_map(dst_filename, long_name)
        assert datac[1] == ldata[1]


class Test_read_waqua_xyz:
    def test_read_waqua_xyz_01(self):
        """
        Read WAQUA xyz file default column 2.
        """
        filename = "tests/files/read_waqua_xyz_test.xyc"
        data = df_io.read_waqua_xyz(filename)
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
        data = df_io.read_waqua_xyz(filename, col)
        datar = np.array([[2.0, 3.0], [5.0, 6.0], [8.0, 9.0], [11.0, 12.0]])
        print("data reference: ", datar)
        print("data read     : ", data)
        assert np.shape(data) == (4, 2)
        assert (data == datar).all() == True


class Test_write_simona_box:
    def test_write_simona_box_01(self):
        """
        Write small SIMONA BOX file.
        """
        filename = "test.box"
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        firstm = 0
        firstn = 0
        df_io.write_simona_box(filename, data, firstm, firstn)
        all_lines = open(filename, "r").read().splitlines()
        all_lines_ref = [
            "      BOX MNMN=(   1,    1,    3,    3), VARIABLE_VAL=",
            "          1.000       2.000       3.000",
            "          4.000       5.000       6.000",
            "          7.000       8.000       9.000",
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
        df_io.write_simona_box(filename, data, firstm, firstn)
        all_lines = open(filename, "r").read().splitlines()
        all_lines_ref = [
            "      BOX MNMN=(   2,    3,    4,    5), VARIABLE_VAL=",
            "          1.000       2.000       3.000",
            "          4.000       5.000       6.000",
            "          7.000       8.000       9.000",
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
        df_io.write_simona_box(filename, data, firstm, firstn)
        all_lines = open(filename, "r").read().splitlines()
        all_lines_ref = ["      BOX MNMN=(   1,    1,   15,   10), VARIABLE_VAL="]
        all_lines_ref.extend(
            [
                "          0.000       0.000       0.000       0.000       0.000       0.000       0.000       0.000       0.000       0.000"
            ]
            * 15
        )
        all_lines_ref.extend(["      BOX MNMN=(   1,   11,   15,   15), VARIABLE_VAL="])
        all_lines_ref.extend(
            ["          0.000       0.000       0.000       0.000       0.000"] * 15
        )
        self.maxDiff = None
        assert all_lines == all_lines_ref


class Test_absolute_path:

    @pytest.mark.skipif(
        platform.system() != "Windows", reason="it will be completely changed"
    )
    def test_absolute_path_01(self):
        """
        Convert absolute path into relative path using relative_path (Windows).
        """
        rootdir = "d:" + os.sep + "some" + os.sep + "dir"
        afile = f"d:{os.sep}some{os.sep}other{os.sep}dir{os.sep}file.ext"
        rfile = ".." + os.sep + "other" + os.sep + "dir" + os.sep + "file.ext"
        assert df_io.absolute_path(rootdir, rfile) == afile

    @pytest.mark.skipif(
        platform.system() != "Windows", reason="it will be completely changed"
    )
    def test_absolute_path_02(self):
        """
        Empty string should not be adjusted by relative_path.
        """
        rootdir = "d:" + os.sep + "some" + os.sep + "dir"
        file = ""
        assert df_io.absolute_path(rootdir, file) == file

    def test_absolute_path_03(self):
        """
        If path on different drive, it shouldn't be adjusted by relative_path (Windows).
        """
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
        assert df_io.absolute_path(rootdir, file) == file

    def test_absolute_path_04(self):
        """
        Convert absolute path into relative path using relative_path (Linux).
        """
        rootdir = os.sep + "some" + os.sep + "dir"
        afile = (
            os.sep + "some" + os.sep + "other" + os.sep + "dir" + os.sep + "file.ext"
        )
        rfile = ".." + os.sep + "other" + os.sep + "dir" + os.sep + "file.ext"
        assert df_io.absolute_path(rootdir, rfile) == afile


class Test_relative_path:
    def test_relative_path_01(self):
        """
        Convert absolute path into relative path using relative_path (Windows).
        """
        rootdir = "d:" + os.sep + "some" + os.sep + "dir"
        afile = (
            "d:"
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
        assert df_io.relative_path(rootdir, afile) == rfile

    def test_relative_path_02(self):
        """
        Empty string should not be adjusted by relative_path.
        """
        rootdir = "d:" + os.sep + "some" + os.sep + "dir"
        file = ""
        assert df_io.relative_path(rootdir, file) == file

    def test_relative_path_03(self):
        """
        If path on different drive, it shouldn't be adjusted by relative_path (Windows).
        """
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
        assert df_io.relative_path(rootdir, file) == file

    def test_relative_path_04(self):
        """
        Convert absolute path into relative path using relative_path (Linux).
        """
        rootdir = os.sep + "some" + os.sep + "dir"
        afile = (
            os.sep + "some" + os.sep + "other" + os.sep + "dir" + os.sep + "file.ext"
        )
        rfile = ".." + os.sep + "other" + os.sep + "dir" + os.sep + "file.ext"
        assert df_io.relative_path(rootdir, afile) == rfile


# TODO: read_xyc
# TODO: write_km_eroded_volumes
# TODO: read_config
# TODO: upgrade_config
# TODO: movepar
# TODO: config_get_xykm
# TODO: clip_chainage_path
# TODO: config_get_bank_guidelines
# TODO: config_get_bank_search_distances
# TODO: config_get_simfile
# TODO: config_get_range
# TODO: config_get_bool
# TODO: config_get_int
# TODO: config_get_float
# TODO: config_get_str
# TODO: config_get_parameter
# TODO: get_kmval
# TODO: read_simdata
