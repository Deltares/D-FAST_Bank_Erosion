import pytest
from unittest.mock import patch
from PySide6.QtWidgets import QApplication, QTabWidget, QMainWindow, QCheckBox, QLineEdit, QLabel
from PySide6.QtGui import QDoubleValidator
from dfastbe.gui.tabs.general import GeneralTab, update_plotting
from dfastbe.gui.state_management import StateStore
from dfastbe.gui.utils import validator

class TestGeneralTab:
    @pytest.fixture(autouse=True)
    def setup_method(self, qapp, qtbot):
        # qapp fixture ensures QApplication exists (do not create manually)
        StateStore._instance = None
        self.state = StateStore.initialize()
        self.window = QMainWindow()
        self.tabs = QTabWidget(self.window)
        yield
        StateStore._instance = None

    def test_widgets_registered(self, qtbot):
        general_tab = GeneralTab(self.tabs, self.window)
        general_tab.create()
        state = StateStore.instance()
        # Check all important widgets are registered
        for key in [
            "saveZoomPlotsEdit", "zoomPlotsRangeEdit", "makePlotsEdit", "savePlotsEdit",
            "closePlotsEdit", "debugOutputEdit", "startRange", "endRange", "bankFileName"
        ]:
            assert key in state

    def test_widget_defaults(self, qtbot):
        general_tab = GeneralTab(self.tabs, self.window)
        general_tab.create()
        state = StateStore.instance()
        # Check saveZoomPlotsEdit is a QCheckBox and unchecked
        save_zoom_checkbox = state["saveZoomPlotsEdit"]
        assert isinstance(save_zoom_checkbox, QCheckBox)
        assert not save_zoom_checkbox.isChecked()
        # Check zoomPlotsRangeEdit is a QLineEdit and has default text "1.0"
        zoom_range_edit = state["zoomPlotsRangeEdit"]
        assert isinstance(zoom_range_edit, QLineEdit)
        assert zoom_range_edit.text() == "1.0"

    def test_zoomplotsrangeedit_has_positive_real_validator(self, qtbot):
        general_tab = GeneralTab(self.tabs, self.window)
        general_tab.create()
        state = StateStore.instance()
        zoom_range_edit = state["zoomPlotsRangeEdit"]
        assert isinstance(zoom_range_edit, QLineEdit)
        edit_validator = zoom_range_edit.validator()
        # Should be a QDoubleValidator and have bottom set to 0 (positive real)
        assert isinstance(edit_validator, QDoubleValidator)
        # The bottom value should be 0 for positive real
        assert edit_validator.bottom() == 0
        # Should be the same type as returned by validator("positive_real")
        ref_validator = validator("positive_real")
        assert type(edit_validator) is type(ref_validator)


class TestUpdatePlotting:
    @pytest.fixture(autouse=True)
    def setup_widgets_and_state(self, qapp):
        # Patch StateStore.instance to return our test state dict
        self.state = {}
        # Helper to create a widget with setEnabled and isEnabled
        def make_checkbox(checked=False):
            cb = QCheckBox()
            cb.setChecked(checked)
            cb.setEnabled(False)
            return cb
        def make_label():
            lbl = QLabel()
            lbl.setEnabled(False)
            return lbl
        def make_lineedit():
            le = QLineEdit()
            le.setEnabled(False)
            return le
        # Set up all widgets as disabled initially
        self.state["makePlotsEdit"] = make_checkbox(checked=False)
        self.state["savePlots"] = make_label()
        self.state["savePlotsEdit"] = make_checkbox(checked=False)
        self.state["saveZoomPlots"] = make_label()
        self.state["saveZoomPlotsEdit"] = make_checkbox(checked=False)
        self.state["zoomPlotsRangeTxt"] = make_label()
        self.state["zoomPlotsRangeEdit"] = make_lineedit()
        self.state["figureDir"] = make_label()
        self.state["figureDirEdit"] = make_lineedit()
        self.state["figureDirEditFile"] = make_label()
        self.state["closePlots"] = make_label()
        self.state["closePlotsEdit"] = make_checkbox(checked=False)
        patcher = patch.object(StateStore, "instance", return_value=self.state)
        self._patcher = patcher
        patcher.start()
        yield
        patcher.stop()

    @pytest.mark.parametrize(
        "make_plots,save_plots,save_zoom_plots,expected",
        [
            # Only makePlotsEdit checked
            (
                True, False, False,
                {
                    "savePlots": True, "savePlotsEdit": True,
                    "saveZoomPlots": False, "saveZoomPlotsEdit": False,
                    "zoomPlotsRangeTxt": False, "zoomPlotsRangeEdit": False,
                    "figureDir": False, "figureDirEdit": False,
                    "figureDirEditFile": False, "closePlots": True,
                    "closePlotsEdit": True
                }
            ),
            # makePlotsEdit and savePlotsEdit checked
            (
                True, True, False,
                {
                    "savePlots": True, "savePlotsEdit": True,
                    "saveZoomPlots": True, "saveZoomPlotsEdit": True,
                    "zoomPlotsRangeTxt": False, "zoomPlotsRangeEdit": False,
                    "figureDir": True, "figureDirEdit": True,
                    "figureDirEditFile": True, "closePlots": True,
                    "closePlotsEdit": True
                }
            ),
            # makePlotsEdit and saveZoomPlotsEdit checked
            # (should not enable zoom fields unless savePlotsEdit is also checked)
            (
                True, False, True,
                {
                    "savePlots": True, "savePlotsEdit": True,
                    "saveZoomPlots": False, "saveZoomPlotsEdit": False,
                    "zoomPlotsRangeTxt": False, "zoomPlotsRangeEdit": False,
                    "figureDir": False, "figureDirEdit": False,
                    "figureDirEditFile": False, "closePlots": True,
                    "closePlotsEdit": True
                }
            ),
            # makePlotsEdit, savePlotsEdit, and saveZoomPlotsEdit all checked
            (
                True, True, True,
                {
                    "savePlots": True, "savePlotsEdit": True,
                    "saveZoomPlots": True, "saveZoomPlotsEdit": True,
                    "zoomPlotsRangeTxt": True, "zoomPlotsRangeEdit": True,
                    "figureDir": True, "figureDirEdit": True,
                    "figureDirEditFile": True, "closePlots": True,
                    "closePlotsEdit": True
                }
            ),
            # None checked
            (
                False, False, False,
                {
                    "savePlots": False, "savePlotsEdit": False,
                    "saveZoomPlots": False, "saveZoomPlotsEdit": False,
                    "zoomPlotsRangeTxt": False, "zoomPlotsRangeEdit": False,
                    "figureDir": False, "figureDirEdit": False,
                    "figureDirEditFile": False, "closePlots": False,
                    "closePlotsEdit": False
                }
            ),
            # Only saveZoomPlotsEdit checked
            # (should have no effect if others are False)
            (
                False, False, True,
                {
                    "savePlots": False, "savePlotsEdit": False,
                    "saveZoomPlots": False, "saveZoomPlotsEdit": False,
                    "zoomPlotsRangeTxt": False, "zoomPlotsRangeEdit": False,
                    "figureDir": False, "figureDirEdit": False,
                    "figureDirEditFile": False, "closePlots": False,
                    "closePlotsEdit": False
                }
            ),
        ],
        ids=[
            "only_makePlotsEdit_checked",
            "makePlotsEdit_and_savePlotsEdit_checked",
            "makePlotsEdit_and_saveZoomPlotsEdit_checked",
            "all_checked",
            "none_checked",
            "only_saveZoomPlotsEdit_checked",
        ]
    )
    def test_plotting_enabling_logic(
        self, make_plots, save_plots, save_zoom_plots, expected
    ):
        """Test update_plotting widget enabling logic for various checkbox combinations."""
        self.state["makePlotsEdit"].setChecked(make_plots)
        self.state["savePlotsEdit"].setChecked(save_plots)
        self.state["saveZoomPlotsEdit"].setChecked(save_zoom_plots)
        update_plotting()
        for key, should_be_enabled in expected.items():
            assert self.state[key].isEnabled() is should_be_enabled, (
                f"{key} enabled={self.state[key].isEnabled()} "
                f"expected={should_be_enabled} "
                f"for makePlotsEdit={make_plots}, "
                f"savePlotsEdit={save_plots}, "
                f"saveZoomPlotsEdit={save_zoom_plots}"
            )
