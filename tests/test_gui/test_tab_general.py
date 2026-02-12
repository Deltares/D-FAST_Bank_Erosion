import pytest
import configparser
from pathlib import Path

from unittest.mock import patch
from PySide6.QtWidgets import QCheckBox, QLineEdit, QLabel
from PySide6.QtGui import QDoubleValidator

from dfastbe.gui.tabs.general import GeneralTab, update_plotting
from dfastbe.gui.state_management import StateStore
from dfastbe.gui.utils import validator
from dfastbe.gui.tabs.main_components import menu_save_configuration


SCENARIOS = [
    (False, False, False, {
        "savePlots": False, "savePlotsEdit": False, "saveZoomPlots": False,
        "saveZoomPlotsEdit": False, "zoomPlotsRangeTxt": False,
        "zoomPlotsRangeEdit": False,
        "figureDir": False, "figureDirEdit": False, "figureDirEditFile": False,
        "closePlots": False, "closePlotsEdit": False
    }),
    (True, False, False, {
        "savePlots": True, "savePlotsEdit": True, "saveZoomPlots": False,
        "saveZoomPlotsEdit": False, "zoomPlotsRangeTxt": False,
        "zoomPlotsRangeEdit": False,
        "figureDir": False, "figureDirEdit": False, "figureDirEditFile": False,
        "closePlots": True, "closePlotsEdit": True
    }),
    (True, True, False, {
        "savePlots": True, "savePlotsEdit": True, "saveZoomPlots": True,
        "saveZoomPlotsEdit": True, "zoomPlotsRangeTxt": False,
        "zoomPlotsRangeEdit": False,
        "figureDir": True, "figureDirEdit": True, "figureDirEditFile": True,
        "closePlots": True, "closePlotsEdit": True
    }),
    (True, True, True, {
        "savePlots": True, "savePlotsEdit": True, "saveZoomPlots": True,
        "saveZoomPlotsEdit": True, "zoomPlotsRangeTxt": True,
        "zoomPlotsRangeEdit": True,
        "figureDir": True, "figureDirEdit": True, "figureDirEditFile": True,
        "closePlots": True, "closePlotsEdit": True
    }),
    (True, False, True, {
        "savePlots": True, "savePlotsEdit": True, "saveZoomPlots": False,
        "saveZoomPlotsEdit": False, "zoomPlotsRangeTxt": False,
        "zoomPlotsRangeEdit": False,
        "figureDir": False, "figureDirEdit": False, "figureDirEditFile": False,
        "closePlots": True, "closePlotsEdit": True
    }),
    (False, False, True, {
        "savePlots": False, "savePlotsEdit": False, "saveZoomPlots": False,
        "saveZoomPlotsEdit": False, "zoomPlotsRangeTxt": False,
        "zoomPlotsRangeEdit": False,
        "figureDir": False, "figureDirEdit": False, "figureDirEditFile": False,
        "closePlots": False, "closePlotsEdit": False
    }),
]

IDS = [
    "none_enabled",
    "only_make_lots_enabled",
    "make_plots_and_save_plots_enabled",
    "all_enabled",
    "make_lots_and_save_zoom_enabled",
    "only_save_zoom_enabled",
]


@pytest.fixture
def initialize_general_tab(setup_tab_state, qtbot):
    general_tab = GeneralTab(
        setup_tab_state['tabs'],
        setup_tab_state['window']
    )
    return general_tab

class TestGeneralTab:
    def test_widgets_registered(self, qtbot, initialize_general_tab):
        general_tab = initialize_general_tab
        general_tab.create()
        state = StateStore.instance()
        # Check all important widgets are registered
        for key in [
            "saveZoomPlotsEdit", "zoomPlotsRangeEdit", "makePlotsEdit", "savePlotsEdit",
            "closePlotsEdit", "debugOutputEdit", "startRange", "endRange", "bankFileName"
        ]:
            assert key in state

    def test_widget_defaults(self, qtbot, initialize_general_tab):
        general_tab = initialize_general_tab
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

    def test_zoom_plots_range_edit_has_positive_real_validator(
            self,
            qtbot,
            initialize_general_tab
    ):
        general_tab = initialize_general_tab
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
        SCENARIOS,
        ids=IDS
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


class TestGuiBehaviourGeneralTab:

    @pytest.mark.parametrize(
        "make_plots,save_plots,save_zoom_plots,expected",
        SCENARIOS,
        ids=IDS
    )
    def test_general_tab_integration_enabling_logic(
            self,
            qtbot,
            setup_tab_state,
            initialize_general_tab,
            make_plots,
            save_plots,
            save_zoom_plots,
            expected):
        """
        Parametrized integration test: Build GeneralTab with real widgets, simulate
        toggling checkboxes, and assert GUI enable/disable state matches update_plotting
        logic.
        """
        window = setup_tab_state['window']
        tabs = setup_tab_state['tabs']
        qtbot.addWidget(window)
        qtbot.addWidget(tabs)
        general_tab = initialize_general_tab
        general_tab.create()
        state = StateStore.instance()
        state["makePlotsEdit"].setChecked(make_plots)
        state["savePlotsEdit"].setChecked(save_plots)
        state["saveZoomPlotsEdit"].setChecked(save_zoom_plots)
        qtbot.wait(10)
        update_plotting()
        qtbot.wait(10)
        for key, should_be_enabled in expected.items():
            widget = state[key]
            assert widget.isEnabled() is should_be_enabled, (
                f"{key} enabled={widget.isEnabled()} expected={should_be_enabled} "
                f"for makePlotsEdit={make_plots}, savePlotsEdit={save_plots}, "
                f"saveZoomPlotsEdit={save_zoom_plots}"
            )


def test_menu_save_configuration_saves_general_tab_state(
        qtbot,
        setup_tab_state,
        initialize_general_tab,
        tmp_path,
        create_widget_configuration):
    """
    Alters widgets in the General tab, calls menu_save_configuration, and checks
    that the saved config file contains the correct state.
    """
    window = setup_tab_state['window']
    tabs = setup_tab_state['tabs']
    qtbot.addWidget(window)
    qtbot.addWidget(tabs)
    general_tab = initialize_general_tab
    general_tab.create()
    state = create_widget_configuration
    # Set only the values needed for this test
    state["makePlotsEdit"].setChecked(True)
    state["savePlotsEdit"].setChecked(True)
    state["saveZoomPlotsEdit"].setChecked(False)
    state["zoomPlotsRangeEdit"].setText("2.5")
    state["figureDirEdit"].setText("my_figures")
    state["closePlotsEdit"].setChecked(True)
    state["debugOutputEdit"].setChecked(False)
    state["bankFileName"].setText("bankfile.txt")
    state["startRange"].setText("10")
    state["endRange"].setText("20")
    state["chainFileEdit"].setText("chain.km")
    state["bankDirEdit"].setText("banks/")
    # Ensure StateStore uses this widget state
    StateStore._instance = state
    # Patch QFileDialog.getSaveFileName to return a temp file path
    save_path = tmp_path / "saved_config.cfg"
    with patch("PySide6.QtWidgets.QFileDialog.getSaveFileName", return_value=(str(save_path), "")):
        menu_save_configuration()
    # Read the saved config and check values
    config = configparser.ConfigParser()
    config.optionxform = str  # preserve case
    config.read(str(save_path))
    assert config["General"]["BankFile"] == "bankfile.txt"
    assert config["General"]["Plotting"] == "True"
    assert config["General"]["SavePlots"] == "True"
    assert config["General"]["SaveZoomPlots"] == "False"
    assert config["General"]["ZoomStepKM"] == "2.5"
    assert Path(config["General"]["FigureDir"]).name == "my_figures"
    assert config["General"]["ClosePlots"] == "True"
    assert Path(config["General"]["RiverKM"]).name == "chain.km"
    assert config["General"]["Boundaries"] == "10:20"
    assert Path(config["General"]["BankDir"]).name == "banks"
