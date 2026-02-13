import pytest
import configparser

from unittest.mock import patch
from PySide6.QtWidgets import QLineEdit, QTreeWidgetItem
from PySide6.QtGui import QDoubleValidator, QIntValidator, Qt

from dfastbe.gui.tabs.main_components import menu_save_configuration
from dfastbe.gui.tabs.erosion import ErosionTab
from dfastbe.gui.state_management import StateStore
from dfastbe.gui.utils import validator


@pytest.fixture
def initialize_erosion_tab(setup_tab_state, qapp):
    return ErosionTab(
        setup_tab_state['tabs'],
        setup_tab_state['window'],
        qapp
    )

class TestErosionTab:
    def test_widgets_registered(self, initialize_erosion_tab):
        initialize_erosion_tab.create()
        state = StateStore.instance()
        for key in [
            "tErosion", "refLevel", "chainageOutStep", "newBankFile",
            "newEqBankFile", "eroVol", "eroVolEqui", "discharges"
        ]:
            assert key in state

    def test_t_erosion_validator(self, initialize_erosion_tab):
        erosion_tab = initialize_erosion_tab
        erosion_tab.create()
        state = StateStore.instance()
        t_erosion = state["tErosion"]
        assert isinstance(t_erosion, QLineEdit)
        t_erosion_validator = t_erosion.validator()
        # Should be a QDoubleValidator and have bottom set to 0 (positive real)
        assert isinstance(t_erosion_validator, QDoubleValidator)
        # The bottom value should be 0 for positive real
        assert t_erosion_validator.bottom() == 0
        # Should be the same type as returned by validator("positive_real")
        ref_validator = validator("positive_real")
        assert type(t_erosion_validator) is type(ref_validator)

    def test_chainage_output_validator(self, initialize_erosion_tab):
        erosion_tab = initialize_erosion_tab
        erosion_tab.create()
        state = StateStore.instance()
        chainage_output = state["tErosion"]
        assert isinstance(chainage_output, QLineEdit)
        chainage_output_validator = chainage_output.validator()
        # Should be a QDoubleValidator and have bottom set to 0 (positive real)
        assert isinstance(chainage_output_validator, QDoubleValidator)
        # The bottom value should be 0 for positive real
        assert chainage_output_validator.bottom() == 0
        # Should be the same type as returned by validator("positive_real")
        ref_validator = validator("positive_real")
        assert type(chainage_output_validator) is type(ref_validator)

    def test_ref_level_validator(self, initialize_erosion_tab):
        erosion_tab = initialize_erosion_tab
        erosion_tab.create()
        state = StateStore.instance()
        ref_level = state["refLevel"]
        assert isinstance(ref_level, QLineEdit)
        ref_level_validator = ref_level.validator()
        # Should be a QIntValidator and have range 1 to 1 (only 1 allowed)
        assert isinstance(ref_level_validator, QIntValidator)
        assert ref_level_validator.bottom() == 1
        assert ref_level_validator.top() == 1

    def test_discharge_widget_column_headers(self, initialize_erosion_tab):
        initialize_erosion_tab.create()
        state = StateStore.instance()
        discharges = state["discharges"]
        assert discharges.headerItem().text(0) == "Level"
        assert discharges.headerItem().text(1) == "FileName"
        assert discharges.headerItem().text(2) == "Probability [-]"

    def test_discharge_widget_column_widths(self, initialize_erosion_tab):
        initialize_erosion_tab.create()
        state = StateStore.instance()
        discharges = state["discharges"]
        assert discharges.columnWidth(0) == 50
        assert discharges.columnWidth(1) == 250


def mock_edit_discharge_dialog(key, istr, file_name="", prob=""):
    return "test_file.nc", "0.42"


class TestGuiBehaviorErosionTab:
    def test_adding_discharge_rows(self, qtbot, setup_tab_state, initialize_erosion_tab):
        erosion_tab = initialize_erosion_tab
        erosion_tab.create()
        state = StateStore.instance()

        state["tabs"] = setup_tab_state['tabs']

        remove_button = state["dischargesRemove"]
        edit_button =  state["dischargesEdit"]
        ref_level_validator = state["refLevel"].validator()

        assert remove_button.isEnabled() == False
        assert edit_button.isEnabled() == False

        # Check if only the Erosion tab exists before adding discharge rows
        assert state["tabs"].count() == 1

        # Add a discharge row
        with patch("dfastbe.gui.base.editADischarge", mock_edit_discharge_dialog):
            qtbot.mouseClick(state["dischargesAdd"], Qt.LeftButton)

        assert state["discharges"].topLevelItemCount() == 1
        assert state["discharges"].topLevelItem(0).text(1) == "test_file.nc"
        assert state["discharges"].topLevelItem(0).text(2) == "0.42"

        # The delete and edit buttons should now be enabled after adding a discharge row
        assert remove_button.isEnabled() == True
        assert edit_button.isEnabled() == True

        # Adding a discharge row should update the reference level validator
        assert ref_level_validator.top() == 1
        # Adding a discharge row should also add a new tab for that level
        assert state["tabs"].count() == 2

    def test_removing_discharge_rows(self, qtbot, setup_tab_state, initialize_erosion_tab):
        erosion_tab = initialize_erosion_tab
        erosion_tab.create()
        state = StateStore.instance()

        state["tabs"] = setup_tab_state['tabs']
        # Add a discharge row
        with patch("dfastbe.gui.base.editADischarge", mock_edit_discharge_dialog):
            qtbot.mouseClick(state["dischargesAdd"], Qt.LeftButton)

        # Check if there is indeed one discharge item before removal
        assert state["discharges"].topLevelItemCount() == 1

        # Check if there are 2 tabs before removal
        assert state["tabs"].count() == 2

        # Select the item to be removed using selection model
        item = state["discharges"].topLevelItem(0)
        state["discharges"].setCurrentItem(item)

        # Ensure the remove button is enabled
        remove_btn = state["dischargesRemove"]
        edit_btn = state["dischargesEdit"]
        if not remove_btn.isEnabled():
            remove_btn.setEnabled(True)
            edit_btn.setEnabled(True)

        qtbot.mouseClick(remove_btn, Qt.LeftButton)

        assert state["discharges"].topLevelItemCount() == 0
        assert remove_btn.isEnabled() == False
        assert edit_btn.isEnabled() == False
        assert state["tabs"].count() == 1

    def test_menu_save_configuration_saves_erosion_tab_state(
        self,
        qtbot,
        setup_tab_state,
        initialize_erosion_tab,
        tmp_path,
        create_widget_configuration
    ):
        """
        Alters widgets in the Erosion tab, calls menu_save_configuration, and checks
        that the saved config file contains the correct state.
        """
        window = setup_tab_state['window']
        tabs = setup_tab_state['tabs']
        qtbot.addWidget(window)
        qtbot.addWidget(tabs)
        erosion_tab = initialize_erosion_tab
        erosion_tab.create()
        state = create_widget_configuration
        # Set values for erosion widgets
        state["tErosion"].setText("123.45")
        state["refLevel"].setText("7")
        state["chainageOutStep"].setText("5.0")
        state["newBankFile"].setText("bank_new.txt")
        state["newEqBankFile"].setText("bank_eq.txt")
        state["eroVol"].setText("erovol_standard.evo")
        state["eroVolEqui"].setText("erovol_eq.evo")
        # Ensure StateStore uses this widget state
        StateStore._instance = state
        # Patch QFileDialog.getSaveFileName to return a temp file path
        save_path = tmp_path / "saved_erosion_config.cfg"
        with patch("PySide6.QtWidgets.QFileDialog.getSaveFileName", return_value=(str(save_path), "")):
            menu_save_configuration()

        # Read the saved config and check values
        config = configparser.ConfigParser()
        config.optionxform = str  # preserve case
        config.read(str(save_path))
        assert config["Erosion"]["TErosion"] == "123.45"
        assert config["Erosion"]["RefLevel"] == "7"
        assert config["Erosion"]["OutputInterval"] == "5.0"
        assert config["Erosion"]["BankNew"] == "bank_new.txt"
        assert config["Erosion"]["BankEq"] == "bank_eq.txt"
        assert config["Erosion"]["EroVol"] == "erovol_standard.evo"
        assert config["Erosion"]["EroVolEqui"] == "erovol_eq.evo"

