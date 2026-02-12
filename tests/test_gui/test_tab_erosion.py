import pytest
from PySide6.QtWidgets import QLineEdit, QTreeWidget
from PySide6.QtGui import QIntValidator
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

    def test_widget_defaults(self, initialize_erosion_tab):
        initialize_erosion_tab.create()
        state = StateStore.instance()
        # tErosion should be QLineEdit with positive_real validator
        t_erosion = state["tErosion"]
        assert isinstance(t_erosion, QLineEdit)
        assert isinstance(t_erosion.validator(), type(validator("positive_real")))
        # refLevel should be QLineEdit with QIntValidator
        ref_level = state["refLevel"]
        assert isinstance(ref_level, QLineEdit)
        assert isinstance(ref_level.validator(), QIntValidator)
        # chainageOutStep should be QLineEdit with positive_real validator
        chainage_out_step = state["chainageOutStep"]
        assert isinstance(chainage_out_step, QLineEdit)
        assert isinstance(chainage_out_step.validator(), type(validator("positive_real")))
        # discharges should be QTreeWidget
        discharges = state["discharges"]
        assert isinstance(discharges, QTreeWidget)

    def test_layout_and_tab_addition(self, setup_tab_state, initialize_erosion_tab):
        tabs = setup_tab_state['tabs']
        initial_count = tabs.count()
        initialize_erosion_tab.create()
        assert tabs.count() == initial_count + 1
        assert tabs.tabText(tabs.count() - 1) == "Erosion"

    def test_new_bank_file_widget(self, initialize_erosion_tab):
        initialize_erosion_tab.create()
        state = StateStore.instance()
        new_bank_file = state["newBankFile"]
        assert isinstance(new_bank_file, QLineEdit)
        assert new_bank_file.text() == ""

    def test_ero_vol_eq_widget(self, initialize_erosion_tab):
        initialize_erosion_tab.create()
        state = StateStore.instance()
        ero_vol_eq = state["eroVolEqui"]
        assert isinstance(ero_vol_eq, QLineEdit)
        assert ero_vol_eq.text() == ""

    def test_discharge_widget_columns(self, initialize_erosion_tab):
        initialize_erosion_tab.create()
        state = StateStore.instance()
        discharges = state["discharges"]
        assert discharges.headerItem().text(0) == "Level"
        assert discharges.headerItem().text(1) == "FileName"
        assert discharges.headerItem().text(2) == "Probability [-]"
