import pytest
from PySide6.QtWidgets import QLineEdit, QComboBox
from PySide6.QtGui import QDoubleValidator

from dfastbe.gui.tabs.bank import BankTab
from dfastbe.gui.state_management import StateStore
from dfastbe.gui.utils import validator

@pytest.fixture
def initialize_bank_tab(setup_tab_state, qtbot):
    bank_tab = BankTab(
        setup_tab_state['tabs'],
        # BankTab only takes tabs, not window or app
    )
    return bank_tab

class TestBankTab:
    def test_widgets_registered(self, qtbot, initialize_bank_tab):
        initialize_bank_tab.create()
        state = StateStore.instance()
        for key in [
            "strengthPar", "bankType", "bankShear", "bankProtect", "bankSlope", "bankReed",
            "velFilterWidth", "velFilterActive", "velFilterTxt",
            "bedFilterWidth", "bedFilterActive", "bedFilterTxt"
        ]:
            assert key in state

    def test_strength_parameter_combo(self, qtbot, initialize_bank_tab):
        bank_tab = initialize_bank_tab
        bank_tab.create()
        state = StateStore.instance()
        combo = state["strengthPar"]
        assert isinstance(combo, QComboBox)
        assert combo.count() == 2
        assert combo.itemText(0) == "Bank Type"
        assert combo.itemText(1) == "Critical Shear Stress"

    @pytest.mark.parametrize("field_key", ["velFilterWidth", "bedFilterWidth"])
    def test_filter_validator(self, qtbot, initialize_bank_tab, field_key):
        """
        Checks that the specified filters field use a QLineEdit with a QDoubleValidator (bottom=0).
        """
        bank_tab = initialize_bank_tab
        bank_tab.create()
        state = StateStore.instance()
        filter_field = state[field_key]
        assert isinstance(filter_field, QLineEdit)
        validator_obj = filter_field.validator()
        assert isinstance(validator_obj, QDoubleValidator)
        assert validator_obj.bottom() == 0
        ref_validator = validator("positive_real")
        assert type(validator_obj) is type(ref_validator)

    def test_switching_strength_par_behaviour(self, qtbot, setup_tab_state, initialize_bank_tab):
        """
        Simulates switching strengthPar from 'Bank Type' to 'Critical Shear Stress' and checks
        whether the Critical Shear Stress field gets enabled/disabled accordingly.
        """
        bank_tab = initialize_bank_tab
        bank_tab.create()
        state = StateStore.instance()
        strength_par = state["strengthPar"]
        bank_shear = state["bankShear"]
        bank_type = state["bankType"]
        # Initially, should be 'Bank Type' (index 0)
        assert strength_par.currentText() == "Bank Type"
        # The bankShear field should be disabled
        assert not bank_shear.isEnabled()
        assert bank_type.isEnabled()
        # Switch to 'Critical Shear Stress' (index 1)
        strength_par.setCurrentIndex(1)
        qtbot.wait(50)  # Allow signal to process
        assert strength_par.currentText() == "Critical Shear Stress"
        # The bankShear field should now be enabled
        assert bank_shear.isEnabled()
        assert not bank_type.isEnabled()
        # Switch back to 'Bank Type'
        strength_par.setCurrentIndex(0)
        qtbot.wait(50)
        assert strength_par.currentText() == "Bank Type"
        # The bankShear field should be disabled again
        assert not bank_shear.isEnabled()
        assert bank_type.isEnabled()

    @pytest.mark.parametrize("filter_key,width_key", [
        ("velFilterActive", "velFilterWidth"),
        ("bedFilterActive", "bedFilterWidth"),
    ])
    def test_filter_checkbox_enables_width(self, qtbot, initialize_bank_tab, filter_key, width_key):
        """
        Checks that checking/unchecking the filter checkbox enables/disables the width field for both velocity and bank elevation filters.
        """
        initialize_bank_tab.create()
        state = StateStore.instance()
        filter_checkbox = state[filter_key]
        filter_width = state[width_key]
        # Initially unchecked, width should be disabled
        assert not filter_checkbox.isChecked()
        assert not filter_width.isEnabled()
        # Check the box, width should become enabled
        filter_checkbox.setChecked(True)
        qtbot.wait(50)
        assert filter_checkbox.isChecked()
        assert filter_width.isEnabled()
        # Uncheck the box, width should become disabled again
        filter_checkbox.setChecked(False)
        qtbot.wait(50)
        assert not filter_checkbox.isChecked()
        assert not filter_width.isEnabled()

    def test_bank_type_select_list(self, qtbot, initialize_bank_tab):
        """
        Checks that the bankType widget is a QComboBox and contains the expected select list items.
        """
        initialize_bank_tab.create()
        state = StateStore.instance()
        bank_type = state["bankTypeSelect"]
        assert isinstance(bank_type, QComboBox)
        expected_items = [
            "0 (Beschermde oeverlijn)",
            "1 (Begroeide oeverlijn)",
            "2 (Goede klei)",
            "3 (Matig / slechte klei)",
            "4 (Zand)",
        ]
        actual_items = [bank_type.itemText(i) for i in range(bank_type.count())]
        assert actual_items == expected_items

    @pytest.mark.parametrize("combo_key", [
        "bankType", "bankProtect", "bankShear", "bankSlope", "bankReed"
    ])
    def test_combo_fields_contain_variable_and_constant(self, qtbot, initialize_bank_tab, combo_key):
        """
        Checks that the specified combo fields contain 'variable' and 'constant' as options.
        """
        initialize_bank_tab.create()
        state = StateStore.instance()
        combo = state.get(f"{combo_key}Type")
        assert combo is not None, f"Combo field {combo_key}Type not found in state."
        assert isinstance(combo, QComboBox)
        items = [combo.itemText(i).lower() for i in range(combo.count())]
        assert "variable" in items, f"'variable' not found in {combo_key}Type options: {items}"
        assert "constant" in items, f"'constant' not found in {combo_key}Type options: {items}"
