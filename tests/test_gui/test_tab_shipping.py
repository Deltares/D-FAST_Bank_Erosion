import pytest
from PySide6.QtWidgets import QLineEdit, QComboBox
from PySide6.QtGui import QDoubleValidator

from dfastbe.gui.tabs.shipping import ShippingTab
from dfastbe.gui.state_management import StateStore
from dfastbe.gui.utils import SHIP_TYPES, validator

@pytest.fixture
def initialize_shipping_tab(setup_tab_state, qtbot):
    shipping_tab = ShippingTab(
        setup_tab_state['tabs'],
    )
    return shipping_tab

class TestShippingTab:
    def test_widgets_registered(self, qtbot, initialize_shipping_tab):
        initialize_shipping_tab.create()
        state = StateStore.instance()
        for key in [
            "shipType", "shipVeloc", "nShips", "shipNWaves", "shipDraught",
            "wavePar0", "wavePar1", "shipTypeSelect"
        ]:
            assert key in state

    def test_ship_type_select_list(self, qtbot, initialize_shipping_tab):
        initialize_shipping_tab.create()
        state = StateStore.instance()
        ship_type = state["shipTypeSelect"]
        assert isinstance(ship_type, QComboBox)
        expected_items = SHIP_TYPES
        actual_items = [ship_type.itemText(i) for i in range(ship_type.count())]
        assert actual_items == expected_items

    @pytest.mark.parametrize("combo_key", [
        "shipType", "shipVeloc", "nShips", "shipNWaves", "shipDraught", "wavePar0", "wavePar1"
    ])
    def test_combo_fields_contain_variable_and_constant(self, qtbot, initialize_shipping_tab, combo_key):
        initialize_shipping_tab.create()
        state = StateStore.instance()
        combo = state.get(f"{combo_key}Type")
        assert combo is not None, f"Combo field {combo_key}Type not found in state."
        assert isinstance(combo, QComboBox)
        items = [combo.itemText(i).lower() for i in range(combo.count())]
        assert "variable" in items, f"'variable' not found in {combo_key}Type options: {items}"
        assert "constant" in items, f"'constant' not found in {combo_key}Type options: {items}"

    @pytest.mark.parametrize("field_key", [
        "shipVeloc", "nShips", "shipNWaves", "shipDraught", "wavePar0", "wavePar1"
    ])
    def test_filter_validator(self, qtbot, initialize_shipping_tab, field_key):
        initialize_shipping_tab.create()
        state = StateStore.instance()
        filter_field = state.get(f"{field_key}Edit")
        assert filter_field is not None, f"Edit field {field_key}Edit not found in state."
        assert isinstance(filter_field, QLineEdit)
        validator_obj = filter_field.validator()
        assert isinstance(validator_obj, QDoubleValidator)
        assert validator_obj.bottom() == 0
        ref_validator = validator("positive_real")
        assert type(validator_obj) is type(ref_validator)
