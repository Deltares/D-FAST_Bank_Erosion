import pytest
from unittest.mock import MagicMock
from dfastbe.gui.configs import ConfigurationExporter

class TestConfigurationExporter:
    @pytest.fixture
    def mock_state_general(self):
        def _make(overrides=None):
            state = {}
            state["chainFileEdit"] = MagicMock(text=MagicMock(return_value="river.km"))
            state["startRange"] = MagicMock(text=MagicMock(return_value="0"))
            state["endRange"] = MagicMock(text=MagicMock(return_value="100"))
            state["bankDirEdit"] = MagicMock(text=MagicMock(return_value="bankdir"))
            state["bankFileName"] = MagicMock(text=MagicMock(return_value="bankfile.txt"))
            state["makePlotsEdit"] = MagicMock(isChecked=MagicMock(return_value=True))
            state["savePlotsEdit"] = MagicMock(isChecked=MagicMock(return_value=False))
            state["saveZoomPlotsEdit"] = MagicMock(isChecked=MagicMock(return_value=True))
            state["zoomPlotsRangeEdit"] = MagicMock(text=MagicMock(return_value="2.5"))
            state["figureDirEdit"] = MagicMock(text=MagicMock(return_value="figures"))
            state["closePlotsEdit"] = MagicMock(isChecked=MagicMock(return_value=False))
            state["debugOutputEdit"] = MagicMock(isChecked=MagicMock(return_value=True))
            if overrides:
                state.update(overrides)
            return state
        return _make

    @pytest.fixture
    def mock_state_detect(self):
        def _make(nbank=2, dlines=None):
            state = {}
            state["simFileEdit"] = MagicMock(text=MagicMock(return_value="sim.nc"))
            state["waterDepth"] = MagicMock(text=MagicMock(return_value="0.0"))
            searchLines = MagicMock()
            searchLines.topLevelItemCount.return_value = nbank
            def make_item(i):
                item = MagicMock()
                item.text.side_effect = lambda idx: [str(i+1), f"line{i+1}.xyc", str((dlines or [10, 20])[i])][idx]
                return item
            searchLines.topLevelItem.side_effect = lambda i: make_item(i)
            state["searchLines"] = searchLines
            return state
        return _make

    @pytest.fixture
    def mock_state_erosion(self):
        def _make(ship_type="Constant", classes="Bank Type", filters=None, nlevel=2, per_level=None):
            state = {}
            # Basic fields
            state["tErosion"] = MagicMock(text=MagicMock(return_value="1"))
            state["riverAxisEdit"] = MagicMock(text=MagicMock(return_value="axis.xyc"))
            state["fairwayEdit"] = MagicMock(text=MagicMock(return_value="fairway.xyc"))
            state["chainageOutStep"] = MagicMock(text=MagicMock(return_value="0.1"))
            state["outDirEdit"] = MagicMock(text=MagicMock(return_value="outdir"))
            state["newBankFile"] = MagicMock(text=MagicMock(return_value="banknew"))
            state["newEqBankFile"] = MagicMock(text=MagicMock(return_value="bankeq"))
            state["eroVol"] = MagicMock(text=MagicMock(return_value="erovol.evo"))
            state["eroVolEqui"] = MagicMock(text=MagicMock(return_value="erovoleq.evo"))
            # Ship params
            state["shipTypeType"] = MagicMock(currentText=MagicMock(return_value=ship_type))
            state["shipTypeSelect"] = MagicMock(currentIndex=MagicMock(return_value=1))
            state["shipTypeEdit"] = MagicMock(text=MagicMock(return_value="2"))
            state["shipVelocEdit"] = MagicMock(text=MagicMock(return_value="5.0"))
            state["nShipsEdit"] = MagicMock(text=MagicMock(return_value="3"))
            state["shipNWavesEdit"] = MagicMock(text=MagicMock(return_value="5"))
            state["shipDraughtEdit"] = MagicMock(text=MagicMock(return_value="1.2"))
            state["wavePar0Edit"] = MagicMock(text=MagicMock(return_value="150.0"))
            state["wavePar1Edit"] = MagicMock(text=MagicMock(return_value="110.0"))
            # Bank strength
            state["strengthPar"] = MagicMock(currentText=MagicMock(return_value=classes))
            state["bankTypeType"] = MagicMock(currentText=MagicMock(return_value="Constant"))
            state["bankTypeSelect"] = MagicMock(currentIndex=MagicMock(return_value=0))
            state["bankTypeEdit"] = MagicMock(text=MagicMock(return_value="banktype.txt"))
            state["bankShearEdit"] = MagicMock(text=MagicMock(return_value="shear.txt"))
            state["bankProtectEdit"] = MagicMock(text=MagicMock(return_value="protect.txt"))
            state["bankSlopeEdit"] = MagicMock(text=MagicMock(return_value="20.0"))
            state["bankReedEdit"] = MagicMock(text=MagicMock(return_value="0.0"))
            # Filters
            state["velFilterActive"] = MagicMock(isChecked=MagicMock(return_value=(filters or {}).get("vel", False)))
            state["velFilterWidth"] = MagicMock(text=MagicMock(return_value="0.3"))
            state["bedFilterActive"] = MagicMock(isChecked=MagicMock(return_value=(filters or {}).get("bed", False)))
            state["bedFilterWidth"] = MagicMock(text=MagicMock(return_value="0.4"))
            # Levels
            state["discharges"] = MagicMock()
            state["discharges"].topLevelItemCount.return_value = nlevel
            def make_level_item(i):
                item = MagicMock()
                item.text.side_effect = lambda idx: [str(i+1), f"sim{i+1}.nc", f"0.{i+1}"][idx]
                return item
            state["discharges"].topLevelItem.side_effect = lambda i: make_level_item(i)
            state["refLevel"] = MagicMock(text=MagicMock(return_value="3"))
            # Per-level overrides
            for i in range(nlevel):
                istr = str(i+1)
                for key in ["shipTypeType", "shipTypeSelect", "shipTypeEdit", "shipVelocType", "shipVelocEdit", "nShipsType", "nShipsEdit", "shipNWavesType", "shipNWavesEdit", "shipDraughtType", "shipDraughtEdit", "bankSlopeType", "bankSlopeEdit", "bankReedType", "bankReedEdit", "eroVolEdit"]:
                    state[f"{istr}_{key}"] = MagicMock()
                # Set per-level types to 'Use Default' unless overridden
                for key in ["shipTypeType", "shipVelocType", "nShipsType", "shipNWavesType", "shipDraughtType", "bankSlopeType", "bankReedType"]:
                    state[f"{istr}_{key}"].currentText.return_value = (per_level or {}).get(f"{istr}_{key}", "Use Default")
                # Set per-level edit fields
                for key in ["shipTypeEdit", "shipVelocEdit", "nShipsEdit", "shipNWavesEdit", "shipDraughtEdit", "bankSlopeEdit", "bankReedEdit", "eroVolEdit"]:
                    state[f"{istr}_{key}"].text.return_value = (per_level or {}).get(f"{istr}_{key}", "")
                # For shipTypeSelect
                state[f"{istr}_shipTypeSelect"].currentIndex.return_value = 0
            return state
        return _make

    def test_build_general_section_fields_and_defaults(self, mock_state_general):
        state = mock_state_general()
        exporter = ConfigurationExporter(state)
        exporter._build_general_section()
        section = exporter.config["General"]
        assert section["Version"] == "1.0"
        assert section["RiverKM"] == "river.km"
        assert section["Boundaries"] == "0:100"
        assert section["BankDir"] == "bankdir"
        assert section["BankFile"] == "bankfile.txt"
        assert section["Plotting"] == "True"
        assert section["SavePlots"] == "False"
        assert section["SaveZoomPlots"] == "True"
        assert section["ZoomStepKM"] == "2.5"
        assert section["FigureDir"] == "figures"
        assert section["ClosePlots"] == "False"
        assert section["DebugOutput"] == "True"

    @pytest.mark.parametrize("nbank,dlines,expected_dlines", [
        (2, [10, 20], "[ 10, 20 ]"),
        (1, [42], "[ 42 ]"),
    ])
    def test_build_detect_section_nbank_lines_dlines(self, mock_state_detect, nbank, dlines, expected_dlines):
        state = mock_state_detect(nbank=nbank, dlines=dlines)
        exporter = ConfigurationExporter(state)
        exporter._build_detect_section()
        section = exporter.config["Detect"]
        assert section["NBank"] == str(nbank)
        for i in range(nbank):
            assert section[f"Line{i+1}"] == f"line{i+1}.xyc"
        assert section["DLines"] == expected_dlines

    @pytest.mark.parametrize("ship_type,expected_shiptype", [
        ("Constant", "2"),
        ("Variable", "2"),
    ])
    def test_build_erosion_section_ship_params(self, mock_state_erosion, ship_type, expected_shiptype):
        state = mock_state_erosion(ship_type=ship_type)
        exporter = ConfigurationExporter(state)
        exporter._build_erosion_section()
        section = exporter.config["Erosion"]
        assert section["ShipType"] == expected_shiptype
        assert section["VShip"] == "5.0"
        assert section["NShip"] == "3"
        assert section["NWaves"] == "5"
        assert section["Draught"] == "1.2"
        assert section["Wave0"] == "150.0"
        assert section["Wave1"] == "110.0"

    @pytest.mark.parametrize("classes,expected_flag,expected_banktype", [
        ("Bank Type", "true", "0"),
        ("Critical Shear Stress", "false", "shear.txt"),
    ])
    def test_build_erosion_section_classes_flag(self, mock_state_erosion, classes, expected_flag, expected_banktype):
        state = mock_state_erosion(classes=classes)
        exporter = ConfigurationExporter(state)
        exporter._build_erosion_section()
        section = exporter.config["Erosion"]
        assert section["Classes"] == expected_flag
        assert section["BankType"] == expected_banktype

    @pytest.mark.parametrize("filters,expected_keys", [
        ({"vel": True, "bed": False}, ["VelFilterDist"]),
        ({"vel": False, "bed": True}, ["BedFilterDist"]),
        ({"vel": True, "bed": True}, ["VelFilterDist", "BedFilterDist"]),
        ({"vel": False, "bed": False}, []),
    ])
    def test_build_erosion_section_filters(self, mock_state_erosion, filters, expected_keys):
        state = mock_state_erosion(filters=filters)
        exporter = ConfigurationExporter(state)
        exporter._build_erosion_section()
        section = exporter.config["Erosion"]
        for key in ["VelFilterDist", "BedFilterDist"]:
            if key in expected_keys:
                assert key in section
            else:
                assert key not in section

    def test_build_erosion_section_levels_and_overrides(self, mock_state_erosion):
        per_level = {
            "1_shipTypeType": "Constant",
            "1_shipTypeEdit": "7",
            "2_shipVelocType": "Constant",
            "2_shipVelocEdit": "9.9",
            "2_eroVolEdit": "vol2.evo",
        }
        state = mock_state_erosion(nlevel=2, per_level=per_level)
        exporter = ConfigurationExporter(state)
        exporter._build_erosion_section()
        section = exporter.config["Erosion"]
        assert section["NLevel"] == "2"
        assert section["RefLevel"] == "3"
        assert section["SimFile1"] == "sim1.nc"
        assert section["PDischarge1"] == "0.1"
        assert section["SimFile2"] == "sim2.nc"
        assert section["PDischarge2"] == "0.2"
        assert section["ShipType1"] == "1"  # shipTypeSelect index + 1
        assert section["VShip2"] == "9.9"
        assert section["EroVol2"] == "vol2.evo"

    def test_bank_type_constant_writes_string_index(self, mock_state_erosion):
        """Verify BankType is serialized as a string in the Bank-Type / Constant branch.

        When the bank-strength selector is set to Bank Type and the bank-type
        selector is set to Constant, the bank-strength builder reads the
        integer index from the bank-type combo box and writes it under the
        BankType key of the Erosion section. The standard library config
        parser only accepts string values for section options, so the index
        must be converted to a string before assignment; otherwise the
        assignment raises a TypeError saying option values must be strings.

        What this test checks:
            * Building the Erosion section runs to completion without raising
              a TypeError when the Bank-Type / Constant branch is exercised.
            * The stored value of BankType equals the string "2" for an index
              of 2. A non-zero index is used so that a regression which drops
              the index or hard-codes "0" would also be caught.
            * The stored value is an instance of str, guarding against any
              future change that assigns the raw integer.
        """
        state = mock_state_erosion(classes="Bank Type")
        state["bankTypeSelect"].currentIndex.return_value = 2
        exporter = ConfigurationExporter(state)
        exporter._build_erosion_section()
        section = exporter.config["Erosion"]
        assert section["BankType"] == "2"
        assert isinstance(section["BankType"], str)

    def test_per_level_ship_type_constant_writes_string_index_plus_one(
        self, mock_state_erosion
    ):
        """Verify per-level ShipType is serialized as the string of (index + 1) in the Constant branch.

        For each discharge level n, when the per-level ship-type selector for
        that level is set to Constant, the discharge-levels builder writes the
        string of the combo-box current index plus one under the ShipTypeN
        key of the Erosion section. The plus-one offset translates the
        zero-based combo-box index into the one-based ship-type identifier
        expected by the configuration file format. The result must be
        converted to a string because the standard library config parser only
        accepts string values for section options.

        What this test checks:
            * Building the Erosion section runs to completion without raising
              a TypeError when the per-level Constant branch is exercised.
            * The stored value of ShipType1 equals the string "3" for an index
              of 2. A non-zero index pins both the string conversion and the
              plus-one arithmetic, so a regression to either would be caught.
            * The stored value is an instance of str, not int.
        """
        per_level = {"1_shipTypeType": "Constant"}
        state = mock_state_erosion(nlevel=1, per_level=per_level)
        state["1_shipTypeSelect"].currentIndex.return_value = 2
        exporter = ConfigurationExporter(state)
        exporter._build_erosion_section()
        section = exporter.config["Erosion"]
        assert section["ShipType1"] == "3"
        assert isinstance(section["ShipType1"], str)
