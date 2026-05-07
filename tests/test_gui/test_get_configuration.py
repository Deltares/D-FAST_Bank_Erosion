"""Unit tests for the ConfigurationExporter class.

These tests cover the GUI-state-to-configuration export pipeline in
dfastbe.gui.configs without spinning up a Qt event loop. The state store
is replaced with plain dicts of MagicMock widgets, exercised both per
section (via the private _build_* helpers) and end-to-end (via the public
build method).
"""

import pytest
from unittest.mock import MagicMock
from dfastbe.gui.configs import ConfigurationExporter


class TestConfigurationExporter:
    """Behavioral tests for ConfigurationExporter.

    The fixtures below construct minimal mock state stores for each top-level
    section (General, Detect, Erosion). Tests either drive a single private
    builder (for fine-grained failure messages) or call the public build
    method (to pin the public API contract).
    """

    @pytest.fixture
    def mock_state_general(self):
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
        return state

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
            per_level_keys = [
                "shipTypeType",
                "shipTypeSelect",
                "shipTypeEdit",
                "shipVelocType",
                "shipVelocEdit",
                "nShipsType",
                "nShipsEdit",
                "shipNWavesType",
                "shipNWavesEdit",
                "shipDraughtType",
                "shipDraughtEdit",
                "bankSlopeType",
                "bankSlopeEdit",
                "bankReedType",
                "bankReedEdit",
                "eroVolEdit",
            ]
            per_level_type_keys = [
                "shipTypeType",
                "shipVelocType",
                "nShipsType",
                "shipNWavesType",
                "shipDraughtType",
                "bankSlopeType",
                "bankReedType",
            ]
            per_level_edit_keys = [
                "shipTypeEdit",
                "shipVelocEdit",
                "nShipsEdit",
                "shipNWavesEdit",
                "shipDraughtEdit",
                "bankSlopeEdit",
                "bankReedEdit",
                "eroVolEdit",
            ]
            overrides = per_level or {}
            for i in range(nlevel):
                istr = str(i + 1)
                for key in per_level_keys:
                    state[f"{istr}_{key}"] = MagicMock()
                for key in per_level_type_keys:
                    state[f"{istr}_{key}"].currentText.return_value = overrides.get(
                        f"{istr}_{key}", "Use Default"
                    )
                for key in per_level_edit_keys:
                    state[f"{istr}_{key}"].text.return_value = overrides.get(
                        f"{istr}_{key}", ""
                    )
                state[f"{istr}_shipTypeSelect"].currentIndex.return_value = 0
            return state
        return _make

    def test_build_general_section_fields_and_defaults(self, mock_state_general):
        exporter = ConfigurationExporter(mock_state_general)
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
        (0, [], "[ ]"),
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

    def test_build_detect_section_zero_banks_emits_no_line_keys(self, mock_state_detect):
        """Verify that the Detect section contains no per-bank Line keys when there are zero banks.

        When the search-lines widget reports zero entries, the detect builder
        must skip the per-bank loop entirely. The only Detect-section keys
        produced should be the four scalar keys SimFile, WaterDepth, NBank and
        DLines. Producing a Line1 (or any other Line<n>) key with no bank to
        back it would create a malformed configuration.

        What this test checks:
            * Building the Detect section runs to completion.
            * No key whose name starts with the literal prefix Line is
              present in the resulting section, regardless of how the section
              is iterated.
            * The four scalar keys are still populated. NBank reads zero and
              DLines reads as the empty-list literal.
        """
        state = mock_state_detect(nbank=0, dlines=[])
        exporter = ConfigurationExporter(state)
        exporter._build_detect_section()
        section = exporter.config["Detect"]
        assert section["NBank"] == "0"
        assert section["DLines"] == "[ ]"
        assert section["SimFile"] == "sim.nc"
        assert section["WaterDepth"] == "0.0"
        line_keys = [key for key in section.keys() if key.startswith("Line")]
        assert line_keys == []

    def test_build_erosion_section_zero_levels_emits_no_per_level_keys(
        self, mock_state_erosion
    ):
        """Verify that the Erosion section contains no per-level keys when there are zero levels.

        When the discharges widget reports zero entries, the discharge-levels
        builder writes NLevel and RefLevel but skips the per-level loop. None
        of the per-level keys (SimFile<n>, PDischarge<n>, ShipType<n>,
        VShip<n>, NShip<n>, NWaves<n>, Draught<n>, Slope<n>, Reed<n>,
        EroVol<n>) should appear in the resulting section.

        What this test checks:
            * Building the full Erosion section runs to completion.
            * NLevel reads zero and RefLevel still reads from the validator
              field, demonstrating that the two scalar keys are written
              before the per-level loop is reached.
            * No key matches any of the per-level prefixes followed by a
              digit. This guards against a future refactor that mistakenly
              moves a per-level write outside the loop body.
        """
        state = mock_state_erosion(nlevel=0)
        exporter = ConfigurationExporter(state)
        exporter._build_erosion_section()
        section = exporter.config["Erosion"]
        assert section["NLevel"] == "0"
        assert section["RefLevel"] == "3"
        per_level_prefixes = (
            "SimFile",
            "PDischarge",
            "ShipType",
            "VShip",
            "NShip",
            "NWaves",
            "Draught",
            "Slope",
            "Reed",
            "EroVol",
        )
        offending = [
            key
            for key in section.keys()
            for prefix in per_level_prefixes
            if key.startswith(prefix) and key[len(prefix):].isdigit()
        ]
        assert offending == []

    @pytest.mark.parametrize(
        "ship_type,select_index,edit_text,expected_shiptype",
        [
            ("Constant", 2, "ignored-edit-text", "3"),
            ("Variable", 99, "custom-edit-text", "custom-edit-text"),
        ],
    )
    def test_build_erosion_section_ship_params(
        self,
        mock_state_erosion,
        ship_type,
        select_index,
        edit_text,
        expected_shiptype,
    ):
        """Verify the two branches of the top-level ship-type writer in the Erosion section.

        The ship-parameters builder picks one of two sources for the ShipType
        value depending on the value of the ship-type type selector:

            * When the type selector reads Constant, the value is the string
              of the ship-type combo-box current index plus one. The plus-one
              offset translates the zero-based combo-box index into the
              one-based ship-type identifier expected by the configuration
              file format. The free-form edit field is not consulted.
            * When the type selector reads anything else (commonly Variable),
              the value is the literal text of the free-form edit field. The
              combo-box index is not consulted.

        What this parametrization checks:
            * For the Constant row, the test sets the combo-box index to a
              non-zero value and the edit field to a sentinel string. The
              expected output is the string of index plus one. A regression
              that reads from the edit field instead would produce the
              sentinel and fail the assertion. A regression that drops the
              plus-one would produce a different number and also fail.
            * For the Variable row, the test sets the edit field to a
              non-numeric sentinel string and the combo-box index to a value
              that, if accidentally used, would produce a clearly different
              numeric output. The expected output is the sentinel string. A
              regression that reads the index plus one would produce a number
              and fail the assertion.

        The remaining ship-parameter keys (VShip, NShip, NWaves, Draught,
        Wave0, Wave1) are populated from fields that do not depend on the
        ship-type branch and are checked here as a sanity guard against
        accidental misordering or omission.
        """
        state = mock_state_erosion(ship_type=ship_type)
        state["shipTypeSelect"].currentIndex.return_value = select_index
        state["shipTypeEdit"].text.return_value = edit_text
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

    def test_build_returns_complete_configuration_with_three_sections_in_order(
        self, mock_state_general, mock_state_detect, mock_state_erosion
    ):
        """Verify build() returns a complete configuration with the expected sections in order.

        The build method is the only public entry point on the exporter and is
        what the get_configuration helper delegates to when extracting the GUI
        state into a configuration. Downstream callers depend on a single call
        producing a fully-populated configuration object.

        What this test checks:
            * build returns a config parser whose sections are exactly
              General, Detect and Erosion, in that declaration order.
            * Each of the three sections contains at least one representative
              key drawn from the corresponding builder, demonstrating that
              every per-section builder ran and contributed to the result.
            * The Erosion section contains keys produced by every nested
              builder it invokes (basic erosion parameters, ship parameters,
              bank-strength parameters, and per-level discharge parameters).
        """
        state = {}
        state.update(mock_state_general)
        state.update(mock_state_detect(nbank=1, dlines=[10]))
        state.update(mock_state_erosion(nlevel=1))
        exporter = ConfigurationExporter(state)

        config = exporter.build()

        assert config.sections() == ["General", "Detect", "Erosion"]
        assert config["General"]["Version"] == "1.0"
        assert config["General"]["RiverKM"] == "river.km"
        assert config["Detect"]["NBank"] == "1"
        assert config["Detect"]["Line1"] == "line1.xyc"
        assert config["Detect"]["DLines"] == "[ 10 ]"
        assert config["Erosion"]["TErosion"] == "1"
        assert config["Erosion"]["VShip"] == "5.0"
        assert config["Erosion"]["Classes"] == "true"
        assert config["Erosion"]["NLevel"] == "1"
        assert config["Erosion"]["SimFile1"] == "sim1.nc"
