from typing import List
from unittest.mock import MagicMock, patch

import pytest
from shapely.geometry import LineString, MultiLineString, Polygon

from dfastbe.bank_lines.data_models import BankLinesRiverData, SearchLines


class TestSearchLines:
    def test_mask_with_multilinestring(self):
        """Test the mask method with a LineString and a MultiLineString."""
        # Define a LineString and a MultiLineString
        line1 = LineString([(0, 0), (1, 1), (2, 2)])
        line2 = MultiLineString([[(3, 3), (4, 4)], [(5, 5), (6, 6)]])

        # Combine them into search_lines
        search_lines = [line1, line2]

        # Define a river center line
        river_center_line = LineString([(0, 0), (6, 6)])

        # Call the mask method
        masked_lines, max_distance = SearchLines.mask(
            search_lines, river_center_line, max_river_width=10
        )

        # Assertions
        assert len(masked_lines) == 2
        assert isinstance(masked_lines[0], LineString)
        assert isinstance(masked_lines[1], LineString)
        assert max_distance > 0

    @pytest.fixture
    def lines(self) -> List[LineString]:
        return [LineString([(0, 0), (1, 1)]), LineString([(2, 2), (3, 3)])]

    def test_d_lines(self, lines):
        search_lines = SearchLines(lines)
        search_lines.d_lines = lines
        assert search_lines.d_lines == lines

    @patch("dfastbe.io.LineGeometry")
    def test_searchlines_with_center_line(self, mock_center_line, lines):
        mask = LineString([(0, 0), (2, 2)])
        mock_center_line.values = mask
        search_lines = SearchLines(lines, mask=mock_center_line)
        assert search_lines.max_distance == pytest.approx(2.0)

    def test_d_lines_not_set(self, lines):
        search_lines = SearchLines(lines)
        with pytest.raises(
                ValueError, match="The d_lines property has not been set yet."
        ):
            search_lines.d_lines

    def test_to_polygons(self):
        """Test the to_polygons method of the SearchLines class."""
        line1 = LineString([(0, 0), (1, 1), (2, 2)])
        line2 = LineString([(3, 3), (4, 4), (5, 5)])
        search_lines = [line1, line2]

        search_lines_obj = SearchLines(search_lines)
        search_lines_obj.d_lines = [1.0, 2.0]
        polygons = search_lines_obj.to_polygons()

        assert len(polygons) == 2
        assert isinstance(polygons[0], Polygon)
        assert isinstance(polygons[1], Polygon)
        assert polygons[0].buffer(-1.0).is_empty
        assert polygons[1].buffer(-2.0).is_empty


class TestBankLinesRiverData:
    @pytest.fixture
    def mock_config_file(self):
        """Fixture to create a mock ConfigFile object."""
        mock_config = MagicMock()
        mock_config.get_search_lines.return_value = [
            LineString([(0, 0), (1, 1)]),
            LineString([(2, 2), (3, 3)]),
        ]
        mock_config.get_bank_search_distances.return_value = [10, 20]
        mock_config.get_sim_file.return_value = "mock_sim_file"
        mock_config.get_float.return_value = 0.5
        return mock_config

    @pytest.fixture
    def mock_simulation_data(self):
        """Fixture to create a mock BaseSimulationData object."""
        mock_sim_data = MagicMock()
        mock_sim_data.dry_wet_threshold = 0.3
        return mock_sim_data

    @patch("dfastbe.bank_lines.data_models.BaseSimulationData.read")
    def test_get_bank_lines_simulation_data(
        self, mock_read, mock_config_file, mock_simulation_data
    ):
        """Test the _get_bank_lines_simulation_data method."""
        mock_read.return_value = mock_simulation_data

        with patch("dfastbe.io.LineGeometry"):
            river_data = BankLinesRiverData(mock_config_file)
            simulation_data, h0 = river_data._get_bank_lines_simulation_data()

        mock_read.assert_called_once_with("mock_sim_file")
        assert simulation_data == mock_simulation_data
        assert h0 == 0.8  # 0.5 (critical water depth) + 0.3 (dry_wet_threshold)

    @patch("dfastbe.bank_lines.data_models.BaseSimulationData.read")
    def test_simulation_data(self, mock_read, mock_config_file, mock_simulation_data):
        """Test the simulation_data method."""
        mock_read.return_value = mock_simulation_data

        with patch("dfastbe.io.LineGeometry") as mock_line_geometry, patch.object(
            BankLinesRiverData, "search_lines"
        ) as mock_search_lines:
            river_data = BankLinesRiverData(mock_config_file)
            mock_search_lines.max_distance = 50
            simulation_data, h0 = river_data.simulation_data()

            mock_read.assert_called_once_with("mock_sim_file")
            simulation_data.clip.assert_called_once_with(
                river_data.river_center_line.values, 50
            )
            assert simulation_data == mock_simulation_data
            assert h0 == 0.8
