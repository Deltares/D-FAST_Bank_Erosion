from typing import List, Tuple

from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.polygon import Polygon

from dfastbe.io.config import log_text
from dfastbe.io.data_models import BaseRiverData, BaseSimulationData, LineGeometry

MAX_RIVER_WIDTH = 1000


class SearchLines:

    def __init__(self, lines: List[LineString], mask: LineGeometry = None):
        """Search lines initialization.

        Args:
            lines (List[LineString]):
                List of search lines.
            mask (LineGeometry, optional):
                Center line for masking the search lines. Defaults to None.
        """
        if mask is None:
            self.values = lines
            self.max_distance = None
        else:
            self.values, self.max_distance = self.mask(lines, mask.values)

        self.size = len(lines)

    @property
    def d_lines(self) -> List[float]:
        if hasattr(self, "_d_lines"):
            return self._d_lines
        else:
            raise ValueError("The d_lines property has not been set yet.")

    @d_lines.setter
    def d_lines(self, value: List[float]):
        self._d_lines = value

    @staticmethod
    def mask(
        search_lines: List[LineString],
        river_center_line: LineString,
        max_river_width: float = MAX_RIVER_WIDTH,
    ) -> Tuple[List[LineString], float]:
        """
        Clip the list of lines to the envelope of a certain size surrounding a reference line.

        Args:
            search_lines (List[LineString]):
                List of lines to be clipped.
            river_center_line (LineString):
                Reference line to which the search lines are clipped.
            max_river_width: float
                Maximum distance away from river_profile.

        Returns:
            Tuple[List[LineString], float]:
                - List of clipped search lines.
                - Maximum distance from any point within line to reference line.

        Examples:
            ```python
            >>> from shapely.geometry import LineString
            >>> search_lines = [LineString([(0, 0), (1, 1)]), LineString([(2, 2), (3, 3)])]
            >>> river_center_line = LineString([(0, 0), (2, 2)])
            >>> search_lines_clipped, max_distance = SearchLines.mask(search_lines, river_center_line)
            >>> max_distance
            2.0

            ```
        """
        num = len(search_lines)
        profile_buffer = river_center_line.buffer(max_river_width, cap_style=2)

        # The algorithm uses simplified geometries for determining the distance between lines for speed.
        # Stay accurate to within about 1 m
        profile_simplified = river_center_line.simplify(1)

        max_distance = 0
        for ind in range(num):
            # Clip the bank search lines to the reach of interest (indicated by the reference line).
            search_lines[ind] = search_lines[ind].intersection(profile_buffer)

            # If the bank search line breaks into multiple parts, select the part closest to the reference line.
            if search_lines[ind].geom_type == "MultiLineString":
                search_lines[ind] = SearchLines._select_closest_part(
                    search_lines[ind], profile_simplified, max_river_width
                )

            # Determine the maximum distance from a point on this line to the reference line.
            line_simplified = search_lines[ind].simplify(1)
            max_distance = max(
                [Point(c).distance(profile_simplified) for c in line_simplified.coords]
            )

            # Increase the value of max_distance by 2 to account for error introduced by using simplified lines.
            max_distance = max(max_distance, max_distance + 2)

        return search_lines, max_distance

    @staticmethod
    def _select_closest_part(
            search_lines_segments: MultiLineString,
            reference_line: LineString,
            max_river_width: float,
    ) -> LineString:
        """Select the closest part of a MultiLineString to the reference line.

        Args:
            search_lines_segments (MultiLineString):
                The MultiLineString containing multiple line segments to evaluate.
            reference_line (LineString):
                The reference line to calculate distances.
            max_river_width (float):
                Maximum allowable distance.

        Returns:
            LineString: The closest part of the MultiLineString.
        """
        closest_part = search_lines_segments.geoms[0]
        min_distance = max_river_width

        for part in search_lines_segments.geoms:
            simplified_part = part.simplify(1)
            distance = simplified_part.distance(reference_line)
            if distance < min_distance:
                min_distance = distance
                closest_part = part

        return closest_part

    def to_polygons(self) -> List[Polygon]:
        """
        Construct a series of polygons surrounding the bank search lines.

        Returns:
            bank_areas:
                Array containing the areas of interest surrounding the bank search lines.

        Examples:
            ```python
            >>> search_lines = [LineString([(0, 0), (1, 1)]), LineString([(2, 2), (3, 3)])]
            >>> search_lines_clipped = SearchLines(search_lines)
            >>> search_lines_clipped.d_lines = [10, 20]
            >>> bank_areas = search_lines_clipped.to_polygons()
            >>> len(bank_areas)
            2

            ```
        """
        bank_areas = [
            self.values[b].buffer(distance, cap_style=2)
            for b, distance in enumerate(self.d_lines)
        ]
        return bank_areas


class BankLinesRiverData(BaseRiverData):

    @property
    def search_lines(self) -> SearchLines:
        """Get search lines for bank lines.

        Returns:
            SearchLines:
                Search lines for bank lines.

        Examples:
            ```python
            >>> from dfastbe.io.config import ConfigFile
            >>> config_file = ConfigFile.read("tests/data/bank_lines/meuse_manual.cfg")
            >>> bank_lines_river_data = BankLinesRiverData(config_file)
            No message found for read_chainage
            No message found for clip_chainage
            >>> search_lines = bank_lines_river_data.search_lines
            No message found for read_search_line
            No message found for read_search_line
            >>> len(search_lines.values)
            2

            ```
        """
        search_lines = SearchLines(self.config_file.get_search_lines(), self.river_center_line)
        search_lines.d_lines = self.config_file.get_bank_search_distances(search_lines.size)
        return search_lines

    def _get_bank_lines_simulation_data(self) -> Tuple[BaseSimulationData, float]:
        """read simulation data and drying flooding threshold dh0

        Returns:
            Tuple[BaseSimulationData, float]:
                simulation data and critical water depth (h0).
        """
        sim_file = self.config_file.get_sim_file("Detect", "")
        log_text("read_simdata", data={"file": sim_file})
        simulation_data = BaseSimulationData.read(sim_file)
        # increase critical water depth h0 by flooding threshold dh0
        # get critical water depth used for defining bank line (default = 0.0 m)
        critical_water_depth = self.config_file.get_float(
            "Detect", "WaterDepth", default=0
        )
        h0 = critical_water_depth + simulation_data.dry_wet_threshold
        return simulation_data, h0

    def simulation_data(self) -> Tuple[BaseSimulationData, float]:
        """Get simulation data and critical water depth and clip to river center line.

        Returns:
            Tuple[BaseSimulationData, float]:
                simulation data and critical water depth (h0).

        Examples:
            ```python
            >>> from dfastbe.io.config import ConfigFile
            >>> from unittest.mock import patch
            >>> config_file = ConfigFile.read("tests/data/bank_lines/meuse_manual.cfg")
            >>> bank_lines_river_data = BankLinesRiverData(config_file)  # doctest: +ELLIPSIS
            N...e
            >>> simulation_data, h0 = bank_lines_river_data.simulation_data()
            N...e
            >>> h0
            0.1

            ```
        """
        simulation_data, h0 = self._get_bank_lines_simulation_data()
        # clip simulation data to boundaries ...
        log_text("clip_data")
        simulation_data.clip(self.river_center_line.values, self.search_lines.max_distance)

        return simulation_data, h0
