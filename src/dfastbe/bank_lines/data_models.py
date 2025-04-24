from typing import List, Tuple, Dict
from shapely.geometry import LineString, Point
from shapely.geometry.polygon import Polygon
from dfastbe.io import BaseRiverData, ConfigFile, CenterLine, log_text, SimulationData


MAX_RIVER_WIDTH = 1000


class SearchLines:

    def __init__(self, lines: List[LineString], mask: CenterLine = None):
        """Search lines initialization.

        Args:
            lines (List[LineString]):
                List of search lines.
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

        Arg:
            max_river_width: float
                Maximum distance away from river_profile.

        Returns:
            List[LineString]: List of clipped search lines.
            float: Maximum distance from any point within line to reference line.
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
                distance_min = max_river_width
                i_min = 0
                for i in range(len(search_lines[ind])):
                    line_simplified = search_lines[ind][i].simplify(1)
                    distance_min_i = line_simplified.distance(profile_simplified)
                    if distance_min_i < distance_min:
                        distance_min = distance_min_i
                        i_min = i
                search_lines[ind] = search_lines[ind][i_min]

            # Determine the maximum distance from a point on this line to the reference line.
            line_simplified = search_lines[ind].simplify(1)
            max_distance = max(
                [Point(c).distance(profile_simplified) for c in line_simplified.coords]
            )

            # Increase the value of max_distance by 2 to account for error introduced by using simplified lines.
            max_distance = max(max_distance, max_distance + 2)

        return search_lines, max_distance

    def to_polygons(self) -> List[Polygon]:
        """
        Construct a series of polygons surrounding the bank search lines.

        Returns:
            bank_areas:
                Array containing the areas of interest surrounding the bank search lines.
        """
        bank_areas = [None] * self.size
        for b, distance in enumerate(self.d_lines):
            bank_areas[b] = self.values[b].buffer(distance, cap_style=2)

        return bank_areas


class BankLinesRiverData(BaseRiverData):

    def __init__(self, config_file: ConfigFile):
        super().__init__(config_file)

    @property
    def search_lines(self) -> SearchLines:
        search_lines = SearchLines(self.config_file.get_search_lines(), self.river_center_line)
        search_lines.d_lines = self.config_file.get_bank_search_distances(search_lines.size)
        return search_lines

    def _get_bank_lines_simulation_data(self) -> Tuple[SimulationData, float]:
        """
        read simulation data and drying flooding threshold dh0
        """
        sim_file = self.config_file.get_sim_file("Detect", "")
        log_text("read_simdata", data={"file": sim_file})
        simulation_data = SimulationData.read(sim_file)
        # increase critical water depth h0 by flooding threshold dh0
        # get critical water depth used for defining bank line (default = 0.0 m)
        critical_water_depth = self.config_file.get_float(
            "Detect", "WaterDepth", default=0
        )
        h0 = critical_water_depth + simulation_data.dry_wet_threshold
        return simulation_data, h0

    def simulation_data(self, bank_lines: bool = True) -> Dict[SimulationData, float]:
        simulation_data, h0 = self._get_bank_lines_simulation_data()
        # clip simulation data to boundaries ...
        log_text("clip_data")
        simulation_data.clip(self.river_center_line.values, self.search_lines.max_distance)
        data = {
            "simulation_data": simulation_data,
            "h0": h0,
        }

        return data