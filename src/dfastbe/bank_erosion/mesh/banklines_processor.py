import numpy as np
from dfastbe.bank_erosion.mesh.data_models import MeshData
from dfastbe.bank_erosion.data_models.calculation import (
    BankData,
    SingleBank,
    FairwayData
)
from dfastbe.bank_erosion.data_models.inputs import ErosionRiverData
from dfastbe.io.data_models import LineGeometry
from dfastbe.io.logger import log_text
from dfastbe.utils import on_right_side
from dfastbe.bank_erosion.mesh.processor import MeshProcessor


class BankLinesProcessor:
    """Class to process bank lines and intersect them with a mesh."""

    def __init__(self, river_data: ErosionRiverData, mesh_data: MeshData):
        """Constructor for BankLinesProcessor."""
        self.bank_lines = river_data.bank_lines
        self.river_center_line = river_data.river_center_line.as_array()
        self.num_bank_lines = len(self.bank_lines)
        self.mesh_data = mesh_data
        self.river_data = river_data

    def get_fairway_data(self, river_axis: LineGeometry) :
        log_text("chainage_to_fairway")
        # intersect fairway and mesh
        fairway_intersection_coords, fairway_face_indices = MeshProcessor(
            river_axis.as_array(), self.mesh_data
        ).intersect_line_mesh()

        if self.river_data.debug:
            arr = (
                          fairway_intersection_coords[:-1] + fairway_intersection_coords[1:]
                  ) / 2
            line_geom = LineGeometry(arr, crs=river_axis.crs)
            line_geom.to_file(
                file_name=f"{str(self.river_data.output_dir)}/fairway_face_indices.shp",
                data={"iface": fairway_face_indices},
            )

        return FairwayData(fairway_face_indices, fairway_intersection_coords)

    def intersect_with_mesh(self) -> BankData:
        """Intersect bank lines with a mesh and return bank data.

        Args:
            mesh_data: Mesh data containing face coordinates and connectivity information.

        Returns:
            BankData object containing bank line coordinates, face indices, and other bank-related data.
        """
        n_bank_lines = len(self.bank_lines)

        bank_line_coords = []
        bank_face_indices = []
        for bank_index in range(n_bank_lines):
            line_coords = np.array(self.bank_lines.geometry[bank_index].coords)
            log_text("bank_nodes", data={"ib": bank_index + 1, "n": len(line_coords)})

            coords_along_bank, face_indices = MeshProcessor(
                line_coords, self.mesh_data
            ).intersect_line_mesh()
            bank_line_coords.append(coords_along_bank)
            bank_face_indices.append(face_indices)

        # linking bank lines to chainage
        log_text("chainage_to_banks")
        bank_chainage_midpoints = [None] * n_bank_lines
        is_right_bank = [True] * n_bank_lines
        for bank_index, coords in enumerate(bank_line_coords):
            segment_mid_points = LineGeometry((coords[:-1, :] + coords[1:, :]) / 2)
            chainage_mid_points = segment_mid_points.intersect_with_line(
                self.river_center_line
            )

            # check if the bank line is defined from low chainage to high chainage
            if chainage_mid_points[0] > chainage_mid_points[-1]:
                # if not, flip the bank line and all associated data
                chainage_mid_points = chainage_mid_points[::-1]
                bank_line_coords[bank_index] = bank_line_coords[bank_index][::-1, :]
                bank_face_indices[bank_index] = bank_face_indices[bank_index][::-1]

            bank_chainage_midpoints[bank_index] = chainage_mid_points

            # check if the bank line is a left or right bank
            # when looking from low-to-high chainage
            is_right_bank[bank_index] = on_right_side(
                coords, self.river_center_line[:, :2]
            )
            if is_right_bank[bank_index]:
                log_text("right_side_bank", data={"ib": bank_index + 1})
            else:
                log_text("left_side_bank", data={"ib": bank_index + 1})

        bank_order = tuple("right" if val else "left" for val in is_right_bank)
        data = {
            'is_right_bank': is_right_bank,
            'bank_line_coords': bank_line_coords,
            'bank_face_indices': bank_face_indices,
            'bank_chainage_midpoints': bank_chainage_midpoints
        }
        return BankData.from_column_arrays(
            data,
            SingleBank,
            bank_lines=self.bank_lines,
            n_bank_lines=n_bank_lines,
            bank_order=bank_order,
        )