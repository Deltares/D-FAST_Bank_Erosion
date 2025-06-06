from unittest.mock import MagicMock

import numpy as np
import pytest

from dfastbe.bank_erosion.mesh_processor import (
    _get_slices,
    _get_slices_core,
    enlarge,
    get_slices_ab,
    intersect_line_mesh,
)


class TestMeshProcessor:

    @pytest.fixture
    def mesh_data(self) -> MagicMock:
        """Fixture to provide a mock mesh data object."""
        mesh_data = MagicMock()
        # Define two adjacent square faces (quads) with 4 nodes each, sharing an edge
        mesh_data.x_face_coords = np.array(
            [
                [209253.125, 209252.734375, 209271.921875, 209273.3125],
                [209252.734375, 209253.046875, 209271.3125, 209271.921875],
                [209271.921875, 209271.3125, 209290.453125, 209292.125],
                [209271.3125, 209271.40625, 209289.546875, 209290.453125],
            ]
        )
        mesh_data.y_face_coords = np.array(
            [
                [389624.40625, 389663.96875, 389664.25, 389625.09375],
                [389663.96875, 389704.0, 389703.96875, 389664.25],
                [389664.25, 389703.96875, 389704.34375, 389665.0625],
                [389703.96875, 389744.0625, 389744.09375, 389704.34375],
            ]
        )
        mesh_data.face_node = np.array(
            [[2, 0, 5, 6], [0, 1, 3, 5], [5, 3, 8, 9], [3, 4, 7, 8]]
        )
        mesh_data.n_nodes = np.array([4] * 4)
        mesh_data.x_edge_coords = np.array(
            [
                [209253.125, 209273.3125],
                [209253.125, 209252.734375],
                [209252.734375, 209271.921875],
                [209273.3125, 209271.921875],
                [209252.734375, 209271.921875],
                [209252.734375, 209253.046875],
                [209253.046875, 209271.3125],
                [209271.921875, 209271.3125],
                [209271.921875, 209292.125],
                [209271.921875, 209271.3125],
                [209271.3125, 209290.453125],
                [209292.125, 209290.453125],
                [209271.3125, 209290.453125],
                [209271.3125, 209271.40625],
                [209271.40625, 209289.546875],
                [209290.453125, 209289.546875],
            ]
        )
        mesh_data.y_edge_coords = np.array(
            [
                [389624.40625, 389625.09375],
                [389624.40625, 389663.96875],
                [389663.96875, 389664.25],
                [389625.09375, 389664.25],
                [389663.96875, 389664.25],
                [389663.96875, 389704.0],
                [389704.0, 389703.96875],
                [389664.25, 389703.96875],
                [389664.25, 389665.0625],
                [389664.25, 389703.96875],
                [389703.96875, 389704.34375],
                [389665.0625, 389704.34375],
                [389703.96875, 389704.34375],
                [389703.96875, 389744.0625],
                [389744.0625, 389744.09375],
                [389704.34375, 389744.09375],
            ]
        )
        mesh_data.edge_node = np.array(
            [
                [2, 6],
                [2, 0],
                [0, 5],
                [6, 5],
                [0, 5],
                [0, 1],
                [1, 3],
                [5, 3],
                [5, 9],
                [5, 3],
                [3, 8],
                [9, 8],
                [3, 8],
                [3, 4],
                [4, 7],
                [8, 7],
            ]
        )
        mesh_data.edge_face_connectivity = np.array(
            [
                [0, -1],
                [0, -1],
                [0, 1],
                [0, -1],
                [-1, -1],
                [1, -1],
                [1, -1],
                [1, 2],
                [2, -1],
                [-1, -1],
                [2, 3],
                [2, -1],
                [-1, -1],
                [3, -1],
                [3, -1],
                [3, -1],
            ]
        )
        mesh_data.face_edge_connectivity = np.array(
            [[1, 2, 3, 0], [5, 6, 7, 2], [7, 10, 11, 8], [13, 14, 15, 10]]
        )
        mesh_data.boundary_edge_nrs = np.arange(16)
        return mesh_data

    @pytest.mark.parametrize(
        "line,expected_coords,expected_idx",
        [
            (
                np.array(
                    [
                        [209266.44709443, 389650.16238121],
                        [209266.44709443, 389651.16238121],
                    ]
                ),
                np.array(
                    [
                        [209266.44709443, 389650.16238121],
                        [209266.44709443, 389651.16238121],
                    ]
                ),
                [0],
            ),
            (
                np.array(
                    [
                        [209266.44709443, 389650.16238121],
                        [209269.67183787, 389664.217019],
                    ]
                ),
                np.array(
                    [
                        [209266.44709443, 389650.16238121],
                        [209269.67183787, 389664.217019],
                    ]
                ),
                [0],
            ),
            (
                np.array(
                    [
                        [209266.44709443, 389650.16238121],
                        [209269.67183787, 389664.217019],
                        [209271.7614607, 389674.70572161],
                    ]
                ),
                np.array(
                    [
                        [209266.44709443, 389650.16238121],
                        [209269.67183787, 389664.217019],
                        [209271.7614607, 389674.70572161],
                    ]
                ),
                [0, 1],
            ),
            (
                np.array(
                    [
                        [209266.44709443, 389650.16238121],
                        [209269.67183787, 389664.217019],
                        [209271.7614607, 389674.70572161],
                        [209278.48314731, 389704.10923615],
                    ]
                ),
                np.array(
                    [
                        [209266.44709443, 389650.16238121],
                        [209269.67183787, 389664.217019],
                        [209271.7614607, 389674.70572161],
                        [209278.48314731, 389704.10923615],
                    ]
                ),
                [0, 1, 2],
            ),
            (
                np.array(
                    [
                        [209266.44709443, 389650.16238121],
                        [209260.0, 389660.0],
                        [209271.7614607, 389674.70572161],
                    ]
                ),
                np.array(
                    [
                        [209266.44709443, 389650.16238121],
                        [209260.0, 389660.0],
                        [209263.2979992, 389664.1235914],
                        [209271.7614607, 389674.70572161],
                    ]
                ),
                [0, 0, 1],
            ),
            (
                np.array(
                    [
                        [209266.44709443, 389650.16238121],
                        [209260.0, 389660.0],
                        [209261.0, 389660.0],
                        [209271.7614607, 389674.70572161],
                    ]
                ),
                np.array(
                    [
                        [209266.44709443, 389650.16238121],
                        [209260.0, 389660.0],
                        [209261.0, 389660.0],
                        [209264.02539435, 389664.13425354],
                        [209271.7614607, 389674.70572161],
                    ]
                ),
                [0, 0, 0, 1],
            ),
            (
                np.array([[100, 100], [120, 120]]),
                np.array([[100, 100], [120, 120]]),
                [-1],
            ),
        ],
        ids=[
            "Within one quad",
            "Match one quad",
            "Match two quads",
            "Match three quads",
            "Match two quads, outside one",
            "Match two quads, outside two",
            "Outside mesh",
        ],
    )
    def test_intersect_line_mesh(self, mesh_data, line, expected_coords, expected_idx):
        crds, idx = intersect_line_mesh(line, mesh_data)
        assert np.allclose(crds, expected_coords)
        assert np.array_equal(idx, expected_idx)

    @pytest.mark.parametrize(
        "line,shape,expected_enlarged_line",
        [
            (
                np.array([[2.2, 4.3], [3.2, 4.3]]),
                (2, 4),
                np.array([[2.2, 4.3, 0.0, 0.0], [3.2, 4.3, 0.0, 0.0]]),
            ),
            (
                np.array([[2.4, 4.1], [2.4, 3.6]]),
                (4, 2),
                np.array([[2.4, 4.1], [2.4, 3.6], [0.0, 0.0], [0.0, 0.0]]),
            ),
            (np.array([1.2, 3.1]), (4,), np.array([1.2, 3.1, 0.0, 0.0])),
        ],
        ids=["Enlarge to (2, 4)", "Enlarge to (4, 2)", "Enlarge to (4,)"],
    )
    def test_enlarge(self, line, shape, expected_enlarged_line):
        # Test with a line that intersects the mesh
        enlarged_line = enlarge(line, shape)
        assert np.allclose(enlarged_line, expected_enlarged_line)

    def test_get_slices_ab(self):
        X0 = np.ma.array([209171.296875, 209171.296875, 209171.484375, 209188.1875])
        Y0 = np.ma.array([389625.15625, 389625.15625, 389665.5625, 389624.96875])
        X1 = np.ma.array([209188.1875, 209171.484375, 209189.09375, 209189.09375])
        Y1 = np.ma.array([389624.96875, 389665.5625, 389665.375, 389665.375])
        xi0 = 209186.621094
        xi1 = 209189.367188
        yi0 = 389659.99609375
        yi1 = 389673.75
        a, b, slices = get_slices_ab(X0, Y0, X1, Y1, xi0, yi0, xi1, yi1, 0)
        expected_a = np.ma.array([0.9207387758922553])
        expected_b = np.ma.array([0.3921626068608838])
        expected_slices = np.array([2])
        assert np.allclose(a, expected_a)
        assert np.allclose(b, expected_b)
        assert np.array_equal(slices, expected_slices)

    def test_get_slices(self, mesh_data):
        index = 1
        prev_b = 0.0
        bpj = np.array([209266.44709443, 389670.16238121])
        bpj1 = np.array([209266.44709443, 389651.16238121])
        b, edges, nodes = _get_slices(index, prev_b, bpj, bpj1, mesh_data)
        assert np.allclose(b, np.array([0.6845984]))
        assert np.array_equal(edges, np.array([2]))
        assert np.array_equal(nodes, np.array([-1]))
