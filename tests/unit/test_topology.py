"""Connectivity, immutability, and generic edge-qubit stabilizer checks."""

from dataclasses import FrozenInstanceError
import unittest

import numpy as np

from quantum_lattice.lattice import Lattice, square_lattice
from quantum_lattice.lattice.square import star_syndrome as legacy_star_syndrome


def tetrahedral_lattice() -> Lattice:
    """A closed sphere fixture: degree-three vertices and triangular faces."""
    return Lattice(
        4,
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        [(0, 3, 1), (0, 4, 2), (1, 5, 2), (3, 5, 4)],
        geometry="tetrahedral_fixture",
        boundary="closed",
    )


class TopologyTests(unittest.TestCase):
    def test_square_ids_degrees_and_object_navigation(self):
        lattice = square_lattice(3)
        self.assertEqual((lattice.n_vertices, lattice.n_edges, lattice.n_faces), (9, 18, 9))
        self.assertEqual(lattice.shape, (3, 3))
        self.assertEqual([vertex.id for vertex in lattice.vertices], list(range(9)))
        self.assertEqual([edge.id for edge in lattice.edges], list(range(18)))
        self.assertEqual([face.id for face in lattice.faces], list(range(9)))
        np.testing.assert_array_equal(lattice.vertex_degrees, np.full(9, 4))
        np.testing.assert_array_equal(lattice.face_degrees, np.full(9, 4))
        self.assertEqual(lattice.vertices[0].position, (0.0, 0.0))
        self.assertEqual(lattice.vertices[0].edge_ids, (0, 1, 4, 13))
        self.assertEqual(lattice.faces[0].edge_ids, (0, 3, 6, 1))
        for vertex in lattice.vertices:
            for edge in vertex.incident_edges:
                self.assertIs(edge, lattice.edges[edge.id])
                self.assertIn(vertex, edge.endpoints)
        for edge in lattice.edges:
            self.assertEqual(len(edge.incident_faces), 2)
            for face in edge.incident_faces:
                self.assertIs(face, lattice.faces[face.id])
                self.assertIn(edge, face.boundary_edges)

    def test_periodic_wrap_and_original_edge_flatten_order(self):
        lattice = square_lattice(4)
        self.assertEqual(lattice.edges[2 * 3].vertex_ids, (3, 0))
        self.assertEqual(lattice.edges[2 * 12 + 1].vertex_ids, (12, 0))
        for y in range(4):
            for x in range(4):
                vertex = 4 * y + x
                self.assertEqual(lattice.edges[2 * vertex].vertex_ids, (vertex, 4 * y + (x + 1) % 4))
                self.assertEqual(lattice.edges[2 * vertex + 1].vertex_ids, (vertex, 4 * ((y + 1) % 4) + x))

    def test_objects_and_compiled_arrays_are_immutable(self):
        lattice = square_lattice(3)
        for instance, attribute, value in (
            (lattice, "geometry", "changed"),
            (lattice.vertices[0], "edge_ids", ()),
            (lattice.edges[0], "vertex_ids", (1, 2)),
            (lattice.faces[0], "edge_ids", ()),
        ):
            with self.subTest(attribute=attribute):
                with self.assertRaises(FrozenInstanceError):
                    setattr(instance, attribute, value)
        for name in (
            "edge_vertex_indices", "vertex_degrees", "face_degrees",
            "vertex_edge_indices", "vertex_edge_mask", "face_edge_indices", "face_edge_mask",
        ):
            array = getattr(lattice, name)
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.flat[0] = 0
            with self.assertRaises(ValueError):
                array.setflags(write=True)

    def test_constructor_does_not_alias_input_connectivity(self):
        endpoints = np.array([(0, 1), (1, 2), (2, 0)])
        boundary = [0, 1, 2]
        positions = [[0, 0], [1, 0], [0, 1]]
        lattice = Lattice(3, endpoints, [boundary], positions=positions, boundary="open")
        endpoints[:] = 0
        boundary[:] = []
        positions[0][0] = 99
        self.assertEqual(lattice.edges[0].vertex_ids, (0, 1))
        self.assertEqual(lattice.faces[0].edge_ids, (0, 1, 2))
        self.assertEqual(lattice.vertices[0].position, (0.0, 0.0))

    def test_generic_square_matches_original_for_identical_error_flags(self):
        rng = np.random.default_rng(4123)
        for length in (3, 4, 8):
            lattice = square_lattice(length)
            errors = rng.integers(0, 2, size=(2, 7, length, length, 2), dtype=np.uint8)
            flat = errors.reshape(2, 7, lattice.n_edges)
            np.testing.assert_array_equal(
                lattice.star_syndrome(flat).reshape(2, 7, length, length),
                legacy_star_syndrome(errors),
            )
            np.testing.assert_array_equal(
                lattice.defect_fractions(flat),
                legacy_star_syndrome(errors).mean(axis=(-2, -1)),
            )

    def test_each_single_edge_flips_exactly_its_two_vertices(self):
        lattice = square_lattice(3)
        errors = np.eye(lattice.n_edges, dtype=bool)
        syndromes = lattice.star_syndrome(errors)
        for edge in lattice.edges:
            np.testing.assert_array_equal(np.flatnonzero(syndromes[edge.id]), sorted(edge.vertex_ids))

    def test_closed_topology_even_global_star_and_plaquette_parity(self):
        rng = np.random.default_rng(910)
        for lattice in (square_lattice(5), tetrahedral_lattice()):
            errors = rng.integers(0, 2, size=(100, lattice.n_edges))
            self.assertTrue(np.all(lattice.star_syndrome(errors).sum(axis=-1) % 2 == 0))
            self.assertTrue(np.all(lattice.plaquette_syndrome(errors).sum(axis=-1) % 2 == 0))

    def test_star_and_face_supports_commute_for_square_and_tetrahedron(self):
        for lattice in (square_lattice(3), tetrahedral_lattice()):
            face_patterns = np.zeros((lattice.n_faces, lattice.n_edges), dtype=bool)
            for face in lattice.faces:
                face_patterns[face.id, list(face.edge_ids)] = True
            self.assertFalse(np.any(lattice.star_syndrome(face_patterns)))
            star_patterns = np.zeros((lattice.n_vertices, lattice.n_edges), dtype=bool)
            for vertex in lattice.vertices:
                star_patterns[vertex.id, list(vertex.edge_ids)] = True
            self.assertFalse(np.any(lattice.plaquette_syndrome(star_patterns)))

    def test_tetrahedral_non_square_operators_exhaustively(self):
        lattice = tetrahedral_lattice()
        self.assertEqual((lattice.n_vertices, lattice.n_edges, lattice.n_faces), (4, 6, 4))
        np.testing.assert_array_equal(lattice.vertex_degrees, [3, 3, 3, 3])
        np.testing.assert_array_equal(lattice.face_degrees, [3, 3, 3, 3])
        patterns = (np.arange(64)[:, None] >> np.arange(6)) & 1
        expected_stars = np.array([
            [sum(pattern[list(vertex.edge_ids)]) % 2 for vertex in lattice.vertices]
            for pattern in patterns
        ], dtype=bool)
        expected_faces = np.array([
            [sum(pattern[list(face.edge_ids)]) % 2 for face in lattice.faces]
            for pattern in patterns
        ], dtype=bool)
        np.testing.assert_array_equal(lattice.star_syndrome(patterns), expected_stars)
        np.testing.assert_array_equal(lattice.plaquette_syndrome(patterns), expected_faces)
        self.assertTrue(np.all(lattice.star_syndrome(np.ones(6, dtype=bool))))
        self.assertTrue(np.all(lattice.plaquette_syndrome(np.ones(6, dtype=bool))))

    def test_irregular_graph_padding_and_isolated_vertices(self):
        lattice = Lattice(4, [(0, 1), (1, 2)], boundary="graph")
        np.testing.assert_array_equal(lattice.vertex_degrees, [1, 2, 1, 0])
        patterns = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        np.testing.assert_array_equal(lattice.star_syndrome(patterns), [
            [0, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0],
        ])
        self.assertEqual(lattice.plaquette_syndrome(patterns).shape, (4, 0))

    def test_empty_graph_operators(self):
        lattice = Lattice(2, [])
        np.testing.assert_array_equal(lattice.star_syndrome(np.empty((3, 0), dtype=bool)), np.zeros((3, 2), dtype=bool))
        np.testing.assert_array_equal(lattice.defect_fractions(np.empty(0, dtype=bool)), 0.0)

    def test_parallel_edges_are_distinct_qubits(self):
        lattice = Lattice(2, [(0, 1), (0, 1)])
        np.testing.assert_array_equal(lattice.vertex_degrees, [2, 2])
        np.testing.assert_array_equal(lattice.star_syndrome([1, 1]), [False, False])

    def test_reject_invalid_edges_and_metadata(self):
        bad_inputs = [
            (0, [], {}), (True, [], {}), (2, [(0, 0)], {}),
            (2, [(0, 2)], {}), (2, [(-1, 1)], {}),
            (2, [(0.0, 1.0)], {}), (2, [[0, 1, 0]], {}),
            (2, [(0, 1)], {"boundary": "unknown"}),
            (2, [(0, 1)], {"positions": [(0, 0)]}),
            (2, [(0, 1)], {"positions": [(0, np.inf), None]}),
            (2, [(0, 1)], {"shape": (2, 0)}),
        ]
        for vertices, edges, kwargs in bad_inputs:
            with self.subTest(vertices=vertices, edges=edges, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    Lattice(vertices, edges, **kwargs)

    def test_reject_invalid_face_boundaries(self):
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        for face in ([0, 1], [0, 1, 1], [0, 1, 5], [0, 2, 1, 3, 4], [0, 1.0, 2]):
            with self.subTest(face=face):
                with self.assertRaises(ValueError):
                    Lattice(5, edges, [face], boundary="open")
        with self.assertRaisesRegex(ValueError, "two faces per edge"):
            Lattice(5, edges, [range(5)], boundary="periodic")
        with self.assertRaisesRegex(ValueError, "more than two faces"):
            Lattice(5, edges, [range(5), range(5), range(5)], boundary="graph")
        with self.assertRaisesRegex(ValueError, "no isolated vertices"):
            Lattice(2, [], boundary="closed")

    def test_reject_invalid_error_shapes_and_nonbinary_flags(self):
        lattice = tetrahedral_lattice()
        for errors in (0, np.zeros(5), np.zeros((2, 7)), [0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, np.nan]):
            with self.subTest(errors=errors):
                with self.assertRaises(ValueError):
                    lattice.star_syndrome(errors)

    def test_square_builder_rejects_degenerate_tori(self):
        for length in (True, 2, 3.0):
            with self.assertRaises(ValueError):
                square_lattice(length)


if __name__ == "__main__":
    unittest.main()
