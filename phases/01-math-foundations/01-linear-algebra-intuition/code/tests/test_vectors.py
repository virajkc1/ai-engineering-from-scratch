"""Tests for the Linear Algebra Intuition lesson.
Lesson: phases/01-math-foundations/01-linear-algebra-intuition/docs/en.md
Run from the lesson's code/ directory:
    python3 -m unittest discover tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectors import Matrix, Vector, gram_schmidt, is_independent


class TestVector(unittest.TestCase):
    def test_add_and_subtract(self):
        a, b = Vector([1, 2, 3]), Vector([4, 5, 6])
        self.assertEqual((a + b).components, [5, 7, 9])
        self.assertEqual((b - a).components, [3, 3, 3])

    def test_dot_product(self):
        self.assertEqual(Vector([1, 2, 3]).dot(Vector([4, 5, 6])), 32)
        self.assertEqual(Vector([1, 0]).dot(Vector([0, 1])), 0)

    def test_magnitude_and_normalize(self):
        self.assertAlmostEqual(Vector([3, 4]).magnitude(), 5.0)
        self.assertAlmostEqual(Vector([3, 4]).normalize().magnitude(), 1.0)

    def test_angle_between(self):
        self.assertAlmostEqual(Vector([1, 0]).angle_between(Vector([0, 1])), 90.0)
        self.assertAlmostEqual(Vector([1, 0]).angle_between(Vector([1, 0])), 0.0)

    def test_projection_residual_is_orthogonal(self):
        a, b = Vector([3, 4]), Vector([1, 0])
        proj = a.project_onto(b)
        self.assertEqual(proj.components, [3.0, 0.0])
        self.assertAlmostEqual((a - proj).dot(b), 0.0)


class TestIndependenceAndBasis(unittest.TestCase):
    def test_standard_basis_is_independent(self):
        e = [Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1])]
        self.assertTrue(is_independent(e))

    def test_dependent_set_detected(self):
        v1, v2 = Vector([1, 0, 0]), Vector([0, 1, 0])
        self.assertFalse(is_independent([v1, v2, Vector([2, 1, 0])]))

    def test_gram_schmidt_is_orthonormal(self):
        basis = gram_schmidt([Vector([1, 1, 0]), Vector([1, 0, 1]), Vector([0, 1, 1])])
        self.assertEqual(len(basis), 3)
        for u in basis:
            self.assertAlmostEqual(u.magnitude(), 1.0, places=9)
        for i in range(len(basis)):
            for j in range(i + 1, len(basis)):
                self.assertAlmostEqual(basis[i].dot(basis[j]), 0.0, places=9)


class TestMatrix(unittest.TestCase):
    def test_rotation_90_degrees(self):
        rotated = Matrix([[0, -1], [1, 0]]) @ Vector([3, 1])
        self.assertEqual(rotated.components, [-1, 3])

    def test_matmul_and_transpose(self):
        a = Matrix([[1, 2], [3, 4]])
        self.assertEqual((a @ Matrix([[1, 0], [0, 1]])).rows, [[1, 2], [3, 4]])
        self.assertEqual(a.transpose().rows, [[1, 3], [2, 4]])

    def test_rank(self):
        self.assertEqual(Matrix([[1, 0], [0, 1]]).rank(), 2)
        self.assertEqual(Matrix([[1, 2], [2, 4]]).rank(), 1)
        self.assertEqual(Matrix([[1, 0, 0], [0, 1, 0]]).rank(), 2)


if __name__ == "__main__":
    unittest.main()
