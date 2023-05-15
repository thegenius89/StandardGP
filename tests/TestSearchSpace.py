from numpy import sin
from numpy.random import rand
from os import path
from sys import path as sys_path
import unittest

sys_path.append(
    path.abspath(path.join(path.dirname(__file__), "../standardgp/"))
)

from SearchSpace import SearchSpace
from GP import dotdict


class TestSearchSpace(unittest.TestCase):
    def test_numpy_math_equivalence(self):
        x = rand(200, 5)
        y = sin(x[:, 0] * x[:, 0])
        cfg: dotdict = dotdict({
            "max_nodes": 24,
        })
        space = SearchSpace(x, y, cfg)
        self.assertEqual(space.sq(1), space.hsq(1))
        self.assertEqual(space.sq(0), space.hsq(0))
        self.assertEqual(space.sq(-1), space.hsq(-1))
        self.assertEqual(space.sq(50000), space.hsq(50000))
        self.assertEqual(space.psqrt(1), space.hpsqrt(1))
        self.assertEqual(space.psqrt(0), space.hpsqrt(0))
        self.assertEqual(space.psqrt(-1), space.hpsqrt(-1))
        self.assertEqual(space.psqrt(50000), space.psqrt(50000))
        self.assertEqual(space.plog(1), space.hplog(1))
        self.assertEqual(space.plog(0), space.hplog(0))
        self.assertEqual(space.plog(-1), space.hplog(-1))
        self.assertEqual(space.plog(50000), space.hplog(50000))
        self.assertEqual(space.pexp(1), space.hpexp(1))
        self.assertEqual(space.pexp(0), space.hpexp(0))
        self.assertEqual(space.pexp(-1), space.hpexp(-1))
        self.assertEqual(space.pexp(5), space.hpexp(5))
        self.assertEqual(space.pexp(-5), space.hpexp(-5))
        self.assertEqual(space.pexp(50000), space.hpexp(50000))
        self.assertEqual(space.pdiv(1, 0), space.hpdiv(1, 0))
        self.assertEqual(space.pdiv(0, 1), space.hpdiv(0, 1))
        self.assertEqual(space.pdiv(-1, -1), space.hpdiv(-1, -1))
        self.assertEqual(space.pdiv(50000, 0.1), space.hpdiv(50000, 0.1))



if __name__ == "__main__":
    unittest.main()
