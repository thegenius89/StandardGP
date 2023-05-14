import unittest
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../standardgp/"))
)

from Individual import Individual
from SearchSpace import SearchSpace
from GP import GP

from numpy import array


class TestStringMethods(unittest.TestCase):
    def test_size(self):
        space = SearchSpace(
            array([[1, 2], [2, 3], [3, 4], [6, 5]]), array([1, 2, 3, 4]), GP.cfg
        )
        indi = Individual(GP.cfg, space)
        self.assertEqual(indi.tree_cnt, indi.genome.node_cnt)
        self.assertEqual(indi.tree_cnt, len(indi.node_refs))

    def test_delete(self):
        space = SearchSpace(
            array([[1, 2], [2, 3], [3, 4], [6, 5]]), array([1, 2, 3, 4]), GP.cfg
        )
        for x in range(100):
            indi = Individual(GP.cfg, space)
            fit_before = indi.get_fit()
            indi.simplify().simplify().simplify()
            self.assertEqual(fit_before, indi.get_fit())

    def test_stupid_configs(self):
        gp = GP(
            array([[1, 2], [2, 3], [3, 4], [6, 5]]),
            array([1, 2, 3, 4]),
            cfg={"crossovers": 0.0, "mutations": 0.0, "gens": 2},
        )
        gp.run()
        gp = GP(
            array([[1, 2], [2, 3], [3, 4], [6, 5]]),
            array([1, 2, 3, 4]),
            cfg={"pop_size": 0, "gens": 2},
        )
        gp.run()
        gp = GP(
            array([[1, 2], [2, 3], [3, 4], [6, 5]]),
            array([1, 2, 3, 4]),
            cfg={"gens": 0},
        )
        gp.run()
        gp = GP(
            array([[1, 2], [2, 3], [3, 4], [6, 5]]),
            array([1, 2, 3, 4]),
            cfg={"max_nodes": 0, "gens": 2},
        )
        gp.run()


if __name__ == "__main__":
    unittest.main()
