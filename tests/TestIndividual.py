from numpy import sin
from numpy.random import rand
from os import path
from sys import path as sys_path
import unittest

sys_path.append(path.abspath(path.join(path.dirname(__file__), "../standardgp/")))

from Individual import Individual
from SearchSpace import SearchSpace
from GP import dotdict


class TestIndividual(unittest.TestCase):
    def test_size(self):
        x = rand(200, 5)
        y = sin(x[:, 0] * x[:, 0])
        cfg: dotdict = dotdict(
            {
                "max_nodes": 24,
                "cache_size": 500000,
            }
        )
        space = SearchSpace(x, y, cfg)
        indi = Individual(cfg, space)
        self.assertEqual(indi.tree_cnt, indi.genome.node_cnt)
        self.assertEqual(indi.tree_cnt, len(indi.node_refs))

    def test_delete(self):
        x = rand(200, 5)
        y = sin(x[:, 0] * x[:, 0])
        cfg: dotdict = dotdict(
            {
                "max_nodes": 24,
                "cache_size": 500000,
            }
        )
        space = SearchSpace(x, y, cfg)
        for x in range(100):
            indi = Individual(cfg, space)
            fit_before = indi.get_fit()
            indi.simplify().simplify().simplify()
            self.assertEqual(fit_before, indi.get_fit())


if __name__ == "__main__":
    unittest.main()
