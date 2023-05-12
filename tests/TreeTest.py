import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../standardgp/'
)))

try:
    from standardgp.Individual import Individual
    from standardgp.SearchSpace import SearchSpace
    from standardgp.GP import Config, GP
except Exception:
    from Individual import Individual
    from SearchSpace import SearchSpace
    from GP import Config, GP

from numpy import array


class TestStringMethods(unittest.TestCase):

    def test_size(self):
        cfg = Config()
        space = SearchSpace(
            array([[1, 2], [2, 3], [3, 4], [6, 5]]),
            array([1, 2, 3, 4]),
            cfg
        )
        indi = Individual(cfg, space)
        self.assertEqual(indi.tree_cnt, indi.genome.node_cnt)
        self.assertEqual(indi.tree_cnt, len(indi.node_refs))

    def test_delete(self):
        cfg = Config()
        cfg.operators = 0.4
        cfg.functions = 0.4
        space = SearchSpace(
            array([[1, 2], [2, 3], [3, 4], [6, 5]]),
            array([1, 2, 3, 4]),
            cfg
        )
        for x in range(100):
            indi = Individual(cfg, space)
            fit_before = indi.get_fit(indi.genome)
            indi.simplify().simplify().simplify()
            self.assertEqual(fit_before, indi.get_fit(indi.genome))

    def test_stupid_configs(self):
        gp = GP(
            array([[1, 2], [2, 3], [3, 4], [6, 5]]),
            array([1, 2, 3, 4]),
            cfg={"mutation": 0.0}
        )
        gp = GP(
            array([[1, 2], [2, 3], [3, 4], [6, 5]]),
            array([1, 2, 3, 4]),
            cfg={"pop_size": 0}
        )
        gp = GP(
            array([[1, 2], [2, 3], [3, 4], [6, 5]]),
            array([1, 2, 3, 4]),
            cfg={"operators": 1, "functions": 1}
        )


if __name__ == '__main__':
    unittest.main()
