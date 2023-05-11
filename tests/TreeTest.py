import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../standardgp/'
)))

from Individual import Individual
from SearchSpace import SearchSpace
from StandardGP import Config

from numpy import array


class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        cfg = Config()
        space = SearchSpace(array([[1, 2], [2, 3], [2, 4], [6, 5]]),
            array([1, 2, 3, 4]), cfg)
        indi = Individual(cfg, space)
        self.assertEqual(indi.tree_cnt, indi.genome.node_cnt)

if __name__ == '__main__':
    unittest.main()
