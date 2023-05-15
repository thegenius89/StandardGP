from numpy import sin
from numpy.random import rand
from os import path
from sys import path as sys_path
import unittest

sys_path.append(
    path.abspath(path.join(path.dirname(__file__), "../standardgp/"))
)

from Individual import Individual
from SearchSpace import SearchSpace
from GP import GP


class TestGP(unittest.TestCase):
    def test_stupid_configs(self):
        x = rand(200, 5)
        y = sin(x[:, 0] * x[:, 0])
        gp = GP(x, y, cfg={
            "crossovers": 0.0,
            "mutations": 0.0,
            "gens": 2
        })
        gp.run()

        gp = GP(x, y, cfg={
            "pop_size": 0,
            "gens": 2
        })
        gp.run()

        gp = GP(x, y, cfg={
            "gens": 0
        })
        gp.run()
        
        gp = GP(x, y, cfg = {
            "max_nodes": 0,
            "gens": 2
        })
        gp.run()


if __name__ == "__main__":
    unittest.main()
