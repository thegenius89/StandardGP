# Copyright (c) 2023, Thure Foken.
# All rights reserved.

from itertools import cycle
from numpy import subtract, multiply, add
from numpy.random import rand
from random import choice, randint
from math import sin, cos, tan

from Helper import hsq, hneg, hpsqrt, hplog, hpexp
from Helper import hpdiv, hadd, hsub, hmul


class TreeGrammar:

    def __init__(self, cfg, problem):
        self.cfg, self.space = cfg, problem.namespace
        self.problem = problem
        self.ts = [
            ("0", 0.0), ("0.5", 0.5), ("1", 1.0), ("2", 2.0),
            ("pi", self.space["pi"]), ("pih", self.space["pih"]),
        ]
        for input in range(problem.x_train.shape[0]):
            self.ts.append(("x{}".format(input), problem.x_train[input]))
        self.tnts = [
            ("+", add), ("+", add), ("-", subtract), ("*", multiply),
            ("/", self.space["div"]), ("-", subtract), ("*", multiply),
        ]
        self.onts = [
            ("sin", self.space["sin"]), ("cos", self.space["cos"]),
            ("abs", self.space["abs"]), ("neg", self.space["neg"]),
            ("log", self.space["log"]), ("exp", self.space["exp"]),
            ("sqrt", self.space["sqrt"]), ("sq", self.space["sq"]),
            ("tan", self.space["tan"]),
        ]
        self.mapper = {}
        self.add_keys(self.space)
        self.ts_iter = cycle([choice(self.ts) for _ in range(1223)])
        self.tnts_iter = cycle([choice(self.tnts) for _ in range(937)])
        self.onts_iter = cycle([choice(self.onts) for _ in range(1069)])
        self.rand_nodes = {}
        for s in range(1, self.cfg.max_nodes * 2):
            self.rand_nodes[s] = cycle([randint(0, s - 1) for _ in range(59)])

    def add_keys(self, space, cnt=0) -> None:
        hash_calculators = {
            "pi": space["pi"], "pih": space["pih"], "tan": tan, "0.5": 0.5,
            "1": 1.0, "2": 2.0, "+": hadd, "-": hsub, "*": hmul, "0": 0.0,
            "/": hpdiv, "log": hplog, "exp": hpexp, "sqrt": hpsqrt,
            "sq": hsq, "abs": abs, "neg": hneg, "sin": sin, "cos": cos,
        }
        for k, v in space.items():
            if k not in hash_calculators:
                hash_calculators[k] = rand()
        for arr in [self.ts, self.tnts, self.onts]:
            for s in arr:
                self.mapper[s[0]] = hash_calculators[s[0]]
                cnt += 1
