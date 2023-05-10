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

    def __init__(self, cfg, space: dict):
        self.cfg, self.space = cfg, space
        self.ts = [
            ("x", space["x"]), ("y", space["y"]), ("0", 0.0),
            ("v", space["v"]), ("w", space["w"]), ("0.5", 0.5),
            ("z", space["z"]), ("pi", space["pi"]), ("2", 2.0),
            ("pih", space["pih"]), ("1", 1.0),
        ]
        self.tnts = [
            ("+", add), ("+", add), ("-", subtract), ("*", multiply),
            ("/", space["div"]), ("-", subtract), ("*", multiply),
        ]
        self.onts = [
            ("sin", space["sin"]), ("cos", space["cos"]),
            ("abs", space["abs"]), ("neg", space["neg"]),
            ("log", space["log"]), ("exp", space["exp"]),
            ("sqrt", space["sqrt"]), ("sq", space["sq"]),
            ("tan", space["tan"]),
        ]
        self.mapper = {}
        self.add_keys(space)
        self.ts_iter = cycle([choice(self.ts) for _ in range(1223)])
        self.tnts_iter = cycle([choice(self.tnts) for _ in range(937)])
        self.onts_iter = cycle([choice(self.onts) for _ in range(1069)])
        self.rand_nodes = {}
        for s in range(1, self.cfg.max_nodes * 2):
            self.rand_nodes[s] = cycle([randint(0, s - 1) for _ in range(59)])

    def add_keys(self, space, cnt=0) -> None:
        hash_calculators = {
            "x": rand(), "y": rand(), "z": rand(), "v": rand(), "w": rand(),
            "0": 0.0, "0.5": 0.5,
            "pi": space["pi"], "pih": space["pih"], "tan": tan,
            "1": 1.0, "2": 2.0, "+": hadd, "-": hsub, "*": hmul,
            "/": hpdiv, "log": hplog, "exp": hpexp, "sqrt": hpsqrt,
            "sq": hsq, "abs": abs, "neg": hneg, "sin": sin, "cos": cos,
        }
        for arr in [self.ts, self.tnts, self.onts]:
            for s in arr:
                self.mapper[s[0]] = hash_calculators[s[0]]
                cnt += 1
