# Copyright (c) 2023, Thure Foken.
# All rights reserved.

from itertools import cycle
from numpy import ndarray, add, tan, subtract, multiply, divide
from numpy import cos, sin, absolute, pi, negative, where, log, sqrt, exp
from numpy.random import rand
from random import choice, randint


class SearchSpace:
    def __init__(self, x: ndarray, y: ndarray, cfg):
        self.x_train: ndarray = x.T
        self.y_train: ndarray = y
        self.cfg: dict = cfg
        self.space: dict = {
            "sin": sin,
            "cos": cos,
            "tan": tan,
            "abs": absolute,
            "pi": pi,
            "exp": self.pexp,
            "div": self.pdiv,
            "sqrt": self.psqrt,
            "log": self.plog,
            "sq": self.sq,
            "neg": negative,
            "pih": pi * 0.5,
        }
        self.size: float = self.x_train.shape[1] * 1.0
        self.target: ndarray = self.normalize(y)
        self.ts: list = [
            ("0.5", 0.5),
            ("1", 1.0),
            ("2", 2.0),
            ("pi", self.space["pi"]),
            ("pih", self.space["pih"]),
        ]
        for input in range(self.x_train.shape[0]):
            self.space["x{}".format(input)] = self.x_train[input]
            self.ts.append(("x{}".format(input), self.x_train[input]))
        self.tnts: list = [
            ("+", add),
            ("+", add),
            ("-", subtract),
            ("*", multiply),
            ("/", self.space["div"]),
            ("-", subtract),
            ("*", multiply),
        ]
        self.onts: list = [
            ("sin", self.space["sin"]),
            ("cos", self.space["cos"]),
            ("log", self.space["log"]),
            ("exp", self.space["exp"]),
            ("sqrt", self.space["sqrt"]),
            ("sq", self.space["sq"]),
            ("tan", self.space["tan"]),
        ]
        self.build_hash_system(self.space)
        # provide random nodes that the algorithm just need to call
        # next(...) for fast execution
        self.ts_iter = cycle([choice(self.ts) for _ in range(1223)])
        self.tnts_iter = cycle([choice(self.tnts) for _ in range(937)])
        self.onts_iter = cycle([choice(self.onts) for _ in range(1069)])
        self.rand_nodes: dict = {}
        for s in range(1, self.cfg.max_nodes * 3):
            self.rand_nodes[s] = cycle([randint(0, s - 1) for _ in range(119)])

    def normalize(self, pred: ndarray) -> ndarray:
        pred: ndarray = pred - add.reduce(pred) / self.size
        pred: ndarray = pred / (add.reduce(pred * pred) ** 0.5)
        return pred

    def reconstruct_invariances(self, repr) -> str:
        try:
            from scipy.stats import linregress

            x: ndarray = eval(repr, self.space)
            y: ndarray = self.y_train
            slope, intercept, _r, _p, _std_err = linregress(x, y)
            slope, intercept = round(slope, 8), round(intercept, 8)
            return "{} + {} * ({})".format(intercept, slope, repr)
        except Exception:
            return repr

    def build_hash_system(self, space, cnt=0) -> None:
        import math

        # topological hash functions
        hash_calculators = {
            "pi": space["pi"],
            "pih": space["pih"],
            "tan": math.tan,
            "1": 1.0,
            "2": 2.0,
            "+": self.hadd,
            "-": self.hsub,
            "*": self.hmul,
            "0": 0.0,
            "/": self.hpdiv,
            "log": self.hplog,
            "exp": self.hpexp,
            "sq": self.hsq,
            "abs": abs,
            "neg": self.hneg,
            "sin": math.sin,
            "cos": math.cos,
            "0.5": 0.5,
            "sqrt": self.hpsqrt,
        }
        for k, v in space.items():
            if k not in hash_calculators:
                hash_calculators[k] = rand()
        self.mapper: dict = {}
        for arr in [self.ts, self.tnts, self.onts]:
            for s in arr:
                self.mapper[s[0]] = hash_calculators[s[0]]
                cnt += 1

    # protected numpy functions <
    def sq(self, a: ndarray) -> ndarray:  # against too large squares
        return a * where((a > 1000), 1, a)

    def psqrt(self, a: ndarray) -> ndarray:  # against negative sqrt
        return sqrt(absolute(a))

    def plog(self, a: ndarray) -> ndarray:  # against log of <= 0
        return log(absolute(where((a <= 0), 1, a)))

    def pexp(self, a: ndarray) -> ndarray:  # against too large results
        return exp(where((absolute(a) > 5), 0, a))

    def pdiv(self, a: ndarray, b: ndarray) -> ndarray:  # against x/0
        return divide(a, where((absolute(b) <= 0.000001), 1, b))

    # >

    # equivalent hash functions <
    def hsq(self, a: float) -> float:
        return a * a if a <= 1000 else a

    def hneg(self, a: float) -> float:
        return -a

    def hpsqrt(self, a: float) -> float:
        return sqrt(abs(a))

    def hplog(self, a: float) -> float:
        return log(abs(a)) if a > 0 else 0

    def hpexp(self, a: float) -> float:
        return exp(a) if abs(a) <= 5 else 1

    def hpdiv(self, a: float, b: float) -> float:
        return a / b if abs(b) > 0.000001 else a

    def hadd(self, a: float, b: float) -> float:
        return a + b

    def hsub(self, a: float, b: float) -> float:
        return a - b

    def hmul(self, a: float, b: float) -> float:
        return a * b

    # >
