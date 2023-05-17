# Copyright (c) 2023, Thure Foken.
# All rights reserved.

from itertools import cycle
from numpy import ndarray, add, tan, subtract, multiply, divide
from numpy import cos, sin, absolute, pi, where, log, sqrt, exp
from numpy.random import rand
from random import choice, randint

import math


class SearchSpace:
    """
    SearchSpace includes the function space and the search space.
    The function space is defined as every mathematical function.
    The search space is defined as any combination of given operators.
    SearchSpace also defines how to map Node labels to the corresponding
    data or operator and the corresponding hash value or operator.
    """

    def __init__(self, x: ndarray, y: ndarray, cfg):
        self.x_train: ndarray = x.T
        self.y_train: ndarray = y
        self.cfg: dict = cfg
        # defining the search space
        self.space: dict = {
            "sin": sin,
            "cos": cos,
            "tan": tan,
            "exp": self.pexp,
            "div": self.pdiv,
            "sqrt": self.psqrt,
            "log": self.plog,
            "sq": self.sq,
            "pih": pi * 0.5,
            "pi": pi,
        }
        self.terminals: list = [
            ("0.5", 0.5),
            ("1", 1.0),
            ("2", 2.0),
            ("pi", self.space["pi"]),
            ("pih", self.space["pih"]),
        ]
        self.binary: list = [
            ("+", add),
            ("+", add),
            ("-", subtract),
            ("-", subtract),
            ("*", multiply),
            ("*", multiply),
            ("/", self.space["div"]),
        ]
        self.unary: list = [
            ("sin", self.space["sin"]),
            ("cos", self.space["cos"]),
            ("log", self.space["log"]),
            ("exp", self.space["exp"]),
            ("sqrt", self.space["sqrt"]),
            ("sq", self.space["sq"]),
            ("tan", self.space["tan"]),
        ]
        # bring the input and output in the right form for GP
        self.size: float = self.x_train.shape[1] * 1.0
        self.target: ndarray = self.normalize(y)
        for input in range(self.x_train.shape[0]):
            input_name = "x{}".format(input)
            self.space[input_name] = self.x_train[input]
            self.terminals.append((input_name, self.x_train[input]))
        self.build_hash_system(self.space)
        # provide fast random access
        self.term_iter = cycle([choice(self.terminals) for _ in range(1223)])
        self.binary_iter = cycle([choice(self.binary) for _ in range(937)])
        self.unary_iter = cycle([choice(self.unary) for _ in range(1069)])
        self.rand_nodes: dict = {}
        for s in range(1, self.cfg.max_nodes * 3):
            self.rand_nodes[s] = cycle([randint(0, s - 1) for _ in range(119)])

    def normalize(self, pred: ndarray) -> ndarray:
        # location and scale invariance
        pred: ndarray = pred - add.reduce(pred) / self.size
        pred: ndarray = pred / (add.reduce(pred * pred) ** 0.5)
        return pred

    def reconstruct_invariances(self, repr) -> str:
        # reconstructs the invariants after the run
        try:
            from scipy.stats import linregress

            x: ndarray = eval(repr, self.space)
            y: ndarray = self.y_train
            slope, intercept, _r, _p, _std_err = linregress(x, y)
            slope, intercept = round(slope, 8), round(intercept, 8)
            return "{} + {} * ({})".format(intercept, slope, repr)
        except Exception:
            return repr

    def build_hash_system(self, space) -> None:
        # topological hash functions
        label_hash = {
            "0.5": 0.5,
            "1": 1.0,
            "2": 2.0,
            "pi": space["pi"],
            "pih": space["pih"],
            "tan": math.tan,
            "sin": math.sin,
            "cos": math.cos,
            "+": self.hadd,
            "-": self.hsub,
            "*": self.hmul,
            "/": self.hpdiv,
            "log": self.hplog,
            "exp": self.hpexp,
            "sq": self.hsq,
            "sqrt": self.hpsqrt,
        }
        # float64 random variable for each input
        label_hash.update({k: rand() for k, _ in space.items() if k not in label_hash})
        # mapping the Node labels to the functions
        self.mapper: dict = {}
        for arr in self.terminals + self.binary + self.unary:
            self.mapper[arr[0]] = label_hash[arr[0]]

    # protected numpy functions <
    def sq(self, a: ndarray) -> ndarray:  # against too large squares
        return a * where((a > 1000), 1, a)

    def psqrt(self, a: ndarray) -> ndarray:  # against negative sqrt
        return sqrt(absolute(a))

    def plog(self, a: ndarray) -> ndarray:  # against log of <= 0
        return log(absolute(where((a <= 0), 1, a)))

    def pexp(self, a: ndarray) -> ndarray:  # against too large results
        return exp(where((absolute(a) > 5), 0, a))

    def pdiv(self, a: ndarray, b: ndarray) -> ndarray:  # against x/lim(0)
        return divide(a, where((absolute(b) <= 0.000001), 1, b))

    # >

    # equivalent protected hash functions <
    def hsq(self, a: float) -> float:
        return a * a if a <= 1000 else a

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
