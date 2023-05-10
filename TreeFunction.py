# Copyright (c) 2023, Thure Foken.
# All rights reserved.

from numpy import ndarray, add, tan
from numpy import cos, sin, abs, pi, negative
from numpy.random import rand

from Helper import sq, psqrt, plog, pexp, pdiv


class TreeFunction:

    def __init__(self, cfg, function: str, n: int):
        self.cfg = cfg
        self.repr = function
        self.namespace = {
            "v": rand(n) * 4 - 2, "w": rand(n) * 4 - 2, "x": rand(n) * 4 - 2,
            "y": rand(n) * 4 - 2, "z": rand(n) * 4 - 2, "sin": sin, "cos": cos,
            "exp": pexp, "div": pdiv, "sqrt": psqrt, "pih": pi * 0.5,
            "log": plog, "sq": sq, "neg": negative, "abs": abs, "pi": pi,
            "tan": tan,
        }
        self.size = n * 1.0
        self.tar = self.eval_expr(function)

    def eval_expr(self, repr: str) -> ndarray:
        pred = eval(repr, self.namespace)
        pred = pred - add.reduce(pred) / self.size
        pred += rand(int(self.size)) * self.cfg.noise
        pred = pred / (add.reduce(pred * pred) ** 0.5)
        return pred

    def reconstruct_invariances(self, repr) -> str:
        try:
            from scipy.stats import linregress
            x = eval(repr, self.namespace)
            y = eval(self.repr, self.namespace)
            slope, intercept, _r, _p, _std_err = linregress(x, y)
            slope, intercept = round(slope, 8), round(intercept, 8)
            return "{} + {} * ({})".format(intercept, slope, repr)
        except Exception as ex:
            return str(ex)
