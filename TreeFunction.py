# Copyright (c) 2023, Thure Foken.
# All rights reserved.

from numpy import ndarray, add, tan
from numpy import cos, sin, abs, pi, negative
from numpy.random import rand

from Helper import sq, psqrt, plog, pexp, pdiv


class TreeFunction:

    def __init__(self, x: ndarray, y: ndarray, cfg):
        self.x_train = x.T
        self.y_train = y
        self.cfg = cfg
        self.namespace = {
            "sin": sin, "cos": cos, "tan": tan, "abs": abs, "pi": pi,
            "exp": pexp, "div": pdiv, "sqrt": psqrt, "pih": pi * 0.5,
            "log": plog, "sq": sq, "neg": negative,
        }
        for input in range(self.x_train.shape[0]):
            self.namespace["x{}".format(input)] = self.x_train[input]
        self.size = self.x_train.shape[1] * 1.0
        self.tar = self.normalize(y)

    def normalize(self, pred: ndarray) -> ndarray:
        pred = pred - add.reduce(pred) / self.size
        pred += rand(int(self.size)) * self.cfg.noise
        pred = pred / (add.reduce(pred * pred) ** 0.5)
        return pred

    def reconstruct_invariances(self, repr) -> str:
        try:
            from scipy.stats import linregress
            x = eval(repr, self.namespace)
            y = self.y_train
            slope, intercept, _r, _p, _std_err = linregress(x, y)
            slope, intercept = round(slope, 8), round(intercept, 8)
            return "{} + {} * ({})".format(intercept, slope, repr)
        except Exception:
            return repr
