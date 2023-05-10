# Copyright (c) 2023, Thure Foken.
# All rights reserved.

from math import log, exp, sqrt
from numpy import abs as npabs
from numpy import exp as npexp
from numpy import log as nplog
from numpy import sqrt as npsqrt
from numpy import where, divide


# define protected versions of functions


def sq(a) -> float:
    return a * where((a > 1000), 1, a)


def psqrt(a) -> float:
    return npsqrt(npabs(a))


def plog(a) -> float:
    return nplog(npabs(where((a <= 0), 1, a)))


def pexp(a) -> float:
    return npexp(where((npabs(a) > 5), 0, a))


def pdiv(a, b) -> float:
    return divide(a, where((abs(b) <= 0.000001), 1, b))


# define equivalent versions for the subtree cache


def hsq(a) -> float:
    return a * a if a <= 1000 else a


def hneg(a) -> float:
    return -a


def hpsqrt(a) -> float:
    return sqrt(abs(a))


def hplog(a) -> float:
    return log(abs(a)) if a > 0 else 0


def hpexp(a) -> float:
    return exp(a) if abs(a) <= 5 else 1


def hpdiv(a, b) -> float:
    return a / b if abs(b) > 0.000001 else a


def hadd(a, b) -> float:
    return a + b


def hsub(a, b) -> float:
    return a - b


def hmul(a, b) -> float:
    return a * b
