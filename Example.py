# Copyright (c) 2023, Thure Foken.
# All rights reserved.

from StandardGP import GP


def main() -> None:
    # hidden function to find
    func = "sin(x * x) * cos(y - 1)"
    # func = "sin(x * x + 1) * cos(y * 2)"
    # func = "x ** 4 - x ** 3 - x ** 2 - x"
    # func = "x ** 6 + x ** 5 + x ** 4 + x ** 3 + x ** 2 + x"
    # func = "log(x + 1) + log(x ** 2 + 1)"
    # func = "(1 - x) ** 2 + (y - x ** 2) ** 2"
    # func = "x ** 4 - x ** 3 + 0.5 * y ** 2 - y"
    # func = "x * y + sin((x - 1) * (y - 1))"
    # func = "x ** 3 + y ** 3 - y - x"
    # func = "((tan(x) / exp(y)) * (log(z) - tan(v)))"
    gp = GP(func)
    best, repr = gp.run(show=True)
    print("Gen: {}, Fit: {}, Expr: {}".format(gp.gen, round(best, 9), repr))


if __name__ == "__main__":
    main()
