# Copyright (c) 2023, Thure Foken.
# All rights reserved.

import numpy as np

from Helper import plog, pexp
from StandardGP import GP


def main() -> None:
    x = np.random.rand(200, 5) * 4 - 2
    y = np.sin(x[:, 0] * x[:, 0]) * np.cos(x[:, 1] - 1) * 188 - 243
    # y = plog(x[:, 0] + 1) + plog(x[:, 0] ** 2 + 1)
    # y = np.sin(x[:, 0] * x[:, 0] + 1) * np.cos(x[:, 1] * 2)
    # y = x[:, 0] ** 4 - x[:, 0] ** 3 - x[:, 0] ** 2 - x[:, 0]
    # y = x[:, 0] ** 6 + x[:, 0] ** 5 + x[:, 0] ** 4 + x[:, 0] ** 3 + x[:, 0] ** 2 + x[:, 0]
    # y = (1 - x[:, 0]) ** 2 + (x[:, 1] - x[:, 0] ** 2) ** 2
    # y = x[:, 0] ** 4 - x[:, 0] ** 3 + 0.5 * x[:, 1] ** 2 - x[:, 1]
    # y = x[:, 0] * x[:, 1] + np.sin((x[:, 0] - 1) * (x[:, 1] - 1))
    # y = x[:, 0] ** 3 + x[:, 1] ** 3 - x[:, 1] - x[:, 0]
    # y = ((np.tan(x[:, 0]) / pexp(x[:, 1])) * (plog(x[:, 2]) - np.tan(x[:, 3])))
    gp = GP(x, y)
    best_fit, best_model = gp.run(show=True)
    print("Epochs: {}, Fit: {}, Model: {}".format(gp.gen, best_fit, best_model))
    print("RMSE:", np.sqrt(np.mean(np.square(gp.predict(x) - y))))


if __name__ == "__main__":
    main()
