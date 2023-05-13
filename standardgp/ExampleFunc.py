import numpy as np

from standardgp import GP


if __name__ == "__main__":
    # GP.seed(129)
    # StandardGP is able to find them all exactly
    x = np.random.rand(200, 5) * 4 - 2
    y = np.sin(x[:, 0] * x[:, 0]) * np.cos(x[:, 1] - 1) * 188 - 243
    # y = x[:, 0] ** 4 - x[:, 0] ** 3 - x[:, 0] ** 2 - x[:, 0]
    # y = (1 - x[:, 0]) ** 2 + (x[:, 1] - x[:, 0] ** 2) ** 2
    # y = x[:, 0] ** 4 - x[:, 0] ** 3 + 0.5 * x[:, 1] ** 2 - x[:, 1]
    # y = x[:, 0] * x[:, 1] + np.sin((x[:, 0] - 1) * (x[:, 1] - 1))
    # y = x[:, 0] ** 3 + x[:, 1] ** 3 - x[:, 1] - x[:, 0]
    # even more complicated functions
    # x = np.random.rand(200, 5)
    # y = ((np.tan(x[:, 0]) / np.exp(x[:, 1])) * (np.log(x[:, 2]) - np.tan(x[:, 3])))
    gp = GP(x, y)
    best_fit, model = gp.run(show=True, threads=8)
    print("Error: {}, Model: {}".format(best_fit, model))
    print("RMSE:", np.sqrt(np.mean(np.square(gp.predict(x) - y))))
