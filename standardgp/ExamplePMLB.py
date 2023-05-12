from pmlb import fetch_data

import numpy as np

from standardgp import GP


def get_data(dataset_name) -> np.ndarray:
    # data from https://github.com/EpistasisLab/pmlb
    data = fetch_data(dataset_name, local_cache_dir="./datasets")
    y = data["target"].to_numpy()
    x = []
    for i, key in enumerate(data):
        if key == "target":
            continue
        x.append(data[key].to_numpy())
    return np.array(x).T, y


if __name__ == "__main__":
    GP.seed(149)
    # example for regression tasks
    # see all regression datasets available at:
    # https://epistasislab.github.io/pmlb/
    x, y = get_data("yeast")
    gp = GP(x[::2, :], y[::2])  # half of the data to train
    best_fit, model = gp.run(show=True, threads=8)
    print("Epochs: {}, Fit: {}, Model: {}".format(gp.gen, best_fit, model))
    print("RMSE train:", np.sqrt(np.mean(np.square(gp.predict(x) - y))))
    x, y = x[1::2, :], y[1::2]  # rest of the data to test on unknown data
    print("RMSE test:", np.sqrt(np.mean(np.square(gp.predict(x) - y))))
