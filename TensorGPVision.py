from numpy.random import rand, shuffle, randint
from numpy import sin, cos
from numpy import arange, tile, apply_along_axis, full
from numpy import ndarray, where


class TensorGPVision:
    def __init__(self, x: ndarray, y: ndarray):
        self.ps, self.depth, self.gs = 4000, 6, 20
        self.tree_matrix = tile(arange(self.ps) + 1, (self.depth, 1))
        self.sizes = full((self.ps, self.depth), 1)
        self.hashes = full((self.ps, self.depth), 0)
        self.labels = randint(0, self.gs, size=(self.ps, self.depth))
        apply_along_axis(shuffle, arr=self.tree_matrix, axis=1)
        print(self.tree_matrix)

    def mutation(self) -> None:
        mutation_mask = rand(self.ps, self.depth) < 0.001
        idxs = where(mutation_mask)
        new_genes = randint(1, idxs[1].size, size=idxs[1].size)
        self.tree_matrix[idxs[1], idxs[0]] = new_genes

    def train(self, gens=1) -> None:
        for gen in range(gens):
            self.mutation()

    def predict(self, x) -> float:
        return x


if __name__ == "__main__":
    x = rand(5, 200) * 2
    y = sin(x[0, :] * x[0, :]) * cos(x[1, :] - 1)
    model = TensorGPVision(x, y)
    model.train()
    model.predict(x)
