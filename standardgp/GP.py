# Copyright (c) 2023, Thure Foken.
# All rights reserved.

from multiprocessing import Process, Manager
from numpy import argsort, arange, sum, average, fromiter
from numpy import full, array, ndarray, where, unique
from numpy.random import choice
from time import time, sleep

try:
    from Individual import Individual
    from SearchSpace import SearchSpace
except Exception:
    from standardgp.Individual import Individual
    from standardgp.SearchSpace import SearchSpace


def run_wrapper(gp_args, store, seed) -> None:
    """
    Initialize a new GP instance for each process.
    """
    if seed != 0:
        GP.seed(seed)
    gp = GP(gp_args[0], gp_args[1], gp_args[2])
    gp.init_pop()
    gp.run_threaded(store)


def output_stream(store, cfg) -> None:
    """
    A thread that prints the best model over all cores to the console.
    """
    while not store["done"] and store["best"] > cfg.precision:
        if store["repr"]:
            form = "Error: {:1.12f}, Model: {}"
            print(form.format(store["best"], store["repr"]))
        sleep(0.3)


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class GP:
    """
    StandardGP is a multicore symbolic regression algorithm
    with vectorized fitness evaluation and a reduced function space.
    """

    cfg: dotdict = dotdict(
        {
            "gens": 100,
            "pop_size": 4000,
            "elites": 0.15,
            "crossovers": 0.70,
            "mutations": 0.10,
            "max_nodes": 24,
            "cache_size": 500000,  # make sure your RAM supports this
            "precision": 1e-8,
        }
    )

    mutations: int = 0
    crossovers: int = 0
    cx_rejected: int = 0
    init_seed: int = 0

    def __init__(self, x, y, cfg={}):
        self.cfg.update(cfg)
        self.x, self.y = x, y  # training data
        self.space = SearchSpace(self.x, self.y, self.cfg)
        self.gen: int = 0
        self.ps: int = self.cfg.pop_size
        self.elites: int = int(self.ps * self.cfg.elites)
        self.pop: array = array([0 for i in range(self.ps)], dtype=object)
        self.fits: array = full(self.ps, 1.0)
        self.sizes: array = full(self.ps, 1)
        self.upper_probs: ndarray = self.get_rank_field(self.ps, 0.9)
        self.lower_probs: ndarray = self.get_rank_field(self.ps, 0.3)
        self.elite_range: indices = arange(self.ps - self.elites, self.ps)
        self.elite_probs: ndarray = self.get_rank_field(self.elites, 2.0)
        self.best, self.best_repr, self.best_indi = 1.0, "", self.new()

    def get_rank_field(self, size, rank_exponent) -> ndarray:
        exp_ranks = arange(size) ** rank_exponent
        return exp_ranks / sum(exp_ranks)

    @staticmethod
    def seed(value: int) -> None:
        from numpy.random import seed
        from random import seed as std_lib_seed

        seed(value)
        std_lib_seed(value)
        GP.init_seed = value

    def new(self, min_d=2, max_d=4) -> Individual:
        return Individual(self.cfg, self.space, min_d, max_d)

    def init_pop(self) -> None:
        """
        Fill population with non-zero-fitness trees
        """
        cnt: int = 0
        while cnt < self.ps:
            indi = self.new()
            fit = indi.get_fit()
            if fit != 1.0:
                self.pop[cnt] = indi
                self.fits[cnt] = fit
                self.sizes[cnt] = indi.tree_size
                self.update_best(cnt)
                cnt += 1

    def mutate(self, pos: int) -> None:
        if self.pop[pos].subtree_mutate():
            self.fits[pos] = self.pop[pos].get_fit()
            self.sizes[pos] = self.pop[pos].tree_size
            self.update_best(pos)
            GP.mutations += 1

    def sort_pop(self) -> None:
        sort_fit: indices = argsort(self.fits)[::-1]
        self.pop[:] = self.pop[sort_fit]
        self.fits[:] = self.fits[sort_fit]
        self.sizes[:] = self.sizes[sort_fit]

    def mutate_pop(self) -> None:
        # subtree mutation based on fitness
        size: int = int(self.ps * self.cfg.mutations)
        if size == 0:
            return
        self.sort_pop()
        upper: indices = choice(self.ps, p=self.upper_probs, size=size)
        iter = map(lambda a: self.mutate(a), upper)
        fromiter(iter, None)

    def copy_elites(self, elites: int) -> None:
        if elites == 0:
            return
        copies = choice(self.elite_range, p=self.elite_probs, size=elites)
        for idx, elite in enumerate(copies):
            self.pop[idx].copy(self.pop[elite])
            self.fits[idx] = self.fits[elite]
            self.sizes[idx] = self.sizes[elite]

    def cx(self, i: int, j: int) -> None:
        if self.pop[i].crossover(self.pop[j]):
            self.fits[i] = self.pop[i].get_fit()
            self.fits[j] = self.pop[j].get_fit()
            self.sizes[i] = self.pop[i].tree_size
            self.sizes[j] = self.pop[j].tree_size
            self.update_best(i)
            self.update_best(j)
            GP.crossovers += 1
            return
        GP.cx_rejected += 1

    def reproduction(self) -> None:
        self.sort_pop()
        self.copy_elites(self.elites)
        # more then 2 parents possible
        size: int = int(self.ps * self.cfg.crossovers)
        if size == 0:
            return
        lower: indices = choice(self.ps, p=self.upper_probs, size=size)
        upper: indices = choice(self.ps, p=self.lower_probs, size=size)
        idxs: indices = where(lower != upper)[0]
        iter = map(lambda a, b: self.cx(a, b), lower[idxs], upper[idxs])
        fromiter(iter, None)

    def print_stats(self) -> None:
        for i, indi in enumerate(choice(self.pop, 20)):
            print(indi.as_expression(indi.genome))
        print("Gen           :", self.gen)
        print("Mean          :", round(average(self.fits[self.fits != 1]), 5))
        print("Min           :", round(min(self.fits), 10))
        print("Fit calls     :", Individual.fit_calls)
        print("Unique exprs  :", len(Individual.tree_cache))
        print("Unique subtrs :", len(Individual.subtree_cache))
        print("Unique fits   :", len(Individual.unique_fits))
        print("Diversity     :", len(unique(self.fits)))
        print("Avg tree size :", round(average(self.sizes), 3))
        print("Mutations     :", GP.mutations)
        print("Crossovers    :", GP.crossovers)
        print("CX Rejected   :", GP.cx_rejected)
        print("Used          :", round(time() - self.start, 5))
        print("Best fit      :", round(self.best, 10))
        print("Best expr     :", self.best_repr)
        print()

    def termination(self, shared_dict: dict) -> bool:
        # found solution with given precision on another core
        if self.update_dict(shared_dict):
            return True
        # found solution with given precision
        if self.best <= self.cfg.precision:
            return True
        return False

    def update_best(self, pos: int) -> None:
        if self.fits[pos] < self.best:
            self.best = self.fits[pos]
            indi: Individual = self.pop[pos]
            self.best_repr = indi.as_expression(indi.genome)
            self.best_indi = self.new()
            self.best_indi.copy(indi)

    def get_default_stats(self) -> dict:
        return {
            "best": 1.0,
            "repr": "",
            "gen": 1,
            "indi": self.new(),
            "done": False,
        }

    def rand_config(self) -> dict:
        from random import randint

        cfg = self.cfg
        cfg.update(
            {
                "elites": randint(5, 15) / 100.0,
                "crossovers": randint(20, 60) / 100.0,
                "mutations": randint(3, 15) / 100.0,
            }
        )
        return cfg

    def run(self, show=False, threads=1) -> tuple:
        # run single threaded
        if threads <= 1:
            self.init_pop()
            shared_dict: dict = self.get_default_stats()
            return self.run_threaded(shared_dict, show=show, simplify=True)
        # run multiple instances and stop if any solution is found
        shared_dict: dict = Manager().dict(self.get_default_stats())
        if show:
            printer = Process(target=output_stream, args=(shared_dict, self.cfg))
            printer.start()
        processes: list = []
        gp_args = (self.x, self.y, self.cfg)
        for i in range(threads):
            proc_seed = GP.init_seed * (i + 1)
            processes.append(
                Process(target=run_wrapper, args=(gp_args, shared_dict, proc_seed))
            )
            processes[-1].start()
        [p.join() for p in processes]
        shared_dict["done"] = True
        if show:
            printer.join()
        self.best = shared_dict["best"]
        self.best_repr = shared_dict["repr"]
        self.gen = shared_dict["gen"]
        self.best_indi = shared_dict["indi"]
        print("Original model:", self.best_repr)
        self.simplify_last_model()
        return round(self.best, 16), self.best_repr

    def run_threaded(self, shared_dict, show=False, simplify=False) -> tuple:
        self.start = time()
        self.gen, gens = 1, self.cfg.gens + 1
        for gen in range(1, gens):
            self.gen = gen
            if (self.gen % (int(gens / 10) + 1) == 0) and show:
                self.print_stats()
            if self.termination(shared_dict):
                break
            self.mutate_pop()
            self.reproduction()
        if show:
            self.print_stats()
        if simplify:
            self.simplify_last_model()
        self.update_dict(shared_dict)
        return round(self.best, 16), self.best_repr

    def simplify_last_model(self) -> None:
        self.best_indi.simplify().simplify().simplify()
        self.best_repr = self.best_indi.as_expression(self.best_indi.genome)
        self.best_repr = self.space.reconstruct_invariances(self.best_repr)

    def update_dict(self, shared_dict) -> bool:
        if self.best < shared_dict["best"]:
            shared_dict["best"] = self.best
            shared_dict["repr"] = self.best_repr
            shared_dict["gen"] = self.gen
            shared_dict["indi"] = self.best_indi
        if shared_dict["best"] <= self.cfg.precision:
            return True
        return False

    def predict(self, x) -> ndarray:
        for input in range(x.T.shape[0]):
            self.space.space["x{}".format(input)] = x.T[input]
        return eval(self.best_repr, self.space.space)
