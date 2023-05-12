# Copyright (c) 2023, Thure Foken.
# All rights reserved.

from multiprocessing import Process, Manager
from numpy import argsort, arange, sum, max, average, fromiter
from numpy import full, array, ndarray, where
from numpy.random import choice
from time import time, sleep

from Individual import Individual
from SearchSpace import SearchSpace


class Config:

    def __init__(self):
        self.precision     = 0.99999999
        self.gens          = 100     # [1, n] after 100 gens a solution is rare
        self.elites        = 0.07    # [0, 0.5] elite copies per gen
        self.crossovers    = 0.60    # [0, 2.0] inplace crossover on population
        self.mutations     = 0.09    # [0, 1.0] probabillity per tree per gen
        self.high_pressure = 0.9     # [0.1, 4.0] rank exponent - pick
        self.low_pressure  = 0.3     # [0.1, 4.0] rank exponent - spread
        self.pop_size      = 4000    # [1, n] number of trees
        self.grow_limit    = 4       # [2, n] how fast individuals can grow
        self.max_nodes     = 24      # [8, n] max nodes per tree
        self.operators     = 0.2     # [0, 0.4] probabillity to select operator
        self.functions     = 0.2     # [0, 0.4] probabillity to select function
        self.noise         = 0.000   # make the dataset noisy
        self.debug_pop     = False   # pick a sample and show while running
        self.constants     = True    # insert random variables
        self.cache_size    = 500000  # max ndarrays in cache

    def update(self, cfg) -> None:
        if isinstance(cfg, Config):
            self = cfg
        if isinstance(cfg, dict):
            for k, v in cfg.items():
                setattr(self, k, v)


def run_wrapper(gp, shared_dict, seed) -> None:
    if seed != 0:
        GP.seed(seed)
    gp = GP(gp[0], gp[1], gp[2])
    gp.run_threaded(shared_dict)


def output_stream(shared_dict, cfg) -> None:
    while not shared_dict["done"]:
        if shared_dict["best"] >= cfg.precision:
            break
        print("Fit: {}, Model: {}".format(
            shared_dict["best"], shared_dict["repr"]))
        sleep(0.5)


class GP:

    mutations = 0
    crossovers = 0
    cx_rejected = 0
    init_seed = 0

    def __init__(self, x, y, cfg={}):
        self.cfg = Config()
        self.cfg.update(cfg)
        self.space = SearchSpace(x, y, self.cfg)
        self.ps = self.cfg.pop_size
        self.gen = 0
        self.pop = array([self.new() for i in range(self.ps)])
        self.fits = full(self.ps, 0.0)
        exp_ranks = arange(1, self.ps + 1) ** self.cfg.high_pressure
        self.sel_probs = exp_ranks / sum(exp_ranks)
        exp_ranks = arange(1, self.ps + 1) ** self.cfg.low_pressure
        self.lower_sel_probs = exp_ranks / sum(exp_ranks)
        self.init_pop()

    @staticmethod
    def seed(value: int) -> None:
        from numpy.random import seed
        from random import seed as std_lib_seed
        seed(value)
        std_lib_seed(value)
        GP.init_seed = value

    def new(self) -> Individual:
        return Individual(self.cfg, self.space)

    def init_pop(self) -> None:
        # fill population with non-zero-fitness trees
        cnt = 0
        while cnt < self.ps:
            indi = self.new()
            fit = indi.get_fit(self.gen)
            if fit != 0:
                self.pop[cnt] = indi
                self.fits[cnt] = fit
                cnt += 1

    def mutate(self, pos: int) -> None:
        self.pop[pos].subtree_mutate()
        self.fits[pos] = self.pop[pos].get_fit(self.gen)
        self.update_best(pos)

    def mutate_pop(self) -> None:
        # mutation based on fitness
        # the higher the fitness, the higher the probabillity to mutate
        sort_indices = argsort(self.fits)
        self.pop[:] = self.pop[sort_indices]
        self.fits[:] = self.fits[sort_indices]
        size = int(self.ps * self.cfg.mutations)
        if size == 0:
            return
        upper = choice(self.ps, p=self.sel_probs, size=size)
        iter = map(lambda a: self.mutate(a), upper)
        fromiter(iter, None)
        GP.mutations += size

    def copy_elites(self, indices: ndarray) -> None:
        for c, i in enumerate(indices):
            self.pop[i].copy(self.pop[self.ps - c - 1])
            self.fits[i] = self.fits[self.ps - c - 1]

    def cx(self, i: int, j: int) -> None:
        if self.pop[i].crossover(self.pop[j]):
            self.fits[i] = self.pop[i].get_fit(self.gen)
            self.fits[j] = self.pop[j].get_fit(self.gen)
            self.update_best(i)
            self.update_best(j)
            GP.crossovers += 1
            return
        GP.cx_rejected += 1

    def reproduction(self) -> None:
        sort_indices = argsort(self.fits)
        self.pop[:] = self.pop[sort_indices]
        self.fits[:] = self.fits[sort_indices]
        # copy some of the best trees directly into the new population
        # by overwriting the worst trees
        until = int(self.ps * self.cfg.elites)
        if until > 0:
            self.copy_elites(range(until))
            self.copy_elites(range(until, until * 2))
        # swap subtrees between many individuals
        # (more then 2 parents possible)
        size = int(self.ps * self.cfg.crossovers)
        if size == 0:
            return
        lower = choice(self.ps, p=self.sel_probs, size=size)
        upper = choice(self.ps, p=self.lower_sel_probs, size=size)
        idxs = where(lower != upper)[0]
        iter = map(lambda a, b: self.cx(a, b), lower[idxs], upper[idxs])
        fromiter(iter, None)

    def print_stats(self, gen: int) -> None:
        unique_fits = set(self.fits)
        if self.cfg.debug_pop:
            for i, indi in enumerate(choice(self.pop, 20)):
                print(indi.as_expression(indi.genome))
        tree_size = average([indi.tree_cnt for indi in self.pop])
        fit_calls = GP.mutations + GP.crossovers * 2 + self.ps
        print("Gen           :", gen)
        print("Mean          :", round(average(self.fits[self.fits != 0]), 5))
        print("Max           :", round(max(self.fits), 10))
        print("Fit calls     :", fit_calls)
        print("Unique exprs  :", len(Individual.tree_cache))
        print("Unique subtrs :", len(Individual.subtree_cache))
        print("Unique fits   :", len(Individual.unique_fits))
        print("Diversity     :", len(unique_fits))
        print("Avg tree size :", round(tree_size, 3))
        print("Mutations     :", GP.mutations)
        print("Crossovers    :", GP.crossovers)
        print("CX Rejected   :", GP.cx_rejected)
        print("Used          :", round(time() - self.start, 5))
        print("Best fit      :", round(self.best, 10))
        print("Best expr     :", self.best_repr)
        print()

    def termination(self, gen: int, shared_dict: dict) -> bool:
        # found solution with given precision on another core
        if self.update_dict(shared_dict):
            return True
        # found solution with given precision
        if self.best >= self.cfg.precision:
            return True
        return False

    def update_best(self, pos: int) -> None:
        if self.fits[pos] > self.best:
            self.best = self.fits[pos]
            indi = self.pop[pos]
            self.best_repr = indi.as_expression(indi.genome)
            self.best_indi = self.new()
            self.best_indi.copy(indi)

    def run(self, show=False, threads=1) -> tuple:
        # run single threaded
        shared_dict = {
            "best": 0, "repr": "", "gen": 0, "indi": "", "done": False,
        }
        if threads <= 1:
            return self.run_threaded(shared_dict, show=show, simplify=True)
        # run multiple instances and stop if any solution is found
        shared_dict = Manager().dict({
            "best": 0, "repr": "", "gen": 0, "indi": "", "done": False,
        })
        processes = []
        for i in range(threads):
            p = Process(
                target=run_wrapper,
                args=(
                    (self.space.x_train.T, self.space.y_train, self.cfg),
                    shared_dict, GP.init_seed * (i + 1),
                )
            )
            processes.append(p)
            p.start()
        printer = Process(target=output_stream, args=(shared_dict, self.cfg, ))
        printer.start()
        [p.join() for p in processes]
        shared_dict["done"] = True
        printer.join()
        self.best = shared_dict["best"]
        self.best_repr = shared_dict["repr"]
        self.gen = shared_dict["gen"]
        self.best_indi = shared_dict["indi"]
        print("Original model:", self.best_repr)
        self.simplify_last_model()
        return self.best, self.best_repr

    def run_threaded(self, shared_dict, show=False, simplify=False) -> tuple:
        self.start = time()
        self.best, self.best_repr, self.best_indi = 0, "", self.new()
        gen, gens = 1, self.cfg.gens + 1
        for gen in range(1, gens):
            self.gen = gen
            if (gen % (int(gens / 10) + 1) == 0) and show:
                self.print_stats(gen)
            if self.termination(gen, shared_dict):
                break
            self.mutate_pop()
            self.reproduction()
        if show:
            self.print_stats(gen)
        if simplify:
            self.simplify_last_model()
        self.update_dict(shared_dict)
        return self.best, self.best_repr

    def simplify_last_model(self) -> None:
        self.best_indi.simplify().simplify().simplify()
        self.best_repr = self.best_indi.as_expression(self.best_indi.genome)
        self.best_repr = self.space.reconstruct_invariances(self.best_repr)

    def update_dict(self, shared_dict) -> bool:
        if self.best > shared_dict["best"]:
            shared_dict["best"] = self.best
            shared_dict["repr"] = self.best_repr
            shared_dict["gen"] = self.gen
            shared_dict["indi"] = self.best_indi
        if shared_dict["best"] >= self.cfg.precision:
            return True
        return False

    def predict(self, x) -> ndarray:
        for input in range(x.T.shape[0]):
            self.space.space["x{}".format(input)] = x.T[input]
        return eval(self.best_repr, self.space.space)
