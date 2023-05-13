# Copyright (c) 2023, Thure Foken.
# All rights reserved.

# meassure time
# python -m cProfile -o out ExampleFunc.py
# python -m pstats out
# check code style
# flake8 --ignore E221,E251 --exclude ./tests

from multiprocessing import Process, Manager
from numpy import argsort, arange, sum, max, average, fromiter
from numpy import full, array, ndarray, where, unique
from numpy.random import choice
from time import time, sleep

from Individual import Individual
from SearchSpace import SearchSpace


def run_wrapper(gp, shared_dict, seed) -> None:
    if seed != 0:
        GP.seed(seed)
    gp = GP(gp[0], gp[1], gp[2])
    gp.run_threaded(shared_dict)


def output_stream(shared_dict, cfg) -> None:
    while not shared_dict["done"]:
        if shared_dict["best"] >= 1 - cfg.precision:
            break
        print("Fit: {}, Model: {}".format(
            shared_dict["best"], shared_dict["repr"]))
        sleep(0.5)


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class GP:

    cfg = dotdict({
        "gens"      : 100,     # [1, n] stop after n generations
        "pop_size"  : 4000,    # [1, n] number of trees in population
        "elites"    : 0.15,    # [0, 0.5] % elite copies per generation
        "crossovers": 0.60,    # [0, 2.0] % inplace crossover on population
        "mutations" : 0.10,    # [0, 1.0] % probabillity per tree per gen
        "max_nodes" : 24,      # [8, n] max nodes per tree
        "cache_size": 500000,  # max ndarrays in cache (look at your RAM)
        "precision" : 1e-8,    # precision termination condition
    })

    mutations: int = 0
    crossovers: int = 0
    cx_rejected: int = 0
    init_seed: int = 0

    def __init__(self, x, y, cfg={}):
        self.cfg.update(cfg)
        self.space = SearchSpace(x, y, self.cfg)
        self.ps: int = self.cfg.pop_size
        self.gen: int = 0
        self.pop: array = array([self.new() for i in range(self.ps)])
        self.fits: array = full(self.ps, 0.0)
        self.sizes: array = full(self.ps, 1)
        exp_ranks = arange(1, self.ps + 1) ** 0.9
        self.sel_probs: array = exp_ranks / sum(exp_ranks)
        exp_ranks = arange(1, self.ps + 1) ** 0.3
        self.lower_sel_probs: array = exp_ranks / sum(exp_ranks)
        self.until: int = int(self.ps * self.cfg.elites)
        self.elite_range: indices = arange(self.ps - self.until, self.ps)
        exp_ranks = arange(1, self.until + 1) ** 2.0
        self.elite_probs: array = exp_ranks / sum(exp_ranks)
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
        cnt: int = 0
        while cnt < self.ps:
            indi = self.new()
            fit = indi.get_fit()
            if fit != 0:
                self.pop[cnt] = indi
                self.fits[cnt] = fit
                self.sizes[cnt] = indi.tree_cnt
                cnt += 1

    def mutate(self, pos: int) -> None:
        self.pop[pos].subtree_mutate()
        self.fits[pos] = self.pop[pos].get_fit()
        self.sizes[pos] = self.pop[pos].tree_cnt
        self.update_best(pos)

    def mutate_pop(self) -> None:
        # mutation based on fitness
        # the higher the fitness, the higher the probabillity to mutate
        size: int = int(self.ps * self.cfg.mutations)
        if size == 0:
            return
        sort_fit = argsort(self.fits)
        self.pop[:] = self.pop[sort_fit]
        self.fits[:] = self.fits[sort_fit]
        self.sizes[:] = self.sizes[sort_fit]
        upper: indices = choice(self.ps, p=self.sel_probs, size=size)
        iter = map(lambda a: self.mutate(a), upper)
        fromiter(iter, None)
        GP.mutations += size

    def copy_elites(self, until: int) -> None:
        if until == 0:
            return
        copies = choice(self.elite_range, p=self.elite_probs, size=until)
        for idx, elite in enumerate(copies):
            self.pop[idx].copy(self.pop[elite])
            self.fits[idx] = self.fits[elite]
            self.sizes[idx] = self.sizes[elite]

    def cx(self, i: int, j: int) -> None:
        if self.pop[i].crossover(self.pop[j]):
            self.fits[i] = self.pop[i].get_fit()
            self.fits[j] = self.pop[j].get_fit()
            self.sizes[i] = self.pop[i].tree_cnt
            self.sizes[j] = self.pop[j].tree_cnt
            self.update_best(i)
            self.update_best(j)
            GP.crossovers += 1
            return
        GP.cx_rejected += 1

    def reproduction(self) -> None:
        # indices = where(self.sizes > average(self.sizes) * 1.5)[0]
        # self.fits[indices] = 0
        sort_fit: indices = argsort(self.fits)
        self.pop[:] = self.pop[sort_fit]
        self.fits[:] = self.fits[sort_fit]
        self.sizes[:] = self.sizes[sort_fit]
        # copy some of the best trees directly into the new population
        # by overwriting the worst trees
        self.copy_elites(self.until)
        # swap subtrees between many individuals
        # (more then 2 parents possible)
        size: int = int(self.ps * self.cfg.crossovers)
        if size == 0:
            return
        lower: indices = choice(self.ps, p=self.sel_probs, size=size)
        upper: indices = choice(self.ps, p=self.lower_sel_probs, size=size)
        idxs: indices = where(lower != upper)[0]
        iter = map(lambda a, b: self.cx(a, b), lower[idxs], upper[idxs])
        fromiter(iter, None)

    def print_stats(self) -> None:
        for i, indi in enumerate(choice(self.pop, 20)):
            print(indi.as_expression(indi.genome))
        print("Gen           :", self.gen)
        print("Mean          :", round(average(self.fits[self.fits != 0]), 5))
        print("Max           :", round(max(self.fits), 10))
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
        if self.best >= 1 - self.cfg.precision:
            return True
        return False

    def update_best(self, pos: int) -> None:
        if self.fits[pos] > self.best:
            self.best = self.fits[pos]
            indi: Individual = self.pop[pos]
            self.best_repr = indi.as_expression(indi.genome)
            self.best_indi = self.new()
            self.best_indi.copy(indi)

    def run(self, show=False, threads=1) -> tuple:
        # run single threaded
        shared_dict: dict = {
            "best": 0, "repr": "", "gen": 0, "indi": "", "done": False,
        }
        if threads <= 1:
            return self.run_threaded(shared_dict, show=show, simplify=True)
        # run multiple instances and stop if any solution is found
        shared_dict: dict = Manager().dict({
            "best": 0, "repr": "", "gen": 0, "indi": "", "done": False,
        })
        processes: list = []
        for i in range(threads):
            processes.append(Process(
                target=run_wrapper,
                args=(
                    (self.space.x_train.T, self.space.y_train, self.cfg),
                    shared_dict, GP.init_seed * (i + 1),
                )
            ))
            processes[-1].start()
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
        if shared_dict["best"] >= 1 - self.cfg.precision:
            return True
        return False

    def predict(self, x) -> ndarray:
        for input in range(x.T.shape[0]):
            self.space.space["x{}".format(input)] = x.T[input]
        return eval(self.best_repr, self.space.space)
