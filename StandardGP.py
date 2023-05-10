# Copyright (c) 2023, Thure Foken.
# All rights reserved.

from multiprocessing import Process
from numpy import argsort, arange, sum, max, average, fromiter
from numpy import full, array, ndarray, where
from numpy.random import choice
from time import time

from Config import Config
from TreeFunction import TreeFunction
from TreeGrammar import TreeGrammar
from Individual import Individual


class GP:

    mutations = 0
    crossovers = 0
    cx_rejected = 0

    def __init__(self, problem):
        self.cfg = Config()
        self.problem = TreeFunction(self.cfg, problem, self.cfg.probl_size)
        self.gram = TreeGrammar(self.cfg, self.problem.namespace)
        self.ps = self.cfg.pop_size
        self.gen = 0
        self.pop = array([self.new() for i in range(self.ps)])
        self.fits = full(self.ps, 0.0)
        exp_ranks = arange(1, self.ps + 1) ** self.cfg.high_pressure
        self.sel_probs = exp_ranks / sum(exp_ranks)
        exp_ranks = arange(1, self.ps + 1) ** self.cfg.low_pressure
        self.lower_sel_probs = exp_ranks / sum(exp_ranks)
        self.init_pop()

    def new(self) -> Individual:
        return Individual(self.cfg, self.gram, self.problem.tar)

    def init_pop(self) -> None:
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
        until = int(self.ps * self.cfg.elites)
        if until > 0:
            self.copy_elites(range(until))
            self.copy_elites(range(until, until * 2))
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

    def termination(self, gen: int) -> bool:
        if self.best >= self.cfg.precision:
            return True
        return False

    def update_best(self, pos: int) -> None:
        if self.fits[pos] > self.best:
            self.best = self.fits[pos]
            indi = self.pop[pos]
            self.best_repr = indi.as_expression(indi.genome)

    def run(self, show=False) -> tuple:
        self.start = time()
        self.best, self.best_repr = 0, ""
        gen, gens = 1, self.cfg.gens + 1
        for gen in range(1, gens):
            self.gen = gen
            if (gen % (int(gens / 10) + 1) == 0) and show:
                self.print_stats(gen)
            if self.termination(gen):
                break
            self.mutate_pop()
            self.reproduction()
        if show:
            self.print_stats(gen)
        repr = self.best_repr
        if self.cfg.linregress:
            repr = self.problem.reconstruct_invariances(repr)
        return self.best, repr
