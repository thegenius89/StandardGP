# Copyright (c) 2023, Thure Foken.
# All rights reserved.

from collections import defaultdict
from numpy import add, ndarray
from random import random


class Node:

    ID = 0

    def __init__(self, perent):
        self.value    = 0
        self.label    = ""
        self.left     = None
        self.right    = None
        self.parent   = perent
        self.hash_l   = 0
        self.hash_r   = 0
        self.hash     = 0
        self.node_cnt = 1
        self.id       = Node.ID
        Node.ID += 1


class Individual:

    unique_fits = defaultdict(int)
    tree_cache = {}
    subtree_cache = {}

    def __init__(self, cfg, grammar, tar: ndarray, min_d=2):
        self.gram = grammar
        self.map = self.gram.mapper
        self.cfg = cfg
        self.node_refs = {}
        self.genome = self.rand_tree(Node(None), min_d=min_d)
        self.tree_cnt = self.genome.node_cnt
        self.tar = tar
        self.probl_size = cfg.probl_size
        self.hash = self.genome.hash

    def get_fit(self, gen) -> float:
        cfg = self.cfg
        maxi = cfg.max_nodes
        size = self.tree_cnt
        if size >= maxi or size < 3:
            return 0
        if self.hash in Individual.tree_cache:
            return Individual.tree_cache[self.hash]
        pred = self.parse(self.genome)
        pred = pred - add.reduce(pred) / self.probl_size
        norm_arr = (add.reduce(pred * pred) ** 0.5)
        if norm_arr == 0:
            Individual.tree_cache[self.hash] = 0
            return 0
        error = add.reduce((self.tar - (pred / norm_arr)) ** 2)
        fit = 1 - (error / self.probl_size)
        Individual.unique_fits[fit] += 1
        Individual.tree_cache[self.hash] = fit
        return fit

    def parse(self, root: Node) -> float:
        if not root.left and not root.right:
            return root.value
        elif root.left and not root.right:
            key = root.hash
            if key in Individual.subtree_cache:
                return Individual.subtree_cache[key]
            new_result = root.value(self.parse(root.left))
            Individual.subtree_cache[key] = new_result
            return new_result
        else:
            key = root.hash
            if key in Individual.subtree_cache:
                return Individual.subtree_cache[key]
            new_result = root.value(self.parse(root.left),
                                    self.parse(root.right))
            Individual.subtree_cache[key] = new_result
            return new_result

    def rand_tree(self, root: Node, min_d=2, max_d=4) -> Node:
        if random() < self.cfg.operators and max_d > 1:
            root.label, root.value = next(self.gram.tnts_iter)
            root.left = self.rand_tree(Node(root), min_d - 1, max_d - 1)
            root.right = self.rand_tree(Node(root), min_d - 1, max_d - 1)
            root.node_cnt += root.left.node_cnt + root.right.node_cnt
            root.hash_l = root.left.hash
            root.hash_r = root.right.hash
            root.hash = self.map[root.label](root.hash_l, root.hash_r)
        elif random() < self.cfg.functions and max_d > 1:
            root.label, root.value = next(self.gram.onts_iter)
            root.left = self.rand_tree(Node(root), min_d - 1, max_d - 1)
            root.node_cnt += root.left.node_cnt
            root.hash_l = root.left.hash
            root.hash = self.map[root.label](root.hash_l)
        else:
            if min_d > 1:
                self.rand_tree(root, min_d, max_d)
                return root
            root.label, root.value = next(self.gram.ts_iter)
            root.hash = self.map[root.label]
            if self.cfg.constants and random() < 0.08:
                root.value = round(random() * 10, 1)
                root.label = str(root.value)
                root.hash = root.value
        self.node_refs[root.id] = root
        return root

    def fix_sizes(self, root: Node, diff: int) -> None:
        next_par = root
        while next_par.parent:
            next_par = next_par.parent
            next_par.node_cnt += diff
        self.tree_cnt += diff
        assert self.tree_cnt == next_par.node_cnt

    def fix_hashs(self, root: Node) -> None:
        n_p = root
        while n_p.parent:
            n_p = n_p.parent
            if n_p.left and n_p.right:
                n_p.hash_l = n_p.left.hash
                n_p.hash_r = n_p.right.hash
                n_p.hash = self.map[n_p.label](n_p.hash_l, n_p.hash_r)
            elif n_p.left and not n_p.right:
                n_p.hash_l = n_p.left.hash
                n_p.hash = self.map[n_p.label](n_p.hash_l)
        self.hash = self.genome.hash

    def remove_refs(self, root: Node) -> None:
        if root.left and root.right:
            del self.node_refs[root.left.id]
            del self.node_refs[root.right.id]
            self.remove_refs(root.left)
            self.remove_refs(root.right)
        elif root.left and not root.right:
            del self.node_refs[root.left.id]
            self.remove_refs(root.left)

    def fix_refs(self, root: Node, other: Node) -> None:
        if root.left and root.right:
            k_0, k_1 = root.left.id, root.right.id
            self.node_refs[k_0] = other.node_refs.pop(k_0)
            self.node_refs[k_1] = other.node_refs.pop(k_1)
            self.fix_refs(root.left, other)
            self.fix_refs(root.right, other)
        elif root.left and not root.right:
            k_0 = root.left.id
            self.node_refs[k_0] = other.node_refs.pop(k_0)
            self.fix_refs(root.left, other)

    def subtree_mutate(self) -> None:
        pos = next(self.gram.rand_nodes[self.tree_cnt])
        node = list(self.node_refs.values())[pos]
        parent = node.parent
        new_node = self.rand_tree(Node(parent), min_d=2)
        del self.node_refs[node.id]
        self.remove_refs(node)
        if parent is None:
            self.genome = new_node
        else:
            if parent.left.id == node.id:
                parent.left = new_node
            else:
                parent.right = new_node
        self.fix_sizes(new_node, new_node.node_cnt - node.node_cnt)
        self.fix_hashs(new_node)

    def crossover(self, other) -> bool:
        rng = self.gram.rand_nodes
        own_nodes = list(self.node_refs.values())
        other_nodes = list(other.node_refs.values())
        maxi = self.cfg.max_nodes
        cnt = 0
        while True:
            pos_1 = next(rng[self.tree_cnt])
            pos_2 = next(rng[other.tree_cnt])
            subt_1 = own_nodes[pos_1]
            subt_2 = other_nodes[pos_2]
            new_size_1 = self.tree_cnt + subt_2.node_cnt - subt_1.node_cnt
            new_size_2 = other.tree_cnt + subt_1.node_cnt - subt_2.node_cnt
            if subt_1.node_cnt == 1 or subt_2.node_cnt == 1:
                pass
            elif subt_1.hash == subt_2.hash:
                pass
            elif new_size_1 < 3:
                pass
            elif new_size_2 < 3:
                pass
            elif new_size_1 >= maxi:
                pass
            elif new_size_2 >= maxi:
                pass
            elif abs(new_size_1 - new_size_2) > self.cfg.grow_limit:
                pass
            else:
                break
            if cnt > 8:
                return False
            cnt += 1
        self.fix_refs(subt_2, other)
        other.fix_refs(subt_1, self)
        subt_2.value, subt_1.value = subt_1.value, subt_2.value
        subt_2.label, subt_1.label = subt_1.label, subt_2.label
        subt_2.hash_l, subt_1.hash_l = subt_1.hash_l, subt_2.hash_l
        subt_2.hash_r, subt_1.hash_r = subt_1.hash_r, subt_2.hash_r
        subt_2.hash, subt_1.hash = subt_1.hash, subt_2.hash
        subt_2.node_cnt, subt_1.node_cnt = subt_1.node_cnt, subt_2.node_cnt
        subt_2.left, subt_1.left = subt_1.left, subt_2.left
        subt_2.right, subt_1.right = subt_1.right, subt_2.right
        self.fix_sizes(subt_1, subt_1.node_cnt - subt_2.node_cnt)
        other.fix_sizes(subt_2, subt_2.node_cnt - subt_1.node_cnt)
        self.fix_hashs(subt_1)
        other.fix_hashs(subt_2)
        if subt_2.left:
            subt_2.left.parent = subt_2
        if subt_2.right:
            subt_2.right.parent = subt_2
        if subt_1.left:
            subt_1.left.parent = subt_1
        if subt_1.right:
            subt_1.right.parent = subt_1
        return True

    def copy_rec(self, root: Node, other) -> None:
        root.value, root.label = other.value, other.label
        root.node_cnt = other.node_cnt
        root.hash_l = other.hash_l
        root.hash_r = other.hash_r
        root.hash = other.hash
        root.id = Node.ID
        Node.ID += 1
        if not other.left:
            root.left = None
        if not other.right:
            root.right = None
        self.node_refs[root.id] = root
        if other.left:
            if not root.left:
                root.left = Node(root)
            self.copy_rec(root.left, other.left)
        if other.right:
            if not root.right:
                root.right = Node(root)
            self.copy_rec(root.right, other.right)

    def copy(self, other) -> None:
        self.node_refs = {}
        self.tree_cnt = other.tree_cnt
        self.hash = other.hash
        self.genome.id = Node.ID
        Node.ID += 1
        self.copy_rec(self.genome, other.genome)

    def as_expression(self, root, expr="") -> str:
        if not root.left and not root.right:
            return root.label
        elif root.left and not root.right:
            subtree = self.as_expression(root.left, expr)
            return root.label + "(" + subtree + ")"
        else:
            subt_1 = self.as_expression(root.left, expr)
            subt_2 = self.as_expression(root.right, expr)
            return "(" + subt_1 + ")" + root.label + "(" + subt_2 + ")"
