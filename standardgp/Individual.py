# Copyright (c) 2023, Thure Foken.
# All rights reserved.

from collections import defaultdict
from numpy import add, ndarray
from random import random


class Node:
    ID: int = 0

    def __init__(self, perent):
        self.value: int = 0
        self.label: str = ""
        self.left: Node = None
        self.right: Node = None
        self.parent: Node = perent
        self.hash_l: float = 0.0
        self.hash_r: float = 0.0
        self.hash: float = 0.0
        self.node_cnt: int = 1
        self.id: int = Node.ID
        Node.ID += 1


class Individual:
    unique_fits: defaultdict = defaultdict(int)
    tree_cache: dict = {}
    subtree_cache: dict = {}
    fit_calls: int = 0

    def __init__(self, cfg, space, min_d=2, max_d=4):
        self.space: SearchSpace = space
        self.map: dict = self.space.mapper
        self.cfg: dotdict = cfg
        self.node_refs: dict = {}
        self.const_prob: flaot = 1 / len(self.space.ts)
        self.genome = self.rand_tree(Node(None), min_d=min_d, max_d=max_d)
        self.tree_cnt: int = self.genome.node_cnt
        self.target: ndarray = space.target
        self.probl_size: float = space.size
        self.hash: float = self.genome.hash

    # fitness with reduced function space <
    def get_fit(self) -> float:
        Individual.fit_calls += 1
        cfg: dotdict = self.cfg
        maxi: int = cfg.max_nodes
        size: int = self.tree_cnt
        if self.hash in Individual.tree_cache:
            return Individual.tree_cache[self.hash]
        if size >= maxi or size < 3:
            Individual.tree_cache[self.hash] = 1.0
            return 1.0
        pred: array = self.parse(self.genome)
        pred: array = pred - add.reduce(pred) / self.probl_size
        norm_arr: float = add.reduce(pred * pred) ** 0.5
        if norm_arr == 0:
            Individual.tree_cache[self.hash] = 1.0
            return 1.0
        error = add.reduce((self.target - (pred / norm_arr)) ** 2)
        fit: float = error / (self.probl_size + 1)
        Individual.unique_fits[fit] += 1
        Individual.tree_cache[self.hash] = fit
        return fit

    def parse(self, root: Node) -> ndarray:
        if not root.left and not root.right:
            return root.value
        elif root.left and not root.right:
            key: float = root.hash
            if key in Individual.subtree_cache:
                return Individual.subtree_cache[key]
            new_result = root.value(self.parse(root.left))
            if len(Individual.subtree_cache) < self.cfg.cache_size:
                Individual.subtree_cache[key] = new_result
            return new_result
        else:
            key: float = root.hash
            if key in Individual.subtree_cache:
                return Individual.subtree_cache[key]
            new_result = root.value(
                self.parse(root.left), self.parse(root.right)
            )
            if len(Individual.subtree_cache) < self.cfg.cache_size:
                Individual.subtree_cache[key] = new_result
            return new_result

    # >

    # creates a new random tree <
    def rand_tree(self, root: Node, min_d=2, max_d=4) -> Node:
        if random() < 0.2 and max_d > 1:
            root.label, root.value = next(self.space.tnts_iter)
            root.left = self.rand_tree(Node(root), min_d - 1, max_d - 1)
            root.right = self.rand_tree(Node(root), min_d - 1, max_d - 1)
            root.node_cnt += root.left.node_cnt + root.right.node_cnt
            root.hash_l = root.left.hash
            root.hash_r = root.right.hash
            root.hash = self.map[root.label](root.hash_l, root.hash_r)
        elif random() < 0.2 and max_d > 1:
            root.label, root.value = next(self.space.onts_iter)
            root.left = self.rand_tree(Node(root), min_d - 1, max_d - 1)
            root.node_cnt += root.left.node_cnt
            root.hash_l = root.left.hash
            root.hash = self.map[root.label](root.hash_l)
        else:
            if min_d > 1:
                self.rand_tree(root, min_d, max_d)
                return root
            root.label, root.value = next(self.space.ts_iter)
            root.hash = self.map[root.label]
            if self.cfg.constants and random() < self.const_prob:
                root.value = round(random() * 10, 1)
                root.label = str(root.value)
                root.hash = root.value
        self.node_refs[root.id] = root
        return root

    # >

    # repair tree after change for fast access and hash calculation <
    def fix_sizes(self, root: Node, diff: int) -> None:
        next_par: Node = root
        while next_par.parent:
            next_par = next_par.parent
            next_par.node_cnt += diff
        self.tree_cnt += diff
        assert self.tree_cnt == next_par.node_cnt

    def fix_hashs(self, root: Node) -> None:
        n_p: Node = root
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

    # >

    # genetic operators <
    def subtree_mutate(self) -> bool:
        pos: int = next(self.space.rand_nodes[self.tree_cnt])
        node: Node = list(self.node_refs.values())[pos]
        maxi: int = self.cfg.max_nodes
        if self.tree_cnt + 16 >= maxi * 3:
            return False
        parent: Node = node.parent
        new_node: Node = self.rand_tree(Node(parent), min_d=2)
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
        return True

    def crossover(self, other) -> bool:
        rng: dict = self.space.rand_nodes
        own_nodes: list = list(self.node_refs.values())
        other_nodes: list = list(other.node_refs.values())
        maxi: int = self.cfg.max_nodes
        cnt: int = 0
        while True:
            # search for a good crossover node
            pos_1: int = next(rng[self.tree_cnt])
            pos_2: int = next(rng[other.tree_cnt])
            subt_1: Node = own_nodes[pos_1]
            subt_2: Node = other_nodes[pos_2]
            new_size_1 = self.tree_cnt + subt_2.node_cnt - subt_1.node_cnt
            new_size_2 = other.tree_cnt + subt_1.node_cnt - subt_2.node_cnt
            if subt_1.node_cnt == 1 or subt_2.node_cnt == 1:
                pass  # not allowed because no internal nodes
            elif subt_1.hash == subt_2.hash:
                pass  # not allowed because there is no effect except blow
            elif new_size_1 < 3:
                pass  # result would be too small
            elif new_size_2 < 3:
                pass  # result would be too small
            elif new_size_1 >= maxi:
                pass  # result would be too large
            elif new_size_2 >= maxi:
                pass  # result would be too large
            elif abs(new_size_1 - new_size_2) > 4:
                pass  # results would grow too fast
            else:
                break  # found good nodes
            if cnt > 8:
                return False  # no crossover point found -> reject
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

    # >

    # tree copy <
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
        self.node_refs: dict = {}
        self.tree_cnt: int = other.tree_cnt
        self.hash: float = other.hash
        self.genome.id: int = Node.ID
        Node.ID += 1
        self.copy_rec(self.genome, other.genome)

    # >

    # for printing only <
    def as_expression(self, root, expr="") -> str:
        if not root.left and not root.right:
            return root.label
        elif root.left and not root.right:
            subtree = self.as_expression(root.left, expr)
            return root.label + "(" + subtree + ")"
        else:
            subt_1 = self.as_expression(root.left, expr)
            subt_2 = self.as_expression(root.right, expr)
            if root.label == "/":
                return "div(" + subt_1 + ", " + subt_2 + ")"
            return "(" + subt_1 + ")" + root.label + "(" + subt_2 + ")"

    # >

    # simplification <
    def simplify(self) -> object:
        self.simplify_rec(self.genome)
        return self

    def simplify_rec(self, root: Node) -> float:
        if not root.left and not root.right:
            return root.value
        elif root.left and not root.right:
            new_result = root.value(self.simplify_rec(root.left))
            if not isinstance(new_result, ndarray):
                if root.parent is None:
                    return root.value
                self.remove_node(root, new_result)
            else:
                if root.hash == root.left.hash:
                    self.remove_function(root, root.left)
            return new_result
        else:
            new_result = root.value(
                self.simplify_rec(root.left), self.simplify_rec(root.right)
            )
            if not isinstance(new_result, ndarray):
                if root.parent is None:
                    return root.value
                self.remove_node(root, new_result)
            else:
                if root.hash == root.left.hash:
                    self.remove_function(root, root.left)
                elif root.hash == root.right.hash:
                    self.remove_function(root, root.right)
            return new_result

    def remove_node(self, root, new_const) -> None:
        self.remove_refs(root)
        self.fix_sizes(root, -(root.node_cnt - 1))
        root.node_cnt = 1
        root.left = None
        root.right = None
        root.value = new_const
        root.label = str(root.value)
        root.hash_l = 0
        root.hash_r = 0
        root.hash = new_const
        self.fix_hashs(root)

    def remove_function(self, root, child) -> None:
        if root.parent is None:
            return
        del self.node_refs[child.id]
        self.fix_sizes(root, -1)
        root.node_cnt = child.node_cnt
        root.left = child.left
        root.right = child.right
        root.value = child.value
        root.label = child.label
        if child.left:
            child.left.parent = root
        if child.right:
            child.right.parent = root

    # >
