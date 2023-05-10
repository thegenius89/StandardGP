# Copyright (c) 2023, Thure Foken.
# All rights reserved.

class Config:

    def __init__(self):
        self.precision     = 0.99999999
        self.gens          = 100    # [1, n] after 100 gens a solution is rare
        self.elites        = 0.07   # [0, 0.5] elite copies per gen
        self.crossovers    = 0.60   # [0, 2.0] inplace crossover on population
        self.mutations     = 0.09   # [0, 1.0] probabillity per tree per gen
        self.high_pressure = 0.9    # [0.1, 4.0] rank exponent - pick
        self.low_pressure  = 0.3    # [0.1, 4.0] rank exponent - spread
        self.pop_size      = 4000   # [1, n] number of trees
        self.grow_limit    = 4      # [2, n] how fast individuals can grow
        self.max_nodes     = 24     # [8, n] max nodes per tree
        self.operators     = 0.2    # [0, 0.4] probabillity to select operator
        self.functions     = 0.2    # [0, 0.4] probabillity to select function
        self.noise         = 0.000  # make the dataset noisy
        self.debug_pop     = False  # pick a sample and show while running
        self.constants     = True   # insert random variables
