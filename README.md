# StandardGP
StandardGP is a genetic programming regression algorithm based on original GP operators.
There is a huge debate what StandardGP can do on its own and what all the new linear and hybrid variants can do better or worse.
To provide a comparison algorithm between new GP variants and StandardGP - this is an easy to use pure Python SGP algorithm.

## Features
 - True vectorized fitness evaluation
 - Blow control without fitness intervention via stable Crossover
 - Subtree-Cache that includes the isomorphism between functions
 - Subtree-Mutation based on fitness
 - Subtree-Crossover determines a good internal node based on tree-size
 - Function-space reduction via scale and location invariance

## Usage:
```
import numpy as np
from StandardGP import GP

x = np.random.rand(200, 5) * 2
y = np.sin(x[:, 0] * x[:, 0]) * np.cos(x[:, 1] - 1) * 188 - 243
gp = GP(x, y)
best_fitness, best_model = gp.run(show=True)
print("Generations: {}, Fitness: {}, Model: {}".format(gp.gen, best_fitness, best_model))
print("RMSE:", np.sqrt(np.mean(np.square(gp.predict(x) - y))))
```
You can see how huge your dataset can be without losing performance by changing the 200 to say 2000.

## Advanced usage:
All the hyperparameters are very good chosen - for many cases there is no need to adjust them.
The [...,...] are recommended ranges.
```
class Config:
    def __init__(self):
        self.precision     = 0.99999999
        self.gens          = 100    # [1, n] number of generations
        self.elites        = 0.07   # [0, 0.3] reproduction copies per generation
        self.crossovers    = 0.60   # [0, 2.0] inplace crossover on population
        self.mutations     = 0.09   # [0, 1.0] probabillity per tree per generation to mutate
        self.pop_size      = 4000   # [1, n] number of trees
        self.grow_limit    = 4      # [2, n] how fast individuals can grow
        self.max_nodes     = 24     # [8, n] max nodes per tree
        self.constants     = True   # insert random variables
```

## Coming soon
- Full vectorized GP-Operators that are easy to automatically adapt
- Visualisation of evolved Operators
- There are just 500 lines of code and it will be reduced to around 250 lines for easy understanding and experimenting with new operators
