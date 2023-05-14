<html>
<head>
<meta name="google-site-verification" content="O0VCZ4RSpoJQ-lD0PgLqruxw8QYePHl5jxtbAVEgF60" />
</head>
<body>

# StandardGP
StandardGP is a genetic programming regression algorithm based on original GP operators.
To provide a comparison algorithm between new linear GP variants and StandardGP - this is an easy to use pure Python SGP algorithm.
The goal of this project is to get a better understanding of GP algorithms through visualizations.

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

## Features
 - True vectorized fitness evaluation
 - Blow control without fitness intervention via semi-stable Crossover
 - Multithreaded optimization
 - Subtree-Cache that includes the isomorphism between functions
 - Subtree-Mutation based on fitness
 - Subtree-Crossover determines a good internal node based on tree-size
 - Function-space reduction via scale and location invariance

## Install:
```bash
git clone https://github.com/thegenius89/StandardGP.git
cd StandardGP
python setup.py install
```

## Usage:
```python
import numpy as np
from standardgp import GP

x = np.random.rand(200, 5) * 4 - 2
y = np.sin(x[:, 0] * x[:, 0]) * np.cos(x[:, 1] - 1) * 188 - 243
gp = GP(x, y)
best_fit, model = gp.run(show=True, threads=8)
print("Fit: {}, Model: {}".format(best_fit, model))
print("RMSE:", np.sqrt(np.mean(np.square(gp.predict(x) - y))))
```
You can see how huge your dataset can be without losing performance by changing the 200 to say 2000.

## Advanced usage:
All the hyperparameters are very good chosen - for many cases there is no need to adjust them.
The [...,...] are recommended ranges.
```python
{
    "gens"       : 100,     # [1, n] stop after n generations
    "pop_size"   : 4000,    # [1, n] number of trees in population
    "elites"     : 0.15,    # [0, 0.5] % elite copies per generation
    "crossovers" : 0.60,    # [0, 2.0] % inplace crossover on population
    "mutations"  : 0.10,    # [0, 1.0] % probabillity per tree per gen
    "max_nodes"  : 24,      # [8, n] max nodes per tree
    "cache_size" : 500000,  # max ndarrays in cache (look at your RAM)
    "precision"  : 1e-8,    # precision termination condition
}
```
Just pass a config to the GP
```python
gp = GP(x, y, cfg={"mutations": 0.25, ...})
```

## Supported Operators/Functions/Constants
    +, -, *, /[p]
    sin, cos, tan, exp[p], sqrt[abs], log[p], square[max]
    pi, pi/2, 1, 2, 0.5, uniform(0, 10)

## Coming
- blow control also in the copy reproduction operator to have complete control over size dynamics
- register repo for the "Living Benchmark Suite"
  [https://github.com/cavalab/srbench](https://github.com/cavalab/srbench)
- pip install standardgp

## Coming soon
- Full vectorized GP-Operators that are easy to automatically adapt
- Visualisation of evolved operators
- A full vectorized version will just contain 250 lines of algorithmic code instead of 500
- Allowing dynamic hyperparameter like mutation rates that depends on position and generation -> cfg={"mutations": lambda p, g: 1 / p * sin(g)}

## Visions
GP will one day be able to find optimal Neural Network architectures, many equivalent forms of Einsteins
field equation and Schrödingers equation that are very interesting to study or may be able to find solutions
to fundamental mathematical questions. Once there are operators that work globally for all problems to navigate a
search through the infinite function space to find near optimal solutions the GP algorithms will begin to alter
its own operators and algorithms to find even better and novel approaches to problems.

</body>
</html>
