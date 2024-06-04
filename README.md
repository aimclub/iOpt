<p align="center">
  <img src="/docs/iOpt_logo.png" width="200" height="150"/>
</p>

[![SAI](https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/SAI_badge_flat.svg)](https://sai.itmo.ru/)
[![ITMO](https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/ITMO_badge_flat.svg)](https://en.itmo.ru/en/)

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-green)](LICENSE)
[![python: 3.9](https://img.shields.io/badge/python-3.9-44cc12?style=flat-square&logo=python)](https://www.python.org/downloads/release/python-390/)
[![python: 3.8](https://img.shields.io/badge/python-3.8-44cc12?style=flat-square&logo=python)](https://www.python.org/downloads/release/python-380/)
[![docs: ](https://readthedocs.org/projects/ebonite/badge/?style=flat-square)](https://iopt.readthedocs.io/ru/latest/)
[![build:](https://github.com/UNN-ITMM-Software/iOpt/actions/workflows/python-app.yml/badge.svg)](https://github.com/UNN-ITMM-Software/iOpt/actions)
[![rus:](https://img.shields.io/badge/lang-ru-yellow.svg)](README_ru.md)



iOpt is an open source framework for automatic selection of parameter values both for mathematical models of complex industrial processes and for AI and ML methods used in industry. The framework is distributed under the 3-Clause BSD license.


# **Key features of the framework**
- Automatic selection of parameter values both for mathematical models and for AI and ML methods used in industry.
- Intelligent control of the process of choosing the optimal parameters for industrial applications.
- Integration with external artificial intelligence and machine learning libraries or frameworks as well as applied models.
- Automation of the preliminary analysis of the models under study, e.g., by identifying different types of model dependencies on different groups of parameters.
- Visualization of the process of choosing optimal parameters.


# **Installation**



## Automatic installation

The simplest way to install **iOpt** is using *pip*:

```
pip install iOpt
``` 

## Manual installation

### On Unix-like systems:

```
git clone https://github.com/aimclub/iOpt
cd iOpt
pip install virtualenv
virtualenv ioptenv
source ioptenv/bin/activate
python setup.py install
```

### On Windows:

```
git clone https://github.com/aimclub/iOpt
cd iOpt
pip install virtualenv
virtualenv ioptenv
ioptenv\Scripts\activate.bat
python setup.py install
```
## Docker

Download the image:

```
docker pull aimclub/iopt:latest
```

Using the iOpt image:

```
docker run -it aimclub/iopt:latest
```


# **How to Use**

Using the iOpt framework to minimize the Rastrigin test function.

```python
from problems.rastrigin import Rastrigin
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.output_system.listeners.static_painters import StaticPainterNDListener
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener

from subprocess import Popen, PIPE, STDOUT

if __name__ == "__main__":
    """
    Minimization of the Rastrigin test function with visualization
    """
    # Create a test task
    problem = Rastrigin(2)
    # Setup a solver options
    params = SolverParameters(r=2.5, eps=0.01, iters_limit=300, refine_solution=True)
    # Create the solver
    solver = Solver(problem, parameters=params)
    # Print results to console while solving
    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)
    # 3D visualization at the end of the solution
    spl = StaticPainterNDListener("rastrigin.png", "output", vars_indxs=[0, 1], mode="surface", calc="interpolation")
    solver.add_listener(spl)
    # Run problem solution
    sol = solver.solve()
```

# **Examples**

Let’s demonstrate the use of the iOpt framework when tuning the hyperparameters of one of the machine learning methods. In the support vector machine ([SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)), we find the optimal hyperparameters (the regularization parameter **C**, the kernel coefficient **gamma**) in the problem of breast cancer classification ([detailed description of the data](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))).

```python
import numpy as np
from sklearn.utils import shuffle
from sklearn.datasets import load_breast_cancer

from iOpt.output_system.listeners.static_painters import StaticPainterNDListener
from iOpt.output_system.listeners.animate_painters import AnimatePainterNDListener
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from examples.Machine_learning.SVC._2D.Problems import SVC_2d


def load_breast_cancer_data():
    dataset = load_breast_cancer()
    x_raw, y_raw = dataset['data'], dataset['target']
    inputs, outputs = shuffle(x_raw, y_raw ^ 1, random_state=42)
    return inputs, outputs


if __name__ == "__main__":
    x, y = load_breast_cancer_data()
    regularization_value_bound = {'low': 1, 'up': 6}
    kernel_coefficient_bound = {'low': -7, 'up': -3}

    problem = SVC_2d.SVC_2D(x, y, regularization_value_bound, kernel_coefficient_bound)

    method_params = SolverParameters(r=np.double(3.0), iters_limit=100)
    solver = Solver(problem, parameters=method_params)

    apl = AnimatePainterNDListener("svc2d_anim.png", "output", vars_indxs=[0, 1], to_paint_obj_func=False)
    solver.add_listener(apl)

    spl = StaticPainterNDListener("svc2d_stat.png", "output", vars_indxs=[0, 1], mode="surface", calc="interpolation")
    solver.add_listener(spl)

    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)

    solver_info = solver.solve()

```

Let's consider an example of using multicriteria optimization. We use optimization for two float objectives: precision and recall. The result of the process is a Pareto set chart.

```python
from examples.Machine_learning.SVC._2D.Problems import mco_breast_cancer

from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    problem = mco_breast_cancer.mco_breast_cancer(X, y, X_train, y_train)

    params = SolverParameters(r=3.0, eps=0.01, iters_limit=200, number_of_lambdas=50,
                              start_lambdas=[[0, 1]], is_scaling=False)

    solver = Solver(problem=problem, parameters=params)

    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)

    sol = solver.solve()

    var = [trial.point.float_variables for trial in sol.best_trials]
    val = [[-trial.function_values[i].value for i in range(2)] for trial in sol.best_trials]

    print("size pareto set: ", len(var))
    for fvar, fval in zip(var, val):
        print(fvar, fval)

    fv1 = [-trial.function_values[0].value for trial in sol.best_trials]
    fv2 = [-trial.function_values[1].value for trial in sol.best_trials]
    plt.plot(fv1, fv2, 'ro')
    plt.show()

```


# **Project Structure**

The latest stable release of iOpt is in the [main](https://github.com/UNN-ITMM-Software/iOpt/tree/main) branch. The repository includes the following directories:
- The [iOpt](https://github.com/UNN-ITMM-Software/iOpt/tree/main/iOpt) directory contains the framework core in the form of Python classes.
- The [examples](https://github.com/UNN-ITMM-Software/iOpt/tree/main/examples) directory contains examples of using the framework for both test and applied problems.
- Unit tests are located in the [test](https://github.com/UNN-ITMM-Software/iOpt/tree/main/test) directory.
- Documentation source files are located in the [docs](https://github.com/UNN-ITMM-Software/iOpt/tree/main/docs) directory.

# **Documentation**

A detailed description of the iOpt framework API is available at [Read the Docs](https://iopt.readthedocs.io/ru/latest/).

# **Supported by**

The study is supported by the [Research Center Strong Artificial Intelligence in Industry](https://sai.itmo.ru/) 
of [ITMO University](https://en.itmo.ru/) as part of the plan of the center's program: Framework of intelligent heuristic optimization methods.
