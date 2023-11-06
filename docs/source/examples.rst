Examples of using
=====================

Tuning the parameters of a genetic algorithm to solve the traveling salesman problem
____________________________________________________________________________________


Traveling Salesman Problem (TSP) is an NP-hard combinatorial optimization problem, important 
in theoretical computer science and operations research. The essence of the problem statement 
can be formulated as follows: “For a given list of cities and distances find the shortest route 
between each pair of cities, which passes through each city exactly once and returns to the home 
city."


Problem statement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let the numbers :math:`1,2,{\dots},n` correspond to cities, the values :math:`c_{\mathit{ij}}>0` 
correspond to the distances between the cities :math:`i` и :math:`j`, and the value :math:`x_{\mathit{ij}}=1`,
if there is a path from :math:`i` to :math:`j`, and :math:`x_{\mathit{ij}}=0` otherwise.
Then the travelling salesman problem can be formulated as follows:

.. math::

   \min\sum _{i=1}^n\sum _{i{\neq}j,j=1}^nc_{\mathit{ij}}x_{\mathit{ij}}:

.. math::

   \sum _{i=1,i{\neq}j}^nx_{\mathit{ij}}=1,\sum _{j=1,i{\neq}j}^nx_{\mathit{ij}}=1,i=1,{\dots},n;j=1,{\dots},n;

.. math::

   \sum _{i{\in}Q}^n\sum_{i{\neq}j,j{\in}Q}^nx_{\mathit{ij}}{\leq}\left|Q\right|-1,{\forall}Q{\subsetneq}\left\{1,{\dots},n\right\},\left|Q\right|{\geq}2.

The last constraint ensures that no subset of Q can form a subroute, so the solution returned is 
a single route, not a union of smaller routes.

Known methods for solving TSP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are many algorithms to obtain the optimal distance to be travelled by the salesman: 
exhaustive search, random search, nearest neighbour method, cheapest inclusion method, 
minimum spanning tree method, branch and bound method. We will consider the application 
of a genetic algorithm for solving TSP.

Framework for solving TSP
~~~~~~~~~~~~~~~~~~~~~~~~~

We used `scikit-opt <https://github.com/guofei9987/scikit-opt>`_ as a framework for solving TSP 
using the genetic algorithm. 
This framework is often used in scientific research, has a good structure, documentation 
and technical support. In particular, scikit-opt implements a genetic algorithm for solving 
the travelling salesman problem. This method is called **GA_TSP** and has the following parameters:

* **func**: function for calculating the length of the travelling salesman's path;
* **n_dim**: number of vertices considered in the problem;
* **size_pop**: population size;
* **max_iter**: number of iterations of the genetic algorithm;
* **prob_mut**: probability of occurrence of a mutation.

Parameter optimization problem statement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Obviously, with different sets of values for the population size, 
the maximum number of iterations of the algorithm, and the probability 
of a mutation occurring, the genetic algorithm will produce different solutions. 
It is required to find such a set of algorithm parameters that will provide 
the best solution with a limited resource of iterations.

Initial data
~~~~~~~~~~~~

The **GA_TSP** method from the scikit-opt library accepts an objective function 
for calculating the length of the traveling salesman’s route as one of the parameters. 
This function processes the initial data. 
The data itself is presented in a matrix, where the intersection of the :math:`i`-th row and 
the :math:`j`-th column of the matrix is the distance between the cities :math:`i` and :math:`j`.

The `TSPLIB <http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/>`_ portal
contains the well-known matrix database for travelling salesman problems in XML format. 
As an input to the method, the path to the xml-file pre-loaded from TSPLIB is provided.

.. code-block::
   :caption: Converting an XML file with vertex distances to a two-dimensional numpy array

   import xml.etree.ElementTree as ET
   import numpy as np

   def load_TSPs_matrix(file_name):
      root = ET.parse(file_name).getroot()
      columns = root.findall('graph/vertex')
      num_cols = len(columns)
      trans_matrix = np.zeros((num_cols, num_cols))
      for i, v in enumerate(columns):
         for e in v:
               j = int(e.text)
               trans_matrix[i, j] = float(e.get('cost'))
      return trans_matrix

Finding the optimal path for fixed parameter values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider working with the **GA_TSP** method of the Scikit-opt library. 
Before starting, let’s load the att48.xml task from the open TSPLIB task bank.

First, we define the objective function for calculating the distance of the current 
travelling salesman route. The **calc_total_distance** function is used to calculate 
the distance between the vertices specified in the **routine** array, 
the input parameter of this function.

Next, we initialize the object designed to store the service information of 
the genetic algorithm for solving the travelling salesman problem. 
In particular, we call the **GA_TSP** class constructor and specify the parameters:

* set the **func** parameter to the objective function **calc_total_distance**;
* the **n_dim** parameter stores information about the number of nodes in the distance matrix, 
  so we use the dimension of the matrix to initialize it;
* the **size_pop** parameter is initialized with the starting value 50 (we take into account 
  that it is necessary to specify the population size as a multiple of two);
* the probability of mutation (**prob_mut**) and the iterations number of 
  the genetic algorithm (**max_iter**) are set to 100 and 0.9, respectively.

To run the method for finding the minimum distance for an object of the **GA_TSP** class, 
one must call the **run** method. Upon completion of the work, the specified method returns 
the optimal trajectory and the corresponding value of the distance traveled.
The result obtained with the current parameters is 16237.

.. code-block::
   :caption: An example of working with the scikit-opt library to find a solution 
             to the traveling salesman problem using a genetic algorithm

   import xml.etree.ElementTree as ET
   import numpy as np
   from sko.GA import GA_TSP

   def load_TSPs_matrix(file_name):
      root = ET.parse(file_name).getroot()
      columns = root.findall('graph/vertex')
      num_cols = len(columns)
      trans_matrix = np.zeros((num_cols, num_cols))
      for i, v in enumerate(columns):
         for e in v:
               j = int(e.text)
               trans_matrix[i, j] = float(e.get('cost'))
      return trans_matrix

   def cal_total_distance(routine):
      num_points, = routine.shape
      return sum([trans_matr[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

   trans_matr = load_TSPs_matrix('att48.xml')
   num_cols = trans_matr.shape[0]
   ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_cols, size_pop=50, max_iter=100, prob_mut=0.9)
   best_points, best_distance = ga_tsp.run()

Finding the optimal path when tuning the probability of mutation using the iOpt framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To solve the travelling salesman problem by means of a genetic algorithm, for the iOpt framework, 
we have developed classes focused on finding the best trajectory and minimum distance, by redefining 
the base class **Problem**.

The **GA_TSP_Vary_Mutation** class assumes as input parameters the distance matrix, 
the number of iterations of the genetic algorithm, the population size, and 
the bounds on the variation of the mutation probability.

The class implements an objective function for calculating the total distance between vertices, 
as well as a function for calculating the current distance for fixed user-specified values 
of the number of iterations of the genetic algorithm and the population size. 
The **Calculate** method returns the path length found by the genetic algorithm for a fixed value 
of the population size, the number of iterations, and a variable value of the mutation probability.

.. code-block::
   :caption: Adaptation of a genetic algorithm for the traveling salesman problem

   import numpy as np
   from sko.GA import GA_TSP
   from typing import Dict

   class GA_TSP_Vary_Mutation(Problem):
      def __init__(self, cost_matrix: np.ndarray, num_iteration: int,
                  population_size: int,
                  mutation_probability_bound: Dict[str, float]):
         self.dimension = 1
         self.number_of_float_variables = 1
         self.number_of_discrete_variables = 0
         self.number_of_objectives = 1
         self.number_of_constraints = 0
         self.costMatrix = cost_matrix
         if num_iteration <= 0:
               raise ValueError('The number of iterations cannot be zero or negative.')
         if population_size <= 0:
               raise ValueError('Population size cannot be negative or zero')
         self.populationSize = population_size
         self.numberOfIterations = num_iteration
         self.float_variable_names = np.array(["Mutation probability"],
               dtype=str)
         self.lower_bound_of_float_variables =
               np.array([mutation_probability_bound['low']], dtype=np.double)
         self.upper_bound_of_float_variables =
               np.array([mutation_probability_bound['up']], dtype=np.double)
         self.n_dim = cost_matrix.shape[0]

      def calc_total_distance(self, routine):
         num_points, = routine.shape
         return sum([self.costMatrix[routine[i % num_points], 
               routine[(i + 1) % num_points]] for i in range(num_points)])

      def Calculate(self, point: Point, 
                     functionValue: FunctionValue) -> FunctionValue:
         mutation_prob = point.float_variables[0]
         ga_tsp = GA_TSP(func=self.calc_total_distance,
                         n_dim=self.n_dim, size_pop=self.populationSize,
                         max_iter=self.numberOfIterations, 
                         prob_mut=mutation_prob)
         best_points, best_distance = ga_tsp.run()
         functionValue.value = best_distance[0]
         return functionValue

Below is the code to run the solver of the iOpt framework:

#. loading data from xml file;
#. setting the values of the method iteration number and population size;
#. setting limits for varying mutation probability values;
#. initialization of the problem under study;
#. setting solver parameters;
#. starting the solution process - searching for the optimal distance value.

.. code-block::
   :caption: An example of choosing the optimal parameter GA_TSP using the iOpt framework solver

   import numpy as np
   import xml.etree.ElementTree as ET

   def load_TSPs_matrix(file_name):
      root = ET.parse(file_name).getroot()
      columns = root.findall('graph/vertex')
      num_cols = len(columns)
      trans_matrix = np.zeros((num_cols, num_cols))
      for i, v in enumerate(columns):
         for e in v:
               j = int(e.text)
               trans_matrix[i, j] = float(e.get('cost'))
      return trans_matrix

   if __name__ == "__main__":
      tsp_matrix = load_TSPs_matrix('att48.xml')
      num_iteration = 100
      population_size = 50
      mutation_probability_bound = {'low': 0.0, 'up': 1.0}
      problem = ga_tsp_vary_mutation.GA_TSP_Vary_Mutation(tsp_matrix,
         num_iteration, population_size, mutation_probability_bound)
      method_params = SolverParameters(r=np.double(3.0), iters_limit=20)
      solver = Solver(problem, parameters=method_params)

      solver_info = solver.Solve()

Results
~~~~~~~

When solving TSP using a genetic algorithm tuned by iOpt, it was possible to find 
a better estimate of the optimum than that obtained using the same algorithm tuned 
by means of the uniform grid technique. The distance found using iOpt was 13333, 
while 35 calls to the objective function were made. At the same time, the solution 
with the parameters found by the uniform grid technique was 13958; in this case, 
100 calls to the objective function were made.

.. figure:: images/gatsp.png
   :width: 500
   :align: center

   Demonstration of how the iOpt framework works when setting up the parameters of the traveling salesman problem


Tuning support vector machine hyperparameters for a classification problem in machine learning
______________________________________________________________________________________________

In machine learning problems, in order to obtain a high-quality prediction it is necessary 
to optimize the hyperparameters of the model. 
We consider the support vector machine (SVC_) - a method for constructing a separating surface. 

.. _SVC: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

The method has two real parameters: the regularization coefficient (C) and the kernel coefficient (gamma). 
The task is as follows: to select the hyperparameters of the model to maximize the objective metric.

Dataset used
~~~~~~~~~~~~

We will use the `breast cancer_` dataset. The dataset includes 569 examples, each with 30 
numerical characteristics. Characteristics are calculated from a digitized fine needle aspiration 
(FNA) image of the breast mass. They describe the characteristics of the nuclei of the cells present 
in the image. The distribution by class is as follows: 212 malignant, 357 benign tumors.

.. _`breast cancer`: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic) 

Solving the problem with default parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's solve the classification problem using the SVC_ method with the hyperparameters that 
the scikit-learn framework uses by default. The code includes loading a shuffled dataset 
with a fixed random_state, as well as applying cross-validation.

.. _SVC: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

.. code-block::
    :caption: Solving the problem with default parameters

    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import f1_score
    from sklearn.svm import SVC

    def get_sklearn_breast_cancer_dataset():
        dataset = load_breast_cancer()
        x, y = dataset['data'], dataset['target']
        return shuffle(x, y ^ 1, random_state=42)

    x, y = get_sklearn_breast_cancer_dataset()

    cross_val_score(SVC(), x, y,
                    scoring=lambda model, x, y: f1_score(y, model.predict(x))).mean()

With the default hyperparameters, we solved the problem with an average f1-score 
across all cross-validation experiments of 0.87.

Calculation of the averaged f1-score on a grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let us see if this problem can be solved better by varying two continuous parameters of the algorithm. 
To do this, we calculate the average value of cross-validation at each point of a uniform 20 by 20 grid:

#. regularization parameter **C**: [10\ :sup:`1`, 10\ :sup:`6`];
#. kernel coefficient **gamma**: [10\ :sup:`-7`, 10\ :sup:`-3`].

For convenience, we use a logarithmic scale and the **np.logspace** function to obtain 
the corresponding grid values.

.. code-block::
    :caption: Calculating f1-score value on a 20x20 grid

    import numpy as np

    cs = np.logspace(1, 6, 20)
    gamms = np.logspace(-7, -3, 20)

    params = {'C': cs, 'gamma': gamms}

    search = GridSearchCV(SVC(), cv=5, param_grid=params, 
                        scoring=lambda model, x, y: f1_score(y, model.predict(x)))
    search.fit(x, y)

Let's display the results of the experiment on a graph. To reduce the maximization problem 
to a minimization problem, we multiply the objective value by minus one.

.. figure:: images/cancer_svc_f1.png
    :width: 500
    :align: center

    Average f1-score values on the grid

It can be seen from the graph that there are SVC hyperparameters that solve the problem 
with an average value of 0.94 f1-score, which significantly improves the quality of prediction.

Finding optimal parameters using the iOpt framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An example of working with the framework when varying two continuous parameters
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
Let’s run the iOpt framework to find the optimal point on the grid, maximizing the f1-score. 
To do this, we will declare a class that is a successor of the **Problem** class 
with an abstract **Calculate** method.

.. code-block:: 

    import numpy as np
    from iOpt.trial import Point
    from iOpt.trial import FunctionValue
    from iOpt.problem import Problem
    from sklearn.metrics import f1_score
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from typing import Dict

    class SVC_2D(Problem):
        def __init__(self, x_dataset: np.ndarray, y_dataset: np.ndarray,
                    regularization_bound: Dict[str, float],
                    kernel_coefficient_bound: Dict[str, float]):
            
            self.dimension = 2
            self.number_of_float_variables = 2
            self.number_of_discrete_variables = 0
            self.number_of_objectives = 1
            self.number_of_constraints = 0
            if x_dataset.shape[0] != y_dataset.shape[0]:
                raise ValueError('The input and output sample sizes do not match.')
            self.x = x_dataset
            self.y = y_dataset
            self.float_variable_names = np.array(["Regularization parameter",
                "Kernel coefficient"], dtype=str)
            self.lower_bound_of_float_variables =
                np.array([regularization_bound['low'], 
                kernel_coefficient_bound['low']], dtype=np.double)
            self.upper_bound_of_float_variables =
                np.array([regularization_bound['up'], 
                kernel_coefficient_bound['up']], dtype=np.double)

        def Calculate(self, point: Point, 
                      functionValue: FunctionValue) -> FunctionValue:
            cs, gammas = point.float_variables[0], point.float_variables[1]
            clf = SVC(C=10**cs, gamma=10**gammas)
            clf.fit(self.x, self.y)
            functionValue.value = -cross_val_score(clf, self.x, self.y,
                scoring=lambda model, x, y: f1_score(y, model.predict(x))).mean()
            return functionValue


The SVC_2D class accepts the following constructor parameters:

#. **x_dataset** – an array of objects and their attributes wrapped in **np.ndarray**;
#. **y_dataset** – target labels of each of the **x_dataset** objects in the **np.ndarray** format;
#. **regularization_bound** – maximum and minimum values for **C** as a dictionary;
#. **kernel_coefficient_bound** – maximum and minimum values for **gamma** as a dictionary.

The **Calculate**  method implements the logic of calculating the objective function at **Point**. 
To do this, an SVC classifier is created and trained with the passed hyperparameters, 
then the average value of f1-score by cross-validation is calculated with the opposite sign.

To start the optimization process, we created an object of the **SVC_2D class**, as well as 
an object of the **Solver** class with the passed objective function object. 
To render, we called the **AddListener** method, passing objects of the **AnimationNDPaintListener** 
and **StaticNDPaintListener** classes.

.. code-block:: 
    :caption: Running optimization of the SVC_2D object serving as the objective function

    from iOpt.method.listener import StaticNDPaintListener, AnimationNDPaintListener
    from sklearn.datasets import load_breast_cancer
    from iOpt.solver import Solver
    from iOpt.solver_parametrs import SolverParameters
    from examples.Machine_learning.SVC._2D.Problems import SVC_2d

    if __name__ == "__main__":
        x, y = load_breast_cancer_data()
        regularization_value_bound = {'low': 1, 'up': 6}
        kernel_coefficient_bound = {'low': -7, 'up': -3}

        problem = SVC_2d.SVC_2D(x, y, regularization_value_bound, 
            kernel_coefficient_bound)

        method_params = SolverParameters(r=np.double(3.0), iters_limit=10)
        solver = Solver(problem, parameters=method_params)

        apl = AnimationNDPaintListener("svc2d_anim.png", "output", 
            vars_indxs=[0, 1], to_paint_obj_func=False)
        solver.AddListener(apl)

        spl = StaticNDPaintListener("svc2d_stat.png", "output", vars_indxs=[0, 1],
            mode="surface", calc="interpolation")
        solver.AddListener(spl)

        solver_info = solver.Solve()
        print(solver_info.number_of_global_trials)
        print(solver_info.number_of_local_trials)
        print(solver_info.solving_time)

        print(solver_info.best_trials[0].point.float_variables)
        print(solver_info.best_trials[0].function_values[0].value)

After the experiment, the program displays the total search time for the optimum, 
the point on the grid at which the optimum is reached, the found maximum value of the f1-score metric, 
and also the graph of the objective function.

With a limit on the number of iterations **iterLimits**=10, the framework finds hyperparameters 
where the target metric reaches 0.94, the total calculation time is less than 5 seconds.

For visual interpolation of the graph, the parameter **iterLimits**=100 was set.

.. figure:: images/cancer_iopt_interpol.png
   :width: 500
   :align: center

   Objective function interpolation

The blue dots on the graph represent the points of exploratory trials, 
the red dot marks the found optimum corresponding to the hyperparameters 
at which the f1-score reaches its maximum.

An example of working with the framework when varying two continuous and one discrete parameters
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
Previously, when searching for the optimal set of parameters, an example was considered 
with varying two continuous parameters of the SVC method.
As a target discrete parameter, we will select the type of algorithm kernel: **kernel**. 
This categorical parameter takes one of 5 values: linear, poly, rbf, sigmoid, precomputed.
However, when considering the parameters C and gamma, only three of them are available: 
poly, rbf, sigmoid.

Let's consider the behavior of the quality metric f1-score on each specified core separately. 
Let's set the following areas for parameters:

#. regularization parameter **C**: [10\ :sup:`1`, 10\ :sup:`10`];
#. kernel coefficient **gamma**: [10\ :sup:`-9`, 10\ :sup:`-6.7`].

When running on a uniform grid of 40 by 40 points with the rbf kernel type, 
it is clear that the graph has several local minima.
With this search, the optimal metric value was found equal to -0.949421.

.. figure:: images/rbf_kernel.JPG
   :width: 500
   :align: center

   Graph of f1-score metric values on a specified area with kernel = rbf

When changing the kernel type from rbf to sigmoid, the graph changed, 
but still has a multi-extreme nature.
The minimum metric value on the specified grid is -0.93832.

.. figure:: images/sigmoid_kernel.JPG
   :width: 500
   :align: center

   Graph of f1-score metric values on a specified area with kernel = sigmoid

When studying the behavior of the metric graph on a uniform grid with a poly kernel, 
weak multiextremality is observed.
However, the minimum value of the metric was not found, and the method stopped at the value -0.9337763.

.. figure:: images/poly_kernel.JPG
   :width: 500
   :align: center

   Graph of f1-score metric values on a specified area with kernel = poly

When varying the kernel type for the SVC algorithm, different metric values are observed. 
Thus, it becomes possible to explore the behavior of the metric using the iOpt framework, 
varying the parameters **C**, **gamma** and **kernel** in the specified area.

Let's prepare a problem to solve. To do this, as in the two-dimensional case, 
it is necessary to declare a class that is a descendant of the **Problem** class 
with the abstract **Calculate** method. The code for this class is presented below:

.. code-block:: 

   class SVC_3D(Problem):
      def __init__(self, x_dataset: np.ndarray, y_dataset: np.ndarray,
                  regularization_bound: Dict[str, float],
                  kernel_coefficient_bound: Dict[str, float],
                  kernel_type: Dict[str, List[str]]
                  ):
         super(SVC_3D, self).__init__()
         self.dimension = 3
         self.number_of_float_variables = 2
         self.number_of_discrete_variables = 1
         self.number_of_objectives = 1
         self.number_of_constraints = 0
         if x_dataset.shape[0] != y_dataset.shape[0]:
               raise ValueError('The input and output sample sizes do not match.')
         self.x = x_dataset
         self.y = y_dataset
         self.float_variable_names = np.array(["Regularization parameter", "Kernel coefficient"], dtype=str)
         self.lower_bound_of_float_variables = np.array([regularization_bound['low'], kernel_coefficient_bound['low']],
                                                      dtype=np.double)
         self.upper_bound_of_float_variables = np.array([regularization_bound['up'], kernel_coefficient_bound['up']],
                                                      dtype=np.double)
         self.discrete_variable_names.append('kernel')
         self.discrete_variable_values.append(kernel_type['kernel'])

      def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
         cs, gammas = point.float_variables[0], point.float_variables[1]
         kernel_type = point.discrete_variables[0]
         clf = SVC(C=10 ** cs, gamma=10 ** gammas, kernel=kernel_type)
         functionValue.value = -cross_val_score(clf, self.x, self.y, scoring='f1').mean()
         return functionValue


The SVC_3D class accepts the following constructor parameters:

#. **x_dataset** – an array of objects and their attributes wrapped in **np.ndarray**;
#. **y_dataset** – target labels of each of the **x_dataset** objects in the **np.ndarray** format;
#. **regularization_bound** – maximum and minimum values for **C** as a dictionary;
#. **kernel_coefficient_bound** – maximum and minimum values for **gamma** as a dictionary.
#. **kernel_type** – the kernel type used in the SVC algorithm.

The **Calculate** method implements the logic for calculating the objective function 
at the **Point** point. It is worth noting that the **Point** contains two real parameters 
and one discrete one, which are used to train the SVC classifier.
To obtain the value of the optimized function, the average f1-score value is calculated 
based on cross-validation.

To start the optimization process, you need to create an object of the **SVC_3D** class, 
as well as an object of the **Solver** class with the passed target function object.

When searching for the optimal combination of discrete and continuous parameters, 
the following areas were considered:

#. regularization parameter **C**: [10\ :sup:`1`, 10\ :sup:`10`];
#. kernel coefficient **gamma**: [10\ :sup:`-9`, 10\ :sup:`-6.7`];
#. kernel type **kernel**: [rbf, sigmoid, poly].

To perform the prepared problem, a script has been developed in which the previously specified area 
is set to search for the optimal combination of hyperparameters. As part of the script, data is loaded 
on which the operation of the SVC algorithm is analyzed. Listeners have been added to the script, 
which provide additional information about the search process. Thus, **ConsoleOutputListener** allows 
you to track the process of searching for the optimal set of hyperparameters, visualizing test points 
and metric values at a given point in the console.
**StaticDiscreteListener** with mode='analysis' provides summary statistics in graphical form
according to the study, which displays:

* Graph of the dependence of the value of the objective function on the number of tests
* The metric value at each iteration depending on the selected value of the discrete parameter
* Graph of the minimum values of the objective function at each iteration
* Values of the objective function when using a specific value of the discrete parameter

.. figure:: images/statictic.JPG
   :width: 500
   :align: center

   Statistics on finding the optimal combination of parameters using iOpt

**StaticDiscreteListener** with mode='bestcombination' visualizes level lines that correspond 
to the graph with the value of the discrete parameter at which the optimum for the metric was found. 
The blue dots on the graph indicate the test points of the solver for the best value 
of the discrete parameter, and the gray dots indicate all the others. The red dot is 
the found minimum value of the objective function.

.. figure:: images/best_solve.JPG
   :width: 500
   :align: center

   Level lines of the graph corresponding to the function with the “best” value of the discrete parameter

.. code-block:: 
    :caption: Script for finding the optimal combination of hyperparameters
.. code-block:: 
   
   from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
   from iOpt.output_system.listeners.static_painters import StaticDiscreteListener
   from sklearn.datasets import load_breast_cancer
   from iOpt.solver import Solver
   from iOpt.solver_parametrs import SolverParameters
   from examples.Machine_learning.SVC._3D.Problem import SVC_3D
   from sklearn.utils import shuffle

   def load_breast_cancer_data():
      dataset = load_breast_cancer()
      x_raw, y_raw = dataset['data'], dataset['target']
      inputs, outputs = shuffle(x_raw, y_raw ^ 1, random_state=42)
      return inputs, outputs

   if __name__ == "__main__":
      x, y = load_breast_cancer_data()
      regularization_value_bound = {'low': 1, 'up': 10}
      kernel_coefficient_bound = {'low': -9, 'up': -6.7}
      kernel_type = {'kernel': ['rbf', 'sigmoid', 'poly']}
      problem = SVC_3D.SVC_3D(x, y, regularization_value_bound, kernel_coefficient_bound, kernel_type)
      method_params = SolverParameters(iters_limit=400)
      solver = Solver(problem, parameters=method_params)
      apl = StaticDiscreteListener("experiment1.png", mode='analysis')
      solver.AddListener(apl)
      apl = StaticDiscreteListener("experiment2.png", mode='bestcombination', calc='interpolation', mrkrs=4)
      solver.AddListener(apl)
      cfol = ConsoleOutputListener(mode='full')
      solver.AddListener(cfol)
      solver_info = solver.Solve()

During the work of the framework, a minimum f1-score was found equal to **-0.95157487**. 
Number of solver iterations **iterLimits**=400.


Tuning hyperparameters of the support vector machine for the problem of classifying the state of the air pressure system of trucks
__________________________________________________________________________________________________________________________________

The use of machine learning methods is relevant not only in medicine, but also in industry. 
From an algorithmic point of view, the solution of the problem of classifying the state 
of machine units or the quality of manufactured products does not differ from the classification 
of neoplasms and human conditions. Let’s demonstrate the work of the iOpt framework when tuning 
the hyperparameters (regularization coefficient **C** and kernel coefficient **gamma**) of 
the support vector machine (SVC) in order to maximize the f1-score metric.

Dataset used
~~~~~~~~~~~~

We will use an industrial data set that describes failures in the compressed air supply system 
for Scania_ trucks: braking system, gearshift system, etc.
The initial data consists of 60,000 samples, with each sample characterized by a set of 171 attributes. 

.. _Scania:  http://archive.ics.uci.edu/ml/datasets/IDA2016Challenge

There are two classes in the dataset:

#. the Positiv class characterizes a sample in which, by the combination of attributes, 
   it is possible to establish a failure of the compressed air injection system;
#. the Negative class characterizes the system in which a failure has occurred that is not related 
   to the compressed air injection system.

The attributes in the data set are de-identified in order to respect the confidentiality 
of the characteristics of the Scania truck system. Some of the table cells have an undefined value. 
The data set contains 59 thousand samples, whose set of attributes describes the failure of 
the APS (air pressure system), and one thousand samples describing the failure of other systems. 
In further experiments (to obtain a result in a reasonable time), we will use a subset 
of the initial data consisting of two thousand samples.


Solving the problem with default parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's solve the classification problem using the SVC_ method from the scikit-learn package. 
We will use the default hyperparameters.

.. _SVC: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

.. code-block::
   :caption: Solving the problem with default parameters

   from sklearn.model_selection import cross_val_score
   from sklearn.metrics import f1_score
   from sklearn.svm import SVC
   import pandas as pd

   def get_SCANIA_dataset():
      xls = pd.read_excel(r"aps_failure_training_set1.xls", header=None)
      data = xls.values[1:]
      row, col = data.shape
      _x = data[:,1:col]
      _y = data[:, 0]
      y = np.array(_y, dtype=np.double)
      x = np.array(_x, dtype=np.double)
      return shuffle(x, y, random_state=42)

   X, Y = get_SCANIA_dataset()
   x = X[:2000]
   y = Y[:2000]

   model = Pipeline([('scaler', StandardScaler()), ('model', SVC())])
   cross_val_score(model, x, y, cv=3, scoring="f1").mean()

With the default hyperparameters, we solved the problem with an average f1-score over all 
cross-validation experiments of 0.1068376. 


Calculation of the averaged f1-score on a grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let’s make sure that this problem can be solved better by varying two continuous parameters 
of the algorithm. To do this, we calculate the average value of cross-validation at each point 
of a uniform 20 by 20 grid:

#. regularization parameter **C**: [10\ :sup:`1`, 10\ :sup:`10`];
#. kernel coefficient **gamma**: [10\ :sup:`-8`, 10\ :sup:`-1`].

For convenience, we use a logarithmic scale and the **np.logspace** function to obtain 
the corresponding grid values.

.. code-block::
   :caption: Solving the problem with default parameters

   import numpy as np

   model = Pipeline([('scaler', StandardScaler()), ('model', SVC())])

   cs = np.logspace(1, 10, 20)
   gamms = np.logspace(-8, -1, 20)

   params = {'model__C': cs, 'model__gamma': gamms}

   search = GridSearchCV(model, cv=3, param_grid=params, scoring='f1')
   search.fit(x, y)

Let’s display the results of the experiment on a graph. To reduce the maximization problem 
to a minimization problem, we multiply the objective function by minus one.

.. figure:: images/scania_svc_f1.png
    :width: 800
    :align: center
    
    Graph of average f1-score on a 20x20 grid for the APS failure problem


Finding optimal parameters using the iOpt framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's launch the iOpt framework to find the optimal hyperparameters of the SVC method 
that minimize the f1-score. To do this, one must declare a class that is a successor 
of the **Problem** class with an abstract **Calculate** method.

To start the optimization process, it is necessary to create an object of the **SVC_2D** class, 
as well as an object of the **Solver** class with the passed objective function object. 
To render the results, we called the **AddListener** method, passing objects 
of the **AnimationNDPaintListener** and **StaticNDPaintListener** classes.

.. code-block::
   :caption: Running optimization of the SVC_2D object serving as the objective function

   from iOpt.method.listener import StaticNDPaintListener, AnimationNDPaintListener, ConsoleFullOutputListener
   from iOpt.solver import Solver
   from iOpt.solver_parametrs import SolverParameters
   from examples.Machine_learning.SVC._2D.Problems import SVC_2d
   from sklearn.utils import shuffle
   import numpy as np
   import pandas as pd

   def get_SCANIA_dataset():
      xls = pd.read_excel(r"../Datasets/aps_failure_training_set1.xls", header=None)
      data = xls.values[1:]
      row, col = data.shape
      _x = data[:, 1:col]
      _y = data[:, 0]
      y = np.array(_y, dtype=np.double)
      x = np.array(_x, dtype=np.double)
      return shuffle(x, y, random_state=42)

   if __name__ == "__main__":
      X, Y = get_SCANIA_dataset()
      x = X[:2000]
      y = Y[:2000]
      regularization_value_bound = {'low': 1, 'up': 10}
      kernel_coefficient_bound = {'low': -8, 'up': -1}
      problem = SVC_2d.SVC_2D(x, y, regularization_value_bound, kernel_coefficient_bound)
      method_params = SolverParameters(r=np.double(2.0), iters_limit=200)
      solver = Solver(problem, parameters=method_params)
      apl = AnimationNDPaintListener(vars_indxs=[0, 1], to_paint_obj_func=False)
      solver.AddListener(apl)
      spl = StaticNDPaintListener(vars_indxs=[0, 1], mode="surface", calc="interpolation")
      solver.AddListener(spl)
      cfol = ConsoleFullOutputListener(mode='full')
      solver.AddListener(cfol)
      solver_info = solver.Solve()


After the experiment, the program displays the total search time for the optimum, 
the point at which the optimum is reached, the found optimal value of the f1-score metric, 
as well as the graph of the objective function based on the points of the search trials. 
With a limit on the number of iterations **iterLimits**=200, the framework finds hyperparameters 
where the target metric reaches 0.5723, the total calculation time is less than 1 minute.

.. figure:: images/scania_iopt_interpol.png
    :width: 500
    :align: center
    
    Graph of the objective function based on test points


The blue dots on the graph represent the points of exploratory trials, the red dot marks 
the found optimum corresponding to the hyperparameters at which the value of f1-score reaches a minimum.

To compare the quality of work, experiments were carried out on the same data set using 
the well-known Scikit-Optimize framework.

.. code-block::
   :caption: Starting the search for the optimal value of f1-score using the Scikit-Optimize framework methods

   from skopt.space import Real
   from skopt.utils import use_named_args
   import skopt

   space  = [Real(1e1, 1e10, name='C'),
            Real(1e-8, 1e-1, name='gamma')]

   @use_named_args(space)
   def objective(**p):
      model = Pipeline([('scaler', StandardScaler()), ('model', SVC(**p))])
      return -np.mean(cross_val_score(model, x, y, scoring='f1'))

   results = skopt.gbrt_minimize(objective, space, n_calls=500)
   print(results.fun)


Over 500 iterations, the Scikit-Optimize framework found a combination of parameters that provides 
a solution to the classification problem with f1-score value of 0.4492, which is 21% worse than 
the optimal value obtained using iOpt over 200 iterations.