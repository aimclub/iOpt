import numpy as np
from iOpt.trial import Point


class SolverParameters:
    """
    The SolverParameters class allows you to define the parameters for searching the optimal solution
    """

    def __init__(self,
                 eps: np.double = 0.01,
                 r: np.double = 2.0,
                 iters_limit: int = 20000,
                 evolvent_density: int = 10,
                 eps_r: np.double = 0.01,
                 refine_solution: bool = False,
                 start_point: Point = [],
                 number_of_parallel_points: int = 1,
                 timeout: int = -1,
                 proportion_of_global_iterations: float = 0.95
                 ):
        r"""
        Constructor of SolverParameters class

        :param eps: The accuracy of the solution to the task at hand. Smaller values -- higher search accuracy,
             less likely to stop prematurely.
        :param r: Reliability parameter. Higher value of r -- slower convergence,
             higher probability of finding a global minimum.
        :param iters_limit: maximum number of search trials.
        :param evolvent_density: density of evolvent construction.
             The default density is :math:`2^{-10}` on the hypercube :math:`[0,1]^N`,
             which means that the maximum search accuracy is :math:`2^{-10}`.
        :param eps_r: parameter affecting the speed of solving the problem with constraints. eps_r = 0 - slow convergence
             to the exact solution, eps_r>0 - fast convergence to the neighbourhood of the solution.
        :param refine_solution: if true, the solution will be refined using the local method.
        :param start_point: point of initial approximation to the solution.
        :param number_of_parallel_points: number of parallel computed trials.
        :param timeout: calculation time limit in minutes.
        :param proportion_of_global_iterations: share of global iterations in the search when using the local method.
        """
        self.eps = eps
        self.r = r
        self.iters_limit = iters_limit
        self.proportion_of_global_iterations = proportion_of_global_iterations
        if refine_solution:
            self.global_method_iteration_count = int(self.iters_limit * self.proportion_of_global_iterations)
            self.local_method_iteration_count = self.iters_limit - self.global_method_iteration_count
        else:
            self.global_method_iteration_count = self.iters_limit
            self.local_method_iteration_count = 0

        self.evolvent_density = evolvent_density
        self.eps_r = eps_r
        self.refine_solution = refine_solution
        self.start_point = start_point
        self.number_of_parallel_points = number_of_parallel_points
        self.timeout = timeout
