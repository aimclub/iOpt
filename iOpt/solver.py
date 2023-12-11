from time import time
from typing import List

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.grid_search_method import GridSearchMethod
from iOpt.method.listener import Listener
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchData
from iOpt.method.solverFactory import SolverFactory
from iOpt.problem import Problem
from iOpt.routine.timeout import timeout
from iOpt.solution import Solution
from iOpt.solver_parametrs import SolverParameters


class Solver:
    """
    The Solver class is designed to select optimal (in a given metric) values of parameters of complex objects and processes
    Solver class is intended for selecting optimal (in a given metric) values of parameters of complex objects and processes, e.g., methods of artificial intelligence and
    machine learning and heuristic optimization methods
    """

    def __init__(self,
                 problem: Problem,
                 parameters: SolverParameters = SolverParameters()
                 ):
        """
        Solver class constructor

        :param problem: Optimization problem formulation.
        :param parameters: Parameters of search for optimal solutions.
        """

        self.problem = problem
        self.parameters = parameters

        Solver.check_parameters(self.problem, self.parameters)

        self.__listeners: List[Listener] = []

        self.search_data = SearchData(problem)
        self.evolvent = Evolvent(problem.lower_bound_of_float_variables, problem.upper_bound_of_float_variables,
                                 problem.number_of_float_variables)
        self.task = OptimizationTask(problem)
        self.method = SolverFactory.create_method(parameters, self.task, self.evolvent, self.search_data)
        self.process = SolverFactory.create_process(parameters=parameters, task=self.task, evolvent=self.evolvent,
                                                    search_data=self.search_data, method=self.method,
                                                    listeners=self.__listeners)

    def solve(self) -> Solution:
        """
        Solve an optimization problem. The search is stopped according to the criterion,
        specified when creating the Solver class

        :return: optimization problem solution.
        """
        Solver.check_parameters(self.problem, self.parameters)
        if self.parameters.timeout < 0:
            sol = self.process.solve()
        else:
            solv_with_timeout = timeout(seconds=self.parameters.timeout * 60)(self.process.solve)
            try:
                solv_with_timeout()
                sol = self.get_results()
            except Exception as exc:
                print(exc)
                sol = self.get_results()
                sol.solving_time += self.parameters.timeout * 60
                self.method.recalcR = True
                self.method.recalcM = True
                status = self.method.check_stop_condition()
                for listener in self.__listeners:
                    listener.on_method_stop(self.search_data, self.get_results(), status)
        return sol

    def do_global_iteration(self, number: int = 1):
        """
        Perform several iterations of the global search

        :param number: number of global search iterations.
        """
        Solver.check_parameters(self.problem, self.parameters)
        self.process.do_global_iteration(number)

    def do_local_refinement(self, number: int = 1):
        """
        Perform several iterations of local search

        :param number: number of local search iterations.
        """
        Solver.check_parameters(self.problem, self.parameters)
        self.process.do_local_refinement(number)

    def get_results(self) -> Solution:
        """
        Provide a current estimate of the solution to the optimization problem

        :return: Solving the optimization problem.
        """
        return self.process.get_results()

    def save_progress(self, file_name: str = None, mode = 'full') -> str:
        """
        Save the optimization process to a file

        :param file_name: file name.
        """

        if file_name is None:
            file_name = "log_" + self.parameters.to_string() + "_" + str(time())

        self.process.save_progress(file_name=file_name, mode=mode)

        return file_name

    def load_progress(self, file_name: str, mode = 'full') -> None:
        """
        Load the optimization process from a file

        :param file_name: file name.
        """
        Solver.check_parameters(self.problem, self.parameters)
        self.process.load_progress(file_name=file_name, mode=mode)

        if (self.problem.number_of_discrete_variables > 0):
            self.process.method.iterations_count = self.process.search_data.get_count() - (
                    len(self.method.GetDiscreteParameters(self.problem)) + 1)  # -2
        else:
            self.process.method.iterations_count = self.process.search_data.get_count() - 2

        if mode == 'only search_data':
            self.process.search_data.solution.number_of_global_trials = self.process.method.iterations_count



    def refresh_listener(self) -> None:
        """
        Notify observers of an event that has occurred
        """

        pass

    def add_listener(self, listener: Listener) -> None:
        """
        Additions of an optimization process observer

        :param listener: class object implementing observation methods.
        """

        self.__listeners.append(listener)

    @staticmethod
    def check_parameters(problem: Problem,
                         parameters: SolverParameters = SolverParameters()) -> None:
        """
        Check the parameters of the solver

        :param problem: Optimization problem formulation.
        :param parameters: Parameters of search for optimal solutions.

        """

        if parameters.eps <= 0:
            raise Exception("search precision is incorrect, parameters.eps <= 0")
        if parameters.r <= 1:
            raise Exception("The reliability parameter should be greater 1. r>1")
        if parameters.iters_limit < 1:
            raise Exception("The number of iterations must not be negative. iters_limit>0")
        if parameters.evolvent_density < 2 or parameters.evolvent_density > 20:
            raise Exception("Evolvent density should be within [2,20]")
        if parameters.eps_r < 0 or parameters.eps_r >= 1:
            raise Exception("The epsilon redundancy parameter must be within [0, 1)")

        if problem.number_of_float_variables < 1:
            raise Exception("Must have at least one float variable")
        if problem.number_of_discrete_variables < 0:
            raise Exception("The number of discrete parameters must not be negative")
        if problem.number_of_objectives < 1:
            raise Exception("At least one criterion must be defined")
        if problem.number_of_constraints < 0:
            raise Exception("The number of сonstraints must not be negative")

        if len(problem.float_variable_names) != problem.number_of_float_variables:
            raise Exception("Floaf parameter names are not defined")

        if len(problem.lower_bound_of_float_variables) != problem.number_of_float_variables:
            raise Exception("List of lower bounds for float search variables defined incorrectly")
        if len(problem.upper_bound_of_float_variables) != problem.number_of_float_variables:
            raise Exception("List of upper bounds for float search variables defined incorrectly")

        for lower_bound, upper_bound in zip(problem.lower_bound_of_float_variables,
                                            problem.upper_bound_of_float_variables):
            if lower_bound >= upper_bound:
                raise Exception("For floating point search variables, "
                                "the upper search bound must be greater than the lower.")

        if problem.number_of_discrete_variables > 0:
            if len(problem.discrete_variable_names) != problem.number_of_discrete_variables:
                raise Exception("Discrete parameter names are not defined")

            for discreteValues in problem.discrete_variable_values:
                if len(discreteValues) < 1:
                    raise Exception("Discrete variable values not defined")

        if parameters.start_point:
            if len(parameters.start_point.float_variables) != problem.number_of_float_variables:
                raise Exception("Incorrect start point size")
            if parameters.start_point.discrete_variables:
                if len(parameters.start_point.discrete_variables) != problem.number_of_discrete_variables:
                    raise Exception("Incorrect start point discrete variables")
            for lower_bound, upper_bound, y in zip(problem.lower_bound_of_float_variables,
                                                   problem.upper_bound_of_float_variables,
                                                   parameters.start_point.float_variables):
                if y < lower_bound or y > upper_bound:
                    raise Exception("Incorrect start point coordinate")

    def grid_search(self) -> Solution:
        """
        Search optimal value using grid search algorithm.

        :return: optimization problem solution.
        """

        temp_method = self.method
        temp_process = self.process

        self.method = GridSearchMethod(self.parameters, self.task, self.evolvent, self.search_data)
        self.process = SolverFactory.create_process(parameters=self.parameters, task=self.task, evolvent=self.evolvent,
                                                    search_data=self.search_data, method=self.method,
                                                    listeners=self.__listeners)

        sol = self.solve()

        self.method = temp_method
        self.process = temp_process

        return sol

    def release_all_listener(self):
        """
        Force all listeners to start.
        """

        for listener in self.__listeners:
            listener.on_end_iteration(self.search_data.get_last_items(self.search_data.get_count() - 2),
                                      self.get_results())

        status = self.method.check_stop_condition()
        for listener in self.__listeners:
            listener.on_method_stop(self.search_data, self.get_results(), status)