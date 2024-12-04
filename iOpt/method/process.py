from datetime import datetime
from typing import List

import traceback
import json

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.calculator import Calculator
from iOpt.method.listener import Listener
from iOpt.method.local_optimizer import local_optimize
from iOpt.method.method import Method
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.solution import Solution
from iOpt.solver_parametrs import SolverParameters
from iOpt.trial import FunctionValue, FunctionType


class Process:
    """
    The Process class hides the internal implementation of the Solver class
    """

    def __init__(self,
                 parameters: SolverParameters,
                 task: OptimizationTask,
                 evolvent: Evolvent,
                 search_data: SearchData,
                 method: Method,
                 listeners: List[Listener],
                 calculator: Calculator = None
                 ):
        """
        Constructor of the Process class

        :param parameters: Parameters of the solution to the optimization problem.
        :param task: The wrapper of the problem to be solved.
        :param evolvent: Peano-Hilbert evolvent mapping the segment [0,1] to the multidimensional region D.
        :param search_data: A data structure for storing accumulated search information.
        :param method: An optimization method that performs search trials according to given rules.
        :param listeners: List of "observers" (used to display current information).
        :param calculator: class containing trial methods (parallel and/or inductive circuit)
        """
        self.parameters = parameters
        self.task = task
        self.evolvent = evolvent
        self.search_data = search_data
        self.method = method
        self._listeners = listeners
        self._first_iteration = True
        if calculator is None:
            self.calculator = method.calculator
        else:
            self.calculator = calculator

    def solve(self) -> Solution:
        """
        Solve an optimization problem. The search is stopped according to the criterion,
        specified when creating the Solver class

        :return: Current evaluation of the solution to the optimization problem.
        """

        start_time = datetime.now()

        try:
            while not self.method.check_stop_condition():
                self.do_global_iteration()

        except Exception:
            print('Exception was thrown')
            print(traceback.format_exc())

        if self.parameters.refine_solution:
            self.do_local_refinement(self.parameters.local_method_iteration_count)

        result = self.get_results()
        result.solving_time += (datetime.now() - start_time).total_seconds()

        for listener in self._listeners:
            status = self.method.check_stop_condition()
            listener.on_method_stop(self.search_data, self.get_results(), status)

        return result

    def do_global_iteration(self, number: int = 1):
        """
        Perform several iterations of the global search

        :param number: Number of iterations of global search.
        """
        number_ = number
        done_trials = []
        if self._first_iteration is True:
            for listener in self._listeners:
                listener.before_method_start(self.method)
            done_trials = self.method.first_iteration()
            self._first_iteration = False
            number -= 1

        for _ in range(number):
            newpoint, oldpoint = self.method.calculate_iteration_point()
            self.method.calculate_functionals(newpoint)
            self.method.update_optimum(newpoint)
            self.method.renew_search_data(newpoint, oldpoint)
            self.method.finalize_iteration()
            done_trials = self.search_data.get_last_items(self.parameters.number_of_parallel_points * number_)

        for listener in self._listeners:
            listener.on_end_iteration(done_trials, self.get_results())

    def do_local_refinement(self, number: int = 1):
        """
        Perform several iterations of local search

        :param number: Number of iterations of local search.
        """
        try:
            local_method_iteration_count = number
            if number == -1:
                local_method_iteration_count = int(self.parameters.local_method_iteration_count)

            result = self.get_results()
            # start_point = result.bestTrials[0].point.floatVariables

            local_solution = local_optimize(self.task,
                                            method="Hooke-Jeeves", start_point=result.best_trials[0].point,
                                            max_calcs=local_method_iteration_count,
                                            args={"eps": self.parameters.eps / 100, "step_mult": 2,
                                                  "max_iter": local_method_iteration_count}
                                            )
            # scipy.optimize.minimize(self.problemCalculate, x0=start_point, method='Nelder-Mead',
            #                        options={'maxiter': local_method_iteration_count})
            # local_solution = LocalOptimize(LocalTaskWrapper(self.task, result.bestTrials[0].point.discreteVariables),
            #                               method="Nelder-Mead", start_point=start_point,
            #                               args={"options": {'maxiter': local_method_iteration_count}})

            if local_method_iteration_count > 0:
                result.best_trials[0].point.float_variables = local_solution["x"]

                point: SearchDataItem = SearchDataItem(result.best_trials[0].point,
                                                       self.evolvent.get_inverse_image(
                                                           result.best_trials[0].point.float_variables),
                                                       function_values=[FunctionValue()] *
                                                                       (self.task.problem.number_of_constraints +
                                                                        self.task.problem.number_of_objectives)
                                                       )

                number_of_constraints = self.task.problem.number_of_constraints
                for i in range(number_of_constraints):
                    point.function_values[i] = FunctionValue(FunctionType.CONSTRAINT, i)
                    point.function_values[i] = self.task.problem.calculate(point.point, point.function_values[i])
                    point.set_z(point.function_values[i].value)
                    point.set_index(i)
                    if point.get_z() > 0:
                        break
                point.function_values[number_of_constraints] = FunctionValue(FunctionType.OBJECTIV,
                                                                             number_of_constraints)
                point.function_values[number_of_constraints] = \
                    self.task.problem.calculate(point.point, point.function_values[number_of_constraints])
                point.set_z(point.function_values[number_of_constraints].value)
                point.set_index(number_of_constraints)

                result.best_trials[0].function_values = point.function_values

            result.number_of_local_trials = local_solution["fev"]
        except Exception:
            print("Local Refinement is not possible")

    def get_results(self) -> Solution:
        """
        Return the best solution to the optimization problem

        :return: Optimization problem solution.
        """
        return self.search_data.solution

    def save_progress(self, file_name: str, mode='full') -> None:
        """
        Save the optimization process from a file

        :param mode: 'full' - save all optimization information
        :param file_name: file name.
        """
        data = self.search_data.searchdata_to_json(mode=mode)
        data['Parameters'] = []
        data['Parameters'].append({
            'eps': self.parameters.eps,
            'r': self.parameters.r,
            'iters_limit': self.parameters.iters_limit,
            'start_point': self.parameters.start_point,
            'number_of_parallel_points': self.parameters.number_of_parallel_points
        })
        with open(file_name, 'w') as f:
            json.dump(data, f, indent='\t', separators=(',', ':'))
            f.write('\n')

    def load_progress(self, file_name: str, mode='full') -> None:
        """
        Load the optimization process from a file

        :param file_name: file name.
        """
        with open(file_name) as json_file:
            data = json.load(json_file)

        self.search_data.json_to_searchdata(data=data, mode=mode)
        self.method.iterations_count = self.search_data.get_count() - 2

        for ditem in self.search_data:
            if ditem.get_index() >= 0:
                self.method.update_optimum(ditem)

        self.method.recalc_m()
        self.method.recalc_all_characteristics()
        self._first_iteration = False

        for listener in self._listeners:
            listener.before_method_start(self.method)

    '''
    def RefreshListener(self):
        pass

    def AddListener(self, listener: Listener):
        #self.__listeners.append(listener)
        pass
    '''
