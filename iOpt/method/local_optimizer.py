import sys
from typing import List, Iterable, Callable

import scipy

from iOpt.method.optim_task import OptimizationTask
from iOpt.trial import Point, FunctionValue, FunctionType


class LocalTaskWrapper:
    """
    The LocalTaskWrapper class (the name is temporary) wraps the function computation for further application of local methods
    """

    def __init__(self, task: OptimizationTask, discrete_variables=None, max_calcs=-1):
        self.discrete_variables = discrete_variables
        self.task = task
        self.calcs_count = 0
        self.max_calcs = max_calcs  # В globalizer используется именно ограничение по количеству вычислений функции

    def evaluate_function(self, y: List[float]) -> float:
        """
        Calculate the value of the objective function

        :param y: The point at which you need to calculate the value of the function.

        :return: returns the value of the objective function or
                sys.float_info.max, if:
                the point lies outside the search area
                OR the restrictions have not been met
                OR an exception was thrown (the function cannot be calculated at this point)
                OR the number of calculations has exceeded the limit (if set).
        """
        point = Point(y, self.discrete_variables)
        function_value = FunctionValue(FunctionType.OBJECTIV)
        if self.max_calcs != -1 and self.calcs_count >= self.max_calcs:
            function_value.value = sys.float_info.max
            return function_value.value
        for i in range(self.task.problem.number_of_float_variables):
            if (y[i] < self.task.problem.lower_bound_of_float_variables[i]) \
                    or (y[i] > self.task.problem.upper_bound_of_float_variables[i]):
                function_value.value = sys.float_info.max
                return function_value.value

        self.calcs_count += 1
        try:
            for i in range(self.task.problem.number_of_constraints):
                function_constraint_value = FunctionValue(FunctionType.CONSTRAINT, i)
                function_constraint_value = self.task.problem.calculate(point, function_constraint_value)
                if function_constraint_value.value > 0:
                    function_value.value = sys.float_info.max
                    return function_value.value

            function_value = self.task.problem.calculate(point, function_value)
        except Exception:
            function_value.value = sys.float_info.max

        return function_value.value


class HookeJeevesOptimizer:
    """
    The HookeJeevesOptimizer class implements the Hooke Jeeves method
    """

    def __init__(self, func: Callable[[List[float]], float], start_point: Iterable[float],
                 step_mult: float, eps: float, max_iter: float):
        self.nfev = 0
        self.cur_point = None
        self.minf = None
        self.best_point = None
        self.pr_resdir = None
        self.cur_resdir = None
        self.dim = len(start_point)
        self.f = func
        self.start_point = start_point
        self.max_iter = max_iter
        self.eps = min(eps, 0.0001)
        self.step = self.eps * 2
        self.step_mult = step_mult

    def minimize(self) -> List[float]:
        need_restart: bool = True
        # self.best_point = self.start_point
        # self.minf = self.f(self.start_point)
        k, i, curr_f = 0, 0, 0.0
        while i < self.max_iter:
            i += 1
            if need_restart:
                k = 0
                self.cur_point = self.start_point.copy()
                self.cur_resdir = self.start_point.copy()
                curr_f = self.f(self.cur_point)
                need_restart = False

            self.pr_resdir = [el for el in self.cur_resdir]
            self.cur_resdir = [el for el in self.cur_point]
            next_f_value = self._make_research(self.cur_resdir)

            if curr_f > next_f_value:
                self._do_step()
                k += 1
                curr_f = next_f_value
            elif self.step > self.eps:
                if k != 0:
                    self.start_point = self.pr_resdir.copy()
                else:
                    self.step /= self.step_mult
                need_restart = True
            else:
                break

        return self.pr_resdir

    def _make_research(self, point) -> float:
        best_value = self.f(point)  # в globalizer сделано так, хотя это значение уже было вычислено...

        for i in range(self.dim):
            point[i] += self.step
            right_f_val = self.f(point)
            if right_f_val > best_value:
                point[i] -= 2 * self.step
                left_f_val = self.f(point)

                if left_f_val > best_value:
                    point[i] += self.step
                else:
                    best_value = left_f_val

            else:
                best_value = right_f_val

        return best_value

    def _do_step(self) -> None:
        for i in range(self.dim):
            self.cur_point[i] = (1 + self.step_mult) * self.cur_resdir[i] - self.step_mult * self.pr_resdir[i]


def local_optimize(task: OptimizationTask, method, start_point: Point, args: dict, max_calcs: int = -1) -> dict:
    local_task = LocalTaskWrapper(task=task, discrete_variables=start_point.discrete_variables, max_calcs=max_calcs)
    if method == 'Hooke-Jeeves':
        best_point = HookeJeevesOptimizer(local_task.evaluate_function, start_point.float_variables.copy(),
                                          **args).minimize()
    else:
        best_point = scipy.optimize.minimize(local_task.evaluate_function, x0=start_point.float_variables.copy(),
                                             method=method, **args).x

    return {"x": best_point, "fev": local_task.calcs_count}
