import numpy as np

from iOpt.problem import Problem
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters

from hyperparams import Hyperparameter, Numerical, Categorial

from .interface import Searcher, Point


class Estimator(Problem):
    def __init__(self, searcher: Searcher,
                 float_hyperparams: dict[str, Numerical], discrete_hyperparams: dict[str, Hyperparameter],
                 points: list[Point]):
        super().__init__()

        self.points = points
        self.searcher = searcher

        self.number_of_float_variables = len(float_hyperparams)
        self.number_of_discrete_variables = len(discrete_hyperparams)
        self.dimension = len(float_hyperparams) + len(discrete_hyperparams)
        self.number_of_objectives = 1

        self.float_variables_types, self.is_log_float = [], []
        for name, p in float_hyperparams.items():
            self.float_variable_names.append(name)
            self.float_variables_types.append(p.type)
            self.lower_bound_of_float_variables.append(np.log(p.min_value) if p.is_log_scale else p.min_value)
            self.upper_bound_of_float_variables.append(np.log(p.max_value) if p.is_log_scale else p.max_value)
            self.is_log_float.append(p.is_log_scale)

        for name, p in discrete_hyperparams.items():
            self.discrete_variable_names.append(name)
            if isinstance(p, Numerical):
                assert type == 'int', 'Type must be int'
                assert not p.is_log_scale, 'Log must be off'
                self.discrete_variable_values.append([str(x) for x in range(p.min_value, p.max_value + 1)])
            elif isinstance(p, Categorial):
                self.discrete_variable_values.append(p.values)

    def calculate(self, point, function_value):
        arguments = self.__get_argument_dict(point)
        custom_point = self.searcher._calculate_metric(arguments)
        self.points.append(custom_point)

        function_value.value = -custom_point.value
        return function_value

    def __get_argument_dict(self, point):
        arguments = {}
        for name, type, value, log in zip(self.float_variable_names, self.float_variables_types,
                                          point.float_variables,
                                          self.is_log_float):
            value = np.exp(value) if log else value
            value = int(value + 0.5) if type == 'int' else value
            arguments[name] = value
        if point.discrete_variables is not None:
            for name, value in zip(self.discrete_variable_names, point.discrete_variables):
                arguments[name] = int(value) if value.isnumeric() else value
        return arguments


class iOptSearcher(Searcher):
    def __init__(self, max_iter, *, is_deterministic=True, **kwargs):
        super().__init__(framework_name='iOpt',
                         max_iter=max_iter,
                         is_deterministic=is_deterministic)

        self.kwargs = kwargs

    def _get_points(self):

        floats, discretes = self.split_hyperparams()
        points = []
        problem = Estimator(self, floats, discretes, points)
        framework_params = SolverParameters(iters_limit=self.max_iter, **self.kwargs)
        solver = Solver(problem, parameters=framework_params)
        solver.solve()
        return points

    def _get_searcher_params(self):
        return self.kwargs.copy()

    def split_hyperparams(self):
        floats, discretes = {}, {}
        for name, x in self.hyperparams.items():
            if self.is_discrete_hyperparam(x):
                discretes[name] = x
            else:
                floats[name] = x
        return floats, discretes

    @staticmethod
    def is_discrete_hyperparam(p: Hyperparameter):
        if isinstance(p, Numerical):
            return (p.type == 'int') and (not p.is_log_scale) and (p.max_value - p.min_value + 1 <= 5)
        return True
