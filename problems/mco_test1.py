import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import FunctionType
from iOpt.trial import Trial
from iOpt.problem import Problem
import math


class mco_test1(Problem):
    """

    """

    def __init__(self):
        """
        Конструктор класса mco_test1 problem.
        """
        super(mco_test1, self).__init__()
        self.name = "mco_test1"
        self.dimension = 2
        self.number_of_float_variables = 2
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.float_variable_names = np.ndarray(shape=(self.number_of_float_variables,), dtype=object)
        for i in range(self.number_of_float_variables):
            self.float_variable_names[i] = str(i)

        self.lower_bound_of_float_variables = np.ndarray(shape=(self.number_of_float_variables,), dtype=np.double)
        self.lower_bound_of_float_variables.fill(0)
        self.upper_bound_of_float_variables = np.ndarray(shape=(self.number_of_float_variables,), dtype=np.double)
        self.upper_bound_of_float_variables.fill(1)

        self.known_optimum = np.ndarray(shape=(1,), dtype=Trial)

    def calculateAllFunction(self, point: Point, function_values: np.ndarray(shape=(1), dtype=FunctionValue)) -> \
            np.ndarray(shape=(1), dtype=FunctionValue):
        """
        Calculate all function at a given point.
          For any new problem statement that inherits from :class:`Problem`, this method should be overloaded

        :return: Calculated values of the functions."""
        x = point.float_variables

        # OBJECTIVE 1
        function_values[0].value = np.double((x[0] - 1) * x[1] * x[1] + 1)
        # OBJECTIVE 2
        function_values[1].value = np.double(x[1])

        return function_values
