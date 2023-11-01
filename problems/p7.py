import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import FunctionType
from iOpt.trial import Trial
from iOpt.problem import Problem
import math


class p7(Problem):
    """

    """

    def __init__(self):
        """
        The constructor of the p7 problem class
        """
        super(p7, self).__init__()
        self.name = "p7"
        self.dimension = 3
        self.number_of_float_variables = 2
        self.number_of_discrete_variables = 1
        self.number_of_objectives = 1
        self.number_of_constraints = 4

        self.float_variable_names = np.ndarray(shape=(self.number_of_float_variables,), dtype=object)
        for i in range(self.number_of_float_variables):
            self.float_variable_names[i] = str(i)

        self.discrete_variable_names = np.ndarray(shape=(self.number_of_discrete_variables,), dtype=object)
        for i in range(self.number_of_discrete_variables):
            self.discrete_variable_names[i] = str(i)

        self.lower_bound_of_float_variables = np.ndarray(shape=(self.number_of_float_variables,), dtype=np.double)
        self.lower_bound_of_float_variables.fill(-0.01)
        self.upper_bound_of_float_variables = np.ndarray(shape=(self.number_of_float_variables,), dtype=np.double)
        self.upper_bound_of_float_variables.fill(4)

        self.discrete_variable_values = [["0", "1"] for i in range(self.number_of_discrete_variables)]

        self.known_optimum = np.ndarray(shape=(1,), dtype=Trial)

        pointfv = np.ndarray(shape=(self.number_of_float_variables,), dtype=np.double)
        pointfv = [3.514237, 0]

        pointdv = np.ndarray(shape=(self.number_of_discrete_variables,), dtype=object)
        pointdv[0] = "1"

        KOpoint = Point(pointfv, pointdv)
        KOfunV = np.ndarray(shape=(1,), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0].value = 99.245209
        self.known_optimum[0] = Trial(KOpoint, KOfunV)


    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Calculate the value of the selected function at a given point

        :param point: coordinates of the trial point where the value of the function will be calculated.
        :param function_value: object defining the function number in the task and storing the function value.
        :return: Calculated value of the function at point.
        """
        result: np.double = 0
        x = point.float_variables
        b = int(point.discrete_variables[0])

        if function_value.type == FunctionType.OBJECTIV:
            result = np.double(7.5 * b + 5.5 * (1 - b) + 7 * x[0] + 6 * x[1] +
                               50 * (b / (2 * b - 1)) / (0.9 * (1 - math.exp(-0.5 * x[0]))) +
                               50 * (1 - b / (2 * b - 1)) / (0.8 * (1 - math.exp(-0.4 * x[1]))))
        elif function_value.functionID == 0:  # constraint 1
            result = np.double(0.9 * (1 - math.exp(-0.5 * x[0])) - 2 * b)
        elif function_value.functionID == 1:  # constraint 2
            result = np.double(0.8 * (1 - math.exp(-0.4 * x[1])) - 2 * (1 - b))
        elif function_value.functionID == 2:  # constraint 3
            result = np.double(x[0] - 10 * b - 0.01)
        elif function_value.functionID == 3:  # constraint 4
            result = np.double(x[1] - 10 * (1 - b))

        function_value.value = result
        return function_value
