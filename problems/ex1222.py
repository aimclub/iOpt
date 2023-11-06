import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import FunctionType
from iOpt.trial import Trial
from iOpt.problem import Problem
import math


class ex1222(Problem):
    """

    """

    def __init__(self):
        """
        Constructor of the Ex1222 problem class
        """
        super(ex1222, self).__init__()
        self.name = "ex1222"
        self.dimension = 3
        self.number_of_float_variables = 2
        self.number_of_discrete_variables = 1
        self.number_of_objectives = 1
        self.number_of_constraints = 3

        self.float_variable_names = np.ndarray(shape=(self.number_of_float_variables,), dtype=object)
        for i in range(self.number_of_float_variables):
            self.float_variable_names[i] = str(i)

        self.discrete_variable_names = np.ndarray(shape=(self.number_of_discrete_variables,), dtype=object)
        for i in range(self.number_of_discrete_variables):
            self.discrete_variable_names[i] = str(i)

        self.lower_bound_of_float_variables = np.ndarray(shape=(self.number_of_float_variables,), dtype=np.double)
        self.lower_bound_of_float_variables[0] = 0.2
        self.lower_bound_of_float_variables[1] = -2.22554
        self.upper_bound_of_float_variables = np.ndarray(shape=(self.number_of_float_variables,), dtype=np.double)
        self.upper_bound_of_float_variables[0] = 1
        self.upper_bound_of_float_variables[1] = -1

        self.discrete_variable_values = [["0", "1"] for i in range(self.number_of_discrete_variables)]

        self.known_optimum = np.ndarray(shape=(1,), dtype=Trial)

        pointfv = np.ndarray(shape=(self.number_of_float_variables,), dtype=np.double)
        pointfv = [0.999609, -2.100473]

        pointdv = np.ndarray(shape=(self.number_of_discrete_variables,), dtype=object)
        pointdv[0] = "1"

        KOpoint = Point(pointfv, pointdv)
        KOfunV = np.ndarray(shape=(1,), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0].value = -3.634153
        self.known_optimum[0] = Trial(KOpoint, KOfunV)


    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Calculate the value of the selected function at a given point

        :param point: coordinates of the trial point where the value of the function will be calculated.
        :param function_value: object defining the function number in the task and storing the function value.
        :return: Calculated value of the function at the point.
        """
        result: np.double = 0
        x = point.float_variables
        b = point.discrete_variables

        if function_value.type == FunctionType.OBJECTIV:
            result = np.double(- 5.0 * math.sqrt(x[0] - 0.5) + 0.7 * int(b[0]) - 0.8)
        elif function_value.functionID == 0:  # constraint 1
            result = np.double(-math.exp(x[0] - 0.2) - x[1])
        elif function_value.functionID == 1:  # constraint 2
            result = np.double(x[1] + 1.1 * int(b[0]) + 1)
        elif function_value.functionID == 2:  # constraint 3
            result = np.double(x[0] - 1.2 * int(b[0]))

        function_value.value = result
        return function_value
