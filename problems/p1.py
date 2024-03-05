import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import FunctionType
from iOpt.trial import Trial
from iOpt.problem import Problem
import math


class p1(Problem):
    """

    """

    def __init__(self):
        """
        Constructor of the p1 problem class
        """
        super(p1, self).__init__()
        self.name = "p1"
        self.dimension = 2
        self.number_of_float_variables = 1
        self.number_of_discrete_variables = 1
        self.number_of_objectives = 1
        self.number_of_constraints = 2

        self.float_variable_names = np.ndarray(shape=(self.number_of_float_variables,), dtype=object)
        for i in range(self.number_of_float_variables):
            self.float_variable_names[i] = str(i)

        self.discrete_variable_names = np.ndarray(shape=(self.number_of_discrete_variables,), dtype=object)
        for i in range(self.number_of_discrete_variables):
            self.discrete_variable_names[i] = str(i)

        self.lower_bound_of_float_variables = np.ndarray(shape=(self.number_of_float_variables,), dtype=np.double)
        self.lower_bound_of_float_variables[0] = 0
        self.upper_bound_of_float_variables = np.ndarray(shape=(self.number_of_float_variables,), dtype=np.double)
        self.upper_bound_of_float_variables[0] = 1.6

        self.discrete_variable_values = [["0", "1"] for i in range(self.number_of_discrete_variables)]

        self.known_optimum = np.ndarray(shape=(1,), dtype=Trial)

        pointfv = np.ndarray(shape=(self.number_of_float_variables,), dtype=np.double)
        pointfv[0]= 0.5

        pointdv = np.ndarray(shape=(self.number_of_discrete_variables,), dtype=object)
        pointdv[0] = "1"

        KOpoint = Point(pointfv, pointdv)
        KOfunV = np.ndarray(shape=(1,), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0].value = 2
        self.known_optimum[0] = Trial(KOpoint, KOfunV)


    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Calculate the value of the selected function at a given point

        :param point: coordinates of the trial point where the value of the function will be calculated.
        :param function_value: object defining the function number in the task and storing the function value.
        :return: Calculated value of the function at point.
        """
        result: np.double = 0
        x = point.float_variables[0]
        b = int(point.discrete_variables[0])

        if function_value.type == FunctionType.OBJECTIV:
            result = np.double(2 * x + b)
        elif function_value.functionID == 0:  # constraint 1
            result = np.double(1.25 - x*x - b)
        elif function_value.functionID == 1:  # constraint 2
            result = np.double(x + b - 1.6)

        function_value.value = result
        return function_value
