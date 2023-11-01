import numpy as np
from iOpt.trial import FunctionType
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
import math


class Romeijn5c(Problem):
    def __init__(self):
        """
        Romeijn5c problem class constructor.
        """
        super(Romeijn5c, self).__init__()
        self.name = "Romeijn5c"
        self.dimension: int = 2
        self.number_of_float_variables = self.dimension
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 1
        self.number_of_constraints = 2

        self.float_variable_names = np.ndarray(shape=(self.dimension), dtype=str)
        for i in range(self.dimension):
            self.float_variable_names[i] = i

        self.lower_bound_of_float_variables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.lower_bound_of_float_variables = [-1.5, 0]
        self.upper_bound_of_float_variables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.upper_bound_of_float_variables = [3.5, 15]

        self.known_optimum = np.ndarray(shape=(1), dtype=Trial)

        pointfv = [2.4656, 15]
        KOpoint = Point(pointfv, [])
        KOfunV = np.ndarray(shape=(1), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0] = self.calculate(KOpoint, KOfunV[0])  # -195.37
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

        if function_value.type == FunctionType.OBJECTIV:
            temp = x[1] - 1.275 * pow(x[0], 2) + 5.0 * x[0] - 6.0
            result = np.double(-pow(temp, 2) - 10.0 * (1 - 1 / (8.0 * math.pi)) * math.cos(math.pi * x[0]) - 10.0)
        elif function_value.functionID == 0:  # constraint 1
            result = -math.pi * x[0] - x[1]
        elif function_value.functionID == 1:  # constraint 2
            result = -pow(math.pi * x[0], 2) + 4.0 * x[1]

        function_value.value = result
        return function_value
