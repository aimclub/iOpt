import numpy as np
from iOpt.trial import FunctionType
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
import math


class Stronginc3(Problem):
    def __init__(self):
        """
        Stronginc3 problem class constructor
        """
        super(Stronginc3, self).__init__()
        self.name = "Stronginc3"
        self.dimension: int = 2
        self.number_of_float_variables = self.dimension
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 1
        self.number_of_constraints = 3

        self.float_variable_names = np.ndarray(shape=(self.dimension), dtype=str)
        for i in range(self.dimension):
            self.float_variable_names[i] = i

        self.lower_bound_of_float_variables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.lower_bound_of_float_variables = [0, -1]
        self.upper_bound_of_float_variables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.upper_bound_of_float_variables = [4, 3]

        self.known_optimum = np.ndarray(shape=(1), dtype=Trial)

        pointfv = np.ndarray(shape=(self.dimension), dtype=np.double)
        pointfv = [0.941176, 0.941176]
        KOpoint = Point(pointfv, [])
        KOfunV = np.ndarray(shape=(1), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0].value = -1.489444
        self.known_optimum[0] = Trial(KOpoint, KOfunV)

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Calculating the value of the selected function at a given point

        :param point: coordinates of the trial point where the value of the function will be calculated. 
        :param function_value: object defining the function number in the task and storing the function value.
        :return: Calculated value of the function at point.
        """
        res: np.double = 0
        x: np.double = point.float_variables

        if function_value.type == FunctionType.OBJECTIV:
            t1: np.double = pow(0.5 * x[0] - 0.5, 4.0)
            t2: np.double = pow(x[1] - 1.0, 4.0)
            res = np.double(1.5 * x[0] * x[0] * math.exp(1.0 - x[0] * x[0] - 20.25 * (x[0] - x[1]) * (x[0] - x[1])))
            res = np.double(res + t1 * t2 * math.exp(2.0 - t1 - t2))
            res = np.double(-res)
        elif function_value.functionID == 0:  # constraint 1
            res = np.double(0.01 * ((x[0] - 2.2) * (x[0] - 2.2) + (x[1] - 1.2) * (x[1] - 1.2) - 2.25))
        elif function_value.functionID == 1:  # constraint 2
            res = np.double(100.0 * (1.0 - ((x[0] - 2.0) / 1.2) * ((x[0] - 2.0) / 1.2) - (x[1] / 2.0) * (x[1] / 2.0)))
        elif function_value.functionID == 2:  # constraint 3
            res = np.double(10.0 * (x[1] - 1.5 - 1.5 * math.sin(6.283 * (x[0] - 1.75))))

        function_value.value = res
        return function_value

    def get_name(self):
        return self.name
