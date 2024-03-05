import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
import problems.Hill.hill_generation as hillGen
import math


class Hill(Problem):
    """
    The Hill function is a multimodal, continuous, deterministic function, given by the formula:
       :math:`f(x)=a_{0}+\sum_{i=1}^{m}(a_{i}sin(2i\pi x)+b_{i}cos(2i\pi x))`,
       where :math:`m` is the number of maxima of the function,
       :math:`a, b` - parameters generated randomly.
       In this generator the problem is one-dimensional.
    """

    def __init__(self, function_number: int):
        """
        Constructor of the Hill problem class

        :param functionNumber: task number in the set, :math:`1 <= functionNumber <= 1000`
        """
        super(Hill, self).__init__()
        self.name = "Hill"
        self.dimension = 1
        self.number_of_float_variables = self.dimension
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        self.fn = function_number

        self.float_variable_names = np.ndarray(shape=(self.dimension), dtype=str)
        for i in range(self.dimension):
            self.float_variable_names[i] = i

        self.lower_bound_of_float_variables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.lower_bound_of_float_variables.fill(0)
        self.upper_bound_of_float_variables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.upper_bound_of_float_variables.fill(1)

        self.known_optimum = np.ndarray(shape=(1), dtype=Trial)

        pointfv = np.ndarray(shape=(self.dimension), dtype=np.double)
        pointfv[0] = hillGen.minHill[self.fn][1]
        KOpoint = Point(pointfv, [])
        KOfunV = np.ndarray(shape=(1), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0].value = hillGen.minHill[self.fn][0]
        self.known_optimum[0] = Trial(KOpoint, KOfunV)

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Calculate the value of the selected function at a given point

        :param point: coordinates of the trial point where the value of the function will be calculated.
        :param function_value: object defining the function number in the task and storing the function value.
        :return: Calculated value of the function at point.
        """
        res: np.double = 0
        for i in range(hillGen.NUM_HILL_COEFF):
            res = res + hillGen.aHill[self.fn][i] * math.sin(2 * i * math.pi * point.float_variables[0]) + \
                  hillGen.bHill[self.fn][i] * math.cos(2 * i * math.pi * point.float_variables[0])
        function_value.value = res
        return function_value
