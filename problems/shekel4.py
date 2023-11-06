import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
import problems.Shekel4.shekel4_generation as shekelGen


class Shekel4(Problem):
    """
    The Scheckel function is a multivariate, multimodal, continuous, deterministic function, given by the formula:
       :math:`f(x) = \sum_{i=1}^{m}(c_{i}+\sum_{j=1}^{n}(x-a_{i})^{2})^{-1}`,
       where :math:`m` – number of maxima of the function,
       :math:`a, c` - parameters generated randomly.
       In the generator, the dimensionality of the problem is 4.
    """

    def __init__(self, function_number: int):
        """
        Constructor of the Shekel problem class.

        :param functionNumber: task number in the set, :math:`1 <= functionNumber <= 3`
        """
        super(Shekel4, self).__init__()
        self.name = "Shekel4"
        self.dimension = 4
        self.number_of_float_variables = self.dimension
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        self.fn = function_number

        self.float_variable_names = np.ndarray(shape=(self.dimension,), dtype=str)
        for i in range(self.dimension):
            self.float_variable_names[i] = i

        self.lower_bound_of_float_variables = np.ndarray(shape=(self.dimension,), dtype=np.double)
        self.lower_bound_of_float_variables.fill(0)
        self.upper_bound_of_float_variables = np.ndarray(shape=(self.dimension,), dtype=np.double)
        self.upper_bound_of_float_variables.fill(10)

        self.known_optimum = np.ndarray(shape=(1,), dtype=Trial)

        pointfv = np.ndarray(shape=(self.dimension,), dtype=np.double)
        pointfv.fill(4)
        KOpoint = Point(pointfv, [])
        KOfunV = np.ndarray(shape=(1,), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0] = self.calculate(KOpoint, KOfunV[0])
        self.known_optimum[0] = Trial(KOpoint, KOfunV)

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Calculating the value of the selected function at a given point

        :param point: coordinates of the trial point where the value of the function will be calculated. 
        :param function_value: object defining the function number in the task and storing the function value.
        :return: Calculated value of the function at point.
        """
        res: np.double = 0
        for i in range(shekelGen.maxI[self.fn - 1]):
            den: np.double = 0
            for j in range(self.dimension):
                den = den + pow((point.float_variables[j] - shekelGen.a[i][j]), 2.0)
            res = res - 1 / (den + shekelGen.c[i])

        function_value.value = res
        return function_value
