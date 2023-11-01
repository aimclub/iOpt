import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem


class XSquared(Problem):
    """
    Criterion function :math:`f(x) = x^2`
    """

    def __init__(self, dimension: int):
        super(XSquared, self).__init__()
        self.name = "XSquared"
        self.dimension = dimension
        self.number_of_float_variables = dimension
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 1
        self.number_of_constraints = 0

        self.float_variable_names = np.ndarray(shape=(self.dimension), dtype=str)
        for i in range(self.dimension):
            self.float_variable_names[i] = i

        self.lower_bound_of_float_variables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.lower_bound_of_float_variables.fill(-1)
        self.upper_bound_of_float_variables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.upper_bound_of_float_variables.fill(1)

        self.known_optimum = np.ndarray(shape=(1), dtype=Trial)

        pointfv = np.ndarray(shape=(self.dimension), dtype=np.double)
        pointfv.fill(0)
        KOpoint = Point(pointfv, [])
        KOfunV = np.ndarray(shape=(1), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0].value = 0
        self.known_optimum[0] = Trial(KOpoint, KOfunV)

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Calculation of the criterion value

        :param point: coordinates of the trial point where the value of the function will be calculated. 
        :param function_value: object defining the function number in the task and storing the function value.
        :return: Calculated value of the function at point.
        """
        sum: np.double = 0
        for i in range(self.dimension):
            sum += point.float_variables[i] * point.float_variables[i]

        function_value.value = sum
        return function_value

    def get_name(self):
        return self.name
