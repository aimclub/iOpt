import numpy as np
from iOpt.trial import FunctionType
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
import math


class g2c(Problem):
    def __init__(self):
        """
        Constructor of the gC2 problem class
        """
        super(g2c, self).__init__()
        self.name = "g2c"
        self.dimension: int = 20
        self.number_of_float_variables = self.dimension
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 1
        self.number_of_constraints = 2

        self.float_variable_names = np.ndarray(shape=(self.dimension), dtype=str)
        for i in range(self.dimension):
            self.float_variable_names[i] = i

        self.lower_bound_of_float_variables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.lower_bound_of_float_variables.fill(0)
        self.upper_bound_of_float_variables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.upper_bound_of_float_variables.fill(10)

        self.known_optimum = np.ndarray(shape=(1), dtype=Trial)

        # Optimum is UNDEFINED

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Calculate the value of the selected function at a given point

        :param point: coordinates of the trial point where the value of the function will be calculated.
        :param function_value: object defining the function number in the task and storing the function value.
        :return: Calculated value of the function at the point.
        """
        result: np.double = 0
        x = point.float_variables
        sum1 = 0
        sum2 = 0
        prod = 1

        if function_value.type == FunctionType.OBJECTIV:
            for i in range(0, self.dimension):
                sum1 += pow(math.cos(x[i]), 4)
                sum2 += (i + 1) * pow(x[i], 2)
                prod = prod * pow(x[i], 2)
            result = - abs((sum1 - 2 * prod) / math.sqrt(sum2))
        elif function_value.functionID == 0:  # constraint 1
            for i in range(0, self.dimension):
                prod = prod * x[i]
            result = np.double(-prod + 0.75)
        elif function_value.functionID == 1:  # constraint 2
            for i in range(0, self.dimension):
                sum1 += x[i]
            result = np.double(sum1 - 7.5*self.dimension)

        function_value.value = result
        return function_value
