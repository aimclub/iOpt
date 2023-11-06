import numpy as np
from iOpt.trial import FunctionType
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem


class Romeijn2c(Problem):
    def __init__(self):
        """
        Romeijn2c problem class constructor.
        """
        super(Romeijn2c, self).__init__()
        self.name = "Romeijn2c"
        self.dimension: int = 6
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
        self.upper_bound_of_float_variables = [10, 10, 15, 15, 1, 1]

        self.known_optimum = np.ndarray(shape=(1), dtype=Trial)

        pointfv = [10, 10, 15, 4.609, 0.78511, 0.384]
        KOpoint = Point(pointfv, [])
        KOfunV = np.ndarray(shape=(1), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0] = self.calculate(KOpoint, KOfunV[0])
        self.known_optimum[0] = Trial(KOpoint, KOfunV)

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Calculate the value of the selected function at a given point

        :param point: coordinates of the trial point where the value of the function will be calculated.
        :param function_value: object defining the function number in the task and storing the function value.
        :return: Calculated value of the function at point.
        """
        result: np.double = 1
        x = point.float_variables

        if function_value.type == FunctionType.OBJECTIV:
            result = np.double(-(0.0204 + 0.0607 * pow(x[4], 2)) * x[0] * x[3] * (x[0] + x[1] + x[2]) -
                               (0.0187 + 0.0437 * pow(x[5], 2)) * x[1] * x[2] * (x[0] + 1.57 * x[1] + x[3]))
        elif function_value.functionID == 0:  # constraint 1
            for i in range(0, self.dimension):
                result = result / x[i]
            result = np.double(2070 * result - 1)

        elif function_value.functionID == 1:  # constraint 2
            result = np.double(0.00062 * x[0] * x[3] * pow(x[4], 2) * (x[0] + x[1] + x[2]) +
                               0.00058 * x[1] * x[2] * pow(x[5], 2) * (x[0] + 1.57 * x[1] + x[3]) - 1)

        function_value.value = result
        return function_value
