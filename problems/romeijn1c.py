import numpy as np
from iOpt.trial import FunctionType
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem


class Romeijn1c(Problem):
    def __init__(self):
        """
        RomeijnC1 problem class constructor.
        """
        super(Romeijn1c, self).__init__()
        self.name = "Romeijn1c"
        self.dimension: int = 3
        self.number_of_float_variables = self.dimension
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 1
        self.number_of_constraints = 1

        self.float_variable_names = np.ndarray(shape=(self.dimension), dtype=str)
        for i in range(self.dimension):
            self.float_variable_names[i] = i

        self.lower_bound_of_float_variables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.lower_bound_of_float_variables.fill(0)
        self.upper_bound_of_float_variables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.upper_bound_of_float_variables = [0.18745, 0.16230, 0.42846]

        self.known_optimum = np.ndarray(shape=(1), dtype=Trial)

        # Optimum is UNDEFINED

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
            result = -1 / (pow(x[0], 2) * x[1] * x[2])
        elif function_value.functionID == 0:  # constraint 1
            result = 0.44098 * x[0] + 28.46 * pow(x[0], 2) + 6158.4 * pow(x[0], 2) * x[1] + 0.0037018 * x[2] + \
                     5.4474 * pow(x[2], 2) + 0.032236 * x[0] * x[2] + 2.92 * x[1] * x[2] + 0.44712 * x[1] \
                     + 37.964 * pow(x[1], 2) + 42.876 * x[0] * x[1] - 1

        function_value.value = result
        return function_value
