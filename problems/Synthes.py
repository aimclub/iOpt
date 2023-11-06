import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import FunctionType
from iOpt.trial import Trial
from iOpt.problem import Problem
import math


class Synthes(Problem):
    """

    """

    def __init__(self):
        """
        Constructor of the Synthes problem class
        """
        super(Synthes, self).__init__()
        self.name = "Synthes"
        self.dimension = 6
        self.number_of_float_variables = 3
        self.number_of_discrete_variables = 3
        self.number_of_objectives = 1
        self.number_of_constraints = 6

        self.float_variable_names = np.ndarray(shape=(self.number_of_float_variables,), dtype=object)
        for i in range(self.number_of_float_variables):
            self.float_variable_names[i] = str(i)

        self.discrete_variable_names = np.ndarray(shape=(self.number_of_discrete_variables,), dtype=object)
        for i in range(self.number_of_discrete_variables):
            self.discrete_variable_names[i] = str(i)

        self.lower_bound_of_float_variables = np.ndarray(shape=(self.number_of_float_variables,), dtype=np.double)
        self.lower_bound_of_float_variables.fill(0)
        self.upper_bound_of_float_variables = np.ndarray(shape=(self.number_of_float_variables,), dtype=np.double)
        self.upper_bound_of_float_variables.fill(3)

        self.discrete_variable_values = [[[str(i) for i in range(0, 6)]] for i in range(self.number_of_discrete_variables)]

        self.known_optimum = np.ndarray(shape=(1,), dtype=Trial)
        # UNDEFINED


    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Calculating the value of the selected function at a given point

        :param point: coordinates of the trial point where the value of the function will be calculated. 
        :param function_value: object defining the function number in the task and storing the function value.
        :return: Calculated value of the function at point.
        """
        result: np.double = 0
        x = point.float_variables
        b = [int(x) for x in point.discrete_variables]

        if function_value.type == FunctionType.OBJECTIV:
            result = np.double(5.0 * b[0] + 6.0 * b[1] + 8.0 * b[2] + 10.0 * x[0]
                               - 7.0 * x[2] - 18.0 * math.log(x[1] + 1.0)
                               - 19.2 * math.log(x[0] - x[1] + 1.0) + 10.0)
        elif function_value.functionID == 0:  # constraint 1
            result = np.double(b[0] + b[1] - 1.1)
        elif function_value.functionID == 1:  # constraint 2
            if ((x[0] - x[1] + 1.0) != 0):
                try:
                    result = np.double(-(math.log(x[1] + 1.0) + 1.2*
                                     math.log(x[0] - x[1] + 1.0) - x[2]- 2 * b[2] + 2.0))
                except ValueError:
                    print("CalculateFuncs Error!!!")
                    result = np.NaN
                    pass  # do nothing!
            else:
                result = 1
        elif function_value.functionID == 2:  # constraint 3
            if ((x[0] - x[1] + 1.0) != 0):
                try:
                    result = np.double(-(math.log(x[1] + 1.0) + 1.2*
                                     math.log(x[0] - x[1] + 1.0) - x[2]- 2 * b[2] + 2.0))
                except ValueError:
                    print("CalculateFuncs Error!!!")
                    result = np.NaN
                    pass  # do nothing!

            else:
                result = 1
        elif function_value.functionID == 3:  # constraint 4
            result = np.double(x[1] - x[0])
        elif function_value.functionID == 4:  # constraint 5
            result = np.double(x[1] - 2.0 * b[0])
        elif function_value.functionID == 5:  # constraint 6
            result = np.double(x[0] - x[1] - 2.0 * b[1])

        function_value.value = result
        return function_value
