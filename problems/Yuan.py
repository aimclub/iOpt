import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import FunctionType
from iOpt.trial import Trial
from iOpt.problem import Problem
import math


class Yuan(Problem):
    """

    """

    def __init__(self):
        """
        Constructor of the Yuan problem class.
        """
        super(Yuan, self).__init__()
        self.name = "Yuan"
        self.dimension = 7
        self.number_of_float_variables = 3
        self.number_of_discrete_variables = 4
        self.number_of_objectives = 1
        self.number_of_constraints = 9

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

        self.discrete_variable_values = [["0", "1"] for i in range(self.number_of_discrete_variables)]

        self.known_optimum = np.ndarray(shape=(1,), dtype=Trial)

        pointfv = np.ndarray(shape=(self.number_of_float_variables,), dtype=np.double)
        pointfv = [0.2, 0.8, 1.908]

        pointdv = np.ndarray(shape=(self.number_of_discrete_variables,), dtype=object)
        pointdv = ["1", "1", "0", "1"]

        KOpoint = Point(pointfv, pointdv)
        KOfunV = np.ndarray(shape=(1,), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0].value = 4.5796
        self.known_optimum[0] = Trial(KOpoint, KOfunV)


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
            result = np.double((b[0] - 1.0) * (b[0] - 1.0) + (b[1] - 2.0) * (b[1] - 2.0) +
                               (b[2] - 1.0) * (b[2] - 1.0) - math.log(b[3] + 1.0) +
                               (x[0] - 1.0) * (x[0] - 1.0) + (x[1] - 2.0) * (x[1] - 2.0) +
                               (x[2] - 3.0) * (x[2] - 3.0))
        elif function_value.functionID == 0:  # constraint 1
            result = np.double(b[0] + b[1] + b[2] + x[0] + x[1] + x[2] - 5.0)
        elif function_value.functionID == 1:  # constraint 2
            result = np.double(b[2] * b[2] + x[0] * x[0] + x[1] * x[1] + x[2] * x[2] - 5.5)
        elif function_value.functionID == 2:  # constraint 3
            result = np.double(b[0] + x[0] - 1.2)
        elif function_value.functionID == 3:  # constraint 4
            result = np.double(b[1] + x[1] - 1.8)
        elif function_value.functionID == 4:  # constraint 5
            result = np.double(b[2] + x[2] - 2.5)
        elif function_value.functionID == 5:  # constraint 6
            result = np.double(b[3] + x[0] - 1.2)
        elif function_value.functionID == 6:  # constraint 7
            result = np.double(b[1] * b[1] + x[1] * x[1] - 1.64)
        elif function_value.functionID == 7:  # constraint 8
            result = np.double(b[2] * b[2] + x[2] * x[2] - 4.25)
        elif function_value.functionID == 8:  # constraint 9
            result = np.double(b[1] * b[1] + x[2] * x[2] - 4.64)

        function_value.value = result
        return function_value
