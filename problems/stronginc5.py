import numpy as np
from iOpt.trial import FunctionType
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
import math


class Stronginc5(Problem):
    def __init__(self):
        """
        Stronginc5 problem class constructor.
        """
        super(Stronginc5, self).__init__()
        self.name = "Stronginc5"
        self.dimension: int = 5
        self.number_of_float_variables = self.dimension
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 1
        self.number_of_constraints = 5

        self.float_variable_names = np.ndarray(shape=(self.dimension), dtype=str)
        for i in range(self.dimension):
            self.float_variable_names[i] = i

        self.lower_bound_of_float_variables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.lower_bound_of_float_variables = [-3, -3, -3, -10, -10]
        self.upper_bound_of_float_variables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.upper_bound_of_float_variables = [3, 3, 3, 10, 10]

        self.known_optimum = np.ndarray(shape=(1), dtype=Trial)

        pointfv = [-0.0679, 1.9434, 2.4512, 9.9013, 9.9008]
        KOpoint = Point(pointfv, [])
        KOfunV = np.ndarray(shape=(1), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0] = self.calculate(KOpoint, KOfunV[0])  # -43.298677;
        self.known_optimum[0] = Trial(KOpoint, KOfunV)

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Calculating the value of the selected function at a given point

        :param point: coordinates of the trial point where the value of the function will be calculated. 
        :param function_value: object defining the function number in the task and storing the function value.
        :return: Calculated value of the function at point.
        """
        res: np.double = 0
        x = point.float_variables

        if function_value.type == FunctionType.OBJECTIV:
            res = np.double(math.sin(x[0] * x[2]) - (x[1] * x[4] + x[2] * x[3])*math.cos(x[0] * x[1]))
        elif function_value.functionID == 0:  # constraint 1
            res = np.double(-(x[0] + x[1] + x[2] + x[3] + x[4]))
        elif function_value.functionID == 1:  # constraint 2
            res = np.double(x[1] * x[1] / 9 + x[3] * x[3] / 100 - 1.4)
        elif function_value.functionID == 2:  # constraint 3
            res = np.double(3 - pow(x[0] + 1, 2) - pow(x[1] + 2, 2) - pow(x[2] - 2, 2) - pow(x[4] + 5, 2))
        elif function_value.functionID == 3:  # constraint 4
            res = np.double(4 * x[0] * x[0] * math.sin(x[0]) + x[1] * x[1] * math.cos(x[1] + x[3]) +
                            x[2] * x[2] * (math.sin(x[2] + x[4]) + math.sin(10 * (x[2] - x[3]) / 3)) - 4)
        elif function_value.functionID == 4:  # constraint 5
            res = np.double(x[0] * x[0] + x[1] * x[1] * pow(math.sin((x[0] + x[3]) / 3 + 6.6) +
                                                            math.sin((x[1] + x[4]) / 2 + 0.9), 2) - 17 *
                            pow(math.cos(x[0] + x[2] + 1), 2) + 16)

        function_value.value = res
        return function_value
