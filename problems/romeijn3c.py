import numpy as np
from iOpt.trial import FunctionType
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem

a = np.array([0.5, 0.25, 1.0, 1.0 / 12.0, 2])
c = np.array([0.125, 0.25, 0.1, 0.2, 1.0 / 12.0])
p = np.array([[0, 5], [2, 5], [3, 2], [4, 4], [5, 1]])


class Romeijn3c(Problem):
    def __init__(self):
        """
        Romeijn3c problem class constructor.
        """
        super(Romeijn3c, self).__init__()
        self.name = "Romeijn3c"
        self.dimension: int = 2
        self.number_of_float_variables = self.dimension
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 1
        self.number_of_constraints = 3

        self.float_variable_names = np.ndarray(shape=(self.dimension), dtype=str)
        for i in range(self.dimension):
            self.float_variable_names[i] = i

        self.lower_bound_of_float_variables = np.ndarray(shape=(1, self.dimension), dtype=np.double)
        self.lower_bound_of_float_variables = [-3, -4]
        self.upper_bound_of_float_variables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.upper_bound_of_float_variables = [10, 7]

        self.known_optimum = np.ndarray(shape=(1), dtype=Trial)

        pointfv = [-3, -4]
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
        result: np.double = 0
        x = point.float_variables

        if function_value.type == FunctionType.OBJECTIV:
            for i in range(0, 5):
                temp = 0
                for j in range(0, self.dimension):
                    temp += pow(x[j] - p[i][j], 2)
                temp = a[i] * temp + c[i]
                result += 1 / temp
        elif function_value.functionID == 0:  # constraint 1
            result = np.double(x[0] + x[1] - 5)
        elif function_value.functionID == 1:  # constraint 2
            result = np.double(x[0] - pow(x[1], 2))
        elif function_value.functionID == 2:  # constraint 3
            result = np.double(5 * pow(x[0], 3) - 8 / 5 * pow(x[1], 2))

        function_value.value = result
        return function_value
