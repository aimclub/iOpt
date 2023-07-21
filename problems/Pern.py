import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import FunctionType
from iOpt.trial import Trial
from iOpt.problem import Problem
import math


class Pern(Problem):
    """

    """

    def __init__(self):
        """
        Конструктор класса Pern problem.
        """
        super(Pern, self).__init__()
        self.name = "Pern"
        self.dimension = 2
        self.number_of_float_variables = 1
        self.number_of_discrete_variables = 1
        self.number_of_objectives = 1
        self.number_of_constraints = 3

        self.float_variable_names = np.ndarray(shape=(self.number_of_float_variables,), dtype=object)
        for i in range(self.number_of_float_variables):
            self.float_variable_names[i] = str(i)

        self.discrete_variable_names = np.ndarray(shape=(self.number_of_discrete_variables,), dtype=object)
        for i in range(self.number_of_discrete_variables):
            self.discrete_variable_names[i] = str(i)

        self.lower_bound_of_float_variables = np.ndarray(shape=(self.number_of_float_variables,), dtype=np.double)
        self.lower_bound_of_float_variables[0] = 1
        self.upper_bound_of_float_variables = np.ndarray(shape=(self.number_of_float_variables,), dtype=np.double)
        self.upper_bound_of_float_variables[0] = 10

        self.discrete_variable_values = [[[str(i) for i in range(1, 7)]] for i in range(self.number_of_discrete_variables)]

        self.known_optimum = np.ndarray(shape=(1,), dtype=Trial)

        pointfv = np.ndarray(shape=(self.number_of_float_variables,), dtype=np.double)
        pointfv[0] = 4

        pointdv = np.ndarray(shape=(self.number_of_discrete_variables,), dtype=object)
        pointdv[0] = "1"

        KOpoint = Point(pointfv, pointdv)
        KOfunV = np.ndarray(shape=(1,), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0].value = -17
        self.known_optimum[0] = Trial(KOpoint, KOfunV)


    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Вычисление значения выбранной функции в заданной точке.

        :param point: координаты точки испытания, в которой будет вычислено значение функции
        :param function_value: объект определяющий номер функции в задаче и хранящий значение функции
        :return: Вычисленное значение функции в точке point
        """
        result: np.double = 0
        x = point.float_variables[0]
        b = int(point.discrete_variables[0])

        if function_value.type == FunctionType.OBJECTIV:
            result = np.double(3.0 * b - 5.0 * x)
        elif function_value.functionID == 0:  # constraint 1
            result = np.double(2.0 * b * b - 2.0 * math.sqrt(b) - 2.0 * math.sqrt(x) * b * b
                               + 11.0 * b + 8 * x - 39.0)
        elif function_value.functionID == 1:  # constraint 2
            result = np.double(-b + x - 3.0)
        elif function_value.functionID == 2:  # constraint 3
            result = np.double(2.0 * b + 3 * x - 24.0)

        function_value.value = result
        return function_value

if __name__ == "__main__":
    Problem = Pern()
    point = Problem.known_optimum[0].point
    print(point.float_variables, point.discrete_variables)
    fv = FunctionValue()
    fv = Problem.calculate(point, fv)
    print(fv.value)
    print(abs(fv.value - Problem.known_optimum[0].function_values[0].value))
