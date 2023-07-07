import numpy as np
from iOpt.trial import FunctionType
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
import math


class g8c(Problem):
    def __init__(self):
        """
        Конструктор класса gC8 problem.
        """
        super(g8c, self).__init__()
        self.name = "g8c"
        self.dimension: int = 2
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

        pointfv = [1.2279713, 4.2453733]
        KOpoint = Point(pointfv, [])
        KOfunV = np.ndarray(shape=(1), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0] = self.calculate(KOpoint, KOfunV[0])
        self.known_optimum[0] = Trial(KOpoint, KOfunV)

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Вычисление значения выбранной функции в заданной точке.

        :param point: координаты точки испытания, в которой будет вычислено значение функции
        :param function_value: объект определяющий номер функции в задаче и хранящий значение функции
        :return: Вычисленное значение функции в точке point
        """
        result: np.double = 0
        x = point.float_variables

        if function_value.type == FunctionType.OBJECTIV:
            result = np.double(-pow(math.sin(2 * math.pi*x[0]), 3)*math.sin(2 * math.pi*x[1]) / (pow(x[0], 3)*(x[0] + x[1])))
        elif function_value.functionID == 0:  # constraint 1
            result = np.double(pow(x[0], 2) - x[1] + 1)
        elif function_value.functionID == 1:  # constraint 2
            result = np.double(1 - x[0] + pow(x[1] - 4, 4))

        function_value.value = result
        return function_value
