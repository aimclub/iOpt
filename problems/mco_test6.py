import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import FunctionType
from iOpt.trial import Trial
from iOpt.problem import Problem
import math


class mco_test6(Problem):
    """

    """

    def __init__(self):
        """
        Конструктор класса mco_test1 problem.
        """
        super(mco_test6, self).__init__()
        self.name = "mco_test6"
        self.dimension = 2
        self.number_of_float_variables = 2
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.float_variable_names = np.ndarray(shape=(self.number_of_float_variables,), dtype=object)
        for i in range(self.number_of_float_variables):
            self.float_variable_names[i] = str(i)

        self.lower_bound_of_float_variables = np.ndarray(shape=(self.number_of_float_variables,), dtype=np.double)
        self.lower_bound_of_float_variables.fill(0)
        self.upper_bound_of_float_variables = np.ndarray(shape=(self.number_of_float_variables,), dtype=np.double)
        self.upper_bound_of_float_variables.fill(1)

        self.known_optimum = np.ndarray(shape=(1,), dtype=Trial)

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Вычисление значения выбранной функции в заданной точке.

        :param point: координаты точки испытания, в которой будет вычислено значение функции
        :param function_value: объект, определяющий номер функции в задаче и хранящий значение функции
        :return: Вычисленное значение функции в точке point
        """
        result: np.double = 0
        x = point.float_variables

        if function_value.functionID == 0:  # OBJECTIV 1
            result = np.double(x[0])
        elif function_value.functionID == 1:  # OBJECTIV 2
            result = np.double((1 - (x[0] / (1 + 10 * x[1])) * (x[0] / (1 + 10 * x[1])) -
                                (x[0] / (1 + 10 * x[1])) * math.sin(8 * x[0] * math.pi)) * (1 + 10 * x[1]))

        function_value.value = result
        return function_value
