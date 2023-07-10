import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
import math


class RastriginHiddenConstraint(Problem):
    """
    Функция Растригина со скрытими ограничениями задана формулой:
       :math:`f(y)=(\sum_{i=1}^{N}[x_{i}^{2}-10*cos(2\pi x_{i})])`,
       где :math:`x\in [-2.2, 1.8], N` – размерность задачи.
    """

    def __init__(self, dimension: int):
        """
        Конструктор класса RastriginHiddenConstraint problem.

        :param dimension: Размерность задачи.
        """
        super(RastriginHiddenConstraint, self).__init__()
        self.name = "RastriginHiddenConstraint"
        self.dimension = dimension
        self.number_of_float_variables = dimension
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 1
        self.number_of_constraints = 0

        self.float_variable_names = np.ndarray(shape=(self.dimension), dtype=str)
        for i in range(self.dimension):
            self.float_variable_names[i] = i

        self.lower_bound_of_float_variables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.lower_bound_of_float_variables.fill(-2.2)
        self.upper_bound_of_float_variables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.upper_bound_of_float_variables.fill(1.8)

        self.known_optimum = np.ndarray(shape=(1), dtype=Trial)

        pointfv = np.ndarray(shape=(self.dimension), dtype=np.double)
        pointfv.fill(0)
        KOpoint = Point(pointfv, [])
        KOfunV = np.ndarray(shape=(1), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0].value = 0
        self.known_optimum[0] = Trial(KOpoint, KOfunV)

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Вычисление значения выбранной функции в заданной точке.

        :param point: координаты точки испытания, в которой будет вычислено значение функции
        :param function_value: объект определяющий номер функции в задаче и хранящий значение функции
        :return: Вычисленное значение функции в точке point
        """

        isInf: bool = True
        for i in range(self.dimension):
            if point.float_variables[i] <= 0.5 or point.float_variables[i] > 1.5:
                isInf = False

        if isInf:
            raise Exception("Infinity values")

        sum: np.double = 0
        for i in range(self.dimension):
            sum += point.float_variables[i] * point.float_variables[i] - 10 * math.cos(
                2 * math.pi * point.float_variables[i]) + 10

        function_value.value = sum
        return function_value

    def get_name(self):
        return self.name
