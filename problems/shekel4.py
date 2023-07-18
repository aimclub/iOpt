import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
import problems.Shekel4.shekel4_generation as shekelGen


class Shekel4(Problem):
    """
    Функция Шекеля - это многомерная, мультимодальная, непрерывная, детерминированная функция, задана формулой:
       :math:`f(x) = \sum_{i=1}^{m}(c_{i}+\sum_{j=1}^{n}(x-a_{i})^{2})^{-1}`,
       где :math:`m` – количество максимумов функции,
       :math:`a, c` - параметры, генерируемые случайным образом.
       В генераторе размерность задачи равна 4.
    """

    def __init__(self, function_number: int):
        """
        Конструктор класса Shekel problem.

        :param functionNumber: номер задачи в наборе, :math:`1 <= functionNumber <= 3`
        """
        super(Shekel4, self).__init__()
        self.name = "Shekel4"
        self.dimension = 4
        self.number_of_float_variables = self.dimension
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        self.fn = function_number

        self.float_variable_names = np.ndarray(shape=(self.dimension,), dtype=str)
        for i in range(self.dimension):
            self.float_variable_names[i] = i

        self.lower_bound_of_float_variables = np.ndarray(shape=(self.dimension,), dtype=np.double)
        self.lower_bound_of_float_variables.fill(0)
        self.upper_bound_of_float_variables = np.ndarray(shape=(self.dimension,), dtype=np.double)
        self.upper_bound_of_float_variables.fill(10)

        self.known_optimum = np.ndarray(shape=(1,), dtype=Trial)

        pointfv = np.ndarray(shape=(self.dimension,), dtype=np.double)
        pointfv.fill(4)
        KOpoint = Point(pointfv, [])
        KOfunV = np.ndarray(shape=(1,), dtype=FunctionValue)
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
        res: np.double = 0
        for i in range(shekelGen.maxI[self.fn - 1]):
            den: np.double = 0
            for j in range(self.dimension):
                den = den + pow((point.float_variables[j] - shekelGen.a[i][j]), 2.0)
            res = res - 1 / (den + shekelGen.c[i])

        function_value.value = res
        return function_value
