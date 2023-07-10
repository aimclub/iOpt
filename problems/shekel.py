import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
import problems.Shekel.shekel_generation as shekelGen


class Shekel(Problem):
    """
    Функция Шекеля - это многомерная, мультимодальная, непрерывная, детерминированная функция, задана формулой:
       :math:`f(x) = \sum_{i=1}^{m}(c_{i}+(x-a_{i})^{2})^{-1}`,
       где :math:`m` – количество максимумов функции,
       :math:`a, c` - параметры, генерируемые случайным образом.
       В данном генераторе задача является одномерной.
    """

    def __init__(self, function_number: int):
        """
        Конструктор класса Shekel problem.

        :param functionNumber: номер задачи в наборе, :math:`1 <= functionNumber <= 1000`
        """
        super(Shekel, self).__init__()
        self.name = Shekel
        self.dimension = 1
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
        pointfv[0] = shekelGen.minShekel[self.fn][1]
        KOpoint = Point(pointfv, [])
        KOfunV = np.ndarray(shape=(1,), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0].value = shekelGen.minShekel[self.fn][0]
        self.known_optimum[0] = Trial(KOpoint, KOfunV)

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Вычисление значения выбранной функции в заданной точке.

        :param point: координаты точки испытания, в которой будет вычислено значение функции
        :param function_value: объект определяющий номер функции в задаче и хранящий значение функции
        :return: Вычисленное значение функции в точке point
        """
        res: np.double = 0
        for i in range(shekelGen.NUM_SHEKEL_COEFF):
            res = res - 1 / (
                        shekelGen.kShekel[self.fn][i] * pow(point.float_variables[0] - shekelGen.aShekel[self.fn][i], 2)
                        + shekelGen.cShekel[self.fn][i])

        function_value.value = res
        return function_value
