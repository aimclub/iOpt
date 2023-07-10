import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
from problems.grishagin_function.grishagin_function import GrishaginFunction


class Grishagin(Problem):
    """
    Функция Гришагина задана формулой:
       :math:`f(y) = \{ (\sum_{i=1}^{7}\sum_{i=1}^{7} A_{ij}a_{ij}(x)+B_{ij}b_{ij}(x))^{2}+`
       :math:`+(\sum_{i=1}^{7}\sum_{i=1}^{7} C_{ij}a_{ij}(x)+D_{ij}b_{ij}(x))^{2}\}`,
       где :math:`a_{ij}(x) = sin(i\pi x_{1})sin(j\pi x_{2}),`
       :math:`b_{ij}(x) = cos(i\pi x_{1})cos(j\pi x_{2}),`
       коэффициенты :math:`A_{ij}, B_{ij}, C_{ij}, D_{ij}` - равномерно распределеные величины
       на отрезке :math:`[-1, 1].`
    """

    def __init__(self, function_number: int):
        """
        Конструктор класса Grishagin problem.

        :param functionNumber: номер задачи в наборе, :math:`1 <= functionNumber <= 100`
        """
        super(Grishagin, self).__init__()
        self.name = "Grishagin"
        self.dimension = 2
        self.number_of_float_variables = self.dimension
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        self.float_variable_names = np.ndarray(shape=(self.dimension,), dtype=str)
        for i in range(self.dimension):
            self.float_variable_names[i] = i

        self.lower_bound_of_float_variables = np.ndarray(shape=(self.dimension,), dtype=np.double)
        self.lower_bound_of_float_variables.fill(0)
        self.upper_bound_of_float_variables = np.ndarray(shape=(self.dimension,), dtype=np.double)
        self.upper_bound_of_float_variables.fill(1)

        self.functionNumber = function_number
        self.function: GrishaginFunction = GrishaginFunction(self.functionNumber)
        self.function.SetFunctionNumber()

        self.known_optimum = np.ndarray(shape=(1,), dtype=Trial)
        pointfv = self.function.GetOptimumPoint()
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
        function_value.value = self.function.Calculate(point.float_variables)

        return function_value
