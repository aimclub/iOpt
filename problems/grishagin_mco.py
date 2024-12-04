import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
from problems.grishagin_function.grishagin_function import GrishaginFunction


class Grishagin_mco(Problem):
    """
    Функция Гришагина задана формулой:
       :math:`f(y) = \{ (\sum_{i=1}^{7}\sum_{i=1}^{7} A_{ij}a_{ij}(x)+B_{ij}b_{ij}(x))^{2}+`
       :math:`+(\sum_{i=1}^{7}\sum_{i=1}^{7} C_{ij}a_{ij}(x)+D_{ij}b_{ij}(x))^{2}\}`,
       где :math:`a_{ij}(x) = sin(i\pi x_{1})sin(j\pi x_{2}),`
       :math:`b_{ij}(x) = cos(i\pi x_{1})cos(j\pi x_{2}),`
       коэффициенты :math:`A_{ij}, B_{ij}, C_{ij}, D_{ij}` - равномерно распределеные величины
       на отрезке :math:`[-1, 1].`
    """

    def __init__(self, count_functions: int,
                 function_numbers: np.ndarray(shape=(1), dtype=int) = None
                 ):
        """
        Конструктор класса Grishagin_mco problem.

        :param functionNumber: номер задачи в наборе, :math:`1 <= functionNumber <= 100`
        :param countFunction: количество функций - критериев
        """
        super(Grishagin_mco, self).__init__()
        self.name = "Grishagin_mco"
        self.dimension = 2
        self.number_of_float_variables = self.dimension
        self.number_of_discrete_variables = 0
        self.number_of_objectives = count_functions
        self.number_of_constraints = 0
        self.float_variable_names = np.ndarray(shape=(self.dimension,), dtype=str)
        for i in range(self.dimension):
            self.float_variable_names[i] = i

        self.lower_bound_of_float_variables = np.ndarray(shape=(self.dimension,), dtype=np.double)
        self.lower_bound_of_float_variables.fill(0)
        self.upper_bound_of_float_variables = np.ndarray(shape=(self.dimension,), dtype=np.double)
        self.upper_bound_of_float_variables.fill(1)

        self.count_functions = count_functions
        self.function_numbers = np.ndarray(shape=(self.count_functions, ), dtype=int)
        if function_numbers:
            self.function_numbers = function_numbers #сюда бы проверки всякие запихнуть
        else:
            for i in range(count_functions):
                self.function_numbers[i]=i+1 # мб добавить рандомное заполнение?

        self.functions = np.ndarray(shape=(self.count_functions,), dtype=GrishaginFunction)
        for i in range(count_functions):
            self.functions[i] = GrishaginFunction(self.function_numbers[i])
            self.functions[i].SetFunctionNumber()

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Вычисление значения выбранной функции в заданной точке.

        :param point: координаты точки испытания, в которой будет вычислено значение функции
        :param function_value: объект, определяющий номер функции в задаче и хранящий значение функции
        :return: Вычисленное значение функции в точке point
        """
        function_value.value = self.functions[function_value.functionID].Calculate(point.float_variables)

        return function_value
