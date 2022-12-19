import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
from iOpt.problems.grishagin_function.grishagin_function import GrishaginFunction
import math


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
        self.name = Grishagin
        self.dimension = 2
        self.numberOfFloatVariables = self.dimension
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 0
        self.floatVariableNames = np.ndarray(shape=(self.dimension,), dtype=str)
        for i in range(self.dimension):
            self.floatVariableNames[i] = i

        self.lowerBoundOfFloatVariables = np.ndarray(shape=(self.dimension,), dtype=np.double)
        self.lowerBoundOfFloatVariables.fill(0)
        self.upperBoundOfFloatVariables = np.ndarray(shape=(self.dimension,), dtype=np.double)
        self.upperBoundOfFloatVariables.fill(1)

        self.functionNumber = function_number
        self.function: GrishaginFunction = GrishaginFunction(self.functionNumber)
        self.function.SetFunctionNumber()

        self.knownOptimum = np.ndarray(shape=(1,), dtype=Trial)
        pointfv = self.function.GetOptimumPoint()
        KOpoint = Point(pointfv, [])
        KOfunV = np.ndarray(shape=(1,), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0] = self.Calculate(KOpoint, KOfunV[0])
        self.knownOptimum[0] = Trial(KOpoint, KOfunV)

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        """
        Вычисление значения выбранной функции в заданной точке.

        :param point: координаты точки испытания, в которой будет вычислено значение функции
        :param functionValue: объект определяющий номер функции в задаче и хранящий значение функции
        :return: Вычисленное значение функции в точке point
        """
        functionValue.value = self.function.Calculate(point.floatVariables)

        return functionValue
