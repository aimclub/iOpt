import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
import iOpt.problems.Hill.hill_generation as hillGen
import math


class Hill(Problem):
    """
    Функция Хилла - это мультимодальная, непрерывная, детерминированная функция, задана формулой:
       :math:`f(x)=a_{0}+\sum_{i=1}^{m}(a_{i}sin(2i\pi x)+b_{i}cos(2i\pi x))`,
       где :math:`m` – количество максимумов функции,
       :math:`a, b` - параметры, генерируемые случайным образом.
       В данном генераторе задача является одномерной.
    """

    def __init__(self, function_number: int):
        """
        Конструктор класса Hill problem.

        :param functionNumber: номер задачи в наборе, :math:`1 <= functionNumber <= 1000`
        """
        super(Hill, self).__init__()
        self.name = Hill
        self.dimension = 1
        self.numberOfFloatVariables = self.dimension
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 0
        self.fn = function_number

        self.floatVariableNames = np.ndarray(shape=(self.dimension), dtype=str)
        for i in range(self.dimension):
            self.floatVariableNames[i] = i

        self.lowerBoundOfFloatVariables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.lowerBoundOfFloatVariables.fill(0)
        self.upperBoundOfFloatVariables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.upperBoundOfFloatVariables.fill(1)

        self.knownOptimum = np.ndarray(shape=(1), dtype=Trial)

        pointfv = np.ndarray(shape=(self.dimension), dtype=np.double)
        pointfv[0] = hillGen.minHill[self.fn][1]
        KOpoint = Point(pointfv, [])
        KOfunV = np.ndarray(shape=(1), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0].value = hillGen.minHill[self.fn][0]
        self.knownOptimum[0] = Trial(KOpoint, KOfunV)

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        """
        Вычисление значения выбранной функции в заданной точке.

        :param point: координаты точки испытания, в которой будет вычислено значение функции
        :param functionValue: объект определяющий номер функции в задаче и хранящий значение функции
        :return: Вычисленное значение функции в точке point
        """
        res: np.double = 0
        for i in range(hillGen.NUM_HILL_COEFF):
            res = res + hillGen.aHill[self.fn][i] * math.sin(2 * i * math.pi * point.floatVariables[0]) + \
                  hillGen.bHill[self.fn][i] * math.cos(2 * i * math.pi * point.floatVariables[0])
        functionValue.value = res
        return functionValue
