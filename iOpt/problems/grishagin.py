import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
from iOpt.problems.grishagin_function import GrishaginFunction
import math

class Grishagin(Problem):
    """Base class for optimization problems"""

    def __init__(self, function_number: int):
        """
        Конструктор класса Grishagin.
        :param dimension: Размерность задачи = 2
        :param functionNumber: номер задачи в наборе, :math:`1 <= functionNumber <= 100`
        """
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
        self.function:  GrishaginFunction = GrishaginFunction(self.functionNumber)
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
        Compute selected function at given point.
        :param point: координаты точки испытания, в которой будет вычислено значение функции
        :param functionValue: объект определяющий номер функции в задаче и хранящий значение функции
        :return: Вычисленное значение функции в точке point
        """
        functionValue.value = self.function.Calculate(point.floatVariables)

        return functionValue
