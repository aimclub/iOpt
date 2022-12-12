import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
import iOpt.problems.shekel4_generation as shekelGen

class Shekel4(Problem):
    def __init__(self, function_number: int):
        """
        Конструктор класса Shekel problem.

        :param dimension: Размерность задачи = 4
        :param functionNumber: номер задачи в наборе, :math:`1 <= functionNumber <= 3`
        """
        self.name = Shekel4
        self.dimension = 4
        self.numberOfFloatVariables = self.dimension
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 0
        self.fn = function_number

        self.floatVariableNames = np.ndarray(shape=(self.dimension,), dtype=str)
        for i in range(self.dimension):
            self.floatVariableNames[i] = i

        self.lowerBoundOfFloatVariables = np.ndarray(shape=(self.dimension,), dtype=np.double)
        self.lowerBoundOfFloatVariables.fill(0)
        self.upperBoundOfFloatVariables = np.ndarray(shape=(self.dimension,), dtype=np.double)
        self.upperBoundOfFloatVariables.fill(10)

        self.knownOptimum = np.ndarray(shape=(1,), dtype=Trial)

        pointfv = np.ndarray(shape=(self.dimension,), dtype=np.double)
        pointfv.fill(4)
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
        res: np.double = 0
        for i in range(shekelGen.maxI[self.fn-1]):
            den: np.double = 0
            for j in range(self.dimension):
                den = den + pow((point.floatVariables[j] - shekelGen.a[i][j]), 2.0)
            res = res - 1 / (den + shekelGen.c[i])

        functionValue.value = res
        return functionValue
