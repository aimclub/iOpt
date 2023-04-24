import numpy as np
from iOpt.trial import FunctionType
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
import math


class Stronginc2(Problem):
    def __init__(self):
        """
        Конструктор класса Stronginc2 problem.
        """
        super(Stronginc2, self).__init__()
        self.name = 'Stronginc2'
        self.dimension: int = 2
        self.numberOfFloatVariables = self.dimension
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 2

        self.floatVariableNames = np.ndarray(shape=(self.dimension), dtype=str)
        for i in range(self.dimension):
            self.floatVariableNames[i] = i

        self.lowerBoundOfFloatVariables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.lowerBoundOfFloatVariables = [0, -1]
        self.upperBoundOfFloatVariables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.upperBoundOfFloatVariables = [4, 3]

        self.knownOptimum = np.ndarray(shape=(1), dtype=Trial)

        pointfv = np.ndarray(shape=(self.dimension), dtype=np.double)
        pointfv.fill(1.088)
        KOpoint = Point(pointfv, [])
        KOfunV = np.ndarray(shape=(1), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0] = self.Calculate(KOpoint, KOfunV[0])  # -1.477
        self.knownOptimum[0] = Trial(KOpoint, KOfunV)

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        """
        Вычисление значения выбранной функции в заданной точке.

        :param point: координаты точки испытания, в которой будет вычислено значение функции
        :param functionValue: объект определяющий номер функции в задаче и хранящий значение функции
        :return: Вычисленное значение функции в точке point
        """
        res: np.double = 0
        x: np.double = point.floatVariables

        if functionValue.type == FunctionType.OBJECTIV:
            t1: np.double = pow(0.5 * x[0] - 0.5, 4.0)
            t2: np.double = pow(x[1] - 1.0, 4.0)
            res = np.double(1.5 * x[0] * x[0] * math.exp(1.0 - x[0] * x[0] - 20.25 * (x[0] - x[1]) * (x[0] - x[1])))
            res = np.double(res + t1 * t2 * math.exp(2.0 - t1 - t2))
            res = np.double(-res)
        elif functionValue.functionID == 0:  # constraint 1
            res = np.double(((x[0]- 2.2) * (x[0] - 2.2) + (x[1] - 1.2) * (x[1] - 1.2) - 1.25))
        elif functionValue.functionID == 1:  # constraint 2
            res = np.double(1.21 - (x[0] - 2.2) * (x[0] - 2.2) - (x[1] - 1.2) * (x[1] - 1.2))

        functionValue.value = res
        return functionValue

