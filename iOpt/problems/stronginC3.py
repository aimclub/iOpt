import numpy as np
from iOpt.trial import FunctionType
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
import math


class StronginC3(Problem):
    def __init__(self):
        """
        Конструктор класса StronginC3 problem.
        """
        super(StronginC3, self).__init__()
        self.name = StronginC3
        self.dimension: int = 2
        self.numberOfFloatVariables = self.dimension
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 3

        self.floatVariableNames = np.ndarray(shape=(self.dimension), dtype=str)
        for i in range(self.dimension):
            self.floatVariableNames[i] = i

        self.lowerBoundOfFloatVariables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.lowerBoundOfFloatVariables[0] = 0
        self.lowerBoundOfFloatVariables[1] = -1
        self.upperBoundOfFloatVariables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.upperBoundOfFloatVariables[0] = 4
        self.upperBoundOfFloatVariables[1] = 3

        self.knownOptimum = np.ndarray(shape=(1), dtype=Trial)

        pointfv = np.ndarray(shape=(self.dimension), dtype=np.double)
        pointfv[0] = 0.941176
        pointfv[1] = 0.941176
        KOpoint = Point(pointfv, [])
        KOfunV = np.ndarray(shape=(1), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0].value = -1.489444
        self.knownOptimum[0] = Trial(KOpoint, KOfunV)

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        """
        Вычисление значения выбранной функции в заданной точке.

        :param point: координаты точки испытания, в которой будет вычислено значение функции
        :param functionValue: объект определяющий номер функции в задаче и хранящий значение функции
        :return: Вычисленное значение функции в точке point
        """
        res: np.double = 0
        x1: np.double = point.floatVariables[0]
        x2: np.double = point.floatVariables[1]

        if functionValue.type == FunctionType.OBJECTIV:
            t1: np.double = pow(0.5 * x1 - 0.5, 4.0)
            t2: np.double = pow(x2 - 1.0, 4.0)
            res = np.double(1.5 * x1 * x1 * math.exp(1.0 - x1 * x1 - 20.25 * (x1 - x2) * (x1 - x2)))
            res = np.double(res + t1 * t2 * math.exp(2.0 - t1 - t2))
            res = np.double(-res)
        elif functionValue.functionID == 0:  # constraint 1
            res = np.double(0.01 * ((x1 - 2.2) * (x1 - 2.2) + (x2 - 1.2) * (x2 - 1.2) - 2.25))
        elif functionValue.functionID == 1:  # constraint 2
            res = np.double(100.0 * (1.0 - ((x1 - 2.0) / 1.2) * ((x1 - 2.0) / 1.2) - (x2 / 2.0) * (x2 / 2.0)))
        elif functionValue.functionID == 2:  # constraint 3
            res = np.double(10.0 * (x2 - 1.5 - 1.5 * math.sin(6.283 * (x1 - 1.75))))

        functionValue.value = res
        return functionValue

    def GetName(self):
        return self.name
