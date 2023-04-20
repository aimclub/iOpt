import numpy as np
from iOpt.trial import FunctionType
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
import math


class g8c(Problem):
    def __init__(self):
        """
        Конструктор класса gC8 problem.
        """
        super(g8c, self).__init__()
        self.name = "g8c"
        self.dimension: int = 2
        self.numberOfFloatVariables = self.dimension
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 2

        self.floatVariableNames = np.ndarray(shape=(self.dimension), dtype=str)
        for i in range(self.dimension):
            self.floatVariableNames[i] = i

        self.lowerBoundOfFloatVariables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.lowerBoundOfFloatVariables.fill(0)
        self.upperBoundOfFloatVariables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.upperBoundOfFloatVariables.fill(10)

        self.knownOptimum = np.ndarray(shape=(1), dtype=Trial)

        pointfv = [1.2279713, 4.2453733]
        KOpoint = Point(pointfv, [])
        KOfunV = np.ndarray(shape=(1), dtype=FunctionValue)
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
        result: np.double = 0
        x = point.floatVariables

        if functionValue.type == FunctionType.OBJECTIV:
            result = np.double(-pow(math.sin(2 * math.pi*x[0]), 3)*math.sin(2 * math.pi*x[1]) / (pow(x[0], 3)*(x[0] + x[1])))
        elif functionValue.functionID == 0:  # constraint 1
            result = np.double(pow(x[0], 2) - x[1] + 1)
        elif functionValue.functionID == 1:  # constraint 2
            result = np.double(1 - x[0] + pow(x[1] - 4, 4))

        functionValue.value = result
        return functionValue
