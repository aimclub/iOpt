import numpy as np
from iOpt.trial import FunctionType
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
import math


class Romeijn5c(Problem):
    def __init__(self):
        """
        Конструктор класса Romeijn5c problem.
        """
        super(Romeijn5c, self).__init__()
        self.name = "Romeijn5c"
        self.dimension: int = 2
        self.numberOfFloatVariables = self.dimension
        self.numberOfDiscreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 2

        self.floatVariableNames = np.ndarray(shape=(self.dimension), dtype=str)
        for i in range(self.dimension):
            self.floatVariableNames[i] = i

        self.lowerBoundOfFloatVariables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.lowerBoundOfFloatVariables = [-1.5, 0]
        self.upperBoundOfFloatVariables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.upperBoundOfFloatVariables = [3.5, 15]

        self.knownOptimum = np.ndarray(shape=(1), dtype=Trial)

        pointfv = [2.4656, 15]
        KOpoint = Point(pointfv, [])
        KOfunV = np.ndarray(shape=(1), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0] = self.Calculate(KOpoint, KOfunV[0])  # -195.37
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
            temp = x[1] - 1.275 * pow(x[0], 2) + 5.0 * x[0] - 6.0
            result = np.double(-pow(temp, 2) - 10.0 * (1 - 1 / (8.0 * math.pi)) * math.cos(math.pi * x[0]) - 10.0)
        elif functionValue.functionID == 0:  # constraint 1
            result = -math.pi * x[0] - x[1]
        elif functionValue.functionID == 1:  # constraint 2
            result = -pow(math.pi * x[0], 2) + 4.0 * x[1]

        functionValue.value = result
        return functionValue
