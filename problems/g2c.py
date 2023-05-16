import numpy as np
from iOpt.trial import FunctionType
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
import math


class g2c(Problem):
    def __init__(self):
        """
        Конструктор класса gC2 problem.
        """
        super(g2c, self).__init__()
        self.name = "g2c"
        self.dimension: int = 20
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

        # Optimum is UNDEFINED

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        """
        Вычисление значения выбранной функции в заданной точке.

        :param point: координаты точки испытания, в которой будет вычислено значение функции
        :param functionValue: объект определяющий номер функции в задаче и хранящий значение функции
        :return: Вычисленное значение функции в точке point
        """
        result: np.double = 0
        x = point.floatVariables
        sum1 = 0
        sum2 = 0
        prod = 1

        if functionValue.type == FunctionType.OBJECTIV:
            for i in range(0, self.dimension):
                sum1 += pow(math.cos(x[i]), 4)
                sum2 += (i + 1) * pow(x[i], 2)
                prod = prod * pow(x[i], 2)
            result = - abs((sum1 - 2 * prod) / math.sqrt(sum2))
        elif functionValue.functionID == 0:  # constraint 1
            for i in range(0, self.dimension):
                prod = prod * x[i]
            result = np.double(-prod + 0.75)
        elif functionValue.functionID == 1:  # constraint 2
            for i in range(0, self.dimension):
                sum1 += x[i]
            result = np.double(sum1 - 7.5*self.dimension)

        functionValue.value = result
        return functionValue
