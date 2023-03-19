import numpy as np
from iOpt.trial import FunctionType
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
import math


class StronginC5(Problem):
    def __init__(self):
        """
        Конструктор класса StronginC5 problem.
        """
        super(StronginC5, self).__init__()
        self.name = 'StronginC5'
        self.dimension: int = 5
        self.numberOfFloatVariables = self.dimension
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 5

        self.floatVariableNames = np.ndarray(shape=(self.dimension), dtype=str)
        for i in range(self.dimension):
            self.floatVariableNames[i] = i

        self.lowerBoundOfFloatVariables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.lowerBoundOfFloatVariables.fill(-3)
        self.lowerBoundOfFloatVariables[3] = -10
        self.lowerBoundOfFloatVariables[4] = -10
        self.upperBoundOfFloatVariables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.upperBoundOfFloatVariables.fill(3)
        self.upperBoundOfFloatVariables[3] = 10
        self.upperBoundOfFloatVariables[4] = 10

        self.knownOptimum = np.ndarray(shape=(1), dtype=Trial)

        pointfv = np.ndarray(shape=(self.dimension), dtype=np.double)
        pointfv[0] = -0.0679
        pointfv[1] = 1.9434
        pointfv[2] = 2.4512
        pointfv[3] = 9.9013
        pointfv[4] = 9.9008
        KOpoint = Point(pointfv, [])
        KOfunV = np.ndarray(shape=(1), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0] = self.Calculate(KOpoint, KOfunV[0]) # -43.298677;
        self.knownOptimum[0] = Trial(KOpoint, KOfunV)

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        """
        Вычисление значения выбранной функции в заданной точке.

        :param point: координаты точки испытания, в которой будет вычислено значение функции
        :param functionValue: объект определяющий номер функции в задаче и хранящий значение функции
        :return: Вычисленное значение функции в точке point
        """
        res: np.double = 0
        x = point.floatVariables

        if functionValue.type == FunctionType.OBJECTIV:
            res = np.double(math.sin(x[0] * x[2]) - (x[1] * x[4] + x[2] * x[3])*math.cos(x[0] * x[1]))
        elif functionValue.functionID == 0:  # constraint 1
            res = np.double(-(x[0] + x[1] + x[2] + x[3] + x[4]))
        elif functionValue.functionID == 1:  # constraint 2
            res = np.double(x[1] * x[1] / 9 + x[3] * x[3] / 100 - 1.4)
        elif functionValue.functionID == 2:  # constraint 3
            res = np.double(3 - pow(x[0] + 1, 2) - pow(x[1] + 2, 2) - pow(x[2] - 2, 2) - pow(x[4] + 5, 2))
        elif functionValue.functionID == 3:  # constraint 4
            res = np.double(4 * x[0] * x[0] * math.sin(x[0]) + x[1] * x[1] * math.cos(x[1] + x[3]) +
                            x[2] * x[2] * (math.sin(x[2] + x[4]) + math.sin(10 * (x[2] - x[3]) / 3)) - 4)
        elif functionValue.functionID == 4:  # constraint 5
            res = np.double(x[0] * x[0] + x[1] * x[1] * pow(math.sin((x[0] + x[3]) / 3 + 6.6) +
                                                            math.sin((x[1] + x[4]) / 2 + 0.9), 2)
                            - 17 * pow(math.cos(x[0] + x[2] + 1), 2) + 16)

        functionValue.value = res
        return functionValue

