import numpy as np
from iOpt.trial import FunctionType
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem


class Romeijn2c(Problem):
    def __init__(self):
        """
        Конструктор класса Romeijn2c problem.
        """
        super(Romeijn2c, self).__init__()
        self.name = "Romeijn2c"
        self.dimension: int = 6
        self.numberOfFloatVariables = self.dimension
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 2

        self.floatVariableNames = np.ndarray(shape=(self.dimension), dtype=str)
        for i in range(self.dimension):
            self.floatVariableNames[i] = i

        self.lowerBoundOfFloatVariables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.lowerBoundOfFloatVariables.fill(0)
        self.upperBoundOfFloatVariables = np.ndarray( shape=(self.dimension), dtype=np.double)
        self.upperBoundOfFloatVariables = [10, 10, 15, 15, 1, 1]

        self.knownOptimum = np.ndarray(shape=(1), dtype=Trial)


        pointfv = [10, 10, 15, 4.609, 0.78511, 0.384]
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
        result: np.double = 1
        x = point.floatVariables

        if functionValue.type == FunctionType.OBJECTIV:
            result = np.double(-(0.0204 + 0.0607 * pow(x[4], 2)) * x[0] * x[3] * (x[0] + x[1] + x[2]) - \
                               (0.0187 + 0.0437 * pow(x[5], 2)) * x[1] * x[2] * (x[0] + 1.57 * x[1] + x[3]))
        elif functionValue.functionID == 0:  # constraint 1
            for i in range(0, self.dimension):
                result = result / x[i]
            result = np.double(2070 * result - 1)

        elif functionValue.functionID == 1:  # constraint 2
            result = np.double(0.00062 * x[0] * x[3] * pow(x[4], 2) * (x[0] + x[1] + x[2]) + \
                               0.00058 * x[1] * x[2] * pow(x[5], 2) * (x[0] + 1.57 * x[1] + x[3]) - 1)

        functionValue.value = result
        return functionValue
