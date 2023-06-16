import numpy as np
from iOpt.trial import FunctionType
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem

a = np.array([0.5, 0.25, 1.0, 1.0 / 12.0, 2])
c = np.array([0.125, 0.25, 0.1, 0.2, 1.0 / 12.0])
p = np.array([[0, 5], [2, 5], [3, 2], [4, 4], [5, 1]])


class Romeijn3c(Problem):
    def __init__(self):
        """
        Конструктор класса Romeijn3c problem.
        """
        super(Romeijn3c, self).__init__()
        self.name = "Romeijn3c"
        self.dimension: int = 2
        self.numberOfFloatVariables = self.dimension
        self.numberOfDiscreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 3

        self.floatVariableNames = np.ndarray(shape=(self.dimension), dtype=str)
        for i in range(self.dimension):
            self.floatVariableNames[i] = i

        self.lowerBoundOfFloatVariables = np.ndarray(shape=(1, self.dimension), dtype=np.double)
        self.lowerBoundOfFloatVariables = [-3, -4]
        self.upperBoundOfFloatVariables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.upperBoundOfFloatVariables = [10, 7]

        self.knownOptimum = np.ndarray(shape=(1), dtype=Trial)

        pointfv = [-3, -4]
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
            for i in range(0, 5):
                temp = 0
                for j in range(0, self.dimension):
                    temp += pow(x[j] - p[i][j], 2)
                temp = a[i] * temp + c[i]
                result += 1 / temp
        elif functionValue.functionID == 0:  # constraint 1
            result = np.double(x[0] + x[1] - 5)
        elif functionValue.functionID == 1:  # constraint 2
            result = np.double(x[0] - pow(x[1], 2))
        elif functionValue.functionID == 2:  # constraint 3
            result = np.double(5 * pow(x[0], 3) - 8 / 5 * pow(x[1], 2))

        functionValue.value = result
        return functionValue
