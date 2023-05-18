import numpy as np
from iOpt.trial import FunctionType
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem


class Romeijn1c(Problem):
    def __init__(self):
        """
        Конструктор класса RomeijnC1 problem.
        """
        super(Romeijn1c, self).__init__()
        self.name = "Romeijn1c"
        self.dimension: int = 3
        self.numberOfFloatVariables = self.dimension
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 1

        self.floatVariableNames = np.ndarray(shape=(self.dimension), dtype=str)
        for i in range(self.dimension):
            self.floatVariableNames[i] = i

        self.lowerBoundOfFloatVariables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.lowerBoundOfFloatVariables.fill(0)
        self.upperBoundOfFloatVariables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.upperBoundOfFloatVariables = [0.18745, 0.16230, 0.42846]

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

        if functionValue.type == FunctionType.OBJECTIV:
            result = -1 / (pow(x[0], 2) * x[1] * x[2])
        elif functionValue.functionID == 0:  # constraint 1
            result = 0.44098 * x[0] + 28.46 * pow(x[0], 2) + 6158.4 * pow(x[0], 2) * x[1] + 0.0037018 * x[2] + \
                     5.4474 * pow(x[2], 2) + 0.032236 * x[0] * x[2] + 2.92 * x[1] * x[2] + 0.44712 * x[1] \
                     + 37.964 * pow(x[1], 2) + 42.876 * x[0] * x[1] - 1

        functionValue.value = result
        return functionValue
