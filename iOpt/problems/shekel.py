import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
import iOpt.problems.Shekel.shekel_generation as shekelGen


class Shekel(Problem):
    """
    Функция Шекеля - это многомерная, мультимодальная, непрерывная, детерминированная функция, задана формулой:
       :math:`f(x) = \sum_{i=1}^{m}(c_{i}+(x-a_{i})^{2})^{-1}`,
       где :math:`m` – количество максимумов функции,
       :math:`a, c` - параметры, генерируемые случайным образом.
       В данном генераторе задача является одномерной.
    """

    def __init__(self, function_number: int):
        """
        Конструктор класса Shekel problem.

        :param functionNumber: номер задачи в наборе, :math:`1 <= functionNumber <= 1000`
        """
        super(Shekel, self).__init__()
        self.name = Shekel
        self.dimension = 1
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
        pointfv[0] = shekelGen.minShekel[self.fn][1]
        KOpoint = Point(pointfv, [])
        KOfunV = np.ndarray(shape=(1,), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0].value = shekelGen.minShekel[self.fn][0]
        self.knownOptimum[0] = Trial(KOpoint, KOfunV)

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        """
        Вычисление значения выбранной функции в заданной точке.

        :param point: координаты точки испытания, в которой будет вычислено значение функции
        :param functionValue: объект определяющий номер функции в задаче и хранящий значение функции
        :return: Вычисленное значение функции в точке point
        """
        res: np.double = 0
        for i in range(shekelGen.NUM_SHEKEL_COEFF):
            res = res - 1 / (
                        shekelGen.kShekel[self.fn][i] * pow(point.floatVariables[0] - shekelGen.aShekel[self.fn][i], 2)
                        + shekelGen.cShekel[self.fn][i])

        functionValue.value = res
        return functionValue
