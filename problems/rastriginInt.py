import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
import math


class RastriginInt(Problem):
    """
    Функция  Растригина задана формулой:
       :math:`f(y)=(\sum_{i=1}^{N}[x_{i}^{2}-10*cos(2\pi x_{i})])`,
       где :math:`x\in [-2.2, 1.8], N` – размерность задачи.
    """

    def __init__(self, dimension: int, numberOfDisreteVariables: int):
        """
        Конструктор класса Rastrigin problem.

        :param dimension: Размерность задачи.
        """
        super(RastriginInt, self).__init__()
        self.name = "RastriginInt"
        self.dimension = dimension
        self.numberOfFloatVariables = dimension - numberOfDisreteVariables
        self.numberOfDisreteVariables = numberOfDisreteVariables
        self.numberOfObjectives = 1
        self.numberOfConstraints = 0

        self.floatVariableNames = np.ndarray(shape=(self.numberOfFloatVariables), dtype=str)
        for i in range(self.numberOfFloatVariables):
            self.floatVariableNames[i] = i

        self.discreteVariableNames = np.ndarray(shape=(self.numberOfDisreteVariables), dtype=str)
        for i in range(self.numberOfDisreteVariables):
            self.discreteVariableNames[i] = i

        self.lowerBoundOfFloatVariables = np.ndarray(shape=(self.numberOfFloatVariables), dtype=np.double)
        self.lowerBoundOfFloatVariables.fill(-2.2)
        self.upperBoundOfFloatVariables = np.ndarray(shape=(self.numberOfFloatVariables), dtype=np.double)
        self.upperBoundOfFloatVariables.fill(1.8)

        self.discreteVariableValues = [["A", "B"] for i in range(self.numberOfDisreteVariables)]

        self.knownOptimum = np.ndarray(shape=(1), dtype=Trial)

        pointfv = np.ndarray(shape=(self.numberOfFloatVariables), dtype=np.double)
        pointfv.fill(0)

        pointdv = np.ndarray(shape=(self.numberOfDisreteVariables), dtype=str)
        pointdv.fill("B")

        KOpoint = Point(pointfv, pointdv)
        KOfunV = np.ndarray(shape=(1), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0].value = 0
        self.knownOptimum[0] = Trial(KOpoint, KOfunV)

        self.A = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.A.fill(-2.2)

        self.B = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.B.fill(1.8)

        self.optPoint = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.optPoint = np.append([[0] for i in range(self.numberOfFloatVariables)],
                                  [[1.8] for i in range(self.numberOfDisreteVariables)])

        self.multKoef = 0

        x = np.ndarray(shape=(self.dimension), dtype=np.double)
        count =math.pow(2, self.dimension)
        for i in range(int(count)):
            for j in range(self.dimension):
                x[j] = self.A[j] if (((i >> j) & 1) == 0) else self.B[j]
            v = abs(self.MultFunc(x))
            if v > self.multKoef:  self.multKoef = v

        self.multKoef += 4
        self.optMultKoef = (self.MultFunc(self.optPoint)+self.multKoef)

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        """
        Вычисление значения выбранной функции в заданной точке.

        :param point: координаты точки испытания, в которой будет вычислено значение функции
        :param functionValue: объект определяющий номер функции в задаче и хранящий значение функции
        :return: Вычисленное значение функции в точке point
        """
        sum: np.double = 0
        x = point.floatVariables
        for i in range(self.numberOfFloatVariables):
            sum += x[i] * x[i] - 10 * math.cos(2 * math.pi * x[i]) + 10

        dx = point.discreteVariables
        for i in range(self.numberOfDisreteVariables):
            if dx[i] == "A":
                sum += 2.2
            elif dx[i] == "B":
                sum -= 1.8
            else:
                raise ValueError

        x_arr = self.PointToArray(point)
        sum = sum * (self.MultFunc(x_arr)+self.multKoef)

        functionValue.value = sum
        return functionValue

    def PointToArray(self, point: Point) -> np.ndarray:
        arr = np.ndarray(shape=(self.dimension), dtype=np.double)

        for i in range(0, self.numberOfFloatVariables):
            arr[i] = point.floatVariables[i]

        for i in range(0, self.numberOfDisreteVariables):
            if point.discreteVariables[i] == "A":
                arr[self.numberOfFloatVariables+i] = -2.2
            elif point.discreteVariables[i] == "B":
                arr[self.numberOfFloatVariables+i] = 1.8
        return arr

    def MultFunc(self, x: np.ndarray) -> np.double:
        result: np.double = 0
        a: np.double
        d: np.double

        for i in range(self.dimension):
            d = (self.B[i]-self.A[i])/2
            a = (x[i]-self.optPoint[i])/d
            a = np.double(a * a)
            result = np.double(result + a)
        result = np.double(- result)
        return result

# if __name__ == "__main__":
#     problem = RastriginInt(3, 2)
#     print(problem.A)
#     print(problem.B)
#     print(problem.optPoint)
#     print(problem.optMultKoef)
#     print(problem.multKoef)
#     point = Point([-0.2], ["A", "A"])
#     functionValue = FunctionValue()
#     functionValue = problem.Calculate(point, functionValue)
#     print(functionValue.value)