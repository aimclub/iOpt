import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
import math


class RastriginIntHiddenConstraint(Problem):
    """
    The Rastrigin function is given by the formula:
       :math:`f(y)=(\sum_{i=1}^{N}[x_{i}^{2}-10*cos(2\pi x_{i})])`,
       where :math:`x\in [-2.2, 1.8], N` – problem dimensionality
    """

    def __init__(self, dimension: int, number_of_discrete_variables: int):
        """
        Constructor of the RastriginInt problem class

        :param dimension: problem dimensionality.
        """
        super(RastriginIntHiddenConstraint, self).__init__()
        self.name = "RastriginIntHiddenConstraint"
        self.dimension = dimension
        self.number_of_float_variables = dimension - number_of_discrete_variables
        self.number_of_discrete_variables = number_of_discrete_variables
        self.number_of_objectives = 1
        self.number_of_constraints = 0

        self.float_variable_names = np.ndarray(shape=(self.number_of_float_variables), dtype=object)
        for i in range(self.number_of_float_variables):
            self.float_variable_names[i] = str(i)

        self.discrete_variable_names = np.ndarray(shape=(self.number_of_discrete_variables), dtype=object)
        for i in range(self.number_of_discrete_variables):
            self.discrete_variable_names[i] = str(i)

        self.lower_bound_of_float_variables = np.ndarray(shape=(self.number_of_float_variables), dtype=np.double)
        self.lower_bound_of_float_variables.fill(-2.2)
        self.upper_bound_of_float_variables = np.ndarray(shape=(self.number_of_float_variables), dtype=np.double)
        self.upper_bound_of_float_variables.fill(1.8)

        self.discrete_variable_values = [["A", "B"] for i in range(self.number_of_discrete_variables)]

        self.known_optimum = np.ndarray(shape=(1), dtype=Trial)

        pointfv = np.ndarray(shape=(self.number_of_float_variables), dtype=np.double)
        pointfv.fill(0)

        pointdv = np.ndarray(shape=(self.number_of_discrete_variables), dtype=object)
        pointdv.fill("B")

        KOpoint = Point(pointfv, pointdv)
        KOfunV = np.ndarray(shape=(1), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0].value = 0
        self.known_optimum[0] = Trial(KOpoint, KOfunV)

        self.A = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.A.fill(-2.2)

        self.B = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.B.fill(1.8)

        self.optPoint = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.optPoint = np.append([[0] for i in range(self.number_of_float_variables)],
                                  [[1.8] for i in range(self.number_of_discrete_variables)])

        self.multKoef = 0

        x = np.ndarray(shape=(self.dimension), dtype=np.double)
        count = math.pow(2, self.dimension)
        for i in range(int(count)):
            for j in range(self.dimension):
                x[j] = self.A[j] if (((i >> j) & 1) == 0) else self.B[j]
            v = abs(self.mult_func(x))
            if v > self.multKoef:
                self.multKoef = v

        self.multKoef += 4
        self.optMultKoef = (self.mult_func(self.optPoint) + self.multKoef)

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Calculate the value of the selected function at a given point

        :param point: coordinates of the trial point where the value of the function will be calculated.
        :param function_value: object defining the function number in the task and storing the function value.
        :return: Calculated value of the function at point.
        """

        isInf: bool = True
        for i in range(self.number_of_float_variables):
            if point.float_variables[i] <= 0.5 or point.float_variables[i] > 1.5:
                isInf = False

        if isInf:
            raise Exception("Infinity values")

        sum: np.double = 0
        x = point.float_variables
        for i in range(self.number_of_float_variables):
            sum += x[i] * x[i] - 10 * math.cos(2 * math.pi * x[i]) + 10

        dx = point.discrete_variables
        for i in range(self.number_of_discrete_variables):
            if dx[i] == "A":
                sum += 2.2
            elif dx[i] == "B":
                sum -= 1.8
            else:
                raise ValueError

        x_arr = self.point_to_array(point)
        sum = sum * (self.mult_func(x_arr) + self.multKoef)

        function_value.value = sum
        return function_value

    def point_to_array(self, point: Point) -> np.ndarray:
        arr = np.ndarray(shape=(self.dimension), dtype=np.double)

        for i in range(0, self.number_of_float_variables):
            arr[i] = point.float_variables[i]

        for i in range(0, self.number_of_discrete_variables):
            if point.discrete_variables[i] == "A":
                arr[self.number_of_float_variables+i] = -2.2
            elif point.discrete_variables[i] == "B":
                arr[self.number_of_float_variables+i] = 1.8
        return arr

    def mult_func(self, x: np.ndarray) -> np.double:
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
