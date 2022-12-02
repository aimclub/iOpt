import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
import iOpt.problems.grishagin_generation as grishaginGen
import math

class Grishagin(Problem):
    """Base class for optimization problems"""

    def __init__(self, function_number: int):
        self.name = Grishagin
        self.dimension = 2
        self.numberOfFloatVariables = self.dimension
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 0
        if function_number < 1 or function_number > 100:
            function_number = 1
        self.fn = function_number
        self.icnf = np.ndarray(shape = (45))
        self.af = np.ndarray(shape=(7,7))
        self.bf = np.ndarray(shape=(7, 7))
        self.cf = np.ndarray(shape=(7, 7))
        self.df = np.ndarray(shape=(7, 7))

        self.floatVariableNames = np.ndarray(shape=(self.dimension), dtype=str)
        for i in range(self.dimension):
            self.floatVariableNames[i] = i

        self.lowerBoundOfFloatVariables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.lowerBoundOfFloatVariables.fill(0)
        self.upperBoundOfFloatVariables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.upperBoundOfFloatVariables.fill(1)

        self.knownOptimum = np.ndarray(shape=(1), dtype=Trial)

        pointfv = np.ndarray(shape=(self.dimension), dtype=np.double)
       # pointfv[0] = hillGen.minHill[self.fn][1]
        KOpoint = Point(pointfv, [])
        KOfunV = np.ndarray(shape=(1), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        #KOfunV[0].value = hillGen.minHill[self.fn][0]
        self.knownOptimum[0] = Trial(KOpoint, KOfunV)

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        """Compute selected function at given point."""
        res: np.double = 0
        #for i in range(hillGen.NUM_HILL_COEFF):
           # res = res + hillGen.aHill[self.fn][i] * math.sin(2 * i * math.pi * point.floatVariables[0]) + hillGen.bHill[self.fn][i] * math.cos(2 * i * math.pi * point.floatVariables[0])
        #functionValue.value = res
        return functionValue

    def rndm20(self, k: np.array) -> np.double:
        k1 = np.array([])
        for i in range(0, 38):
            k1[i] = k[i + 7]
        for i in range(38, 45):
            k1[i] = 0
        for i in range(0, 45):
            k[i] = abs(k[i] - k1[i])
        for i in range(27, 45):
            k1[i] = k[i - 27]
        for i in range(0, 27):
            k1[i] = 0

        self.gen(k, k1, 9, 44)
        self.gen(k, k1, 0, 8)

        rndm = 0.
        de2 = 1.
        for i in range(0, 36):
            de2 = de2 / 2
            rndm = rndm + k[i + 9] * de2

        return rndm

    def gen(k: np.array, k1: np.array, kap1: np.int, kap2: np.int):
        jct = 0
        for i in reversed(kap2, kap1):
            j = (k[i] + k1[i] + jct) / 2
            k[i] = k[i] + k1[i] + jct - j * 2
            jct = j
        if jct != 0:
            for i in reversed(kap2, kap1):
                j = (k[i] + jct) / 2
                k[i] = k[i] + jct - j * 2
                jct = j

    def SetFunctionNumber(self, value):
        if value < 1 or value > 100:
            value = 1
        self.fn = value
        self.icnf = np.array([])


        nf = value
        lst = 1
        i1 = (nf -1)/lst
        i2 = i1*lst
       # for i in range(0, 45):




