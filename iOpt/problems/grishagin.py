import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
import iOpt.problems.grishagin_generation as grishaginGen
#from iOpt.problems.hill_function import aHill
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
        self.fn = function_number

        self.floatVariableNames = np.ndarray(shape=(self.dimension), dtype=str)
        for i in range(self.dimension):
            self.floatVariableNames[i] = i

        self.lowerBoundOfFloatVariables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.lowerBoundOfFloatVariables.fill(0)
        self.upperBoundOfFloatVariables = np.ndarray(shape=(self.dimension), dtype=np.double)
        self.upperBoundOfFloatVariables.fill(1)

        self.knownOptimum = np.ndarray(shape=(1), dtype=Trial)

        pointfv = np.ndarray(shape=(self.dimension), dtype=np.double)
        pointfv[0] = hillGen.minHill[self.fn][1]
        KOpoint = Point(pointfv, [])
        KOfunV = np.ndarray(shape=(1), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0].value = hillGen.minHill[self.fn][0]
        self.knownOptimum[0] = Trial(KOpoint, KOfunV)

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        """Compute selected function at given point."""
        res: np.double = 0
        for i in range(hillGen.NUM_HILL_COEFF):
            res = res + hillGen.aHill[self.fn][i] * math.sin(2 * i * math.pi * point.floatVariables[0]) + hillGen.bHill[self.fn][i] * math.cos(2 * i * math.pi * point.floatVariables[0])
        functionValue.value = res
        return functionValue
