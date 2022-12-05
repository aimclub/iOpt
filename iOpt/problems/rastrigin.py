import math

from iOpt.problem import Problem
from iOpt.trial import FunctionValue, Point, Trial


class Rastrigin(Problem):
    """Base class for optimization problems"""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.numberOfFloatVariables = dimension
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 0

        self.floatVariableNames = [str(x) for x in range(dimension)]

        self.lowerBoundOfFloatVariables = dimension * [-2.2]
        self.upperBoundOfFloatVariables = dimension * [1.8]

        KOpoint = Point(dimension * [0.], [])
        KOfunV = FunctionValue()
        KOfunV.value = 0
        self.knownOptimum = [Trial(KOpoint, [KOfunV])]

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        """Compute selected function at given point."""
        functionValue.value = sum(x * x - 10 * math.cos(2 * math.pi * x) + 10 for x in point.floatVariables)
        return functionValue
