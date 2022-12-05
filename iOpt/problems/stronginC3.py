import math

from iOpt.problem import Problem
from iOpt.trial import FunctionType, FunctionValue, Point, Trial


class StronginC3(Problem):
    """Base class for optimization problems"""

    def __init__(self) -> None:
        self.dimension: int = 2
        self.numberOfFloatVariables = self.dimension
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 3

        self.floatVariableNames = [str(x) for x in range(self.dimension)]
        self.lowerBoundOfFloatVariables = [0, -1]
        self.upperBoundOfFloatVariables = [4, 3]

        KOpoint = Point([0.941176, 0.941176], [])
        KOfunV = FunctionValue()
        KOfunV.value = -1.489444
        self.knownOptimum = [Trial(KOpoint, [KOfunV])]

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        """Compute selected function at given point."""
        res = 0.
        x1: float = point.floatVariables[0]
        x2: float = point.floatVariables[1]

        if functionValue.type == FunctionType.OBJECTIV:
            t1: float = pow(0.5 * x1 - 0.5, 4.0)
            t2: float = pow(x2 - 1.0, 4.0)
            res = 1.5 * x1 * x1 * math.exp(1.0 - x1 * x1 - 20.25 * (x1 - x2) * (x1 - x2))
            res = res + t1 * t2 * math.exp(2.0 - t1 - t2)
            res = -res
        elif functionValue.functionID == 0:  # constraint 1
            res = 0.01 * ((x1 - 2.2) * (x1 - 2.2) + (x2 - 1.2) * (x2 - 1.2) - 2.25)
        elif functionValue.functionID == 1:  # constraint 2
            res = 100.0 * (1.0 - ((x1 - 2.0) / 1.2) * ((x1 - 2.0) / 1.2) - (x2 / 2.0) * (x2 / 2.0))
        elif functionValue.functionID == 2:  # constraint 3
            res = 10.0 * (x2 - 1.5 - 1.5 * math.sin(6.283 * (x1 - 1.75)))

        functionValue.value = res
        return functionValue
