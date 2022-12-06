import numpy as np

from iOpt.problem import Problem
from iOpt.problems.GKLS_function.gkls_function import GKLSClass, GKLSFuncionType, GKLSFunction
from iOpt.trial import FunctionValue, Point, Trial


class GKLS(Problem):
    """Base class for optimization problems"""

    def __init__(self, dimension: int,
                 functionNumber: int = 1):
        self.dimension = dimension
        self.numberOfFloatVariables = dimension
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 0

        self.floatVariableNames = [str(i) for i in range(dimension)]
        self.lowerBoundOfFloatVariables = dimension * [-1.]
        self.upperBoundOfFloatVariables = dimension * [1.]

        self.function = GKLSFunction()

        self.mMaxDimension = 50
        self.mMinDimension = 2

        self.function_number = functionNumber
        self.num_minima = 10

        self.problem_class = GKLSClass.Simple
        self.function_class = GKLSFuncionType.TD

        self.function.GKLS_global_value = -1.0
        self.function.SetDimension(self.dimension)
        self.function.mFunctionType = self.function_class

        self.function.SetFunctionClass(self.problem_class, self.dimension)

        self.global_dist = self.function.GKLS_global_dist
        self.global_radius = self.function.GKLS_global_radius

        if (self.function.GKLS_parameters_check() != GKLSFunction.GKLS_OK):
            return

        self.function.SetFunctionNumber(self.function_number)

        KOfunV = FunctionValue()
        KOfunV.value = self.function.GetOptimumValue()

        pointfv = self.function.GetOptimumPoint()
        KOpoint = Point(pointfv, [])
        self.knownOptimum = [Trial(KOpoint, [KOfunV])]

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        """Compute selected function at given point."""
        functionValue.value = self.function.Calculate(np.array(point.floatVariables))
        return functionValue
