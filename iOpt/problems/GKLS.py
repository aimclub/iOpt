import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
from iOpt.problems.GKLS_function.gkls_random import GKLSRandomGenerator
from iOpt.problems.GKLS_function.gkls_function import T_GKLS_Minima
from iOpt.problems.GKLS_function.gkls_function import T_GKLS_GlobalMinima
from iOpt.problems.GKLS_function.gkls_function import GKLSClass
from iOpt.problems.GKLS_function.gkls_function import GKLSFuncionType
from iOpt.problems.GKLS_function.gkls_function import GKLSParameters
from iOpt.problems.GKLS_function.gkls_function import GKLSFunction

import math




class GKLS(Problem):
    """Base class for optimization problems"""

    def __init__(self, dimension: int, 
                 functionNumber: int=1):
        self.dimension = dimension
        self.numberOfFloatVariables = dimension
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 0

        self.floatVariableNames = np.ndarray(shape=(dimension), dtype=str)
        for i in range(self.dimension):
            self.floatVariableNames[i] = i

        self.lowerBoundOfFloatVariables = np.ndarray(shape=(dimension), dtype=np.double)
        self.lowerBoundOfFloatVariables.fill(-1)
        self.upperBoundOfFloatVariables = np.ndarray(shape=(dimension), dtype=np.double)
        self.upperBoundOfFloatVariables.fill(1)

        self.function = GKLSFunction()

        self.mMaxDimension = 50
        self.mMinDimension = 2

        self.function_number = functionNumber
        self.num_minima = 10
        
        self.problem_class = GKLSClass.Simple
        self.function_class = GKLSFuncionType.TD

        self.function.GKLS_global_value = -1.0
        self.function.NumberOfLocalMinima = self.num_minima
        self.function.SetDimension(self.dimension)
        self.function.mFunctionType = self.function_class


        self.function.SetFunctionClass(self.problem_class, self.dimension)

        self.global_dist = self.function.GKLS_global_dist
        self.global_radius = self.function.GKLS_global_radius

        if (self.function.GKLS_parameters_check() != GKLSFunction.GKLS_OK):
            return

        self.function.SetFunctionNumber(self.function_number)
        self.knownOptimum = np.ndarray(shape=(1), dtype=Trial)

        KOfunV = np.ndarray(shape=(1), dtype=FunctionValue)
        
        KOfunV[0] = FunctionValue()
        KOfunV[0].value = self.function.GetOptimumValue()
        

        pointfv = np.ndarray(shape=(dimension), dtype=np.double)
        pointfv.fill(0)
        pointfv = self.function.GetOptimumPoint(pointfv)
        KOpoint = Point(pointfv, [])

        self.knownOptimum[0] = Trial(KOpoint, KOfunV)

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        """Compute selected function at given point."""
        sum : np.double = 0        

        functionValue.value = self.function.Calculate(point.floatVariables)
        return functionValue

